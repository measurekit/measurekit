import {
  bfgs,
  checkArray,
  checkFinite,
  checkOptions,
  checkStopping,
  DataError,
  describe,
  type FieldNames,
  logistic as sigmoid,
  type ValueAndGradient,
} from "@measurekit/core";
import type { EventLog } from "../data/internal.js";
import { checkModel } from "../model.js";
import type { LogisticFit, LogisticModel } from "./logistic.js";

/** Options of {@link searchRates}. */
export interface RateSearchOptions {
  /** The rates to start from, each strictly between its bounds: one per rate the model builder takes. */
  readonly start: readonly number[];
  /** The least value of each rate. */
  readonly lower: readonly number[];
  /** The greatest value of each rate, above its least. */
  readonly upper: readonly number[];
  /**
   * The largest absolute component of the gradient of the log-likelihood, over the rates mapped to the whole line, at
   * which the search has converged.
   *
   * @defaultValue `1e-6`
   */
  readonly tolerance?: number;
  /**
   * The most iterations of the search; each costs a fit for each length its line search tries, and the two fits for each
   * rate of the gradient at the length it accepts.
   *
   * @defaultValue `100`
   */
  readonly maxIterations?: number;
}

const RATE_SEARCH_OPTIONS: FieldNames<RateSearchOptions> = {
  start: true,
  lower: true,
  upper: true,
  tolerance: true,
  maxIterations: true,
};

/** The outcome of {@link searchRates}. */
export interface RateSearch {
  /** The rates that maximize the log-likelihood, found by the search. */
  readonly rates: readonly number[];
  /** The fit of the model built from `rates`. */
  readonly fit: LogisticFit;
  /** Whether the gradient reached the tolerance, or no step could improve the log-likelihood. */
  readonly converged: boolean;
  /** Iterations of the search. */
  readonly iterations: number;
}

const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_MAX_ITERATIONS = 100;

/**
 * The step of the central differences that give the gradient, on the transformed rates: with the error of the
 * differences at about the square of it and the rounding of a fit's log-likelihood divided by it, both near 1e-8.
 */
const DIFFERENCE_STEP = 1e-4;

/**
 * Finds the rates that maximize a model's log-likelihood on a log, by BFGS over the rates mapped to the whole line.
 *
 * @remarks
 * `build` makes the model from the rates, such as the decays and exponents of its features, and each trial fits it to
 * the log, so the cost is the cost of a fit times the trials. The line search fits the model at each length it tries,
 * and the gradient, by central differences, takes two fits for each rate at the length it accepts alone. A rate between
 * its bounds is mapped to the line by the logit of its place between them, so the search never leaves the bounds. The
 * result may be a local optimum: the log-likelihood in the rates need not have one maximum, so a search from other starts
 * can end elsewhere.
 *
 * A fit that does not converge counts as a failed trial, the search steps back from it, as from a length where a fit of
 * the gradient fails, and a search that ends on failed trials has not converged. An error that `build` or the fit throws
 * ends the search with it.
 *
 * @param build - Builds the model for a list of rates, in the order of `start`.
 * @param log - The log to fit each model to.
 * @param options - The starting rates, their bounds and the stopping rules.
 * @returns The best rates, the fit of the model built from them, whether the search converged and the iterations run.
 * @throws `DataError` when the options are out of range, `build` is not a function or does not return a model, or the
 * fit of the model built from `start` is not valid for the log.
 * @see {@link https://doi.org/10.1109/TLT.2021.3113067 | Pavlik, Eglington and Harrell-Williams (2021)}
 */
export function searchRates(
  build: (rates: readonly number[]) => LogisticModel,
  log: EventLog,
  options: RateSearchOptions,
): RateSearch {
  checkOptions(options, RATE_SEARCH_OPTIONS);
  if (typeof build !== "function") {
    throw new DataError(
      `The argument build must be a function from the rates to a logistic model, found ${describe(build)}.`,
    );
  }
  const { start, lower, upper, tolerance = DEFAULT_TOLERANCE, maxIterations = DEFAULT_MAX_ITERATIONS } = options;
  checkStopping(tolerance, maxIterations);
  checkBounds(start, lower, upper);
  const toRates = (x: ArrayLike<number>): number[] =>
    lower.map((least, i) => least + ((upper[i] ?? 0) - least) * sigmoid(x[i] ?? 0));
  const trial = (x: ArrayLike<number>): { fit: LogisticFit; value: number } => {
    const model = build(toRates(x));
    checkModel(model, "The function build must return a logistic model");
    const fit = model.fit(log);
    return {
      fit,
      value: fit.converged && Number.isFinite(fit.logLikelihood) ? -fit.logLikelihood : Number.POSITIVE_INFINITY,
    };
  };
  let lastFailed = false;
  const negativeLogLikelihood: ValueAndGradient = {
    value: (x) => {
      const { value } = trial(x);
      lastFailed = !Number.isFinite(value);
      return value;
    },
    gradient: (x, gradient) => {
      const moved = Float64Array.from(x);
      for (let i = 0; i < x.length; i++) {
        const centre = x[i] ?? 0;
        moved[i] = centre + DIFFERENCE_STEP;
        const up = trial(moved).value;
        moved[i] = centre - DIFFERENCE_STEP;
        const down = trial(moved).value;
        moved[i] = centre;
        // A failed fit leaves the gradient undefined, which rejects the length the line search tried.
        if (!(Number.isFinite(up) && Number.isFinite(down))) {
          lastFailed = true;
          gradient.fill(Number.NaN);
          return;
        }
        gradient[i] = (up - down) / (2 * DIFFERENCE_STEP);
      }
    },
  };
  const origin = Float64Array.from(start, (rate, i) => {
    const place = (rate - (lower[i] ?? 0)) / ((upper[i] ?? 0) - (lower[i] ?? 0));
    return Math.log(place / (1 - place));
  });
  const first = trial(origin);
  if (!Number.isFinite(first.value)) {
    return { rates: [...start], fit: first.fit, converged: false, iterations: 0 };
  }
  const found = bfgs(negativeLogLikelihood, origin, { tolerance, maxIterations });
  const final = trial(found.x);
  return {
    rates: toRates(found.x),
    fit: final.fit,
    converged: found.converged && !lastFailed && Number.isFinite(final.value),
    iterations: found.iterations,
  };
}

/** Checks the starts and bounds of the rates: lists of one finite number per rate, each start inside its bounds. */
function checkBounds(start: readonly number[], lower: readonly number[], upper: readonly number[]): void {
  for (const [name, values] of [
    ["start", start],
    ["lower", lower],
    ["upper", upper],
  ] as const) {
    checkArray(values, `The option ${name} must be an array of rates`);
    if (values.length === 0 || values.length !== start.length) {
      throw new DataError(
        `The option ${name} must hold one number for each rate, as many as the ${start.length} of start, found ${values.length}.`,
      );
    }
    values.forEach((value, i) => {
      checkFinite(value, `The ${name} of rate ${i + 1}`);
    });
  }
  start.forEach((rate, i) => {
    const least = lower[i] ?? 0;
    const greatest = upper[i] ?? 0;
    if (!(least < rate && rate < greatest)) {
      throw new DataError(
        `The start of rate ${i + 1} must lie strictly between its bounds ${least} and ${greatest}, found ${rate}.`,
      );
    }
  });
}

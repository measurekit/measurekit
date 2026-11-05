import {
  checkChoice,
  checkIntegerFrom,
  checkOptions,
  checkStopping,
  DataError,
  describe,
  type FieldNames,
} from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import {
  checkStandardErrorSize,
  gradedResponse,
  type IrtModel,
  maximization,
  oneParameterLogistic,
  STANDARD_ERROR_METHODS,
  type StandardErrorMethod,
  setUp,
  type TestItem,
  twoParameterLogistic,
} from "../irt/internal.js";
import { rectangular } from "../math/internal.js";
import { EM_MAX_ITERATIONS, EM_TOLERANCE } from "./calibration.js";
import { latentDistribution } from "./correlation.js";
import { productGrid } from "./grid.js";
import { arrangeOccasions } from "./longitudinal-data.js";
import { expectationMaximization, type LongitudinalFit } from "./longitudinal-em.js";
import { standardErrorsOf } from "./longitudinal-errors.js";
import { occasionExpectation } from "./longitudinal-expectation.js";

/** Options of {@link calibrateLongitudinal}. */
export interface LongitudinalOptions {
  /**
   * The model of every item: `oneParameterLogistic`, `twoParameterLogistic` or `gradedResponse`.
   *
   * @defaultValue `twoParameterLogistic`
   */
  readonly model?: IrtModel;
  /**
   * The number of nodes of each occasion's axis of the grid, at least 2, from -6 to 6.
   *
   * @defaultValue `31` for 2 occasions, `21` for 3
   */
  readonly points?: number;
  /**
   * The method of the standard errors. `"hessian"` runs two E-steps per estimated parameter; `"cross-product"` one pass
   * over the persons.
   *
   * @defaultValue `"none"`
   */
  readonly standardErrors?: StandardErrorMethod;
  /**
   * EM has converged when no estimated parameter changes by more than this in an iteration.
   *
   * @defaultValue `1e-4`
   */
  readonly tolerance?: number;
  /**
   * The most EM iterations to run.
   *
   * @defaultValue `500`
   */
  readonly maxIterations?: number;
}

const LONGITUDINAL_OPTIONS: FieldNames<LongitudinalOptions> = {
  model: true,
  points: true,
  standardErrors: true,
  tolerance: true,
  maxIterations: true,
};

/** The distribution of the trait on one occasion. */
export interface OccasionMoments {
  /** The mean of the trait. */
  readonly mean: number;
  /** The standard deviation of the trait. */
  readonly sd: number;
}

/** The change of the trait between two consecutive occasions. */
export interface OccasionChange {
  /** The index of the earlier occasion in `occasions`, from 0. */
  readonly from: number;
  /** The index of the later occasion. */
  readonly to: number;
  /** The mean of the change: the later mean less the earlier. */
  readonly mean: number;
  /** The standard deviation of the change. */
  readonly sd: number;
  /** The correlation of the change with the trait at the earlier occasion. */
  readonly correlationWithStart: number;
}

/** Items and the distribution of one trait per occasion, estimated together from the occasions' responses. */
export interface LongitudinalCalibration {
  /** The items, in order of first appearance across the occasions, with the parameters they have on every occasion. */
  readonly items: readonly TestItem[];
  /** The mean and standard deviation of each occasion; the first is 0 and 1. */
  readonly occasions: readonly OccasionMoments[];
  /** The correlations between the occasions, by rows. */
  readonly correlations: readonly (readonly number[])[];
  /** The change between each pair of consecutive occasions: its mean, its standard deviation and its correlation with the start. */
  readonly changes: readonly OccasionChange[];
  /**
   * The standard errors of the estimated parameters on their own scale: the items' parameters in item order, in the
   * order of their model's parameters and with a parameter that the items share once, at its first item; then the
   * correlations of the pairs (1, 0), (2, 0), (2, 1) of occasions, the standard deviations of the occasions after the
   * first and their means. `null` without the option `standardErrors`, and when the information cannot be inverted or
   * gives a variance that is not positive.
   */
  readonly standardErrors: Float64Array | null;
  /** The marginal log-likelihood at the estimates. */
  readonly logLikelihood: number;
  /** Whether the calibration reached `tolerance` within `maxIterations`, with every occasion's distribution on the grid. */
  readonly converged: boolean;
  /** EM iterations run. */
  readonly iterations: number;
}

/** The nodes per axis that 2 and 3 occasions default to: mirt's 31 for two dimensions, and 21 for three. */
const DEFAULT_POINTS: readonly number[] = [31, 21];

/**
 * Estimates the item parameters and the distribution of a trait that changes over 2 or 3 occasions, by marginal
 * maximum likelihood.
 *
 * @remarks
 * The same persons answer the same items on every occasion, or some of them: a person is the row with the same id in
 * each occasion's matrix, an item the column with the same id, and an item has one set of parameters whatever the
 * occasion. A person who took part in some occasions only, and an item that only some occasions have, are allowed. The
 * trait of occasion t is dimension t of a multivariate normal distribution whose means, standard deviations and
 * correlations are estimated, except that the first occasion is N(0, 1) and sets the scale.
 *
 * EM over a grid of `points` nodes per occasion from -6 to 6, weighted by the normal density of the current moments, as
 * `calibrateMultidimensional`. A person's likelihood at a node is the product of the occasions' likelihoods at the
 * node's coordinates, each computed once per axis. The M-step takes a Fisher-scoring step for the items on their
 * expected counts pooled over the occasions, as `calibrate`, and maximizes the moments by BFGS on the weights of the
 * grid. SQUAREM accelerates the iterations.
 *
 * A distribution that runs to the edge of the grid, a mean outside it or a standard deviation below the spacing of its
 * nodes or above its range, leaves `converged` `false`. `changes` gives the moments of the change between consecutive
 * occasions, which determine Embretson's initial ability and modifiability.
 *
 * @param occasions - The responses on each occasion, 2 or 3 matrices of the same persons and items matched by id.
 * @param options - The model, the quadrature, the standard errors and the stopping rules; each option has its default.
 * @returns The items with their parameters, the moments of each occasion, the correlations of the occasions, the change
 *   between occasions, the standard errors, the log-likelihood, whether the estimation converged and the iterations
 *   run.
 * @throws `DataError` when `occasions` is not an array of 2 or 3 response matrices, a matrix is malformed, there are no
 * responses, an occasion shares no item with the first, directly or through the others, an item has responses in fewer
 * than two categories, a model other than the three allowed is given or an option is invalid, or the grid has more than
 * 10⁶ nodes.
 * @see {@link https://doi.org/10.1007/BF02294143 | Andersen (1985)}
 * @see {@link https://doi.org/10.1007/BF02294487 | Embretson (1991)}
 */
export function calibrateLongitudinal(
  occasions: readonly ResponseMatrix[],
  options: LongitudinalOptions = {},
): LongitudinalCalibration {
  checkOptions(options, LONGITUDINAL_OPTIONS);
  const data = arrangeOccasions(occasions);
  const { model = twoParameterLogistic, standardErrors = "none" } = options;
  const {
    points = DEFAULT_POINTS[data.occasions.length - 2] ?? 21,
    tolerance = EM_TOLERANCE,
    maxIterations = EM_MAX_ITERATIONS,
  } = options;
  checkModel(model);
  checkChoice(standardErrors, "The option standardErrors", STANDARD_ERROR_METHODS);
  checkIntegerFrom(points, "The option points", 2);
  checkStopping(tolerance, maxIterations);
  const grid = productGrid(data.occasions.length, points);
  const rule = rectangular(points);
  const answered = Array.from(data.pooled.offsets.subarray(1), (end, row) => end > (data.pooled.offsets[row] ?? 0));
  const setup = setUp(data.pooled, { model }, answered, rule);
  const latent = latentDistribution(grid, { correlations: true, variances: true, means: true, pinned: [0] });
  if (standardErrors !== "none") checkStandardErrorSize(setup.values.length + latent.size);
  const fit: LongitudinalFit<unknown> = {
    setup,
    maximization: maximization(setup, rule.nodes),
    latent,
    eStep: occasionExpectation(data.occasions, setup.items, grid, latent.logWeights),
    grid,
    axis: rule.nodes,
    occasions: data.occasions,
  };
  const { converged, iterations } = expectationMaximization(fit, tolerance, maxIterations);
  const logLikelihood = fit.eStep.run();
  return {
    items: setup.items.map(({ id, model: itemModel, parameters }) => ({ id, model: itemModel, parameters })),
    ...momentsOf(latent.mean, latent.matrix),
    standardErrors: standardErrorsOf(standardErrors, fit),
    logLikelihood,
    converged,
    iterations,
  };
}

/**
 * Checks the model against the three that the calibration validates.
 *
 * @throws `DataError` when it is another.
 */
function checkModel(model: unknown): void {
  if (model !== oneParameterLogistic && model !== twoParameterLogistic && model !== gradedResponse) {
    throw new DataError(
      `The option model must be oneParameterLogistic, twoParameterLogistic or gradedResponse, found ${describe(model)}.`,
    );
  }
}

/** Returns the means, standard deviations, correlations and changes of consecutive occasions of a distribution. */
function momentsOf(
  mean: Float64Array,
  covariance: Float64Array,
): Pick<LongitudinalCalibration, "occasions" | "correlations" | "changes"> {
  const size = mean.length;
  const sd = Array.from(mean, (_, t) => Math.sqrt(covariance[t * size + t] ?? 1));
  const at = (i: number, j: number): number => covariance[i * size + j] ?? 0;
  return {
    occasions: Array.from(mean, (value, t) => ({ mean: value, sd: sd[t] ?? 1 })),
    correlations: Array.from(mean, (_, i) =>
      Array.from(mean, (__, j) => (i === j ? 1 : at(i, j) / ((sd[i] ?? 1) * (sd[j] ?? 1)))),
    ),
    changes: Array.from({ length: size - 1 }, (_, from) => {
      const to = from + 1;
      const spread = Math.sqrt(at(from, from) + at(to, to) - 2 * at(from, to));
      return {
        from,
        to,
        mean: (mean[to] ?? 0) - (mean[from] ?? 0),
        sd: spread,
        correlationWithStart: (at(from, to) - at(from, from)) / ((sd[from] ?? 1) * spread),
      };
    }),
  };
}

import {
  checkArray,
  checkFinite,
  checkOptions,
  DataError,
  describe,
  type FieldNames,
  isRecord,
  normalQuantile,
} from "@measurekit/core";
import { checkOpenUnitInterval } from "../checks.js";
import { largestCategoryCount, logProbability } from "../irt/internal.js";
import type { PoolItem } from "./items.js";
import type { AdaptiveTestContext } from "./selection.js";
import type { StoppingRule } from "./stopping.js";

/** The two-sided level of catR's classification rule, 1 - alpha with its default alpha of 0.05. */
const DEFAULT_LEVEL = 0.95;
/** The error rates of mirtCAT's SPRT. */
const DEFAULT_ALPHA = 0.05;
const DEFAULT_BETA = 0.05;

/** What a classification reads of an adaptive test: its pool, and the responses so far with the estimate from them. */
export type ClassificationContext = Pick<AdaptiveTestContext, "pool" | "administered" | "responses" | "estimate">;

/**
 * A stopping rule that classifies the person into categories separated by cuts on theta: it stops once it has decided
 * the category.
 */
export interface ClassificationRule extends StoppingRule {
  /**
   * Returns the category the rule has decided, from 0, below every cut, to the number of cuts, above every one; `null`
   * while a cut is undecided.
   */
  classify(context: ClassificationContext): number | null;
}

/** Options of {@link classificationThreshold}. */
export interface ClassificationOptions {
  /**
   * The two-sided level of the interval around the estimate, above 0 and below 1.
   *
   * @defaultValue `0.95`
   */
  readonly level?: number;
}

const CLASSIFICATION_OPTIONS: FieldNames<ClassificationOptions> = { level: true };

/**
 * Returns a rule that stops once the confidence interval of the estimate clears every cut, as catR's `classification`
 * rule does for one cut.
 *
 * @remarks
 * The interval is θ̂ ± z SE, z the normal quantile of 1 - (1 - level) / 2. A cut is decided when the interval's lower end
 * is at or above it, or its upper end at or below it; the rule stops when every cut is decided, and the category is the
 * number of cuts at or below the lower end. An estimate whose standard error is infinite decides no cut.
 *
 * @param cuts - The cuts on theta, finite and increasing: one for two categories, two for three.
 * @param options - The level of the interval.
 * @returns The rule, which also classifies.
 * @throws `DataError` when the cuts are not an array of increasing finite numbers, at least one, or the level is not
 * between 0 and 1.
 * @see {@link https://doi.org/10.1007/978-0-387-85461-8_19 | Eggen (2010)}
 */
export function classificationThreshold(
  cuts: readonly number[],
  options: ClassificationOptions = {},
): ClassificationRule {
  const checked = checkCuts(cuts, (cut, i) => {
    checkFinite(cut, `Cut ${i + 1}`);
    return [cut, cut];
  }).map(([cut]) => cut);
  checkOptions(options, CLASSIFICATION_OPTIONS);
  const { level = DEFAULT_LEVEL } = options;
  checkOpenUnitInterval(level, "The option level");
  // catR's z: the quantile of 1 - alpha / 2, alpha = 1 - level.
  const z = normalQuantile(1 - (1 - level) / 2);
  const classify = ({ estimate: { theta, standardError } }: ClassificationContext): number | null => {
    const lower = theta - z * standardError;
    const upper = theta + z * standardError;
    let category = 0;
    for (const cut of checked) {
      if (lower >= cut) category++;
      else if (!(upper <= cut)) return null;
    }
    return category;
  };
  return { stops: (context) => classify(context) !== null, classify };
}

/** One cut of the sequential probability ratio test: the indifference region around it, θ below and above. */
export interface SprtCut {
  /** The theta below the cut that the test tells from `upper`. */
  readonly lower: number;
  /** The theta above the cut that the test tells from `lower`. */
  readonly upper: number;
}

/** Options of {@link sequentialProbabilityRatio}. */
export interface SprtOptions {
  /**
   * The rate of classifying a person at `lower` above the cut, above 0 and below 1.
   *
   * @defaultValue `0.05`
   */
  readonly alpha?: number;
  /**
   * The rate of classifying a person at `upper` below the cut, above 0 and below 1 - `alpha`.
   *
   * @defaultValue `0.05`
   */
  readonly beta?: number;
}

const SPRT_OPTIONS: FieldNames<SprtOptions> = { alpha: true, beta: true };

/**
 * Returns a rule that stops once Wald's sequential probability ratio test has decided at every cut, as mirtCAT's SPRT
 * does for one cut.
 *
 * @remarks
 * At each cut, the log-likelihood ratio ln L(upper) - ln L(lower) of the responses so far is compared with
 * ln(β / (1 - α)) and ln((1 - β) / α): below the first the test decides below the cut, above the second above it, and
 * between them it goes on. The rule stops when every cut is decided, and the category is the number of cuts decided
 * above. Two cuts classify into three categories. The likelihood is that of the pool's items, without a
 * prior, so the estimator does not enter the decisions.
 *
 * @param cuts - The indifference regions, each with `lower` below `upper`, increasing and not overlapping.
 * @param options - The error rates α and β.
 * @returns The rule, which also classifies.
 * @throws `DataError` when the cuts are not an array of such regions, at least one, or the rates are not between 0 and
 * 1 with α + β below 1.
 * @see {@link https://doi.org/10.1177/01466219922031365 | Eggen (1999)}
 * @see {@link https://doi.org/10.1007/978-0-387-85461-8_19 | Eggen (2010)}
 */
export function sequentialProbabilityRatio(cuts: readonly SprtCut[], options: SprtOptions = {}): ClassificationRule {
  const regions = checkCuts(cuts, (cut, i) => {
    if (!isRecord(cut)) throw new DataError(`Cut ${i + 1} must be an object { lower, upper }, found ${describe(cut)}.`);
    checkFinite(cut.lower, `The lower bound of cut ${i + 1}`);
    checkFinite(cut.upper, `The upper bound of cut ${i + 1}`);
    if (!(cut.lower < cut.upper)) {
      throw new DataError(`Cut ${i + 1} needs its lower bound below its upper, found ${cut.lower} and ${cut.upper}.`);
    }
    return [cut.lower, cut.upper];
  });
  checkOptions(options, SPRT_OPTIONS);
  const { alpha = DEFAULT_ALPHA, beta = DEFAULT_BETA } = options;
  checkOpenUnitInterval(alpha, "The option alpha");
  checkOpenUnitInterval(beta, "The option beta");
  if (!(alpha + beta < 1)) {
    throw new DataError(`The options alpha and beta need a sum below 1, found ${alpha} and ${beta}.`);
  }
  const below = Math.log(beta / (1 - alpha));
  const above = Math.log((1 - beta) / alpha);
  const classify = ({ pool, administered, responses }: ClassificationContext) => {
    const answers = administered.flatMap((index, j) => {
      const item = pool[index];
      return item === undefined ? [] : [{ item, response: responses[j] ?? 0 }];
    });
    const probabilities = new Float64Array(largestCategoryCount(answers.map(({ item }) => item)));
    let category = 0;
    for (const [lower, upper] of regions) {
      const ratio = logLikelihood(answers, upper, probabilities) - logLikelihood(answers, lower, probabilities);
      if (ratio > above) category++;
      else if (!(ratio < below)) return null;
    }
    return category;
  };
  return { stops: (context) => classify(context) !== null, classify };
}

/**
 * Checks that the cuts are an array of at least one, each read by `bounds` into its lower and upper bound, every cut's
 * lower bound at or above the previous cut's upper and its upper above the previous one's, and returns their bounds,
 * which a change to the array after the check does not reach.
 *
 * @throws `DataError` naming the first cut that is not.
 */
function checkCuts<TCut>(
  cuts: readonly TCut[],
  bounds: (cut: TCut, i: number) => readonly [number, number],
): (readonly [number, number])[] {
  checkArray(cuts, "The cuts must be an array");
  if (cuts.length === 0) throw new DataError("A classification needs at least one cut.");
  const checked: (readonly [number, number])[] = [];
  cuts.forEach((cut, i) => {
    const current = bounds(cut, i);
    const previous = checked[i - 1];
    if (previous !== undefined && !(current[0] >= previous[1] && current[1] > previous[1])) {
      throw new DataError(`The cuts must increase without overlapping, but cut ${i + 1} is not above cut ${i}.`);
    }
    checked.push(current);
  });
  return checked;
}

/** Returns the log-likelihood of the responses at theta, using `probabilities` for each item's. */
function logLikelihood(
  answers: readonly { readonly item: PoolItem; readonly response: number }[],
  theta: number,
  probabilities: Float64Array,
): number {
  let sum = 0;
  for (const { item, response } of answers) {
    item.model.probabilities(item.parameters, theta, probabilities);
    sum += logProbability(probabilities[response] ?? 0);
  }
  return sum;
}

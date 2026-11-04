import {
  checkArray,
  checkFinite,
  checkOptions,
  checkRecord,
  DataError,
  describe,
  type FieldNames,
  logGamma,
} from "@measurekit/core";
import { checkOpenUnitInterval } from "../checks.js";
import { adaptiveIntegral, roundHalfToEven } from "../math/internal.js";
import { checkCuts } from "./classification.js";

/** Options of {@link livingstonLewisClassification}. */
export interface LivingstonLewisOptions {
  /** The reliability of the scores, above 0 and below 1, such as `cronbachAlpha` of their items. Required. */
  readonly reliability: number;
  /** The highest possible score. Required. */
  readonly maximum: number;
  /**
   * The lowest possible score.
   *
   * @defaultValue `0`
   */
  readonly minimum?: number;
}

const LIVINGSTON_LEWIS_OPTIONS: FieldNames<LivingstonLewisOptions> = {
  reliability: true,
  maximum: true,
  minimum: true,
};

/** The accuracy and consistency of the classification into the scores below a cut and those at or above it. */
export interface CutClassification {
  /** The cut on the score scale. */
  readonly cut: number;
  /** The probability that a person's observed and true classification at the cut agree. */
  readonly accuracy: number;
  /** The probability that two independent administrations classify a person alike at the cut. */
  readonly consistency: number;
}

/** The true-score distribution of Livingston and Lewis: a beta on the proportion scale, and the effective test length. */
export interface BetaTrueScore {
  /** The first shape. */
  readonly alpha: number;
  /** The second shape. */
  readonly beta: number;
  /** The lowest true proportion: 0 when the two-parameter beta takes the place of the four-parameter one. */
  readonly lower: number;
  /** The highest true proportion: 1 when the two-parameter beta takes the place of the four-parameter one. */
  readonly upper: number;
  /** The effective test length, before it is rounded to the trials of the binomial error. */
  readonly effectiveLength: number;
}

/** Livingston and Lewis's classification accuracy and consistency of scores by cuts. */
export interface ScoreClassification {
  /** Each cut's classification into the scores below it and those at or above it, in the order of the cuts. */
  readonly cuts: readonly CutClassification[];
  /** The probability that a person's observed and true categories agree, over all categories at once. */
  readonly accuracy: number;
  /** The probability that two independent administrations give a person one category, over all categories at once. */
  readonly consistency: number;
  /** The fitted true-score distribution. */
  readonly trueScore: BetaTrueScore;
}

/** The relative error of each integral, as the design of the method's reference takes `integrate(rel.tol = 1e-12)`. */
const INTEGRAL_TOLERANCE = 1e-12;
/** The relative offset below which a scaled cut counts as the whole number above it, betafunctions' `.score_threshold`. */
const CUT_OFFSET = 1e-9;
/**
 * The largest effective length after rounding. Each integrand sums the binomial over that many trials, so the time grows
 * with it: 0.2 s for three cuts at 4,639 trials, 2.8 s at 46,379. A test as reliable as 10,000 items is beyond any made.
 */
const MAX_EFFECTIVE_LENGTH = 10_000;

/**
 * Returns Livingston and Lewis's accuracy and consistency of the classification of total scores by cuts, from a
 * four-parameter beta distribution of true scores and a binomial error at the effective test length.
 *
 * @remarks
 * The scores, of any items, are rescaled to proportions of the range from `minimum` to `maximum`. The effective test
 * length N = ((μ - min)(max - μ) - r σ²) / (σ² (1 - r)), from the mean μ, the variance σ² (denominator n - 1) and the
 * reliability r, is the number of dichotomous items of a test as reliable. The first four raw moments of the true
 * proportion follow from the scores times N by Lord's binomial relation, m₁ = mean(x′) / N and m_i = mean(x′₍ᵢ₎) / N₍ᵢ₎
 * with descending factorials, and a four-parameter beta takes their mean, variance, skewness and kurtosis, as Hanson's
 * method of moments does. When
 * that beta is impermissible, a bound outside [0, 1], a shape at most 0 or a value not finite, a two-parameter beta on
 * [0, 1] with the same mean and variance takes its place. N is then rounded to the nearest integer, halves to even, and
 * a person's observed score is binomial with N trials and the true proportion as its probability.
 *
 * Each cut c on the score scale is the true cut at its proportion p = (c - min) / (max - min), and the observed cut at
 * ⌈N p⌉, a score at or above which is above; a cut within 1e-9 of a whole number of trials counts as that number.
 * Accuracy is the probability, integrated over the beta, that the observed category is the true one; consistency, that
 * two independent observed scores fall in one category. Each integral is adaptive Gauss-Kronrod quadrature to a relative
 * 1e-12. Rudner's indices of `classificationAccuracy` suit cuts on theta with estimates and their standard errors;
 * these suit cuts on a total score.
 *
 * @param scores - The total scores of the persons, from `minimum` to `maximum`.
 * @param cuts - The cuts on the score scale, increasing.
 * @param options - The reliability and the range of the scores.
 * @returns Each cut's accuracy and consistency, those over all categories, and the true-score distribution.
 * @throws `DataError` when there are fewer than two scores, a score is not finite or outside the range, the scores do
 * not vary, there are no cuts or they are not finite and increasing, the options are not an object, `reliability` is not
 * between 0 and 1, `minimum` or `maximum` is not finite or `maximum` is not above `minimum`; when the effective length
 * rounds to fewer than 1 trial or more than 10,000; or when neither beta is permissible, as when the
 * reliability is too low for the scores' variance.
 * @see {@link https://doi.org/10.1111/j.1745-3984.1995.tb00462.x | Livingston and Lewis (1995)}
 * @see {@link https://files.eric.ed.gov/fulltext/EJ1443424.pdf | Wolkowitz and Smith (2024)}
 */
export function livingstonLewisClassification(
  scores: readonly number[],
  cuts: readonly number[],
  options: LivingstonLewisOptions,
): ScoreClassification {
  checkRecord(options, "Livingston and Lewis need options { reliability, maximum }");
  checkOptions(options, LIVINGSTON_LEWIS_OPTIONS);
  const { reliability, maximum, minimum = 0 } = options;
  checkOpenUnitInterval(reliability, "The option reliability");
  checkFinite(maximum, "The option maximum");
  checkFinite(minimum, "The option minimum");
  if (!(maximum > minimum)) {
    throw new DataError(`The option maximum must be above the minimum ${minimum}, found ${maximum}.`);
  }
  const proportions = checkScores(scores, minimum, maximum);
  checkCuts(cuts, "Livingston and Lewis need");
  const effectiveLength = lengthOf(scores, minimum, maximum, reliability);
  const trials = roundHalfToEven(effectiveLength);
  if (!(trials >= 1 && trials <= MAX_EFFECTIVE_LENGTH)) {
    throw new DataError(
      `Livingston and Lewis need an effective test length from 1 to ${MAX_EFFECTIVE_LENGTH} after rounding; the scores and the reliability ${reliability} give ${describe(effectiveLength)}.`,
    );
  }
  const shape = trueScoreBeta(proportions, effectiveLength, reliability);
  const trueCuts = cuts.map((cut) => (cut - minimum) / (maximum - minimum));
  const binomial = binomialProbabilities(trials);
  const perCut = trueCuts.map((cut, k) => ({ cut: cuts[k] ?? 0, ...indices(shape, binomial, [cut]) }));
  const [single] = perCut;
  const overall = perCut.length === 1 && single !== undefined ? single : indices(shape, binomial, trueCuts);
  return {
    cuts: perCut,
    accuracy: overall.accuracy,
    consistency: overall.consistency,
    trueScore: { ...shape, effectiveLength },
  };
}

/**
 * Checks the scores and returns them as proportions of the range.
 *
 * @throws `DataError` with fewer than two scores, a score not finite or outside the range, or scores that do not vary.
 */
function checkScores(scores: readonly number[], minimum: number, maximum: number): Float64Array {
  checkArray(scores, "Livingston and Lewis need an array of scores");
  if (scores.length < 2) throw new DataError(`Livingston and Lewis need at least two scores, found ${scores.length}.`);
  const proportions = new Float64Array(scores.length);
  let varies = false;
  for (let i = 0; i < scores.length; i++) {
    const score = scores[i];
    if (!(typeof score === "number" && score >= minimum && score <= maximum)) {
      throw new DataError(`Score ${i + 1} must be a number from ${minimum} to ${maximum}, found ${describe(score)}.`);
    }
    proportions[i] = (score - minimum) / (maximum - minimum);
    varies ||= proportions[i] !== proportions[0];
  }
  if (!varies) throw new DataError("Livingston and Lewis need scores that vary; every score is the same.");
  return proportions;
}

/** Returns the effective test length of the scores, betafunctions' `ETL`, unrounded. */
function lengthOf(scores: readonly number[], minimum: number, maximum: number, reliability: number): number {
  const count = scores.length;
  const mean = scores.reduce((sum, score) => sum + score, 0) / count;
  const variance = scores.reduce((sum, score) => sum + (score - mean) ** 2, 0) / (count - 1);
  return ((mean - minimum) * (maximum - mean) - reliability * variance) / (variance * (1 - reliability));
}

/** A beta distribution on [lower, upper]. */
interface Beta {
  readonly alpha: number;
  readonly beta: number;
  readonly lower: number;
  readonly upper: number;
}

/**
 * Returns the true-score beta: the four-parameter beta of the first four true-score moments, or the two-parameter beta
 * on [0, 1] of the first two where that one is impermissible.
 *
 * @throws `DataError` when neither is permissible.
 */
function trueScoreBeta(proportions: Float64Array, length: number, reliability: number): Beta {
  // Lord's relation: the r-th descending factorial moment of a binomial count with N trials is N₍ᵣ₎ times the r-th
  // raw moment of its probability, so each raw moment of the true proportion is the scores' factorial moment over N₍ᵣ₎.
  const raw = [0, 0, 0, 0];
  for (const proportion of proportions) {
    const x = proportion * length;
    let factorial = 1;
    for (let r = 0; r < 4; r++) {
      factorial *= x - r;
      raw[r] = (raw[r] ?? 0) + factorial;
    }
  }
  let trials = 1;
  for (let r = 0; r < 4; r++) {
    trials *= length - r;
    raw[r] = (raw[r] ?? 0) / proportions.length / trials;
  }
  const [m1 = 0, m2 = 0, m3 = 0, m4 = 0] = raw;
  const variance = m2 - m1 * m1;
  const skewness = (m3 - 3 * m1 * m2 + 2 * m1 ** 3) / variance ** 1.5;
  const kurtosis = (m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1 ** 4) / (variance * variance);
  const four = fourParameterBeta(m1, variance, skewness, kurtosis);
  if (permissible(four)) return four;
  const spread = (m1 * (1 - m1)) / variance - 1;
  const two = { alpha: m1 * spread, beta: (1 - m1) * spread, lower: 0, upper: 1 };
  if (permissible(two)) return two;
  throw new DataError(
    `Livingston and Lewis find no true-score beta for these scores: their true-score variance is ${describe(variance)} at the reliability ${reliability}, and a beta needs one above 0 and below m(1 - m) of the mean m ${describe(m1)}.`,
  );
}

/** Returns the four-parameter beta of a mean, a variance, a skewness and a kurtosis, betafunctions' `Beta.4p.fit`. */
function fourParameterBeta(mean: number, variance: number, skewness: number, kurtosis: number): Beta {
  const r = (6 * (kurtosis - skewness * skewness - 1)) / (6 + 3 * skewness * skewness - 2 * kurtosis);
  const root = Math.sqrt(1 - (24 * (r + 1)) / ((r + 2) * (r + 3) * kurtosis - 3 * (r - 6) * (r + 1)));
  const larger = (r / 2) * (1 + root);
  const smaller = (r / 2) * (1 - root);
  const [alpha, beta] = skewness < 0 ? [larger, smaller] : [smaller, larger];
  const spread = Math.sqrt(variance * (alpha + beta + 1)) / Math.sqrt(alpha * beta);
  return { alpha, beta, lower: mean - alpha * spread, upper: mean + beta * spread };
}

/** Returns whether a beta is a distribution on [0, 1]: finite, with shapes above 0 and its bounds within [0, 1]. */
function permissible({ alpha, beta, lower, upper }: Beta): boolean {
  return (
    [alpha, beta, lower, upper].every(Number.isFinite) &&
    alpha > 0 &&
    beta > 0 &&
    lower >= 0 &&
    upper <= 1 &&
    lower < upper
  );
}

/** The binomial distribution of a number of trials, with the logarithms of its coefficients. */
interface Binomial {
  readonly trials: number;
  readonly logCoefficients: Float64Array;
}

function binomialProbabilities(trials: number): Binomial {
  const logCoefficients = new Float64Array(trials + 1);
  const logTrials = logGamma(trials + 1);
  for (let s = 0; s <= trials; s++) logCoefficients[s] = logTrials - logGamma(s + 1) - logGamma(trials - s + 1);
  return { trials, logCoefficients };
}

/**
 * Writes the probability of each observed category at a true proportion: below the first threshold, between each two,
 * and at or above the last.
 */
function categoryProbabilities(
  binomial: Binomial,
  thresholds: readonly number[],
  proportion: number,
  out: Float64Array,
): void {
  out.fill(0);
  const logSuccess = Math.log(proportion);
  const logFailure = Math.log1p(-proportion);
  let category = 0;
  for (let s = 0; s <= binomial.trials; s++) {
    while (category < thresholds.length && s >= (thresholds[category] ?? 0)) category++;
    out[category] = (out[category] ?? 0) + binomialMass(binomial, s, proportion, logSuccess, logFailure);
  }
}

/**
 * Returns the probability of the observed scores from `low` up to `high`, excluded: the terms of
 * {@link categoryProbabilities} of one category alone, summed in the same order.
 */
function categoryProbability(binomial: Binomial, low: number, high: number, proportion: number): number {
  const logSuccess = Math.log(proportion);
  const logFailure = Math.log1p(-proportion);
  let sum = 0;
  const last = Math.min(binomial.trials, high - 1);
  for (let s = Math.max(0, low); s <= last; s++) sum += binomialMass(binomial, s, proportion, logSuccess, logFailure);
  return sum;
}

/** Returns the binomial probability of s successes, from the logarithms of the proportion and of its complement. */
function binomialMass(
  { trials, logCoefficients }: Binomial,
  s: number,
  proportion: number,
  logSuccess: number,
  logFailure: number,
): number {
  if (proportion <= 0) return Number(s === 0);
  if (proportion >= 1) return Number(s === trials);
  return Math.exp((logCoefficients[s] ?? 0) + s * logSuccess + (trials - s) * logFailure);
}

/** Returns the accuracy and the consistency of the classification by cuts on the proportion scale. */
function indices(shape: Beta, binomial: Binomial, cuts: readonly number[]): { accuracy: number; consistency: number } {
  const thresholds = cuts.map((cut) => {
    const scaled = binomial.trials * cut;
    return Math.ceil(scaled - CUT_OFFSET * Math.max(1, Math.abs(scaled)));
  });
  const categories = new Float64Array(cuts.length + 1);
  const bounds = [Number.NEGATIVE_INFINITY, ...cuts, Number.POSITIVE_INFINITY];
  const scores = [Number.NEGATIVE_INFINITY, ...thresholds, Number.POSITIVE_INFINITY];
  let accuracy = 0;
  for (let k = 0; k < categories.length; k++) {
    const low = scores[k] ?? 0;
    const high = scores[k + 1] ?? 0;
    accuracy += betaIntegral(
      shape,
      (proportion) => categoryProbability(binomial, low, high, proportion),
      bounds[k] ?? 0,
      bounds[k + 1] ?? 0,
    );
  }
  const consistency = betaIntegral(
    shape,
    (proportion) => {
      categoryProbabilities(binomial, thresholds, proportion, categories);
      return categories.reduce((sum, probability) => sum + probability * probability, 0);
    },
    Number.NEGATIVE_INFINITY,
    Number.POSITIVE_INFINITY,
  );
  // Each is an integral of probabilities over a distribution, which the quadrature's rounding can carry a little past
  // 0 or 1, as where every person falls on one side of a cut.
  return { accuracy: probability(accuracy), consistency: probability(consistency) };
}

/** Returns a value within [0, 1], the range of a probability. */
function probability(value: number): number {
  return Math.min(1, Math.max(0, value));
}

/**
 * Returns the integral of the beta's density times `f` from `from` to `to` on the proportion scale, within the beta's
 * support.
 *
 * @remarks
 * The integral is taken on the beta's own scale t = (p - lower) / (upper - lower), split at t = 1/2. A shape below 1
 * makes the density infinite at that end, so the half there is integrated in v = t^α, or w = (1 - t)^β, in which
 * t^(α - 1) dt is dv / α: an exact change of variables that leaves a bounded integrand.
 */
function betaIntegral(shape: Beta, f: (proportion: number) => number, from: number, to: number): number {
  const { alpha, beta, lower, upper } = shape;
  const width = upper - lower;
  const start = Math.max(0, (from - lower) / width);
  const end = Math.min(1, (to - lower) / width);
  if (!(start < end)) return 0;
  const logBeta = logGamma(alpha) + logGamma(beta) - logGamma(alpha + beta);
  const at = (t: number): number => f(lower + width * t);
  const piece = (low: number, high: number): number => {
    if (!(low < high)) return 0;
    if (high <= 0.5 && alpha < 1) {
      const g = (v: number): number => {
        const t = v ** (1 / alpha);
        return (Math.exp((beta - 1) * Math.log1p(-t) - logBeta) / alpha) * at(t);
      };
      return adaptiveIntegral(g, low ** alpha, high ** alpha, INTEGRAL_TOLERANCE);
    }
    if (low >= 0.5 && beta < 1) {
      const g = (w: number): number => {
        const t = 1 - w ** (1 / beta);
        return (Math.exp((alpha - 1) * Math.log(t) - logBeta) / beta) * at(t);
      };
      return adaptiveIntegral(g, (1 - high) ** beta, (1 - low) ** beta, INTEGRAL_TOLERANCE);
    }
    const g = (t: number): number =>
      Math.exp((alpha - 1) * Math.log(t) + (beta - 1) * Math.log1p(-t) - logBeta) * at(t);
    return adaptiveIntegral(g, low, high, INTEGRAL_TOLERANCE);
  };
  return piece(start, Math.min(end, 0.5)) + piece(Math.max(start, 0.5), end);
}

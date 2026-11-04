import { checkArray, checkFinite, checkOptions, DataError, type FieldNames, normalQuantile } from "@measurekit/core";
import { checkOpenUnitInterval } from "../checks.js";
import { percentileRankOf, weightedScores } from "./percentile-ranks.js";

/** An estimate with its standard error and the interval at a level. */
export interface EstimateInterval {
  /** The estimate. */
  readonly estimate: number;
  /** Its standard error, by the delta method. */
  readonly standardError: number;
  /** The estimate less the normal quantile of the level times the standard error. */
  readonly lower: number;
  /** The estimate plus that. */
  readonly upper: number;
}

/** The norms of one score of a sample. */
export interface ScoreNorms {
  /** The score. */
  readonly score: number;
  /** Its z score, (score - mean) / SD. */
  readonly z: EstimateInterval;
  /** Its percentile rank: the percentage of the sample below it plus half the percentage at it. */
  readonly percentile: EstimateInterval;
}

/** The norms of a sample of scores with their standard errors, as the mokken package's `check.norms`, weighted or not. */
export interface NormStatistics {
  /** The mean of the scores, weighted by the persons' weights. */
  readonly mean: EstimateInterval;
  /** The standard deviation of the scores: the root of n / (n - 1) times their weighted mean squared deviation, for n persons. */
  readonly sd: EstimateInterval;
  /** The z score and percentile rank of each distinct score, from the lowest. */
  readonly scores: readonly ScoreNorms[];
  /** The eight boundaries of the stanines, mean + c SD for c from -1.75 to 1.75 by 0.5: element k between k + 1 and k + 2. */
  readonly stanines: readonly EstimateInterval[];
}

/** Options of {@link normStatistics}. */
export interface NormStatisticsOptions {
  /**
   * The level of the intervals, above 0 and below 1.
   *
   * @defaultValue `0.95`
   */
  readonly level?: number;
  /**
   * A survey weight for each score, positive: the norms are then the weighted ones, estimates of the population's that
   * the weighted sample stands for, with standard errors under the weights.
   *
   * @defaultValue every person weighs the same
   */
  readonly personWeights?: readonly number[];
}

const NORM_STATISTICS_OPTIONS: FieldNames<NormStatisticsOptions> = { level: true, personWeights: true };

const DEFAULT_LEVEL = 0.95;
/** The multiples of the SD that bound the stanines: half an SD wide, the fifth centered on the mean. */
const STANINE_BOUNDARIES = [-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75];

/**
 * Returns the norms of a sample of scores, its mean and SD, the z score and the percentile rank of each score and the
 * boundaries of the stanines, each with its standard error and interval, step 9 of a Mokken scale analysis; with
 * `personWeights`, the norms of the population a weighted sample stands for.
 *
 * @remarks
 * The standard errors are the delta method. Each person's weight w_l, 1 without `personWeights`, enters a statistic f
 * through the total weight at each score, and the variance is Σ_l (u_l - ū)², with u_l = w_l ∂f/∂w_l and ū their mean.
 * With equal weights it is the delta method over the numbers of persons with each score, taken as multinomial, of
 * Oosterhuis, van der Ark and Sijtsma's `check.norms`: Σ n_k (∂f/∂n_k)² - (Σ n_k ∂f/∂n_k)² / N, where the standard error
 * of the mean is the SD with denominator N over √N. With weights it is the linearization of a survey's estimators, as
 * the survey package's `svymean` with one stage and no strata computes it, times √((n - 1) / n). The variance is
 * n / (n - 1) times the weighted mean of the squared deviations from the weighted mean, for n persons, as the survey
 * package's `svyvar`: the usual one on n - 1 with equal weights. The intervals are the estimate plus or minus the normal
 * quantile of the level times the standard error.
 *
 * @param scores - The scores of the norm sample: finite numbers, at least two, not all equal.
 * @param options - The level of the intervals and the persons' weights.
 * @returns The mean, the SD, the norms of each distinct score and the stanine boundaries.
 * @throws `DataError` when the scores are not an array of at least two finite numbers with variance, the level is not
 * between 0 and 1, or the weights are not a positive number for each score.
 * @see {@link https://doi.org/10.1111/bmsp.12078 | Sijtsma and van der Ark (2017)}
 * @see {@link https://doi.org/10.1002/9780470580066 | Lumley (2010)}
 */
export function normStatistics(scores: readonly number[], options: NormStatisticsOptions = {}): NormStatistics {
  checkArray(scores, "The scores must be an array of numbers");
  checkOptions(options, NORM_STATISTICS_OPTIONS);
  const { level = DEFAULT_LEVEL, personWeights } = options;
  checkOpenUnitInterval(level, "The option level");
  scores.forEach((score, index) => {
    checkFinite(score, `Score ${index + 1}`);
  });
  const n = scores.length;
  if (n < 2) throw new DataError(`Norm statistics need at least 2 scores, found ${n}.`);
  const sample = weightedScores(scores, personWeights);
  const { scores: distinct, at: weights, squares, total } = sample;
  let sum = 0;
  for (let k = 0; k < distinct.length; k++) sum += (weights[k] ?? 0) * (distinct[k] ?? 0);
  const mean = sum / total;
  const deviations = Array.from(distinct, (score) => score - mean);
  // The weighted mean of the squared deviations, which n / (n - 1) makes the variance.
  const spread = deviations.reduce((sum, deviation, k) => sum + (weights[k] ?? 0) * deviation * deviation, 0) / total;
  const sd = Math.sqrt((spread * n) / (n - 1));
  if (!(sd > 0)) throw new DataError("Norm statistics need scores that vary.");
  const quantile = normalQuantile(1 - (1 - level) / 2);
  const interval = (estimate: number, gradient: readonly number[]): EstimateInterval => {
    const standardError = deltaError(weights, squares, n, gradient);
    return {
      estimate,
      standardError,
      lower: estimate - quantile * standardError,
      upper: estimate + quantile * standardError,
    };
  };
  // The derivatives of the mean and the SD with respect to the weight of a person with each score.
  const ofMean = deviations.map((deviation) => deviation / total);
  const ofSd = deviations.map((deviation) => ((deviation * deviation - spread) * n) / ((n - 1) * total * 2 * sd));
  return {
    mean: interval(mean, ofMean),
    sd: interval(sd, ofSd),
    scores: Array.from(distinct, (score, j) => {
      const deviation = deviations[j] ?? 0;
      const rank = percentileRankOf(sample, score);
      const ofZ = ofMean.map((value, k) => -value / sd - (deviation * (ofSd[k] ?? 0)) / (sd * sd));
      const ofRank = deviations.map((_, k) => ((k < j ? 100 : k === j ? 50 : 0) - rank) / total);
      return { score, z: interval(deviation / sd, ofZ), percentile: interval(rank, ofRank) };
    }),
    stanines: STANINE_BOUNDARIES.map((multiple) =>
      interval(
        mean + multiple * sd,
        ofMean.map((value, k) => value + multiple * (ofSd[k] ?? 0)),
      ),
    ),
  };
}

/**
 * Returns the delta method's standard error of a statistic from its gradient by the weight of a person with each score:
 * Σ_l (w_l g_l - ū)², written with the total weight W_k and the total squared weight Q_k at each score k as
 * Σ_k g_k² Q_k - (Σ_k g_k W_k)² / n.
 */
function deltaError(weights: Float64Array, squares: Float64Array, n: number, gradient: readonly number[]): number {
  let squared = 0;
  let sum = 0;
  for (let k = 0; k < weights.length; k++) {
    const slope = gradient[k] ?? 0;
    squared += (squares[k] ?? 0) * slope * slope;
    sum += (weights[k] ?? 0) * slope;
  }
  return Math.sqrt(Math.max(squared - (sum * sum) / n, 0));
}

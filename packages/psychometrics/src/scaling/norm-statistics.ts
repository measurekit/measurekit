import { checkArray, checkFinite, checkOptions, DataError, type FieldNames, normalQuantile } from "@measurekit/core";
import { checkOpenUnitInterval } from "../checks.js";
import { percentileRanks } from "./percentile-ranks.js";

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

/** The norms of a sample of scores with their standard errors, as the mokken package's `check.norms`. */
export interface NormStatistics {
  /** The mean of the scores. */
  readonly mean: EstimateInterval;
  /** The standard deviation of the scores, with denominator n - 1. */
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
}

const NORM_STATISTICS_OPTIONS: FieldNames<NormStatisticsOptions> = { level: true };

const DEFAULT_LEVEL = 0.95;
/** The multiples of the SD that bound the stanines: half an SD wide, the fifth centered on the mean. */
const STANINE_BOUNDARIES = [-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75];

/**
 * Returns the norms of a sample of scores, its mean and SD, the z score and the percentile rank of each score and the
 * boundaries of the stanines, each with its standard error and interval, step 9 of a Mokken scale analysis.
 *
 * @remarks
 * The standard errors are the delta method over the numbers of persons with each score, taken as multinomial, as
 * Oosterhuis, van der Ark and Sijtsma's `check.norms`: for a statistic f of the counts n_k, the variance is
 * Σ n_k (∂f/∂n_k)² - (Σ n_k ∂f/∂n_k)² / N. The standard error of the mean is the SD with denominator N over √N. The
 * intervals are the estimate plus or minus the normal quantile of the level times the standard error.
 *
 * @param scores - The scores of the norm sample: finite numbers, at least two, not all equal.
 * @param options - The level of the intervals.
 * @returns The mean, the SD, the norms of each distinct score and the stanine boundaries.
 * @throws `DataError` when the scores are not an array of at least two finite numbers with variance, or the level is
 * not between 0 and 1.
 * @see {@link https://doi.org/10.1111/bmsp.12078 | Sijtsma and van der Ark (2017)}
 */
export function normStatistics(scores: readonly number[], options: NormStatisticsOptions = {}): NormStatistics {
  checkArray(scores, "The scores must be an array of numbers");
  checkOptions(options, NORM_STATISTICS_OPTIONS);
  const { level = DEFAULT_LEVEL } = options;
  checkOpenUnitInterval(level, "The option level");
  scores.forEach((score, index) => {
    checkFinite(score, `Score ${index + 1}`);
  });
  const n = scores.length;
  if (n < 2) throw new DataError(`Norm statistics need at least 2 scores, found ${n}.`);
  const distinct = [...new Set(scores)].sort((a, b) => a - b);
  const counts = new Float64Array(distinct.length);
  const position = new Map(distinct.map((score, k) => [score, k]));
  for (const score of scores) counts[position.get(score) ?? 0] = (counts[position.get(score) ?? 0] ?? 0) + 1;
  const mean = scores.reduce((sum, score) => sum + score, 0) / n;
  const deviations = distinct.map((score) => score - mean);
  const squares = deviations.reduce((sum, deviation, k) => sum + (counts[k] ?? 0) * deviation * deviation, 0);
  const sd = Math.sqrt(squares / (n - 1));
  if (!(sd > 0)) throw new DataError("Norm statistics need scores that vary.");
  const quantile = normalQuantile(1 - (1 - level) / 2);
  const interval = (estimate: number, gradient: readonly number[]): EstimateInterval => {
    const standardError = deltaError(counts, gradient);
    return {
      estimate,
      standardError,
      lower: estimate - quantile * standardError,
      upper: estimate + quantile * standardError,
    };
  };
  // The derivatives of the mean and the SD with respect to each count n_k.
  const ofMean = deviations.map((deviation) => deviation / n);
  const ofSd = deviations.map((deviation) => (deviation * deviation - sd * sd) / (2 * sd * (n - 1)));
  const ranks = percentileRanks(scores, distinct);
  return {
    mean: interval(mean, ofMean),
    sd: interval(sd, ofSd),
    scores: distinct.map((score, j) => {
      const deviation = deviations[j] ?? 0;
      const rank = ranks[j] ?? 0;
      const ofZ = ofMean.map((value, k) => -value / sd - (deviation * (ofSd[k] ?? 0)) / (sd * sd));
      const ofRank = distinct.map((_, k) => ((k < j ? 100 : k === j ? 50 : 0) - rank) / n);
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

/** Returns the delta method's standard error of a statistic of multinomial counts, from its gradient by count. */
function deltaError(counts: Float64Array, gradient: readonly number[]): number {
  let squares = 0;
  let sum = 0;
  counts.forEach((count, k) => {
    const slope = gradient[k] ?? 0;
    squares += count * slope * slope;
    sum += count * slope;
  });
  let total = 0;
  for (const count of counts) total += count;
  return Math.sqrt(Math.max(squares - (sum * sum) / total, 0));
}

import { checkArray, checkFinite, checkOptions, DataError, describe, type FieldNames } from "@measurekit/core";
import { checkPersonWeights } from "../irt/internal.js";

/** Options of {@link percentileRanks}. */
export interface PercentileRanksOptions {
  /**
   * A survey weight for each score of the sample, positive: the ranks are then the weighted shares, an estimate of the
   * population's that the weighted sample stands for.
   *
   * @defaultValue every person weighs the same
   */
  readonly personWeights?: readonly number[];
}

const PERCENTILE_RANKS_OPTIONS: FieldNames<PercentileRanksOptions> = { personWeights: true };

/** The distinct scores of a sample from the lowest, with the weight at each and the weight below each. */
export interface WeightedScores {
  /** The distinct scores, ascending. */
  readonly scores: Float64Array;
  /** The weight of the sample at each distinct score. */
  readonly at: Float64Array;
  /** The sum of the squared weights of the persons at each distinct score: the counts when every person weighs 1. */
  readonly squares: Float64Array;
  /** The weight of the sample below each distinct score. */
  readonly below: Float64Array;
  /** The weight of the whole sample. */
  readonly total: number;
}

/**
 * Returns the percentile rank of each score in a norm sample: the percentage of the sample below it plus half the
 * percentage at it, weighted by the persons' survey weights when the option gives them.
 *
 * @remarks
 * PR(x) = 100 (W(s < x) + W(s = x) / 2) / W, with W(·) the weight of the sample's scores s that meet the condition and W
 * the whole sample's; every person weighs 1 without `personWeights`, which gives 100 (#{s < x} + #{s = x} / 2) / n. At
 * integer scores and equal weights it is the percentile rank function of Kolen and Brennan, and the `px` of the equate
 * package times 100; with weights it is the mean of Hmisc's weighted empirical distribution function `wtd.Ecdf` at x and
 * just below x, times 100. The sample is sorted once and each rank is found by binary search, so n scores and m ranks
 * take O((n + m) log n). A rank is of the sample's own scale: for ranks of scale scores, pass the sample's scale scores
 * and, as `scores`, a conversion table's.
 *
 * Valid input: a sample of at least one finite number, finite scores, and a positive weight for each score of the
 * sample; the scores need not occur in the sample.
 *
 * @param sample - The scores of the norm sample.
 * @param scores - The scores to rank.
 * @param options - The persons' survey weights.
 * @returns The percentile rank of each score, from 0 to 100.
 *
 * @example
 * ```ts
 * import { percentileRanks } from "@measurekit/psychometrics/scaling";
 *
 * percentileRanks([10, 20, 20, 30, 40], [20, 25, 50]); // => [40, 60, 100]
 * percentileRanks([10, 20, 20, 30, 40], [20], { personWeights: [3, 1, 1, 1, 2] }); // => [50], as 3 of 8 lie below and 2 at 20
 * ```
 * @throws `DataError` when the sample is not an array of at least one finite number, the scores are not an array of
 * finite numbers, or the weights are not a positive number for each score of the sample.
 * @see {@link https://doi.org/10.1007/978-1-4939-0317-7 | Kolen and Brennan (2014)}
 */
export function percentileRanks(
  sample: readonly number[],
  scores: readonly number[],
  options: PercentileRanksOptions = {},
): readonly number[] {
  checkArray(sample, "The norm sample must be an array of scores");
  checkArray(scores, "The scores must be an array of numbers");
  checkOptions(options, PERCENTILE_RANKS_OPTIONS);
  const weighted = weightedScores(sample, options.personWeights);
  return scores.map((score, index) => {
    checkFinite(score, `Score ${index + 1}`);
    return percentileRankOf(weighted, score);
  });
}

/**
 * Checks a norm sample and its weights and returns its distinct scores with their weights.
 *
 * @throws `DataError` when the sample has no score or a score that is not finite, or the weights are not a positive
 * number for each score.
 */
export function weightedScores(
  sample: readonly number[],
  personWeights: readonly number[] | undefined,
): WeightedScores {
  checkSample(sample);
  const weights = personWeightsOf(personWeights, sample.length);
  const order = Uint32Array.from(sample.keys()).sort((x, y) => (sample[x] ?? 0) - (sample[y] ?? 0));
  const distinct: number[] = [];
  const at: number[] = [];
  const squares: number[] = [];
  for (const person of order) {
    const score = sample[person] ?? 0;
    const weight = weights === null ? 1 : (weights[person] ?? 0);
    if (distinct.at(-1) === score) {
      at[at.length - 1] = (at.at(-1) ?? 0) + weight;
      squares[squares.length - 1] = (squares.at(-1) ?? 0) + weight * weight;
    } else {
      distinct.push(score);
      at.push(weight);
      squares.push(weight * weight);
    }
  }
  const below = new Float64Array(at.length);
  let total = 0;
  for (let k = 0; k < at.length; k++) {
    below[k] = total;
    total += at[k] ?? 0;
  }
  return {
    scores: Float64Array.from(distinct),
    at: Float64Array.from(at),
    squares: Float64Array.from(squares),
    below,
    total,
  };
}

/**
 * Checks that a norm sample has a score, and only finite ones.
 *
 * @throws `DataError` when it does not.
 */
function checkSample(sample: readonly number[]): void {
  if (sample.length === 0) throw new DataError("The norm sample needs at least one score.");
  for (let index = 0; index < sample.length; index++) {
    if (!Number.isFinite(sample[index])) {
      throw new DataError(`The norm sample needs finite scores, found ${describe(sample[index])} at ${index}.`);
    }
  }
}

/**
 * Checks the persons' weights and returns them divided by the power of two at or below the largest, which changes
 * neither their ratios, the only thing the norms use, nor any rounding, and keeps their sum and the sums of their
 * squares from overflowing or underflowing; `null` without weights.
 */
function personWeightsOf(personWeights: readonly number[] | undefined, persons: number): Float64Array | null {
  if (personWeights === undefined) return null;
  checkPersonWeights(personWeights, persons);
  let largest = 0;
  for (const weight of personWeights) largest = Math.max(largest, weight);
  const scale = 2 ** Math.floor(Math.log2(largest));
  return Float64Array.from(personWeights, (weight) => weight / scale);
}

/** Returns the percentile rank of a score among weighted scores: 100 (W(s < x) + W(s = x) / 2) / W. */
export function percentileRankOf({ scores, at, below, total }: WeightedScores, score: number): number {
  let low = 0;
  let high = scores.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if ((scores[middle] ?? 0) < score) low = middle + 1;
    else high = middle;
  }
  if (low === scores.length) return 100;
  const under = below[low] ?? 0;
  const tied = scores[low] === score ? (at[low] ?? 0) : 0;
  return (100 * (under + tied / 2)) / total;
}

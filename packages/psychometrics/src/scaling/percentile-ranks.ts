import { checkArray, checkFinite, DataError, describe } from "@measurekit/core";

/**
 * Returns the percentile rank of each score in a norm sample: the percentage of the sample below it plus half the
 * percentage at it.
 *
 * @remarks
 * PR(x) = 100 (#{s < x} + #{s = x} / 2) / n, for a sample of n scores s. At integer scores it is the percentile rank
 * function of Kolen and Brennan, and the `px` of the equate package times 100. The sample is sorted once and each rank is
 * found by binary search, so n scores and m ranks take O((n + m) log n). A rank is of the sample's own scale: for ranks
 * of scale scores, pass the sample's scale scores and, as `scores`, a conversion table's.
 *
 * Valid input: a sample of at least one finite number, and finite scores; the scores need not occur in the sample.
 *
 * @param sample - The scores of the norm sample.
 * @param scores - The scores to rank.
 * @returns The percentile rank of each score, from 0 to 100.
 *
 * @example
 * ```ts
 * import { percentileRanks } from "@measurekit/psychometrics/scaling";
 *
 * percentileRanks([10, 20, 20, 30, 40], [20, 25, 50]); // [40, 60, 100]
 * ```
 * @throws `DataError` when the sample is not an array of at least one finite number, or the scores are not an array of
 * finite numbers.
 * @see {@link https://doi.org/10.1007/978-1-4939-0317-7 | Kolen and Brennan (2014)}
 */
export function percentileRanks(sample: readonly number[], scores: readonly number[]): readonly number[] {
  checkArray(sample, "The norm sample must be an array of scores");
  checkArray(scores, "The scores must be an array of numbers");
  if (sample.length === 0) throw new DataError("The norm sample needs at least one score.");
  sample.forEach((score, index) => {
    if (!Number.isFinite(score)) {
      throw new DataError(`The norm sample needs finite scores, found ${describe(score)} at ${index}.`);
    }
  });
  const sorted = Float64Array.from(sample).sort();
  return scores.map((score, index) => {
    checkFinite(score, `Score ${index + 1}`);
    const below = firstIndexAtLeast(sorted, score);
    const atOrBelow = firstIndexAbove(sorted, score);
    return (100 * (below + (atOrBelow - below) / 2)) / sorted.length;
  });
}

/** Returns the number of values below `x` in the sorted values. */
function firstIndexAtLeast(sorted: Float64Array, x: number): number {
  return partitionPoint(sorted, (value) => value < x);
}

/** Returns the number of values at most `x` in the sorted values. */
function firstIndexAbove(sorted: Float64Array, x: number): number {
  return partitionPoint(sorted, (value) => value <= x);
}

/** Returns the number of leading values of the sorted values for which `holds` is true. */
function partitionPoint(sorted: Float64Array, holds: (value: number) => boolean): number {
  let low = 0;
  let high = sorted.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (holds(sorted[middle] ?? 0)) low = middle + 1;
    else high = middle;
  }
  return low;
}

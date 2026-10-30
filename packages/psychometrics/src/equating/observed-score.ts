import { checkArray, DataError, describe } from "@measurekit/core";

/**
 * Returns the linear equating of the scores of a form to the scale of another, from random groups.
 *
 * @remarks
 * For each score x of `from`, μ_to + (σ_to / σ_from)(x - μ_from), with the means and standard deviations (denominator n
 * - 1) of the score distributions, as the equate package's `equate(type = "linear")`. Each distribution is the number
 * of persons at each score, from 0 to the highest.
 *
 * @throws `DataError` when a distribution is invalid or has fewer than two persons, or the `from` scores have no
 * variance.
 */
export function linearEquating(from: readonly number[], to: readonly number[]): number[] {
  const source = moments(from, "from");
  const target = moments(to, "to");
  if (!(source.variance > 0)) throw new DataError("Linear equating needs `from` scores with variance.");
  const slope = Math.sqrt(target.variance / source.variance);
  return from.map((_, x) => target.mean + slope * (x - source.mean));
}

/**
 * Returns the equipercentile equating of the scores of a form to the scale of another, from random groups, as the
 * equate package's `equate(type = "equipercentile")`.
 *
 * @remarks
 * For each score x of `from`, the score of `to` with the same percentile rank, scores being continuous with a uniform
 * density over [x - 0.5, x + 0.5]. The percentile rank of x is F(x - 1) + f(x) / 2. Where the inverse is ambiguous
 * because `to` has scores without persons, the result is the mean of the smallest and the largest score with that rank.
 * Ranks of 0 and 1 give -0.5 and the highest `to` score plus 0.5. Each distribution is the number of persons at each
 * score, from 0 to the highest.
 *
 * @throws `DataError` when a distribution is invalid.
 * @see {@link https://doi.org/10.1007/978-1-4939-0317-7 | Kolen and Brennan (2014)}
 */
export function equipercentileEquating(from: readonly number[], to: readonly number[]): number[] {
  const sourceTotal = checkDistribution(from, "from");
  const targetTotal = checkDistribution(to, "to");
  const cumulative = Float64Array.from(cumulativeSums(to));
  const highest = to.length - 1;
  let below = 0;
  return from.map((count) => {
    // The percentile rank as a fraction (2 below + count) / (2 sourceTotal); a rank compares with the cumulative share
    // C(y) / targetTotal of `to` by cross-multiplication, exactly for counts of persons.
    const rank = 2 * below + count;
    below += count;
    const compare = (y: number): number => rank * targetTotal - 2 * sourceTotal * (cumulative[y] ?? 0);
    if (rank === 0) return -0.5;
    if (rank === 2 * sourceTotal) return highest + 0.5;
    const share = rank / (2 * sourceTotal);
    const cumulativeShare = (y: number): number => (y < 0 ? 0 : (cumulative[y] ?? 0) / targetTotal);
    // The smallest score whose cumulative share exceeds the rank, and the largest whose share is below it.
    let upper = 0;
    while (upper < highest && compare(upper) >= 0) upper++;
    const fromUpper =
      upper - 0.5 + (share - cumulativeShare(upper - 1)) / (cumulativeShare(upper) - cumulativeShare(upper - 1));
    let lower = upper;
    while (lower >= 0 && compare(lower) <= 0) lower--;
    if (lower < 0) return fromUpper;
    const fromLower =
      lower + 0.5 + (share - cumulativeShare(lower)) / (cumulativeShare(lower + 1) - cumulativeShare(lower));
    return (fromUpper + fromLower) / 2;
  });
}

/**
 * Checks a score distribution, the number of persons at each score from 0, and returns its number of persons.
 *
 * @param label - Names the distribution in error messages, such as `from`.
 * @throws `DataError` when the counts are not an array, with fewer than two scores, a count that is negative or not
 * finite, or no persons.
 */
export function checkDistribution(counts: readonly number[], label: string): number {
  checkArray(counts, `The ${label} score distribution must be an array of counts`);
  if (counts.length < 2) {
    throw new DataError(`The ${label} score distribution needs counts for at least the scores 0 and 1.`);
  }
  let total = 0;
  counts.forEach((count, score) => {
    if (!(Number.isFinite(count) && count >= 0)) {
      throw new DataError(
        `The ${label} score distribution needs a non-negative count at each score, found ${describe(count)} at ${score}.`,
      );
    }
    total += count;
  });
  if (!(total > 0)) throw new DataError(`The ${label} score distribution needs at least one person.`);
  return total;
}

function cumulativeSums(counts: readonly number[]): number[] {
  let sum = 0;
  return counts.map((count) => {
    sum += count;
    return sum;
  });
}

/** Returns the mean and variance (denominator n - 1) of a score distribution, from Σ f x and Σ f x², as equate. */
function moments(counts: readonly number[], label: string): { mean: number; variance: number } {
  const total = checkDistribution(counts, label);
  if (!(total > 1)) throw new DataError(`Linear equating needs at least two persons in the ${label} distribution.`);
  let sum = 0;
  let squares = 0;
  counts.forEach((count, x) => {
    sum += count * x;
    squares += count * x * x;
  });
  return { mean: sum / total, variance: (squares - (sum * sum) / total) / (total - 1) };
}

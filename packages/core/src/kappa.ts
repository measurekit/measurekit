/**
 * The weights of Cohen's kappa: `"none"`, agreement only; `"linear"`, 1 - |i - j| / (K - 1); `"quadratic"`,
 * 1 - (i - j)² / (K - 1)², for ratings i and j over the range K - 1 of the categories used.
 */
export type KappaWeights = "none" | "linear" | "quadratic";

/**
 * Returns Cohen's kappa of paired categories, (p_o - p_e) / (1 - p_e), with p_o the observed and p_e the chance
 * agreement from the margins, or `null` when p_e is 1, as when every category is the same.
 *
 * @remarks
 * Categories are integers from 0, and both lists have the same length, at least 1; the function does not check them.
 * The weights use the values of the categories over the range between the lowest and the highest used by either list.
 *
 * @param first - The category each pair has in the first list.
 * @param second - The category each pair has in the second list.
 * @param weights - The weights of partial agreement.
 * @returns Kappa, at most 1; `null` where chance agreement is complete, so that kappa is undefined.
 * @see {@link https://doi.org/10.1177/001316446002000104 | Cohen (1960)}
 * @see {@link https://doi.org/10.1037/h0026256 | Cohen (1968)}
 */
export function kappaOfPairs(
  first: ArrayLike<number>,
  second: ArrayLike<number>,
  weights: KappaWeights = "none",
): number | null {
  const { lowest, highest } = categoryRange(first, second);
  const agreement = agreementOf(weights, highest - lowest);
  const n = first.length;
  const { observed, firstTally, secondTally } =
    highest - lowest < n
      ? countInTables(first, second, lowest, highest, agreement)
      : countInMaps(first, second, agreement);
  // Over the categories that occur, so that the work grows with them, not with the range of the categories.
  let chance = 0;
  firstTally.categories.forEach((x, a) => {
    const firstCount = firstTally.counts[a] ?? 0;
    secondTally.categories.forEach((y, b) => {
      chance += (firstCount * (secondTally.counts[b] ?? 0) * agreement(x, y)) / (n * n);
    });
  });
  return chance < 1 ? (observed / n - chance) / (1 - chance) : null;
}

/** The categories that occur in a list, ascending, with the number of times each occurs. */
interface Tally {
  readonly categories: Float64Array;
  readonly counts: Float64Array;
}

/** The observed agreement of the pairs and the tally of each list. */
interface PairCounts {
  readonly observed: number;
  readonly firstTally: Tally;
  readonly secondTally: Tally;
}

/** Counts the pairs in tables over the range of the categories, which is no wider than the pairs. */
function countInTables(
  first: ArrayLike<number>,
  second: ArrayLike<number>,
  lowest: number,
  highest: number,
  agreement: (x: number, y: number) => number,
): PairCounts {
  const firstCounts = new Float64Array(highest - lowest + 1);
  const secondCounts = new Float64Array(highest - lowest + 1);
  let observed = 0;
  for (let i = 0; i < first.length; i++) {
    const x = first[i] ?? 0;
    const y = second[i] ?? 0;
    observed += agreement(x, y);
    firstCounts[x - lowest] = (firstCounts[x - lowest] ?? 0) + 1;
    secondCounts[y - lowest] = (secondCounts[y - lowest] ?? 0) + 1;
  }
  return { observed, firstTally: tableTally(firstCounts, lowest), secondTally: tableTally(secondCounts, lowest) };
}

/** Counts the pairs in maps of the categories that occur, whose range is wider than the pairs. */
function countInMaps(
  first: ArrayLike<number>,
  second: ArrayLike<number>,
  agreement: (x: number, y: number) => number,
): PairCounts {
  const firstCounts = new Map<number, number>();
  const secondCounts = new Map<number, number>();
  let observed = 0;
  for (let i = 0; i < first.length; i++) {
    const x = first[i] ?? 0;
    const y = second[i] ?? 0;
    observed += agreement(x, y);
    firstCounts.set(x, (firstCounts.get(x) ?? 0) + 1);
    secondCounts.set(y, (secondCounts.get(y) ?? 0) + 1);
  }
  return { observed, firstTally: mapTally(firstCounts), secondTally: mapTally(secondCounts) };
}

/** Returns the categories of a table of counts from `lowest` that occur, with their counts. */
function tableTally(counts: Float64Array, lowest: number): Tally {
  const categories: number[] = [];
  const occurring: number[] = [];
  counts.forEach((count, at) => {
    if (count === 0) return;
    categories.push(lowest + at);
    occurring.push(count);
  });
  return { categories: Float64Array.from(categories), counts: Float64Array.from(occurring) };
}

/** Returns the categories of a map of counts, ascending, with their counts. */
function mapTally(counts: ReadonlyMap<number, number>): Tally {
  const sorted = [...counts.entries()].sort(([x], [y]) => x - y);
  return {
    categories: Float64Array.from(sorted, ([category]) => category),
    counts: Float64Array.from(sorted, ([, count]) => count),
  };
}

/** Returns the lowest and the highest category of both lists. */
function categoryRange(first: ArrayLike<number>, second: ArrayLike<number>): { lowest: number; highest: number } {
  let lowest = Number.POSITIVE_INFINITY;
  let highest = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < first.length; i++) {
    const x = first[i] ?? 0;
    const y = second[i] ?? 0;
    lowest = Math.min(lowest, x, y);
    highest = Math.max(highest, x, y);
  }
  return { lowest, highest };
}

/** Returns the agreement of two categories: 1 for equal ones, and for others the weight of their distance in `range`. */
function agreementOf(weights: KappaWeights, range: number): (x: number, y: number) => number {
  if (weights === "none" || range === 0) return (x, y) => (x === y ? 1 : 0);
  return weights === "linear" ? (x, y) => 1 - Math.abs(x - y) / range : (x, y) => 1 - (Math.abs(x - y) / range) ** 2;
}

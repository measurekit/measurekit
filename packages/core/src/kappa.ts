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
 * @see {@link https://doi.org/10.1177/001316446002000104 | Cohen (1960)}
 * @see {@link https://doi.org/10.1037/h0026256 | Cohen (1968)}
 */
export function kappaOfPairs(
  first: ArrayLike<number>,
  second: ArrayLike<number>,
  weights: KappaWeights = "none",
): number | null {
  const { lowest, highest } = categoryRange(first, second);
  const range = highest - lowest;
  const agreement = agreementOf(weights, range);
  const size = range + 1;
  const firstCounts = new Float64Array(size);
  const secondCounts = new Float64Array(size);
  let observed = 0;
  for (let i = 0; i < first.length; i++) {
    const x = first[i] ?? 0;
    const y = second[i] ?? 0;
    observed += agreement(x, y);
    firstCounts[x - lowest] = (firstCounts[x - lowest] ?? 0) + 1;
    secondCounts[y - lowest] = (secondCounts[y - lowest] ?? 0) + 1;
  }
  const n = first.length;
  let chance = 0;
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      chance += ((firstCounts[i] ?? 0) * (secondCounts[j] ?? 0) * agreement(i, j)) / (n * n);
    }
  }
  return chance < 1 ? (observed / n - chance) / (1 - chance) : null;
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

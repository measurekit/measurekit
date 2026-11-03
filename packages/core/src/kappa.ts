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
  const range = highest - lowest;
  const chance =
    weights === "none" || range === 0
      ? exactChance(firstTally, secondTally, n)
      : weights === "linear"
        ? linearChance(positions(firstTally, lowest, range), positions(secondTally, lowest, range), n)
        : quadraticChance(positions(firstTally, lowest, range), positions(secondTally, lowest, range), n);
  return chance < 1 ? (observed / n - chance) / (1 - chance) : null;
}

// The chance agreement Σ_a Σ_b f_a s_b w(x_a, y_b) / n² over the categories that occur, f and s their counts in the two
// lists, in time that grows with the categories rather than with their pairs: 20,000 categories took 4.6 s by pairs.

/** Returns the chance agreement without weights: Σ f_c s_c / n² over the categories of both lists, by a merge. */
function exactChance(first: Tally, second: Tally, n: number): number {
  let sum = 0;
  for (let a = 0, b = 0; a < first.categories.length && b < second.categories.length; ) {
    const x = first.categories[a] ?? 0;
    const y = second.categories[b] ?? 0;
    if (x === y) sum += (first.counts[a++] ?? 0) * (second.counts[b++] ?? 0);
    else if (x < y) a++;
    else b++;
  }
  return sum / (n * n);
}

/** A list's categories as places from 0 to 1 over the range of both lists, with their counts. */
interface Places {
  readonly places: Float64Array;
  readonly counts: Float64Array;
}

function positions({ categories, counts }: Tally, lowest: number, range: number): Places {
  return { places: categories.map((category) => (category - lowest) / range), counts };
}

/**
 * Returns the chance agreement under the weights 1 - |u - v|: 1 - Σ_a f_a D(u_a) / n² with D(u) = Σ_b s_b |u - v_b|,
 * which the places in ascending order give by the counts and sums of the places of the second list below u.
 */
function linearChance(first: Places, second: Places, n: number): number {
  let total = 0;
  for (let b = 0; b < second.places.length; b++) total += (second.counts[b] ?? 0) * (second.places[b] ?? 0);
  let countBelow = 0;
  let sumBelow = 0;
  let distance = 0;
  for (let a = 0, b = 0; a < first.places.length; a++) {
    const u = first.places[a] ?? 0;
    for (; b < second.places.length && (second.places[b] ?? 0) <= u; b++) {
      countBelow += second.counts[b] ?? 0;
      sumBelow += (second.counts[b] ?? 0) * (second.places[b] ?? 0);
    }
    const away = u * countBelow - sumBelow + (total - sumBelow) - u * (n - countBelow);
    distance += (first.counts[a] ?? 0) * away;
  }
  return 1 - distance / (n * n);
}

/**
 * Returns the chance agreement under the weights 1 - (u - v)²: 1 - E[(U - V)²] for independent U and V of the two
 * lists, Var U + Var V + (E U - E V)², each moment about its mean so that no large squares cancel.
 */
function quadraticChance(first: Places, second: Places, n: number): number {
  const moments = ({ places, counts }: Places): { mean: number; variance: number } => {
    let mean = 0;
    for (let k = 0; k < places.length; k++) mean += ((counts[k] ?? 0) * (places[k] ?? 0)) / n;
    let variance = 0;
    for (let k = 0; k < places.length; k++) variance += ((counts[k] ?? 0) * ((places[k] ?? 0) - mean) ** 2) / n;
    return { mean, variance };
  };
  const x = moments(first);
  const y = moments(second);
  return 1 - (x.variance + y.variance + (x.mean - y.mean) ** 2);
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

/** The largest number of item steps of one item that can tie with those of another: a score of 9 has 9 steps. */
const MAX_STEPS = 9;

/** Pascal's triangle of binomial coefficients up to twice the most steps, for the orders of tied item steps. */
const BINOMIALS: readonly (readonly number[])[] = (() => {
  const rows: number[][] = [[1]];
  for (let n = 1; n <= 2 * MAX_STEPS; n++) {
    const previous = rows[n - 1] ?? [];
    rows.push(Array.from({ length: n + 1 }, (_, k) => (previous[k - 1] ?? 0) + (previous[k] ?? 0)));
  }
  return rows;
})();

function binomial(n: number, k: number): number {
  return BINOMIALS[n]?.[k] ?? 0;
}

/**
 * Returns the number of persons who scored at least each step: element s - 1 is the number with a score of at least
 * s, for the steps 1 to `categories` - 1.
 */
function cumulativeCounts(counts: Float64Array): Float64Array {
  const steps = counts.length - 1;
  const cumulative = new Float64Array(steps);
  let sum = 0;
  for (let step = steps; step >= 1; step--) {
    sum += counts[step] ?? 0;
    cumulative[step - 1] = sum;
  }
  return cumulative;
}

/**
 * Returns the probability that step t of the second item comes before step s of the first when the steps of both items
 * are put in the order of how many persons reach them, the most first: 1 where the second's step is reached by more
 * persons and 0 where by fewer. Tied steps of the two items take every order that keeps each item's own steps in
 * order, equally likely, as the mokken package averages over them.
 */
function precedes(first: Float64Array, second: Float64Array, s: number, t: number): number {
  const reachedFirst = first[s - 1] ?? 0;
  const reachedSecond = second[t - 1] ?? 0;
  if (reachedSecond !== reachedFirst) return reachedSecond > reachedFirst ? 1 : 0;
  const tiedFirst = first.reduce((count, reached) => count + (reached === reachedFirst ? 1 : 0), 0);
  const tiedSecond = second.reduce((count, reached) => count + (reached === reachedFirst ? 1 : 0), 0);
  const rankFirst = s - first.indexOf(reachedFirst);
  const rankSecond = t - second.indexOf(reachedFirst);
  // Of the orders of the two runs of tied steps, those with k of the second's before the first's step, over all of them.
  let favorable = 0;
  for (let k = rankSecond; k <= tiedSecond; k++) {
    favorable += binomial(rankFirst - 1 + k, k) * binomial(tiedFirst - rankFirst + tiedSecond - k, tiedSecond - k);
  }
  return favorable / binomial(tiedFirst + tiedSecond, tiedFirst);
}

/**
 * Returns the number of Guttman errors in each pair of scores of two items, by the first item's score, then the
 * second's, as the mokken package's weights.
 *
 * @remarks
 * Each item has a step for every score above 0, which a person passes by scoring at least that high. Steps of both
 * items are put in the order of how many persons pass them. A pair of scores has an error for every step passed that has
 * a step failed before it, and so counts the pairs of a step passed and a step failed that the order says should not
 * happen. Where steps of the two items are reached by as many persons their order is averaged.
 *
 * @param countsFirst - The persons at each score of the first item.
 * @param countsSecond - The persons at each score of the second item.
 */
export function guttmanErrors(countsFirst: Float64Array, countsSecond: Float64Array): Float64Array {
  const categories = countsFirst.length;
  const first = cumulativeCounts(countsFirst);
  const second = cumulativeCounts(countsSecond);
  const errors = new Float64Array(categories * categories);
  for (let s = 1; s < categories; s++) {
    for (let t = 1; t < categories; t++) {
      addStepPair(errors, categories, s, t, precedes(first, second, s, t));
    }
  }
  return errors;
}

/**
 * Adds to each pair of scores the errors of the first item's step s and the second's step t: one where the pair passes
 * the first and fails the second, and the second's step comes first with probability `before`, or passes the second and
 * fails the first, and the first's step comes first.
 */
function addStepPair(errors: Float64Array, categories: number, s: number, t: number, before: number): void {
  for (let a = 0; a < categories; a++) {
    for (let b = 0; b < categories; b++) {
      const error = (a >= s && b < t ? before : 0) + (b >= t && a < s ? 1 - before : 0);
      errors[a * categories + b] = (errors[a * categories + b] ?? 0) + error;
    }
  }
}

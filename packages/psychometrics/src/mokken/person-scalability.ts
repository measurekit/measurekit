import { checkDichotomous, checkResponseMatrix, type ResponseMatrix } from "../data/internal.js";

/** The nonparametric person fit of one person's 0/1 responses. */
export interface PersonScalability {
  /** The person id. */
  readonly id: string;
  /**
   * G over its largest value for the person's score, NC (n - NC) of n answered items: the Guttman errors normed, as
   * PerFit's `Gnormed`; from 0, a Guttman pattern, to 1, its reverse. `null` for an extreme score.
   */
  readonly normedGuttmanErrors: number | null;
  /**
   * Van der Flier's U3: the log-odds of the items answered correctly, set between those of the easiest and the hardest
   * items of the person's score; from 0, a Guttman pattern, to 1, its reverse. `null` for an extreme score.
   */
  readonly u3: number | null;
  /**
   * The person's H in the transposed matrix: the covariance of the person's responses with the other
   * persons' over its largest value; at most 1, low or negative for an aberrant pattern. `null` for an extreme score.
   */
  readonly ht: number | null;
}

/** What every person reads of the items: their proportions correct, log-odds and totals, and their orders. */
interface ItemOrder {
  readonly totals: Float64Array;
  /** The proportion correct of each item over the persons who answered it. */
  readonly proportions: Float64Array;
  /** The items that someone answered, by proportion correct, the largest first. */
  readonly byProportion: Int32Array;
  /** The log-odds of each item, 0 for one that everyone or no one who answered it passed. */
  readonly logOdds: Float64Array;
  /** The items by log-odds, the largest first. */
  readonly byOdds: Int32Array;
}

/**
 * Returns the normed Guttman errors, U3 and Ht of each person, as PerFit's `Gnormed`, `U3` and `Ht`.
 *
 * @remarks
 * For 0/1 items. Items are ordered by their proportions correct over the persons who answered them, PerFit's
 * `"Pairwise"`, and each person's indices use the items the person answered. G counts the pairs of an item failed and a
 * harder one passed, as `guttmanErrors` does on complete data: items of equal proportions take both orders equally
 * likely, so such a pair counts a half. U3 = (Σ_(first NC) λ - Σ x λ) / (Σ_(first NC) λ - Σ_(last NC) λ), λ the
 * log-odds of the items ordered from the largest. Ht is Cov(x_n, Σ_(m ≠ n) x_m) over the person's items, against
 * Σ_(m ≠ n) min(p_n, p_m) - p_n p_m, p the persons' proportions correct. An extreme score, every answered item wrong or
 * every one right, has none of them.
 *
 * @param matrix - The 0/1 responses of the persons.
 * @returns For each person, the normed Guttman errors, U3 and Ht.
 * @throws `DataError` when the matrix is invalid or holds a response other than 0, 1 or `null`.
 * @see {@link https://doi.org/10.1177/01466210122031957 | Meijer and Sijtsma (2001)}
 * @see {@link https://doi.org/10.1177/0022002182013003001 | van der Flier (1982)}
 * @see {@link https://doi.org/10.1207/S15324818AME1604_2 | Karabatsos (2003)}
 */
export function personScalability(matrix: ResponseMatrix): readonly PersonScalability[] {
  checkResponseMatrix(matrix);
  checkDichotomous(matrix, "Person scalability");
  const order = itemOrder(matrix);
  const proportions = matrix.responses.map((row) => {
    let answered = 0;
    let correct = 0;
    for (const response of row) {
      if (response === null) continue;
      answered++;
      correct += response;
    }
    return correct > 0 && correct < answered ? correct / answered : null;
  });
  const maxima = covarianceMaxima(proportions);
  const odds = new Float64Array(matrix.itemIds.length);
  return matrix.responses.map((row, person) => {
    const id = matrix.personIds[person] ?? "";
    const proportion = proportions[person] ?? null;
    if (proportion === null) return { id, normedGuttmanErrors: null, u3: null, ht: null };
    return {
      id,
      normedGuttmanErrors: normedErrors(row, order),
      u3: u3Of(row, order, odds),
      ht: htOf(row, order.totals, maxima[person] ?? 0),
    };
  });
}

/** Returns the items' totals, proportions correct and log-odds, and the items in the order of each. */
function itemOrder({ responses, itemIds }: ResponseMatrix): ItemOrder {
  const count = itemIds.length;
  const answered = new Float64Array(count);
  const totals = new Float64Array(count);
  for (const row of responses) {
    row.forEach((response, item) => {
      if (response === null) return;
      answered[item] = (answered[item] ?? 0) + 1;
      totals[item] = (totals[item] ?? 0) + response;
    });
  }
  const proportions = totals.map((total, item) => total / (answered[item] ?? 0));
  const byProportion = Int32Array.from({ length: count }, (_, item) => item)
    .filter((item) => (answered[item] ?? 0) > 0)
    .sort((a, b) => (proportions[b] ?? 0) - (proportions[a] ?? 0));
  const logOdds = proportions.map((p) => {
    const odds = Math.log(p / (1 - p));
    return Number.isFinite(odds) ? odds : 0;
  });
  const byOdds = Int32Array.from({ length: count }, (_, item) => item).sort(
    (a, b) => (logOdds[b] ?? 0) - (logOdds[a] ?? 0),
  );
  return { totals, proportions, byProportion, logOdds, byOdds };
}

/**
 * Returns a person's Guttman errors over their largest value, NC (n - NC). A pair of an item failed and a harder one
 * passed is an error, and a pair of items of equal proportions half of one, as the step order of `stepPairErrors` counts
 * them for 0/1 items; the counts are of halves, so their sum is exact in any order. The items are taken from the easiest,
 * with the failures among easier ones counted, so that a person costs the items, not their pairs.
 */
function normedErrors(row: readonly (number | null)[], { proportions, byProportion }: ItemOrder): number {
  let errorCount = 0;
  let passed = 0;
  let failed = 0;
  for (let start = 0; start < byProportion.length; ) {
    // The items of one proportion: a pass among them is an error with each failure of an easier item and half of one
    // with each failure among them.
    const proportion = proportions[byProportion[start] ?? 0];
    let end = start;
    let passedHere = 0;
    let failedHere = 0;
    while (end < byProportion.length && proportions[byProportion[end] ?? 0] === proportion) {
      const response = row[byProportion[end] ?? 0] ?? null;
      if (response === 1) passedHere++;
      else if (response === 0) failedHere++;
      end++;
    }
    errorCount += passedHere * failed + 0.5 * passedHere * failedHere;
    passed += passedHere;
    failed += failedHere;
    start = end;
  }
  return errorCount / (passed * failed);
}

/** Returns a person's U3, or `null` when the answered items' log-odds leave its denominator 0. */
function u3Of(row: readonly (number | null)[], { logOdds, byOdds }: ItemOrder, odds: Float64Array): number | null {
  let answered = 0;
  let correct = 0;
  let observed = 0;
  for (const item of byOdds) {
    const response = row[item] ?? null;
    if (response === null) continue;
    const value = logOdds[item] ?? 0;
    odds[answered++] = value;
    correct += response;
    observed += response * value;
  }
  // Each sum runs in the order of the items, as the observed one does, so that the pattern of the easiest items gives 0
  // and that of the hardest 1, exactly.
  let first = 0;
  let last = 0;
  for (let k = 0; k < correct; k++) {
    first += odds[k] ?? 0;
    last += odds[answered - correct + k] ?? 0;
  }
  return first === last ? null : (first - observed) / (first - last);
}

/** Returns a person's Ht from the items' totals and the largest covariance the person's proportion allows. */
function htOf(row: readonly (number | null)[], totals: Float64Array, maximum: number): number | null {
  let answered = 0;
  let sumX = 0;
  let sumRest = 0;
  let sumProduct = 0;
  row.forEach((response, item) => {
    if (response === null) return;
    const rest = (totals[item] ?? 0) - response;
    answered++;
    sumX += response;
    sumRest += rest;
    sumProduct += response * rest;
  });
  const covariance = sumProduct / answered - (sumX / answered) * (sumRest / answered);
  return maximum > 0 ? covariance / maximum : null;
}

/**
 * Returns, for each person of a proportion correct p_n strictly between 0 and 1, Σ_m min(p_n, p_m) - p_n p_m over the
 * other such persons, from the proportions sorted once: (1 - p_n) Σ_(p_m ≤ p_n) p_m + p_n Σ_(p_m > p_n) (1 - p_m).
 */
function covarianceMaxima(proportions: readonly (number | null)[]): Float64Array {
  const scored = proportions.flatMap((p) => (p === null ? [] : [p])).sort((a, b) => a - b);
  const below = new Float64Array(scored.length + 1);
  scored.forEach((p, k) => {
    below[k + 1] = (below[k] ?? 0) + p;
  });
  const above = new Float64Array(scored.length + 1);
  for (let k = scored.length - 1; k >= 0; k--) above[k] = (above[k + 1] ?? 0) + (1 - (scored[k] ?? 0));
  return Float64Array.from(proportions, (p) => {
    if (p === null) return 0;
    // The first index of a proportion above p: every one before it is at most p, the person among them.
    let low = 0;
    let high = scored.length;
    while (low < high) {
      const middle = (low + high) >> 1;
      if ((scored[middle] ?? 0) <= p) low = middle + 1;
      else high = middle;
    }
    return (1 - p) * ((below[low] ?? 0) - p) + p * (above[low] ?? 0);
  });
}

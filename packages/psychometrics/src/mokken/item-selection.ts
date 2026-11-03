import { checkOptions, checkUnitInterval, normalQuantile } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { checkLevel } from "./item-check.js";
import { usableItems } from "./scalability.js";
import { type Covariances, covariances, mokkenData } from "./scores.js";

/** Options of {@link automatedItemSelection}. */
export interface ItemSelectionOptions {
  /**
   * The lower bound c that an item's scalability coefficient H_i must reach to join a scale, from 0 to 1.
   *
   * @defaultValue `0.3`
   */
  readonly lowerBound?: number;
  /**
   * The familywise level of the tests that a pair's or an item's coefficient is above 0, a number above 0 and below 1.
   *
   * @defaultValue `0.05`
   */
  readonly alpha?: number;
}

/** The scales that automated item selection forms, and the items that belong to none. */
export interface MokkenScales {
  /** The item ids of each scale, in the order the scales were formed and, within a scale, the order of the matrix. */
  readonly scales: readonly (readonly string[])[];
  /** The ids of the items in no scale, including items without variance, in the order of the matrix. */
  readonly unscalable: readonly string[];
}

const DEFAULT_LOWER_BOUND = 0.3;
const DEFAULT_ALPHA = 0.05;
/** mokken's tie breaker of pairs with equal coefficients: it favors the pair with the lower first item. */
const ROW_EPSILON = 1e-10;

/**
 * Selects items into Mokken scales one scale at a time by the automated item selection procedure, over the persons
 * who answered every item.
 *
 * @remarks
 * Of the items not yet in a scale, a scale starts with the pair of the largest H_ij among those significantly above 0
 * (one-sided z test of the correlation, Bonferroni-adjusted for the number of pairs), if both items' H_i reach the lower
 * bound. Then the item that makes the scale's H largest is added, among the items that have no negative H_ij with a
 * member, an H_i of at least the lower bound with the scale, and a z test of H_i above 0 that is significant at a level
 * adjusted for all the tests of the scale so far, as long as that H is at least the lower bound. When none qualifies
 * the scale is closed and the next one starts from the items left, until fewer than two remain or no pair qualifies. It
 * is the `normal` search of the mokken package's `aisp`, with its default tests. Items without variance belong to no
 * scale and are left out of the tests. Where mokken stops because the lower bound is above every H_ij, all items are
 * unscalable.
 *
 * @param matrix - The responses of the persons.
 * @param options - The lower bound of H and the level of the tests.
 * @returns The scales, each a list of items, and the items that belong to none.
 * @throws `DataError` when the matrix is invalid, has fewer than two items or more than 1,000, fewer than two persons
 * answered every item, the scores span more than 10 categories, or an option is invalid.
 * @see {@link https://doi.org/10.18637/jss.v020.i11 | van der Ark (2007)}
 * @see {@link https://doi.org/10.4135/9781412984676 | Sijtsma and Molenaar (2002)}
 */
export function automatedItemSelection(matrix: ResponseMatrix, options: ItemSelectionOptions = {}): MokkenScales {
  checkOptions(options);
  const { lowerBound = DEFAULT_LOWER_BOUND, alpha = DEFAULT_ALPHA } = options;
  checkUnitInterval(lowerBound, "The option lowerBound");
  checkLevel(alpha, "The option alpha");
  const data = mokkenData(matrix, "Automated item selection");
  const moments = covariances(data);
  const usable = usableItems(moments, data.items);
  const assigned = searchScales(moments, data.items, usable, data.persons, lowerBound, alpha);
  const scaleCount = assigned.reduce((most, scale) => Math.max(most, scale), 0);
  return {
    scales: Array.from({ length: scaleCount }, (_, scale) =>
      data.itemIds.filter((_, item) => assigned[item] === scale + 1),
    ),
    unscalable: data.itemIds.filter((_, item) => assigned[item] === 0),
  };
}

/** The pairwise statistics that the search reads, by item, then item. */
interface PairStatistics {
  readonly items: number;
  readonly moments: Covariances;
  /** H_ij = Cov / Cov_max. */
  readonly coefficient: (i: number, j: number) => number;
  /** The z statistic of the correlation of two items. */
  readonly z: (i: number, j: number) => number;
  readonly persons: number;
}

/** Returns the scale number of each item, 0 for items in none. */
function searchScales(
  moments: Covariances,
  items: number,
  usable: readonly boolean[],
  persons: number,
  lowerBound: number,
  alpha: number,
): number[] {
  const at = (matrix: Float64Array, i: number, j: number): number => matrix[i * items + j] ?? 0;
  const pairs: PairStatistics = {
    items,
    moments,
    persons,
    coefficient: (i, j) => at(moments.observed, i, j) / at(moments.maximum, i, j),
    z: (i, j) =>
      (at(moments.observed, i, j) * Math.sqrt(persons - 1)) /
      Math.sqrt(at(moments.observed, i, i) * at(moments.observed, j, j)),
  };
  const candidates = Array.from({ length: items }, (_, item) => item).filter((item) => usable[item] === true);
  const inSet = new Array<number>(items).fill(0);
  for (let scale = 1; ; scale++) {
    const free = candidates.filter((item) => inSet[item] === 0);
    if (free.length < 2) break;
    const start = startingPair(pairs, free, candidates, alpha);
    if (start === null || pairs.coefficient(start[0], start[1]) < lowerBound) break;
    for (const item of start) inSet[item] = scale;
    growScale(pairs, { inSet, candidates, scale, freeAtStart: free.length }, lowerBound, alpha);
  }
  return inSet;
}

/** Returns the number of tests that a familywise level is divided by: the pairs of the free items, then each step's items. */
function adjustedLevel(alpha: number, counts: readonly number[]): number {
  const [first = 0, ...rest] = counts;
  return alpha / ((first * (first - 1)) / 2 + rest.reduce((sum, count) => sum + count, 0));
}

/**
 * Returns the pair of free items that starts a scale: the largest H_ij among the pairs with a z test significant at the
 * adjusted level, the pair with the lower first item where H_ij ties; `null` where no pair is significant.
 */
function startingPair(
  pairs: PairStatistics,
  free: readonly number[],
  candidates: readonly number[],
  alpha: number,
): [number, number] | null {
  const critical = Math.abs(normalQuantile(adjustedLevel(alpha, [free.length])));
  let best: [number, number] | null = null;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const row of free) {
    for (const column of free) {
      if (column >= row || Math.abs(pairs.z(row, column)) < critical) continue;
      // mokken subtracts the row number of the full matrix times a small constant before it takes the largest.
      const score = pairs.coefficient(row, column) - (candidates.indexOf(row) + 1) * ROW_EPSILON;
      if (score > bestScore) [best, bestScore] = [[column, row], score];
    }
  }
  return best;
}

/** The state of the search while a scale grows. */
interface Growth {
  /** The scale of each item, 0 for items not yet in one. */
  readonly inSet: number[];
  /** The items with variance, which the search may use. */
  readonly candidates: readonly number[];
  /** The number of the scale being formed. */
  readonly scale: number;
  /** The number of items that were free when the scale started. */
  readonly freeAtStart: number;
}

/** Adds items to the scale that has just started until none qualifies. */
function growScale(pairs: PairStatistics, growth: Growth, lowerBound: number, alpha: number): void {
  const { inSet, candidates, scale } = growth;
  const counts = [growth.freeAtStart];
  for (;;) {
    const members = candidates.filter((item) => inSet[item] === scale);
    // Items with a negative H_ij with a member are out of this scale's search.
    const available = candidates.filter(
      (item) => inSet[item] === 0 && members.every((member) => pairs.coefficient(member, item) >= 0),
    );
    if (available.length === 0) return;
    counts.push(available.length);
    const critical = Math.abs(normalQuantile(adjustedLevel(alpha, counts)));
    const results = available.map((item) => candidateCoefficient(pairs, members, item, lowerBound, critical));
    const best = results.reduce((most, result) => Math.max(most, result), Number.NEGATIVE_INFINITY);
    if (best < lowerBound) return;
    available.forEach((item, k) => {
      if (results[k] === best) inSet[item] = scale;
    });
  }
}

/**
 * Returns H of the scale with an item added, or -98 where the item's H_i with the scale is below the lower bound and -97
 * where its z test is not significant.
 */
function candidateCoefficient(
  pairs: PairStatistics,
  members: readonly number[],
  item: number,
  lowerBound: number,
  critical: number,
): number {
  const { items, moments, persons } = pairs;
  const at = (matrix: Float64Array, i: number, j: number): number => matrix[i * items + j] ?? 0;
  let withItem = 0;
  let maxWithItem = 0;
  let variances = 0;
  for (const member of members) {
    withItem += at(moments.observed, item, member);
    maxWithItem += at(moments.maximum, item, member);
    variances += at(moments.observed, item, item) * at(moments.observed, member, member);
  }
  if (withItem / maxWithItem < lowerBound) return -98;
  if ((withItem * Math.sqrt(persons - 1)) / Math.sqrt(variances) < critical) return -97;
  let observed = withItem;
  let maximum = maxWithItem;
  for (let a = 0; a < members.length; a++) {
    for (let b = a + 1; b < members.length; b++) {
      observed += at(moments.observed, members[a] ?? 0, members[b] ?? 0);
      maximum += at(moments.maximum, members[a] ?? 0, members[b] ?? 0);
    }
  }
  return observed / maximum;
}

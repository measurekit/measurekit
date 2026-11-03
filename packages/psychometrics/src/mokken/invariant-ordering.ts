import { DataError, mean, studentLowerTail } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { normalCdf } from "../math/internal.js";
import { defaultGroupSize, restScoreGroups } from "./groups.js";
import { checkMonotonicityOptions, criterionOf, type ItemCheck, type MonotonicityOptions } from "./item-check.js";
import { itemScalability, scaleCoefficient, usableItems } from "./scalability.js";
import { columnOf, covariances, type MokkenData, mokkenData, scoreOf } from "./scores.js";
import { addViolation, emptyViolations, type Violations } from "./violations.js";

/** The default smallest difference of means that counts as a violation for each step of the highest score. */
const MINIMUM_VIOLATION_PER_STEP = 0.03;

/** Options of {@link invariantItemOrdering}. */
export interface InvariantOrderingOptions extends MonotonicityOptions {
  /**
   * Whether to remove items one at a time, the one that violates the ordering in most pairs first, until none of the
   * remaining items do.
   *
   * @defaultValue `true`
   */
  readonly selection?: boolean;
}

/** The result of checking an invariant item ordering. */
export interface InvariantItemOrdering {
  /** The violations of each item's order against the others, in the order of the matrix's items. */
  readonly items: readonly ItemCheck[];
  /** The ids of the items removed by the selection, in the order removed; none without selection. */
  readonly removed: readonly string[];
  /** The coefficient H^T of the remaining items, the scalability of the persons; `null` where undefined. */
  readonly coefficient: number | null;
}

/**
 * Checks the manifest invariant item ordering: that the items have the same order by their means in every group of
 * persons of similar rest scores.
 *
 * @remarks
 * Items are put in the order of their means, the highest first, ties by the later item first. For each pair, with the
 * rest score the sum of the other items, persons are split into groups of similar rest scores as in {@link
 * monotonicity}. A group in which the harder item has the higher mean by more than `minimumViolation` is a violation. It
 * is tested one-sided at level `alpha`: for dichotomous items by the Freeman-Tukey transformation of the counts of
 * persons passing only one item, for the others by a paired t test, which counts a group in which the harder item is
 * ahead by the same amount for every person as significant, without a statistic. The criterion combines each item's
 * counts over its pairs. With `selection`, items are removed one at a time, the one with violations of most pairs
 * first, by the larger H of the rest where two tie, until no pair has a significant violation; H^T is that of the items
 * left, the scalability of the persons over the items, from the covariances of the persons' response profiles and of
 * their sorted profiles. Persons who answered every item are used. It follows the mokken package's `check.iio` with
 * method MIIO, except that it breaks ties of H by the first item and not by chance.
 *
 * @param matrix - The responses of the persons, with at least one item that the persons answered.
 * @param options - How to select the items, the smallest violation and group size that count, and the level of the
 *   tests.
 * @returns For each item, its violations; the items removed to reach an ordering; and the scalability coefficient of
 *   what is left.
 * @throws `DataError` when the matrix is invalid, has fewer than three items or more than 1,000, fewer than two persons
 * answered every item, the scores span more than 10 categories, an option is invalid, or fewer persons answered every
 * item than the minimum group size.
 * @see {@link https://doi.org/10.1177/0013164409355697 | Ligtvoet, van der Ark, te Marvelde and Sijtsma (2010)}
 */
export function invariantItemOrdering(
  matrix: ResponseMatrix,
  options: InvariantOrderingOptions = {},
): InvariantItemOrdering {
  const checked = checkMonotonicityOptions(options);
  const { selection = true } = options;
  if (typeof selection !== "boolean") {
    throw new DataError(`The option selection must be true or false, found ${String(selection)}.`);
  }
  const data = mokkenData(matrix, "The invariant item ordering check", 3);
  const minimumViolation = checked.minimumViolation ?? MINIMUM_VIOLATION_PER_STEP * (data.categories - 1);
  const minimumGroupSize = checked.minimumGroupSize ?? defaultGroupSize(data.persons);
  if (data.persons < minimumGroupSize || minimumGroupSize < 2) {
    throw new DataError(
      `The invariant item ordering check needs a minimum group size of at least 2, and at least that many persons who answered every item; found ${minimumGroupSize} and ${data.persons}.`,
    );
  }
  const moments = covariances(data);
  const usable = usableItems(moments, data.items);
  const order = itemOrder(data);
  const pairs = pairViolations(data, order, { minimumViolation, minimumGroupSize, alpha: checked.alpha });
  const coefficients = itemScalability(moments, data.items, usable);
  const items = Array.from({ length: data.items }, (_, item) => {
    const total = pairs.items[item] ?? emptyTotals();
    return {
      id: data.itemIds[item] ?? "",
      activeComparisons: total.activeComparisons,
      violations: total.violations,
      maximumViolation: total.maximumViolation,
      violationSum: total.violationSum,
      maximumZ: total.violations > 0 && total.tested ? total.maximumZ : null,
      significantViolations: total.significantViolations,
      criterion: criterionOf({ ...total, itemCoefficient: coefficients[item] ?? null }),
    };
  });
  const removed = selection ? backwardSelection(data, moments, order, pairs.significant) : [];
  const remaining = Array.from({ length: data.items }, (_, item) => item).filter((item) => !removed.includes(item));
  return {
    items,
    removed: removed.map((item) => data.itemIds[item] ?? ""),
    coefficient: personCoefficient(data, remaining),
  };
}

/** Returns the items from the highest mean to the lowest, ties with the later item first. */
function itemOrder(data: MokkenData): number[] {
  const sums = Array.from({ length: data.items }, (_, item) =>
    columnOf(data, item).reduce((sum, score) => sum + score, 0),
  );
  const ascending = sums.map((_, item) => item).sort((a, b) => (sums[a] ?? 0) - (sums[b] ?? 0) || a - b);
  return ascending.reverse();
}

/** What the pairs of items add up to for each item, and the pairs with significant violations. */
interface PairViolations {
  readonly items: readonly Totals[];
  /** The number of significant violations of each pair, by item, then item. */
  readonly significant: Int32Array;
}

/** The violations of an item's order with the others, and whether any had a test statistic. */
interface Totals extends Violations {
  /** Whether any violation had a test statistic. */
  tested: boolean;
}

function emptyTotals(): Totals {
  return { ...emptyViolations(), tested: false };
}

interface PairSettings {
  readonly minimumViolation: number;
  readonly minimumGroupSize: number;
  readonly alpha: number;
}

/** Returns the violations of each pair of items, added up over the items of the pair. */
function pairViolations(data: MokkenData, order: readonly number[], settings: PairSettings): PairViolations {
  const totals = Array.from({ length: data.items }, emptyTotals);
  const significant = new Int32Array(data.items * data.items);
  const rowSums = Float64Array.from({ length: data.persons }, (_, person) => {
    let sum = 0;
    for (let item = 0; item < data.items; item++) sum += scoreOf(data, person, item);
    return sum;
  });
  for (let a = 0; a < order.length - 1; a++) {
    for (let b = a + 1; b < order.length; b++) {
      const easier = order[a] ?? 0;
      const harder = order[b] ?? 0;
      const found = pairResult(data, rowSums, easier, harder, settings);
      significant[easier * data.items + harder] = found.significantViolations;
      significant[harder * data.items + easier] = found.significantViolations;
      for (const item of [easier, harder]) addPair(totals[item] ?? emptyTotals(), found);
    }
  }
  return { items: totals, significant };
}

function addPair(total: Totals, pair: Totals): void {
  total.activeComparisons += pair.activeComparisons;
  total.violations += pair.violations;
  total.violationSum += pair.violationSum;
  total.significantViolations += pair.significantViolations;
  total.maximumViolation = Math.max(total.maximumViolation, pair.maximumViolation);
  total.maximumZ = Math.max(total.maximumZ, pair.maximumZ);
  total.tested = total.tested || pair.tested;
}

/** Returns the violations of the order of two items, the first with the higher mean, across rest score groups. */
function pairResult(
  data: MokkenData,
  rowSums: Float64Array,
  easier: number,
  harder: number,
  settings: PairSettings,
): Totals {
  const rest = Float64Array.from(
    rowSums,
    (sum, person) => sum - scoreOf(data, person, easier) - scoreOf(data, person, harder),
  );
  const groups = restScoreGroups(rest, settings.minimumGroupSize);
  const result = emptyTotals();
  result.activeComparisons = groups.count - 1;
  const members: number[][] = Array.from({ length: groups.count }, () => []);
  for (let person = 0; person < groups.members.length; person++) members[groups.members[person] ?? 0]?.push(person);
  for (const persons of members) {
    const first = persons.map((person) => scoreOf(data, person, easier));
    const second = persons.map((person) => scoreOf(data, person, harder));
    const difference = mean(second) - mean(first);
    if (!(difference > settings.minimumViolation)) continue;
    addViolation(result, difference);
    const test = data.categories > 2 ? pairedTest(first, second) : discordantTest(first, second);
    if (test.pValue < settings.alpha) result.significantViolations += 1;
    if (test.statistic !== null) {
      result.maximumZ = Math.max(result.maximumZ, test.statistic);
      result.tested = true;
    }
  }
  return result;
}

/** A one-sided test of a harder item with the higher mean: the statistic, if it exists, and its p value. */
interface ViolationTest {
  readonly statistic: number | null;
  readonly pValue: number;
}

/**
 * Tests with the Freeman-Tukey transformation of the numbers of persons who pass only the first or only the second of
 * two dichotomous items, as mokken does: z = |√(2k + 2 + b) - √(2n - 2k + b)| with n the number of persons passing one item,
 * k the fewer of the two counts and b = ((2k + 1 - n)² - 10 n) / (12 n), and p = 1 - Φ(z).
 */
function discordantTest(first: readonly number[], second: readonly number[]): ViolationTest {
  let onlyFirst = 0;
  let onlySecond = 0;
  first.forEach((score, person) => {
    const other = second[person] ?? 0;
    if (score >= 1 && other < 1) onlyFirst += 1;
    if (score < 1 && other >= 1) onlySecond += 1;
  });
  const fewer = Math.min(onlyFirst, onlySecond);
  const total = onlyFirst + onlySecond;
  const correction = ((2 * fewer + 1 - total) ** 2 - 10 * total) / (12 * total);
  const statistic = Math.abs(Math.sqrt(2 * fewer + 2 + correction) - Math.sqrt(2 * total - 2 * fewer + correction));
  return { statistic, pValue: 1 - normalCdf(statistic) };
}

/**
 * Tests with a paired t test of the first item's scores against the second's, one-sided for the second being higher. A
 * group in which every person scores the same amount higher on the second item has no statistic; it is significant.
 */
function pairedTest(first: readonly number[], second: readonly number[]): ViolationTest {
  const differences = first.map((score, person) => score - (second[person] ?? 0));
  const count = differences.length;
  const center = mean(differences);
  const variance = differences.reduce((sum, value) => sum + (value - center) ** 2, 0) / (count - 1);
  if (!(variance > 0)) return { statistic: null, pValue: 0 };
  const t = center / Math.sqrt(variance / count);
  return { statistic: Math.abs(t), pValue: studentLowerTail(t, count - 1) };
}

/**
 * Returns the items removed, in order: the one whose pairs have significant violations most often, the item whose
 * removal leaves the largest H of the rest where several tie, the first of those where H ties too.
 */
function backwardSelection(
  data: MokkenData,
  moments: ReturnType<typeof covariances>,
  order: readonly number[],
  significant: Int32Array,
): number[] {
  const { items } = data;
  const removed: number[] = [];
  const active = Int32Array.from(significant);
  for (;;) {
    const counts = order.map((item) => {
      let partners = 0;
      for (let other = 0; other < items; other++) if ((active[other * items + item] ?? 0) > 0) partners += 1;
      return partners;
    });
    if (counts.every((count) => count === 0)) return removed;
    const most = counts.reduce((largest, count) => Math.max(largest, count), 0);
    const tied = order.filter((_, position) => counts[position] === most);
    const chosen = tied.length === 1 ? (tied[0] ?? 0) : tiedByCoefficient(data, moments, tied, removed);
    for (let other = 0; other < items; other++) {
      active[chosen * items + other] = 0;
      active[other * items + chosen] = 0;
    }
    removed.push(chosen);
  }
}

/** Returns the candidate whose removal, with those already removed, leaves the items with the largest H. */
function tiedByCoefficient(
  data: MokkenData,
  moments: ReturnType<typeof covariances>,
  candidates: readonly number[],
  removed: readonly number[],
): number {
  let best = candidates[0] ?? 0;
  let bestH = Number.NEGATIVE_INFINITY;
  for (const candidate of candidates) {
    const rest = Array.from({ length: data.items }, (_, item) => item).filter(
      (item) => item !== candidate && !removed.includes(item),
    );
    const h = scaleCoefficient(moments, data.items, rest) ?? Number.NEGATIVE_INFINITY;
    if (h > bestH) [best, bestH] = [candidate, h];
  }
  return best;
}

/**
 * Returns H^T of the items: the scalability of the persons, Σ Cov(person p, person q) over the pairs of persons over
 * Σ Cov_max, each covariance over the items and the maximum that of the persons' sorted scores. The sums over pairs
 * come from the sums over persons of the centered scores of each item (and of each rank), less the sum of squares of the
 * persons' own scores, which both sums share.
 */
function personCoefficient(data: MokkenData, items: readonly number[]): number | null {
  if (items.length < 2) return null;
  const byItem = new Float64Array(items.length);
  const byRank = new Float64Array(items.length);
  let own = 0;
  for (let person = 0; person < data.persons; person++) {
    const scores = Float64Array.from(items, (item) => scoreOf(data, person, item));
    const center = mean(scores);
    scores.forEach((score, k) => {
      byItem[k] = (byItem[k] ?? 0) + (score - center);
      own += (score - center) ** 2;
    });
    scores.sort().forEach((score, k) => {
      byRank[k] = (byRank[k] ?? 0) + (score - center);
    });
  }
  const sumOfSquares = (values: Float64Array): number => values.reduce((sum, value) => sum + value * value, 0);
  const denominator = sumOfSquares(byRank) - own;
  return denominator > 0 ? (sumOfSquares(byItem) - own) / denominator : null;
}

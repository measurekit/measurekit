import { checkOptions, mean, normalDensity, normalQuantile } from "@measurekit/core";
import { checkResponseMatrix, type ResponseMatrix } from "../data/internal.js";
import { correlation, variance } from "../math/internal.js";
import { completeCases, itemScores } from "./complete-cases.js";
import { GROUP_OPTIONS, type GroupOptions, groupMean, scoreGroups } from "./groups.js";
import { alphaFromVariances } from "./reliability.js";

/**
 * Classical statistics of one item. `count`, `mean` and `sd` use every person who answered the item; the other
 * statistics use the persons who answered every item. A statistic that is undefined for the data is `null`.
 */
export interface ItemStatistics {
  /** The item id. */
  readonly id: string;
  /** Persons who answered the item. */
  readonly count: number;
  /** Mean score: the p-value (difficulty) of a dichotomous item. */
  readonly mean: number | null;
  /** Standard deviation of the scores (denominator n - 1). */
  readonly sd: number | null;
  /** Correlation of the item score with the total score: the point-biserial correlation of a dichotomous item. */
  readonly itemTotal: number | null;
  /** Correlation of the item score with the total score of the other items (corrected item-total correlation). */
  readonly itemRest: number | null;
  /**
   * Biserial correlation of a dichotomous item with the total score, r √(pq) / φ(z), where r is `itemTotal`, p the
   * share of correct responses, q = 1 - p and φ(z) the normal density at the p quantile. `null` for other items.
   */
  readonly biserial: number | null;
  /**
   * Upper-lower discrimination index D: the mean score in the upper group minus the mean score in the lower group,
   * by total score. For a dichotomous item, the difference of the p-values.
   */
  readonly discriminationIndex: number | null;
  /** Cronbach's alpha of the other items. `null` with fewer than three items. */
  readonly alphaIfDeleted: number | null;
}

/**
 * Returns the classical statistics of every item, in item order.
 *
 * @param matrix - The responses of the persons.
 * @param options - The share of persons in the upper and lower groups.
 * @returns The statistics of every item, in item order.
 * @throws `DataError` when the matrix is malformed or `groupFraction` is out of range.
 *
 * @example
 * ```ts
 * import { itemStatistics } from "@measurekit/psychometrics/ctt";
 * import { buildResponseMatrix } from "@measurekit/psychometrics/data";
 *
 * const matrix = buildResponseMatrix([
 *   [1, 1, 0, 1],
 *   [0, 0, 1, 0],
 *   [1, 0, 0, 1],
 *   [1, 1, 1, 1],
 *   [0, 1, 0, 0],
 *   [0, 0, 0, 0],
 * ]);
 * itemStatistics(matrix)[0]; // => { id: "item1", count: 6, mean: 0.5, sd: 0.55, itemTotal: 0.87, itemRest: 0.71, ... }
 * ```
 */
export function itemStatistics(matrix: ResponseMatrix, options: GroupOptions = {}): readonly ItemStatistics[] {
  checkResponseMatrix(matrix);
  checkOptions(options, GROUP_OPTIONS);
  const cases = completeCases(matrix);
  const groups = scoreGroups(cases.totals, options);
  const scores = matrix.itemIds.map((_, item) => itemScores(cases, [item]));
  const variances = scores.map((x) => (x.length >= 2 ? variance(x) : 0));
  const itemVariance = variances.reduce((sum, value) => sum + value, 0);
  return scores.map((x, item) => {
    const answered = answeredScores(matrix, item);
    const rest = cases.totals.map((total, row) => total - (x[row] ?? 0));
    const itemTotal = correlation(x, cases.totals);
    return {
      id: matrix.itemIds[item] ?? "",
      count: answered.length,
      mean: answered.length > 0 ? mean(answered) : null,
      sd: answered.length > 1 ? Math.sqrt(variance(answered)) : null,
      itemTotal,
      itemRest: correlation(x, rest),
      biserial: answered.every((score) => score <= 1) ? biserial(itemTotal, x) : null,
      discriminationIndex: difference(groupMean(groups.upper, x), groupMean(groups.lower, x)),
      alphaIfDeleted: alphaOfRest(scores.length - 1, itemVariance - (variances[item] ?? 0), rest),
    };
  });
}

function answeredScores(matrix: ResponseMatrix, item: number): Float64Array {
  const scores: number[] = [];
  for (const row of matrix.responses) {
    const response = row[item];
    if (response !== null && response !== undefined) scores.push(response);
  }
  return Float64Array.from(scores);
}

/** Converts the point-biserial correlation of a dichotomous item with scores `x` to a biserial correlation. */
function biserial(pointBiserial: number | null, x: Float64Array): number | null {
  if (pointBiserial === null) return null;
  const p = mean(x);
  return (pointBiserial * Math.sqrt(p * (1 - p))) / normalDensity(normalQuantile(p));
}

function difference(upper: number | null, lower: number | null): number | null {
  return upper === null || lower === null ? null : upper - lower;
}

/** Returns alpha of the `itemCount` other items from the sum of their variances and their total, `rest`. */
function alphaOfRest(itemCount: number, itemVariance: number, rest: Float64Array): number | null {
  const restVariance = rest.length >= 2 ? variance(rest) : 0;
  return itemCount >= 2 && restVariance > 0 ? alphaFromVariances(itemCount, itemVariance, restVariance) : null;
}

import { checkChoice, checkOptions, checkUnitInterval, DataError, mean } from "@measurekit/core";
import { checkDichotomous } from "../data/dichotomous.js";
import { checkResponseMatrix, type ResponseMatrix } from "../data/response-matrix.js";
import { correlation, covariance, variance } from "../math/statistics.js";
import { type CompleteCases, completeCases, itemScores } from "./complete-cases.js";

/** How to split the items into two halves. */
export type SplitMethod = "odd-even" | "first-second";

const SPLIT_METHODS: readonly SplitMethod[] = ["odd-even", "first-second"];

/** Options of {@link splitHalf}. */
export interface SplitHalfOptions {
  /**
   * `odd-even` splits items 1, 3, 5, ... from items 2, 4, 6, ...; `first-second` splits the first k / 2 items, rounded
   * down, from the rest.
   *
   * @defaultValue `"odd-even"`
   */
  readonly split?: SplitMethod;
}

/** Split-half reliability. */
export interface SplitHalf {
  /** The correlation r of the half scores. */
  readonly correlation: number;
  /** The correlation stepped up to the full test with the Spearman-Brown formula, 2r / (1 + r). */
  readonly spearmanBrown: number;
  /**
   * Guttman's split-half coefficient (Flanagan-Rulon), 4 cov(A, B) / var(A + B), which does not assume halves of equal
   * variance.
   */
  readonly guttman: number;
}

/**
 * Returns Cronbach's coefficient alpha, k / (k - 1) * (1 - sum of item variances / total score variance), over the
 * persons who answered every item.
 *
 * @throws `DataError` when there are fewer than two items, fewer than two persons answered every item, or
 * their total scores have no variance.
 */
export function cronbachAlpha(matrix: ResponseMatrix): number {
  return alpha(reliabilityCases(matrix, "Alpha"), "Alpha");
}

/**
 * Returns the Kuder-Richardson formula 20 coefficient: {@link cronbachAlpha} of dichotomous items.
 *
 * @throws `DataError` when a response is neither 0 nor 1, or in the cases listed for {@link cronbachAlpha}.
 */
export function kr20(matrix: ResponseMatrix): number {
  const cases = reliabilityCases(matrix, "KR-20");
  checkDichotomous(matrix, "KR-20");
  return alpha(cases, "KR-20");
}

/**
 * Returns the Kuder-Richardson formula 21 coefficient, k / (k - 1) * (1 - M (k - M) / (k s²)), from the mean M and
 * the variance s² (denominator n) of the total scores of the persons who answered every item. It assumes items of
 * equal difficulty and is at most KR-20.
 *
 * @throws `DataError` when a response is neither 0 nor 1, or in the cases listed for {@link cronbachAlpha}.
 */
export function kr21(matrix: ResponseMatrix): number {
  const cases = reliabilityCases(matrix, "KR-21");
  checkDichotomous(matrix, "KR-21");
  const k = matrix.itemIds.length;
  const n = cases.persons.length;
  const populationVariance = (nonzeroTotalVariance(cases, "KR-21") * (n - 1)) / n;
  const m = mean(cases.totals);
  return (k / (k - 1)) * (1 - (m * (k - m)) / (k * populationVariance));
}

/**
 * Returns the split-half reliability over the persons who answered every item.
 *
 * @throws `DataError` when the split is unknown, there are fewer than two items, fewer than two persons answered every
 * item, a half score has no variance, or the half scores correlate at -1.
 */
export function splitHalf(matrix: ResponseMatrix, options: SplitHalfOptions = {}): SplitHalf {
  checkOptions(options);
  const { split = "odd-even" } = options;
  checkChoice(split, "The option split", SPLIT_METHODS);
  const cases = reliabilityCases(matrix, "Split-half reliability");
  const items = matrix.itemIds.map((_, item) => item);
  const half =
    split === "odd-even" ? items.filter((item) => item % 2 === 0) : items.slice(0, Math.floor(items.length / 2));
  const a = itemScores(cases, half);
  const b = itemScores(
    cases,
    items.filter((item) => !half.includes(item)),
  );
  const r = correlation(a, b);
  if (r === null) throw new DataError("Split-half reliability needs half scores with variance.");
  if (r <= -1) throw new DataError("Split-half reliability is undefined for half scores that correlate at -1.");
  return {
    correlation: r,
    spearmanBrown: (2 * r) / (1 + r),
    guttman: (4 * covariance(a, b)) / variance(cases.totals),
  };
}

/**
 * Returns the standard error of measurement, s √(1 - reliability), where s is the standard deviation of the total
 * scores of the persons who answered every item.
 *
 * @param reliability - A reliability coefficient from 0 to 1.
 * @throws `DataError` when `reliability` is outside [0, 1], or in the cases listed for {@link cronbachAlpha}.
 * @defaultValue `reliability` is {@link cronbachAlpha} of `matrix`
 */
export function standardErrorOfMeasurement(matrix: ResponseMatrix, reliability?: number): number {
  const statistic = "The standard error of measurement";
  const cases = reliabilityCases(matrix, statistic);
  const coefficient = reliability === undefined ? alpha(cases, statistic) : reliability;
  checkUnitInterval(coefficient, "A reliability");
  return Math.sqrt(variance(cases.totals) * (1 - coefficient));
}

/** Returns coefficient alpha of `itemCount` items, at least two, from their variances and their total's variance. */
export function alphaFromVariances(itemCount: number, itemVariance: number, totalVariance: number): number {
  return (itemCount / (itemCount - 1)) * (1 - itemVariance / totalVariance);
}

function alpha(cases: CompleteCases, statistic: string): number {
  let itemVariance = 0;
  for (let item = 0; item < cases.itemCount; item++) itemVariance += variance(itemScores(cases, [item]));
  return alphaFromVariances(cases.itemCount, itemVariance, nonzeroTotalVariance(cases, statistic));
}

/**
 * Checks a matrix for a reliability coefficient and returns the persons who answered every item.
 *
 * @throws `DataError` with fewer than two items or fewer than two such persons.
 */
export function reliabilityCases(matrix: ResponseMatrix, statistic: string): CompleteCases {
  checkResponseMatrix(matrix);
  if (matrix.itemIds.length < 2) throw new DataError(`${statistic} needs at least two items.`);
  const cases = completeCases(matrix);
  if (cases.persons.length < 2) {
    throw new DataError(`${statistic} needs at least two persons who answered every item.`);
  }
  return cases;
}

function nonzeroTotalVariance(cases: CompleteCases, statistic: string): number {
  const result = variance(cases.totals);
  if (result === 0) throw new DataError(`${statistic} needs total scores with variance.`);
  return result;
}

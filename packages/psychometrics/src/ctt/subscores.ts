import { DataError } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { covariance, variance } from "../math/statistics.js";
import type { CompleteCases } from "./complete-cases.js";
import { sectionCases, sectionStatistics } from "./sections.js";

/** Whether the subscore of one section is worth reporting: how well it and the total score predict its true score. */
export interface SubscoreValue {
  /** The section's name. */
  readonly section: string;
  /**
   * The proportion reduction in mean squared error (PRMSE) of the observed subscore as a predictor of the true
   * subscore: the subscore's reliability, coefficient alpha of the section's items. `null` for a section whose alpha is
   * not above 0, which has no true-score variance to predict.
   */
  readonly subscorePrmse: number | null;
  /**
   * The PRMSE of the total score as a predictor of the true subscore: the squared correlation of the two. `null` where
   * `subscorePrmse` is.
   */
  readonly totalPrmse: number | null;
  /** Whether the subscore predicts its true score better than the total does: `subscorePrmse` above `totalPrmse`. */
  readonly addedValue: boolean;
}

/**
 * Returns, for each section, whether its subscore has value over the total score, by the proportion reduction in mean
 * squared error of Haberman: the subscore is worth reporting only if it predicts its true score better than the total
 * score does.
 *
 * @remarks
 * A section's subscore s is the sum of its items and the total score x the sum of the items of all sections; items in
 * no section are left out. With ρ the reliability of s, taken as coefficient alpha of its items, and the errors of
 * different sections uncorrelated, the PRMSE of s for the true subscore is ρ and that of x is the squared correlation
 * of the true subscore and x, [cov(s, x) - (1 - ρ) var(s)]² / (ρ var(s) var(x)): the covariance of the true subscore with
 * x is that of s with x less the error variance of s, which x contains. A subscore adds value when ρ exceeds that,
 * which a subscore with many items and a low correlation with the other sections does. The statistics use the persons
 * who answered every item of the sections, as `cronbachAlpha` does for its items.
 *
 * A section needs at least two items, for alpha; sections need at least two, as with one the total is the subscore. A
 * section whose subscore has no variance, or whose alpha is not above 0, gets `null` PRMSEs and no added value.
 *
 * @param matrix - The responses of the persons.
 * @param sections - The ids of the items of each section by the section's name, such as `{ reading: ["r1", "r2"],
 *   listening: ["l1", "l2"] }`. A section's ids are items of the matrix, and no item is in two sections.
 * @returns For each section, the PRMSE of its subscore and of the total score as predictors of the true subscore, and
 *   whether the subscore adds value.
 * @throws `DataError` when the matrix is invalid, the sections are not an object of arrays of item ids, there are fewer
 * than two sections, a section has fewer than two items, an id is not an item of the matrix or is in two places, fewer
 * than two persons answered every item of the sections, or their total scores have no variance.
 * @see {@link https://doi.org/10.3102/1076998607309474 | Haberman (2008)}
 */
export function subscoreValue(
  matrix: ResponseMatrix,
  sections: Readonly<Record<string, readonly string[]>>,
): SubscoreValue[] {
  const { cases, names, indices } = sectionCases(matrix, sections, "Subscore value");
  const totalVariance = variance(cases.totals);
  if (totalVariance === 0) throw new DataError("Subscore value needs total scores with variance.");
  return names.map((section, i) => sectionValue(section, indices[i] ?? [], cases, totalVariance));
}

/** Returns one section's PRMSEs from the complete cases, its items at `indices` of the cases' columns. */
function sectionValue(
  section: string,
  indices: readonly number[],
  cases: CompleteCases,
  totalVariance: number,
): SubscoreValue {
  const { scores, variance: subscoreVariance, alpha: reliability } = sectionStatistics(cases, indices);
  if (!(reliability > 0)) return { section, subscorePrmse: null, totalPrmse: null, addedValue: false };
  const trueCovariance = covariance(scores, cases.totals) - (1 - reliability) * subscoreVariance;
  const totalPrmse = (trueCovariance * trueCovariance) / (reliability * subscoreVariance * totalVariance);
  return { section, subscorePrmse: reliability, totalPrmse, addedValue: reliability > totalPrmse };
}

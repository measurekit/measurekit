import { mean } from "@measurekit/core";
import { standardDeviation } from "../math/internal.js";
import { isConstant, pairOfRaters } from "./ratings.js";

/**
 * Returns the standardized mean difference between two raters: the first rater's mean rating less the second's, over
 * the pooled standard deviation √((s₁² + s₂²) / 2).
 *
 * @remarks
 * The measure of rater leniency used to compare automated with human scores, a difference of the means in standard
 * deviations: positive when the first rater rates higher on average. With the subjects rated by both raters in the same
 * number for both, the pooled standard deviation is that of Cohen's d, so the value is effsize's `cohen.d` of the two
 * raters' ratings as independent samples. Uses the subjects both raters rated, with variances on n - 1. Ratings are
 * finite numbers, not only categories. It is `null` when both raters' ratings are constant, which leaves the pooled
 * standard deviation 0.
 *
 * @param ratings - The ratings of two raters, one row for each subject; `null` when missing.
 * @returns The difference of the means over the pooled standard deviation; `null` where it is undefined.
 * @throws `DataError` when the ratings do not have two raters, a rating is not a finite number or `null`, or fewer than
 * two subjects were rated by both.
 * @see {@link https://doi.org/10.1111/j.1745-3992.2011.00223.x | Williamson, Xi and Breyer (2012)}
 */
export function standardizedMeanDifference(ratings: readonly (readonly (number | null)[])[]): number | null {
  const { first, second } = pairOfRaters(ratings, "The standardized mean difference");
  if (isConstant(first) && isConstant(second)) return null;
  const pooled = Math.sqrt((standardDeviation(first) ** 2 + standardDeviation(second) ** 2) / 2);
  return (mean(first) - mean(second)) / pooled;
}

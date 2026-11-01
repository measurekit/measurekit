import { mean } from "@measurekit/core";
import { isConstant, pairOfRaters } from "./ratings.js";

/**
 * Returns the ratings less their mean, divided by the largest absolute deviation, so that their squares and products
 * neither overflow nor underflow, and a large origin costs no precision. At least one rating differs from the mean.
 */
function scaledDeviations(ratings: Float64Array): Float64Array {
  const center = mean(ratings);
  const deviations = ratings.map((rating) => rating - center);
  const largest = deviations.reduce((most, deviation) => Math.max(most, Math.abs(deviation)), 0);
  return deviations.map((deviation) => deviation / largest);
}

/**
 * Returns the Pearson correlation between two raters' ratings over the subjects both rated, or `null` when either
 * rater's ratings are constant.
 *
 * @remarks
 * The agreement of two raters in rank order, beside the standardized mean difference, which measures leniency: an
 * automated and a human rater can correlate perfectly and still differ in mean. Ratings are finite numbers, not only
 * categories. The value is the sum of the products of the deviations from the means over the square root of the
 * product of their sums of squares, held within -1 and 1 as R's `cor` holds it, where rounding of perfectly correlated
 * ratings could pass them.
 *
 * @throws `DataError` when the ratings do not have two raters, a rating is not a finite number or `null`, or fewer than
 * two subjects were rated by both.
 * @see {@link https://doi.org/10.1111/j.1745-3992.2011.00223.x | Williamson, Xi and Breyer (2012)}
 */
export function ratingCorrelation(ratings: readonly (readonly (number | null)[])[]): number | null {
  const { first, second } = pairOfRaters(ratings, "The rating correlation");
  if (isConstant(first) || isConstant(second)) return null;
  const x = scaledDeviations(first);
  const y = scaledDeviations(second);
  let products = 0;
  let squaresX = 0;
  let squaresY = 0;
  for (let i = 0; i < x.length; i++) {
    products += (x[i] ?? 0) * (y[i] ?? 0);
    squaresX += (x[i] ?? 0) ** 2;
    squaresY += (y[i] ?? 0) ** 2;
  }
  return Math.min(1, Math.max(-1, products / Math.sqrt(squaresX * squaresY)));
}

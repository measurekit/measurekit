import { DataError, describe } from "@measurekit/core";
import { multidimensionalGradedResponse } from "./graded.js";
import { checkMultidimensionalItem } from "./items.js";
import { multidimensionalThreeParameterLogistic, multidimensionalTwoParameterLogistic } from "./logistic.js";
import type { MultidimensionalItem } from "./model.js";

/**
 * Returns an item's multidimensional discrimination (MDISC): the length of its slopes, √(Σ a_d²), which is its
 * largest slope along any direction of the dimensions.
 *
 * @remarks
 * The slopes are the first entries of the item's parameter vector, so that an item of a model implemented outside the
 * package has its MDISC as well. The direction of best measurement is the slopes over their length, and the guessing
 * of the M3PL does not enter, as in mirt's `MDISC`.
 *
 * @param item - The item.
 * @returns The length of the item's slopes.
 * @throws `DataError` when the item is not an item with valid parameters for its model.
 * @see {@link https://doi.org/10.1007/978-0-387-89976-3 | Reckase (2009)}
 */
export function multidimensionalDiscrimination(item: MultidimensionalItem): number {
  checkMultidimensionalItem(item, "The item");
  const slopes = item.model.toVector(item.parameters).subarray(0, item.model.dimensionCount(item.parameters));
  return Math.hypot(...slopes);
}

/**
 * Returns an item's multidimensional difficulty (MDIFF) at each boundary between its categories: -intercept / MDISC,
 * the signed distance from the origin to the point of steepest slope along the item's direction of best measurement.
 *
 * @remarks
 * A dichotomous item has one boundary and a graded item one per intercept, in the order of the intercepts, as in
 * mirt's `MDIFF`. Only the built-in models have the intercepts it divides.
 *
 * @param item - An item of one of the built-in models.
 * @returns One difficulty per boundary between the item's categories.
 * @throws `DataError` when the item is not an item with valid parameters for its model, its model is not built in, or
 * its slopes are all 0, so that it has no direction.
 * @see {@link https://doi.org/10.1007/978-0-387-89976-3 | Reckase (2009)}
 */
export function multidimensionalDifficulty(item: MultidimensionalItem): number[] {
  const discrimination = multidimensionalDiscrimination(item);
  if (discrimination === 0) {
    throw new DataError("An item whose slopes are all 0 has no direction, and so no multidimensional difficulty.");
  }
  return Array.from(interceptsOf(item), (intercept) => -intercept / discrimination);
}

/**
 * Returns the intercepts of an item of a built-in model, one per boundary between its categories: the entries of its
 * parameter vector after the slopes, but for the M3PL's asymptote.
 */
function interceptsOf({ model, parameters }: MultidimensionalItem): Float64Array {
  const vector = model.toVector(parameters);
  const dimensions = model.dimensionCount(parameters);
  if (model === multidimensionalGradedResponse) return vector.subarray(dimensions);
  if (model === multidimensionalTwoParameterLogistic || model === multidimensionalThreeParameterLogistic) {
    return vector.subarray(dimensions, dimensions + 1);
  }
  throw new DataError(
    `Only the built-in models have the intercepts of a multidimensional difficulty, found the ${describe(model.name)} model.`,
  );
}

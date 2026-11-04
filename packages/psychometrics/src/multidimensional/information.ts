import { checkArray, checkFinite, DataError } from "@measurekit/core";
import { checkDimensions, checkMultidimensionalItems } from "./items.js";
import type { MultidimensionalItem } from "./model.js";

/**
 * Returns the test information matrix at `theta`: the sum of the items' information matrices, by rows. Its inverse is
 * the covariance of a maximum likelihood estimate of theta there.
 *
 * @param items - The items of the test.
 * @param theta - The thetas, one for each dimension.
 * @returns The information matrix, by rows.
 * @throws `DataError` when the items are not an array, there are none, an item is invalid, the items differ in their
 * number of dimensions or have more than 100, or theta is not an array of that many finite numbers.
 */
export function multidimensionalInformation(
  items: readonly MultidimensionalItem[],
  theta: readonly number[],
): readonly (readonly number[])[] {
  const dimensions = checkMultidimensionalItems(items, "Information");
  checkDimensions(dimensions, "Information");
  checkArray(theta, "Theta must be an array");
  if (theta.length !== dimensions) {
    throw new DataError(
      `Theta needs ${dimensions} values for items on ${dimensions} dimensions, found ${theta.length}.`,
    );
  }
  theta.forEach((value, d) => {
    checkFinite(value, `Theta ${d + 1}`);
  });
  const point = Float64Array.from(theta);
  const sum = new Float64Array(dimensions * dimensions);
  const matrix = new Float64Array(dimensions * dimensions);
  for (const item of items) {
    item.model.information(item.parameters, point, matrix);
    for (let k = 0; k < sum.length; k++) sum[k] = (sum[k] ?? 0) + (matrix[k] ?? 0);
  }
  return Array.from({ length: dimensions }, (_, i) => Array.from(sum.subarray(i * dimensions, (i + 1) * dimensions)));
}

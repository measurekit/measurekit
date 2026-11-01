import { checkArray, checkFinite } from "@measurekit/core";
import { checkItem } from "./items.js";
import type { IrtItem } from "./model.js";

/**
 * Returns the Fisher information of an item at `theta`.
 *
 * @param item - The item.
 * @param theta - The ability.
 * @returns The information, at least 0.
 * @throws `DataError` when the item's parameters or theta are invalid.
 */
export function itemInformation(item: IrtItem, theta: number): number {
  checkItem(item, "The item");
  checkFinite(theta, "Theta");
  return item.model.information(item.parameters, theta);
}

/**
 * Returns the test information at `theta`: the sum of the items' information.
 *
 * @param items - The items of the test.
 * @param theta - The ability.
 * @returns The information, at least 0.
 * @throws `DataError` when the items are not an array, an item is invalid or theta is not finite.
 */
export function testInformation(items: readonly IrtItem[], theta: number): number {
  checkArray(items, "The items must be an array");
  checkFinite(theta, "Theta");
  let sum = 0;
  for (const [index, item] of items.entries()) {
    checkItem(item, `Item ${index + 1}`);
    sum += item.model.information(item.parameters, theta);
  }
  return sum;
}

/**
 * Returns the conditional standard error of measurement at `theta`, 1 / √I(θ) from the test information I(θ):
 * the standard error of a maximum likelihood estimate of theta there. `Infinity` when the information is 0.
 *
 * @param items - The items of the test.
 * @param theta - The ability.
 * @returns The standard error, in the units of theta.
 * @throws `DataError` when an item's parameters or theta are invalid.
 */
export function conditionalStandardError(items: readonly IrtItem[], theta: number): number {
  return 1 / Math.sqrt(testInformation(items, theta));
}

import { checkArray, DataError } from "@measurekit/core";
import { checkModelledItem, checkModelShape, checkResponses, indexItemsWith } from "../irt/items.js";
import type { MultidimensionalItem, MultidimensionalModel } from "./model.js";

/** The methods of {@link MultidimensionalModel}, which a model must have. */
const MODEL_METHODS = [
  "categoryCount",
  "dimensionCount",
  "parameters",
  "startingValues",
  "toVector",
  "fromVector",
  "probabilities",
  "thetaGradients",
  "parameterDerivatives",
  "information",
] as const;

/**
 * Checks that a value is a multidimensional model: an object with a string name, a number of categories and the
 * methods of {@link MultidimensionalModel}.
 *
 * @param message - Says what the value should be, such as `Item 3 needs a model such as multidimensionalGradedResponse`.
 * @throws `DataError` when it is not.
 */
export function checkMultidimensionalModel(model: unknown, message: string): asserts model is MultidimensionalModel {
  checkModelShape(model, MODEL_METHODS, message);
}

/**
 * Checks that an item has a model and its parameters are valid for it.
 *
 * @param label - Names the item in the error message, such as `Item 3`.
 * @throws `DataError` when the item is not an object with a model, or with the model's message, prefixed by `label`,
 * when the parameters are invalid.
 */
export function checkMultidimensionalItem(item: unknown, label: string): asserts item is MultidimensionalItem {
  checkModelledItem(item, label, checkMultidimensionalModel, "multidimensionalTwoParameterLogistic");
}

/**
 * Checks a list of items and returns their number of dimensions.
 *
 * @param subject - Names what takes the items in the error message, such as `Information`.
 * @throws `DataError` when the items are not an array, there are none, an item is invalid, or the items differ in
 * their number of dimensions.
 */
export function checkMultidimensionalItems(items: readonly MultidimensionalItem[], subject: string): number {
  checkArray(items, `${subject} needs an array of items`);
  if (items.length === 0) throw new DataError(`${subject} needs at least one item.`);
  let dimensions = 0;
  items.forEach((item, index) => {
    checkMultidimensionalItem(item, `Item ${index + 1}`);
    const count = item.model.dimensionCount(item.parameters);
    if (index === 0) dimensions = count;
    else if (count !== dimensions) {
      throw new DataError(`Item ${index + 1} has ${count} dimensions, but item 1 has ${dimensions}.`);
    }
  });
  return dimensions;
}

/**
 * Checks the items of a test and returns their indices by id.
 *
 * @throws `DataError` when the items are not an array, an id is not a string or appears twice, or an item is invalid.
 */
export function indexMultidimensionalItems(
  items: readonly (MultidimensionalItem & { readonly id?: unknown })[],
): Map<string, number> {
  return indexItemsWith(items, checkMultidimensionalItem);
}

/**
 * Checks the items against one response each, `null` when missing, and returns their number of dimensions: without
 * items, the number of `priorDimensions`.
 *
 * @throws `DataError` naming the first invalid item or response, or when there are no items and no prior to give the
 * dimensions.
 */
export function checkMultidimensionalPattern(
  items: readonly MultidimensionalItem[],
  responses: readonly (number | null)[],
  priorDimensions?: number,
): number {
  if (Array.isArray(items) && items.length === 0 && priorDimensions === undefined) {
    throw new DataError("Scoring needs at least one item, or a prior, which gives the dimensions.");
  }
  const dimensions =
    Array.isArray(items) && items.length === 0 && priorDimensions !== undefined
      ? priorDimensions
      : checkMultidimensionalItems(items, "Scoring");
  checkResponses(items, responses);
  return dimensions;
}

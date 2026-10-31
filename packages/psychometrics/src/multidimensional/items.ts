import { checkArray, checkRecord, DataError, describe, isRecord } from "@measurekit/core";
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
  const valid =
    isRecord(model) &&
    typeof model.name === "string" &&
    typeof model.maxCategories === "number" &&
    MODEL_METHODS.every((method) => typeof model[method] === "function");
  if (!valid) throw new DataError(`${message}, found ${describe(model)}.`);
}

/**
 * Checks that an item has a model and its parameters are valid for it.
 *
 * @param label - Names the item in the error message, such as `Item 3`.
 * @throws `DataError` when the item is not an object with a model, or with the model's message, prefixed by `label`,
 * when the parameters are invalid.
 */
export function checkMultidimensionalItem(item: unknown, label: string): asserts item is MultidimensionalItem {
  if (!isRecord(item)) {
    throw new DataError(`${label} must be an item with a model and parameters, found ${describe(item)}.`);
  }
  checkMultidimensionalModel(item.model, `${label} needs a model such as multidimensionalTwoParameterLogistic`);
  try {
    item.model.toVector(item.parameters);
  } catch (error) {
    if (error instanceof DataError) throw new DataError(`${label}: ${error.message}`, { cause: error });
    throw error;
  }
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
  checkArray(items, "The items must be an array");
  const indices = new Map<string, number>();
  items.forEach((item, index) => {
    checkRecord(item, `Item ${index + 1} must be an item with an id, a model and parameters`);
    if (typeof item.id !== "string") {
      throw new DataError(`Item ${index + 1} needs a string id, found ${describe(item.id)}.`);
    }
    if (indices.has(item.id)) throw new DataError(`Item ids must be unique; "${item.id}" appears twice.`);
    checkMultidimensionalItem(item, `Item "${item.id}"`);
    indices.set(item.id, index);
  });
  return indices;
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
  checkArray(responses, "The responses must be an array");
  if (responses.length !== items.length) {
    throw new DataError(`Found ${responses.length} responses to ${items.length} items.`);
  }
  for (const [index, item] of items.entries()) {
    const response = responses[index];
    const count = item.model.categoryCount(item.parameters);
    const valid = typeof response === "number" && Number.isSafeInteger(response) && response >= 0 && response < count;
    if (response !== null && !valid) {
      throw new DataError(
        `The response to item ${index + 1} must be null or an integer from 0 to ${count - 1}, found ${describe(response)}.`,
      );
    }
  }
  return dimensions;
}

/**
 * Checks that an item's responses can be estimated with its model and returns its number of categories.
 *
 * @throws `DataError` when the item has a response above the model's categories, responses in fewer than two
 * categories, or no response in a category below its highest.
 */
export function checkItemCategories(id: string, model: MultidimensionalModel, counts: readonly number[]): number {
  if (counts.length > model.maxCategories) {
    throw new DataError(
      `Item "${id}" has a response of ${counts.length - 1}, but the ${model.name} model allows responses from 0 to ${model.maxCategories - 1}.`,
    );
  }
  if (counts.length < 2) {
    throw new DataError(`Item "${id}" has responses in fewer than two categories; calibration needs at least two.`);
  }
  const empty = counts.indexOf(0);
  if (empty >= 0) {
    throw new DataError(
      `Item "${id}" has no response in category ${empty}; calibration needs responses in every category up to the highest.`,
    );
  }
  return counts.length;
}

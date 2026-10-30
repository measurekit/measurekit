import { checkArray, checkRecord, DataError, describe, isRecord } from "@measurekit/core";
import type { IrtItem, IrtModel } from "./model.js";

/** The methods of {@link IrtModel}, which a model must have. */
const MODEL_METHODS = [
  "categoryCount",
  "parameters",
  "startingValues",
  "toVector",
  "fromVector",
  "probabilities",
  "thetaDerivatives",
  "parameterDerivatives",
  "information",
] as const;

/**
 * Checks that a value is an IRT model: an object with a string name, a number of categories and the methods of
 * {@link IrtModel}.
 *
 * @param message - Says what the value should be, such as `Item 3 needs a model such as twoParameterLogistic`.
 * @throws `DataError` when it is not.
 */
export function checkModel(model: unknown, message: string): asserts model is IrtModel {
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
export function checkItem(item: unknown, label: string): asserts item is IrtItem {
  if (!isRecord(item)) {
    throw new DataError(`${label} must be an item with a model and parameters, found ${describe(item)}.`);
  }
  checkModel(item.model, `${label} needs a model such as twoParameterLogistic`);
  try {
    item.model.toVector(item.parameters);
  } catch (error) {
    if (error instanceof DataError) throw new DataError(`${label}: ${error.message}`, { cause: error });
    throw error;
  }
}

/**
 * Checks that a list of items has at least one item and that every item's parameters are valid.
 *
 * @param subject - Names what takes the items in the error message, such as `A simulation`.
 * @throws `DataError` when the items are not an array, there are none, or an item is invalid.
 */
export function checkItemList(items: readonly IrtItem[], subject: string): void {
  checkArray(items, `${subject} needs an array of items`);
  if (items.length === 0) throw new DataError(`${subject} needs at least one item.`);
  items.forEach((item, index) => {
    checkItem(item, `Item ${index + 1}`);
  });
}

/**
 * Checks the items of a test and returns their indices by id.
 *
 * @throws `DataError` when the items are not an array, an id is not a string or appears twice, or an item is invalid.
 */
export function indexItems(items: readonly (IrtItem & { readonly id?: unknown })[]): Map<string, number> {
  checkArray(items, "The items must be an array");
  const indices = new Map<string, number>();
  items.forEach((item, index) => {
    checkRecord(item, `Item ${index + 1} must be an item with an id, a model and parameters`);
    if (typeof item.id !== "string") {
      throw new DataError(`Item ${index + 1} needs a string id, found ${describe(item.id)}.`);
    }
    if (indices.has(item.id)) throw new DataError(`Item ids must be unique; "${item.id}" appears twice.`);
    checkItem(item, `Item "${item.id}"`);
    indices.set(item.id, index);
  });
  return indices;
}

/**
 * Checks every item and that `responses` has one response per item, each `null` or a category of its item.
 *
 * @throws `DataError` naming the first invalid item or response.
 */
export function checkPattern(items: readonly IrtItem[], responses: readonly (number | null)[]): void {
  checkArray(items, "The items must be an array");
  checkArray(responses, "The responses must be an array");
  if (responses.length !== items.length) {
    throw new DataError(`Found ${responses.length} responses to ${items.length} items.`);
  }
  for (const [index, item] of items.entries()) {
    checkItem(item, `Item ${index + 1}`);
    const response = responses[index];
    const count = item.model.categoryCount(item.parameters);
    const valid = typeof response === "number" && Number.isSafeInteger(response) && response >= 0 && response < count;
    if (response !== null && !valid) {
      throw new DataError(
        `The response to item ${index + 1} must be null or an integer from 0 to ${count - 1}, found ${describe(response)}.`,
      );
    }
  }
}

/**
 * Returns an item's expected score at `theta`, Σ k P(k | θ). `probabilities` is scratch space with room for the item's
 * categories.
 */
export function expectedScore(item: IrtItem, theta: number, probabilities: Float64Array): number {
  item.model.probabilities(item.parameters, theta, probabilities);
  const count = item.model.categoryCount(item.parameters);
  let score = 0;
  for (let k = 1; k < count; k++) score += k * (probabilities[k] ?? 0);
  return score;
}

/** Returns the largest number of categories of the items, the room scratch probabilities need. */
export function largestCategoryCount(items: readonly IrtItem[]): number {
  let largest = 0;
  for (const item of items) largest = Math.max(largest, item.model.categoryCount(item.parameters));
  return largest;
}

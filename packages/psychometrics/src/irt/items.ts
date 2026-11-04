import { checkArray, checkFinite, checkRecord, DataError, describe, isRecord } from "@measurekit/core";
import { checkResponseMatrix, type ResponseMatrix } from "../data/internal.js";
import type { IrtItem, IrtModel } from "./model.js";

/** What the checks of items need of an item of either kind of model: its categories. */
export interface CategorizedItem {
  /** The item's model. */
  readonly model: { categoryCount(parameters: unknown): number };
  /** The item's parameters. */
  readonly parameters: unknown;
}

/** What the check of an item needs of its model, of either kind. */
interface ParameterizedModel {
  toVector(parameters: unknown): unknown;
}

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
 * Checks that a value is a model of either kind: an object with a string name, a number of categories and the given
 * methods.
 *
 * @param methods - The names of the methods the model must have.
 * @param message - Says what the value should be, such as `Item 3 needs a model such as twoParameterLogistic`.
 * @throws `DataError` when it is not.
 */
export function checkModelShape(model: unknown, methods: readonly string[], message: string): void {
  const valid =
    isRecord(model) &&
    typeof model.name === "string" &&
    typeof model.maxCategories === "number" &&
    methods.every((method) => typeof model[method] === "function");
  if (!valid) throw new DataError(`${message}, found ${describe(model)}.`);
}

/**
 * Checks that a value is an IRT model: an object with a string name, a number of categories and the methods of
 * {@link IrtModel}.
 *
 * @param message - Says what the value should be, such as `Item 3 needs a model such as twoParameterLogistic`.
 * @throws `DataError` when it is not.
 */
export function checkModel(model: unknown, message: string): asserts model is IrtModel {
  checkModelShape(model, MODEL_METHODS, message);
}

/**
 * Checks that an item has a model that `checkModel` accepts and that the model takes its parameters.
 *
 * @param label - Names the item in the error message, such as `Item 3`.
 * @param checkModel - Checks the model; its message names the model the item needs.
 * @param example - Names a model that an item can have, such as `twoParameterLogistic`.
 * @throws `DataError` when the item is not an object with a model, or with the model's message, prefixed by `label`,
 * when the parameters are invalid.
 */
export function checkModelledItem(
  item: unknown,
  label: string,
  checkModel: (model: unknown, message: string) => asserts model is ParameterizedModel,
  example: string,
): void {
  if (!isRecord(item)) {
    throw new DataError(`${label} must be an item with a model and parameters, found ${describe(item)}.`);
  }
  checkModel(item.model, `${label} needs a model such as ${example}`);
  try {
    item.model.toVector(item.parameters);
  } catch (error) {
    if (error instanceof DataError) throw new DataError(`${label}: ${error.message}`, { cause: error });
    throw error;
  }
}

/**
 * Checks that an item has a model and its parameters are valid for it.
 *
 * @param label - Names the item in the error message, such as `Item 3`.
 * @throws `DataError` when the item is not an object with a model, or with the model's message, prefixed by `label`,
 * when the parameters are invalid.
 */
export function checkItem(item: unknown, label: string): asserts item is IrtItem {
  checkModelledItem(item, label, checkModel, "twoParameterLogistic");
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
 * Checks a response matrix and its items: one valid item per column, and every response a category of its item.
 *
 * @throws `DataError` naming the first problem found.
 */
export function checkItemsOfMatrix(matrix: ResponseMatrix, items: readonly IrtItem[]): void {
  checkResponseMatrix(matrix);
  checkArray(items, "The items must be an array");
  const { itemIds, personIds, responses } = matrix;
  if (items.length !== itemIds.length) {
    throw new DataError(`Found ${items.length} items for ${itemIds.length} columns of responses.`);
  }
  items.forEach((item, column) => {
    const id = itemIds[column] ?? "";
    checkItem(item, `Item "${id}"`);
    const count = item.model.categoryCount(item.parameters);
    responses.forEach((row, person) => {
      const response = row[column] ?? null;
      if (response !== null && response >= count) {
        throw new DataError(
          `The response of person "${personIds[person]}" to item "${id}" is ${response}, but the item has categories 0 to ${count - 1}.`,
        );
      }
    });
  });
}

/**
 * Checks that there is one finite theta per person.
 *
 * @throws `DataError` when the thetas are not an array, their count differs or a theta is not finite.
 */
export function checkPersonThetas(matrix: ResponseMatrix, thetas: readonly number[]): void {
  checkArray(thetas, "The thetas must be an array");
  if (thetas.length !== matrix.personIds.length) {
    throw new DataError(`Found ${thetas.length} thetas for ${matrix.personIds.length} persons.`);
  }
  thetas.forEach((theta, person) => {
    checkFinite(theta, `The theta of person "${matrix.personIds[person]}"`);
  });
}

/**
 * Checks the items of a test with `checkItem` and returns their indices by id.
 *
 * @param checkItem - Checks one item; `label` names it in the error message.
 * @throws `DataError` when the items are not an array, an id is not a string or appears twice, or an item is invalid.
 */
export function indexItemsWith<TItem extends { readonly id?: unknown }>(
  items: readonly TItem[],
  checkItem: (item: TItem, label: string) => void,
): Map<string, number> {
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
 * Checks the items of a test and returns their indices by id.
 *
 * @throws `DataError` when the items are not an array, an id is not a string or appears twice, or an item is invalid.
 */
export function indexItems(items: readonly (IrtItem & { readonly id?: unknown })[]): Map<string, number> {
  return indexItemsWith(items, checkItem);
}

/**
 * Checks that `responses` has one response per item, each `null` or a category of its item, and calls `checkItem`
 * before the response to the item is checked.
 *
 * @param checkItem - Checks an item, given its index, before its response.
 * @throws `DataError` naming the first invalid item or response.
 */
export function checkResponses<TItem extends CategorizedItem>(
  items: readonly TItem[],
  responses: readonly (number | null)[],
  checkItem: (item: TItem, index: number) => void = () => undefined,
): void {
  checkArray(responses, "The responses must be an array");
  if (responses.length !== items.length) {
    throw new DataError(`Found ${responses.length} responses to ${items.length} items.`);
  }
  for (const [index, item] of items.entries()) {
    checkItem(item, index);
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
 * Checks every item and that `responses` has one response per item, each `null` or a category of its item.
 *
 * @throws `DataError` naming the first invalid item or response.
 */
export function checkPattern(items: readonly IrtItem[], responses: readonly (number | null)[]): void {
  checkArray(items, "The items must be an array");
  checkResponses(items, responses, (item, index) => {
    checkItem(item, `Item ${index + 1}`);
  });
}

/** Returns the items with a response and the responses, in the order of the items. */
export function answeredItems<TItem>(
  items: readonly TItem[],
  responses: readonly (number | null)[],
): { readonly item: TItem; readonly response: number }[] {
  return items.flatMap((item, index) => {
    const response = responses[index];
    return response === null || response === undefined ? [] : [{ item, response }];
  });
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
export function largestCategoryCount(items: readonly CategorizedItem[]): number {
  let largest = 0;
  for (const item of items) largest = Math.max(largest, item.model.categoryCount(item.parameters));
  return largest;
}

/**
 * Returns the model of each item from one model for all or a list with one per item, checking each with `check`.
 *
 * @param check - Checks one model; its message names the model the option needs.
 * @throws `DataError` when a list does not have one model per item, or a model is invalid.
 */
export function modelsOf<TModel>(
  model: TModel | readonly TModel[],
  itemCount: number,
  check: (model: TModel) => void,
): readonly TModel[] {
  const models = isList(model) ? model : Array.from({ length: itemCount }, () => model);
  if (models.length !== itemCount) throw new DataError(`Found ${models.length} models for ${itemCount} items.`);
  for (const each of models) check(each);
  return models;
}

// Array.isArray does not narrow a union with a readonly array type.
function isList<T>(value: T | readonly T[]): value is readonly T[] {
  return Array.isArray(value);
}

/**
 * Checks that each id with fixed parameters is the id of an item.
 *
 * @throws `DataError` when an id is not an item id.
 */
export function checkFixedIds(ids: Iterable<string>, itemIds: readonly string[]): void {
  for (const id of ids) {
    if (!itemIds.includes(id)) throw new DataError(`Fixed parameters are given for "${id}", which is not an item id.`);
  }
}

/**
 * Checks that the fixed parameters of an item have a category for each response to it.
 *
 * @param responseCategories - The number of categories up to the item's highest response.
 * @throws `DataError` when the item has a response beyond its fixed parameters' categories.
 */
export function checkFixedCategories(id: string, responseCategories: number, categoryCount: number): void {
  if (responseCategories > categoryCount) {
    throw new DataError(
      `Item "${id}" has a response of ${responseCategories - 1}, but its fixed parameters have ${categoryCount} categories.`,
    );
  }
}

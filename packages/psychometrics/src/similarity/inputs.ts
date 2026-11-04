import { checkArray, DataError, describe } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { checkItemsOfMatrix, checkPersonThetas, type IrtItem, largestCategoryCount } from "../irt/internal.js";
import { type PersonPair, readPairs } from "./pairs.js";

/**
 * The most items that the generalized binomial test and M4 compare in a pair: their distributions of the matches take
 * time that grows with the square and the cube of the items, and M4's table of identical correct and incorrect
 * responses 16 MB at the limit, against tests that rarely pass 300 items.
 */
const MAX_COMPARED_ITEMS = 1000;

/** A response matrix, its items and thetas, and pairs of its persons, checked. */
export interface ModelInputs {
  readonly matrix: ResponseMatrix;
  readonly items: readonly IrtItem[];
  readonly thetas: readonly number[];
  /** The person indices of the pairs: pair k at 2k and 2k + 1. */
  readonly pairs: Int32Array;
  /** Where each item's categories start in a person's probabilities, and their end after the last item. */
  readonly offsets: Int32Array;
}

/**
 * Checks the inputs of an index that takes items and thetas.
 *
 * @throws `DataError` when the matrix, the items, the thetas or the pairs are invalid, or a response is not a category
 * of its item.
 */
export function checkModelInputs(
  matrix: ResponseMatrix,
  items: readonly IrtItem[],
  thetas: readonly number[],
  pairs: readonly PersonPair[],
): ModelInputs {
  checkItemsOfMatrix(matrix, items);
  checkPersonThetas(matrix, thetas);
  const offsets = new Int32Array(items.length + 1);
  items.forEach((item, column) => {
    offsets[column + 1] = (offsets[column] ?? 0) + item.model.categoryCount(item.parameters);
  });
  return { matrix, items, thetas, pairs: readPairs(matrix.personIds, pairs), offsets };
}

/** The probabilities of every category of every item at one person's theta, kept until another person's replace them. */
export interface PersonProbabilities {
  /** The person whose probabilities `values` holds, or -1 before the first. */
  person: number;
  /** By item, then category, from the offsets of {@link ModelInputs}. */
  readonly values: Float64Array;
  /** Room for one item's probabilities. */
  readonly scratch: Float64Array;
}

/** Returns room for the probabilities of a person. */
export function personProbabilities({ items, offsets }: ModelInputs): PersonProbabilities {
  return {
    person: -1,
    values: new Float64Array(offsets[items.length] ?? 0),
    scratch: new Float64Array(largestCategoryCount(items)),
  };
}

/** Returns the probabilities of `person`, computing them unless they are those `probabilities` holds. */
export function probabilitiesOf(
  { items, thetas, offsets }: ModelInputs,
  probabilities: PersonProbabilities,
  person: number,
): Float64Array {
  const { values, scratch } = probabilities;
  if (probabilities.person === person) return values;
  const theta = thetas[person] ?? 0;
  for (let column = 0; column < items.length; column++) {
    const item = items[column];
    if (item === undefined) continue;
    item.model.probabilities(item.parameters, theta, scratch);
    const offset = offsets[column] ?? 0;
    const count = (offsets[column + 1] ?? 0) - offset;
    for (let k = 0; k < count; k++) values[offset + k] = scratch[k] ?? 0;
  }
  probabilities.person = person;
  return values;
}

/**
 * Returns the probability that two persons who answer alone give an item the same response, Σ_k p_k q_k over the
 * item's categories, which their probabilities hold from `from` to `to`.
 */
export function matchProbability(p: Float64Array, q: Float64Array, from: number, to: number): number {
  let match = 0;
  for (let index = from; index < to; index++) match += (p[index] ?? 0) * (q[index] ?? 0);
  return match;
}

/**
 * Reads the keyed category of each item: the option `key`, or category 1 of every item when it is absent.
 *
 * @param categories - The number of categories of each item, or `null` when only the responses tell them; a keyed
 * category must be below it.
 * @param absent - Says why the key is needed when the option is absent and some item is not dichotomous, or `null`
 * when every item is.
 * @throws `DataError` when the key is not an array of one category per item, or is absent while `absent` is not null.
 */
export function readKey(
  key: readonly number[] | undefined,
  itemIds: readonly string[],
  categories: (column: number) => number | null,
  absent: string | null,
): Int32Array {
  if (key === undefined) {
    if (absent !== null) throw new DataError(absent);
    return new Int32Array(itemIds.length).fill(1);
  }
  checkArray(key, "The option key must be an array of the keyed category of each item");
  if (key.length !== itemIds.length) {
    throw new DataError(`The option key has ${key.length} categories for ${itemIds.length} items.`);
  }
  return Int32Array.from(key, (category, column) => {
    const count = categories(column) ?? Number.POSITIVE_INFINITY;
    if (!(Number.isSafeInteger(category) && category >= 0 && category < count)) {
      const range = Number.isFinite(count) ? `from 0 to ${count - 1}` : "from 0";
      throw new DataError(
        `The key of item "${itemIds[column]}" must be a category, an integer ${range}, found ${describe(category)}.`,
      );
    }
    return category;
  });
}

/**
 * Checks that a pair compares at most {@link MAX_COMPARED_ITEMS} items.
 *
 * @throws `DataError` when it compares more.
 */
export function checkComparedItems(compared: number, index: string): void {
  if (compared > MAX_COMPARED_ITEMS) {
    throw new DataError(
      `${index} compares at most ${MAX_COMPARED_ITEMS} items in a pair, found a pair that answered ${compared} items in common.`,
    );
  }
}

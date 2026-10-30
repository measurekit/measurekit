import { DataError, describe } from "@measurekit/core";
import type { TestItem } from "../irt/model.js";

/**
 * An item of an adaptive test's pool.
 *
 * @typeParam TParameters - The parameters of the item's model.
 */
export interface PoolItem<TParameters = unknown> extends TestItem<TParameters> {
  /** The item's content area, which content balancing uses. */
  readonly contentArea?: string;
  /**
   * The Sympson-Hetter exposure parameter: the probability, from 0 to 1, that the item is administered when it is
   * selected.
   *
   * @defaultValue `1`
   */
  readonly exposure?: number;
}

/**
 * Returns the index of an item id.
 *
 * @throws `DataError` when the id is not one of the test's items.
 */
export function indexOf(indices: ReadonlyMap<string, number>, id: string): number {
  const index = indices.get(id);
  if (index === undefined) throw new DataError(`The test has no item ${describe(id)}.`);
  return index;
}

/**
 * Returns the item with an id.
 *
 * @throws `DataError` when the id is not one of the test's items.
 */
export function itemById<TItem extends TestItem>(
  items: readonly TItem[],
  indices: ReadonlyMap<string, number>,
  id: string,
): TItem {
  const item = items[indices.get(id) ?? -1];
  if (item === undefined) throw new DataError(`The test has no item ${describe(id)}.`);
  return item;
}

/**
 * Checks that a response is a category of an item.
 *
 * @throws `DataError` when it is not.
 */
export function checkResponse(item: TestItem, response: number): void {
  const count = item.model.categoryCount(item.parameters);
  if (!(Number.isSafeInteger(response) && response >= 0 && response < count)) {
    throw new DataError(
      `The response to item "${item.id}" must be an integer from 0 to ${count - 1}, found ${describe(response)}.`,
    );
  }
}

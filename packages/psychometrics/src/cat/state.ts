import { DataError, isRecord } from "@measurekit/core";
import type { AbilityEstimate } from "../irt/ability.js";
import { indexOf, itemById } from "./items.js";

/**
 * What the states of adaptive and multistage tests share: the responses so far and the estimate from them.
 *
 * @typeParam TEstimate - The estimate of what the test measures.
 */
export interface TestState<TEstimate = AbilityEstimate> {
  /** The ids of the administered items, in order. */
  readonly items: readonly string[];
  /** The responses to the administered items. */
  readonly responses: readonly number[];
  /** The estimate from the responses. */
  readonly estimate: TEstimate;
}

/**
 * Checks a state passed back to a test, as from JSON, and returns the indices of its administered items: `items` and
 * `responses` are arrays with one response per item, each accepted by `checkResponse` for its item.
 *
 * @throws `DataError` when they are not, or an id is not one of the test's items.
 */
export function checkState<TItem extends { readonly id: string }>(
  state: TestState<unknown>,
  items: readonly TItem[],
  indices: ReadonlyMap<string, number>,
  checkResponse: (item: TItem, response: number) => void,
): number[] {
  if (!(isRecord(state) && Array.isArray(state.items) && Array.isArray(state.responses))) {
    throw new DataError("A test state needs the arrays items and responses.");
  }
  if (state.responses.length !== state.items.length) {
    throw new DataError(
      `A test state needs one response per item, found ${state.responses.length} responses to ${state.items.length} items.`,
    );
  }
  return state.items.map((id, j) => {
    checkResponse(itemById(items, indices, id), state.responses[j] ?? Number.NaN);
    return indexOf(indices, id);
  });
}

/**
 * Checks that no item index appears twice: each item is administered once, or withheld.
 *
 * @throws `DataError` naming the first item that appears twice.
 */
export function checkDistinct(indices: readonly number[], items: readonly { readonly id: string }[]): void {
  const seen = new Set<number>();
  for (const index of indices) {
    if (seen.has(index)) {
      throw new DataError(
        `A test state names item "${items[index]?.id}" twice; an item is administered or withheld once.`,
      );
    }
    seen.add(index);
  }
}

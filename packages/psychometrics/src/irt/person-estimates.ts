import { DataError } from "@measurekit/core";
import type { ResponseMatrix, ResponseRecords } from "../data/internal.js";
import { compressResponses } from "./responses.js";

/**
 * Returns the estimate of every person, in the order of the data's person ids, from the person's answered items only,
 * so that sparse data, such as adaptive tests on a large pool, need no row per person over all items. A person without
 * responses has the estimate `null`.
 *
 * @param indexItems - Checks the items, of either kind of model, and returns their indices by id.
 * @param estimator - Estimates one person from the answered items and their responses.
 * @throws `DataError` when the data are malformed, `indexItems` throws it, an item of the data has no item with its id,
 * or the estimator throws it.
 */
export function estimatePersons<TItem extends { readonly id: string }, TEstimate>(
  data: ResponseMatrix | ResponseRecords,
  items: readonly TItem[],
  indexItems: (items: readonly TItem[]) => Map<string, number>,
  estimator: (items: readonly TItem[], responses: readonly (number | null)[]) => TEstimate,
): { readonly id: string; readonly estimate: TEstimate | null }[] {
  const responses = compressResponses(data);
  const indices = indexItems(items);
  const columns = responses.itemIds.map((id) => {
    const item = items[indices.get(id) ?? -1];
    if (item === undefined) throw new DataError(`The data have item "${id}", which is not among the items.`);
    return item;
  });
  return responses.personIds.map((id, person) => {
    const start = responses.offsets[person] ?? 0;
    const end = responses.offsets[person + 1] ?? 0;
    if (end === start) return { id, estimate: null };
    const answered = Array.from(responses.items.subarray(start, end)).flatMap((column) => columns[column] ?? []);
    return { id, estimate: estimator(answered, Array.from(responses.codes.subarray(start, end))) };
  });
}

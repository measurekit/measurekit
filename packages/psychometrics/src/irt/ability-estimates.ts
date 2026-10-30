import { DataError } from "@measurekit/core";
import { checkFunction } from "../checks.js";
import type { ResponseMatrix } from "../data/response-matrix.js";
import type { ResponseRecords } from "../data/response-records.js";
import { type AbilityEstimator, posteriorMeanEstimate } from "./ability.js";
import { indexItems } from "./items.js";
import type { PersonEstimate } from "./joint-maximum-likelihood.js";
import type { TestItem } from "./model.js";
import { compressResponses } from "./responses.js";

/**
 * Returns the estimate of theta of every person, in the order of the data's person ids, from the person's answered
 * items only, so that sparse data, such as adaptive tests on a large pool, need no row per person over all items.
 * Items are matched to the data by id, as a calibration's items are. Persons without responses get `null` estimates.
 *
 * @param estimator - The estimator of each person.
 * @throws `DataError` when the data are malformed, an item is invalid, item ids repeat, an item of the data has no item
 * with its id, a response is not a category of its item, the estimator is not a function, or it throws it.
 * @defaultValue `estimator` is `posteriorMeanEstimate`
 */
export function abilityEstimates(
  data: ResponseMatrix | ResponseRecords,
  items: readonly TestItem[],
  estimator: AbilityEstimator = posteriorMeanEstimate,
): PersonEstimate[] {
  checkFunction(estimator, "The estimator must be a function of items and responses, such as posteriorMeanEstimate");
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
    if (end === start) return { id, theta: null, standardError: null };
    const answered = Array.from(responses.items.subarray(start, end)).flatMap((column) => columns[column] ?? []);
    const { theta, standardError } = estimator(answered, Array.from(responses.codes.subarray(start, end)));
    return { id, theta, standardError };
  });
}

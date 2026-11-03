import { checkOptions } from "@measurekit/core";
import { checkFunction } from "../checks.js";
import type { ResponseMatrix, ResponseRecords } from "../data/internal.js";
import { type AbilityEstimator, posteriorMeanEstimate } from "./ability.js";
import { indexItems } from "./items.js";
import type { PersonEstimate } from "./joint-maximum-likelihood.js";
import type { TestItem } from "./model.js";
import { estimatePersons } from "./person-estimates.js";
import { sharedPosteriorMeanEstimator } from "./shared-posterior-mean.js";

/** Options of {@link abilityEstimates}. */
export interface AbilityEstimatesOptions {
  /**
   * The estimator of each person.
   *
   * @defaultValue `posteriorMeanEstimate`
   */
  readonly estimator?: AbilityEstimator;
}

/**
 * Returns the estimate of theta of every person, in the order of the data's person ids, from the person's answered
 * items only, so that sparse data, such as adaptive tests on a large pool, need no row per person over all items.
 * Items are matched to the data by id, as a calibration's items are. Persons without responses get `null` estimates. The
 * posterior mean estimator takes the log-probabilities of the responses at the nodes once for all persons.
 *
 * @param data - The responses of the persons.
 * @param items - The items, matched to the data by id.
 * @param options - The estimator of each person.
 * @returns The estimate and standard error of each person, with `null` for a person without responses.
 * @throws `DataError` when the data are malformed, an item is invalid, item ids repeat, an item of the data has no item
 * with its id, a response is not a category of its item, the options are not an object, the estimator is not a
 * function, or it throws it.
 */
export function abilityEstimates(
  data: ResponseMatrix | ResponseRecords,
  items: readonly TestItem[],
  options: AbilityEstimatesOptions = {},
): PersonEstimate[] {
  checkOptions(options);
  const { estimator = posteriorMeanEstimate } = options;
  checkFunction(estimator, "The estimator must be a function of items and responses, such as posteriorMeanEstimate");
  const estimate = estimator === posteriorMeanEstimate ? sharedPosteriorMeanEstimator() : estimator;
  return estimatePersons(data, items, indexItems, estimate).map(({ id, estimate }) => ({
    id,
    theta: estimate?.theta ?? null,
    standardError: estimate?.standardError ?? null,
  }));
}

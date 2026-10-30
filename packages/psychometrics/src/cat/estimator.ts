import { checkFunction } from "../checks.js";
import type { AbilityEstimator } from "../irt/ability.js";
import type { IrtItem } from "../irt/model.js";

/**
 * Returns an estimator that uses `initial` until the responses are mixed, and `mixed` from then on.
 *
 * @remarks
 * Responses are mixed when one is above its item's lowest category and one below its highest: before that, the
 * likelihood grows toward an end of the theta range and the maximum likelihood estimate is that end. Missing responses
 * are skipped. `untilMixed(posteriorMeanEstimate, maximumLikelihoodEstimate)` gives EAP estimates until the first right
 * and the first wrong answer, and ML estimates after.
 *
 * @throws `DataError` when `initial` or `mixed` is not a function.
 */
export function untilMixed(initial: AbilityEstimator, mixed: AbilityEstimator): AbilityEstimator {
  checkFunction(initial, "The initial estimator must be a function of items and responses");
  checkFunction(mixed, "The mixed estimator must be a function of items and responses");
  return (items, responses) => (isMixed(items, responses) ? mixed : initial)(items, responses);
}

function isMixed(items: readonly IrtItem[], responses: readonly (number | null)[]): boolean {
  let aboveLowest = false;
  let belowHighest = false;
  responses.forEach((response, index) => {
    const item = items[index];
    if (response === null || item === undefined) return;
    aboveLowest ||= response > 0;
    belowHighest ||= response < item.model.categoryCount(item.parameters) - 1;
  });
  return aboveLowest && belowHighest;
}

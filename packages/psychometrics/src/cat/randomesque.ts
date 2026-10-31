import { checkMethod, checkPositiveInteger } from "@measurekit/core";
import type { PoolControls } from "./items.js";
import type { ItemSelectionRule } from "./selection.js";

/**
 * Returns randomesque selection: one of the `count` eligible items that `rule` scores highest, each with the same
 * probability, the first in pool order on ties. It scores those items with uniform draws from `context.random` and the
 * others `-Infinity`. Items `rule` scores `-Infinity` are not selected.
 *
 * @throws `DataError` when `rule` is not a selection rule or `count` is not a positive integer.
 * @see {@link https://doi.org/10.1207/s15324818ame0204_6 | Kingsbury and Zara (1989)}
 */
export function randomesque<TItem extends PoolControls, TEstimate>(
  rule: ItemSelectionRule<TItem, TEstimate>,
  count: number,
): ItemSelectionRule<TItem, TEstimate> {
  checkMethod(rule, "scores", "Randomesque selection needs a selection rule, such as maximumInformation");
  checkPositiveInteger(count, "The number of items of randomesque selection");
  return {
    scores(context, out) {
      rule.scores(context, out);
      const candidates: number[] = [];
      context.eligible.forEach((flag, index) => {
        if (flag === 1 && (out[index] ?? Number.NEGATIVE_INFINITY) > Number.NEGATIVE_INFINITY) candidates.push(index);
      });
      candidates.sort((i, j) => (out[j] ?? 0) - (out[i] ?? 0) || i - j);
      candidates.forEach((index, rank) => {
        out[index] = rank < count ? context.random.uniform() : Number.NEGATIVE_INFINITY;
      });
    },
  };
}

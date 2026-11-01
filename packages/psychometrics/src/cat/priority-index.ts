import { checkMethod, checkPositiveInteger, DataError } from "@measurekit/core";
import {
  type ContentBounds,
  type ContentConstraint,
  countAreas,
  poolAreas,
  readBounds,
} from "./content-constraints.js";
import type { PoolControls } from "./items.js";
import type { ItemSelectionRule } from "./selection.js";

/**
 * Returns the factor of each area, `weight × f1 × f2` of Su, or `weight` where the items outside the area have used up
 * the places that its minimum leaves them (`f2` is 0 or less, or the minimum is the length).
 *
 * `f1 = (maximum - given) / maximum` is the share of the maximum left, and 0 for a maximum of 0;
 * `f2 = ((length - minimum) - (count - given)) / (length - minimum)` is the share of the items outside the area that
 * can still be given.
 */
function factors(
  { minimums, maximums, weights }: ContentBounds,
  length: number,
  given: Int32Array,
  count: number,
): Float64Array {
  return Float64Array.from(weights, (weight, k) => {
    const outside = length - (minimums[k] ?? 0);
    const share = outside > 0 ? (outside - (count - (given[k] ?? 0))) / outside : 0;
    if (!(share > 0)) return weight;
    const maximum = maximums[k] ?? 0;
    const room = maximum > 0 ? Math.max(0, (maximum - (given[k] ?? 0)) / maximum) : 0;
    return weight * room * share;
  });
}

/**
 * Returns the maximum priority index of Cheng and Chang: a rule that multiplies the scores of `rule` by the priority of
 * the content areas each item is in, so that items of areas near their bounds are taken first.
 *
 * @remarks
 * The score of item `i` is `score × Π (weight × f)` over the constrained areas it is in, with `f = f1 × f2` of Su: `f1 = (maximum - given) / maximum` falls to 0 as an area reaches its maximum, and
 * `f2 = ((length - minimum) - (count - given)) / (length - minimum)` falls to 0 as the items outside the area use up the
 * places that the minimum leaves them; `f` is 1 once `f2` is 0 or less. An item in no constrained area keeps its score.
 * The rule scores `-Infinity` where `rule` does, and it throws for a finite score below 0, as a product of a negative
 * score reverses the order: use {@link contentBounds}, alone, with rules such as `aStratified` and `shannonEntropy`
 * that score below 0.
 *
 * The index orders items; it does not hold the bounds, as an item outside an area that is needed can still win. Wrap it
 * in {@link contentBounds}. It differs from the filter alone where three or more areas compete: with two areas and exact
 * bounds `f` is equal for both at every step, so the index gives the filter's test. Areas are counted by the pool
 * array, so a pool must not change after its first use.
 *
 * @param rule - The rule that scores the items.
 * @param constraints - The area, the minimum, the maximum and the weight, the bounds as integers from 0.
 * @param length - The most items a test may have, a positive integer.
 * @throws `DataError` when `rule` is not a selection rule, the constraints are invalid or on one area twice, or
 * `length` is not a positive integer; from the rule, when `rule` scores an eligible item finitely below 0.
 * @see {@link https://doi.org/10.1348/000711008X304376 | Cheng and Chang (2009)}
 * @see {@link https://doi.org/10.1177/0146621616639305 | Su (2016)}
 */
export function maximumPriorityIndex<TItem extends PoolControls, TEstimate>(
  rule: ItemSelectionRule<TItem, TEstimate>,
  constraints: readonly ContentConstraint[],
  length: number,
): ItemSelectionRule<TItem, TEstimate> {
  checkMethod(rule, "scores", "maximumPriorityIndex needs a selection rule, such as maximumInformation");
  const bounds = readBounds(constraints);
  checkPositiveInteger(length, "The length of maximumPriorityIndex");
  const itemAreasOf = poolAreas(bounds.areas);
  return {
    scores(context, out) {
      rule.scores(context, out);
      const { pool, administered, eligible } = context;
      const itemAreas = itemAreasOf(pool);
      const factor = factors(
        bounds,
        length,
        countAreas(administered, itemAreas, bounds.areas.length),
        administered.length,
      );
      eligible.forEach((flag, index) => {
        if (flag !== 1) return;
        const score = out[index] ?? Number.NEGATIVE_INFINITY;
        if (score === Number.NEGATIVE_INFINITY) return;
        if (!(score >= 0)) {
          throw new DataError(
            `The rule scored item "${pool[index]?.id}" ${score}; maximumPriorityIndex multiplies scores of at least 0, so give a rule that scores below 0 to contentBounds alone.`,
          );
        }
        let priority = score;
        for (const area of itemAreas[index] ?? []) priority *= factor[area] ?? 1;
        out[index] = priority;
      });
    },
  };
}

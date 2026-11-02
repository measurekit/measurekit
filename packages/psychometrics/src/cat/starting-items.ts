import { checkArray, checkMethod, DataError, describe, isNonEmptyString } from "@measurekit/core";
import type { PoolControls } from "./items.js";
import type { ItemSelectionRule } from "./selection.js";

/**
 * Returns a rule that administers fixed starting items in order, then follows `rule`.
 *
 * @remarks
 * Each of the first `ids.length` selections of a test takes the first of `ids` that is eligible, scored 0 with every
 * other item `-Infinity`; an id not eligible, such as one administered, withheld or an enemy of an administered item,
 * is skipped. A selection where none of `ids` is eligible, and every later selection, follows `rule`. The starting items
 * take the place of catR's `startItems(fixItems)`; without them, the test's first item is the one `rule` scores
 * highest at the estimate without responses.
 *
 * @param ids - The ids of the starting items, in the order to administer them.
 * @param rule - The rule that selects the items after them.
 * @returns The selection rule, whose `scores` throws `DataError` when an id is not in the pool.
 * @throws `DataError` when `ids` is not an array of distinct non-empty strings or `rule` is not a selection rule.
 */
export function startingItems<TItem extends PoolControls, TEstimate>(
  ids: readonly string[],
  rule: ItemSelectionRule<TItem, TEstimate>,
): ItemSelectionRule<TItem, TEstimate> {
  checkArray(ids, "The starting items must be an array of item ids");
  const seen = new Set<string>();
  ids.forEach((id, k) => {
    if (!isNonEmptyString(id)) {
      throw new DataError(`Starting item ${k + 1} must be a non-empty item id, found ${describe(id)}.`);
    }
    if (seen.has(id)) throw new DataError(`The starting items name item "${id}" twice.`);
    seen.add(id);
  });
  checkMethod(rule, "scores", "Starting items need a selection rule to follow them, such as maximumInformation");
  const positionsByPool = new WeakMap<readonly TItem[], readonly number[]>();
  return {
    scores(context, out) {
      const { pool, administered, eligible } = context;
      if (administered.length < ids.length) {
        let positions = positionsByPool.get(pool);
        if (positions === undefined) {
          positions = poolPositions(pool, ids);
          positionsByPool.set(pool, positions);
        }
        const first = positions.find((index) => eligible[index] === 1);
        if (first !== undefined) {
          eligible.forEach((flag, index) => {
            if (flag === 1) out[index] = index === first ? 0 : Number.NEGATIVE_INFINITY;
          });
          return;
        }
      }
      rule.scores(context, out);
    },
  };
}

/**
 * Returns the pool index of each id.
 *
 * @throws `DataError` when an id is not in the pool.
 */
function poolPositions(pool: readonly PoolControls[], ids: readonly string[]): number[] {
  const indices = new Map(pool.map(({ id }, index) => [id, index]));
  return ids.map((id) => {
    const index = indices.get(id);
    if (index === undefined) throw new DataError(`The starting item "${id}" is not in the pool.`);
    return index;
  });
}

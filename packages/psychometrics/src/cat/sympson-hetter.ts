import { checkUnitInterval } from "@measurekit/core";
import type { PoolControls } from "./items.js";
import { type AdaptiveTestContext, highestScore, type ItemSelectionRule } from "./selection.js";

/**
 * Checks an item's Sympson-Hetter exposure parameter.
 *
 * @throws `DataError` unless it is missing or a number from 0 to 1.
 */
export function checkExposure(item: PoolControls): void {
  if (item.exposure !== undefined) checkUnitInterval(item.exposure, `The exposure parameter of item "${item.id}"`);
}

/**
 * Selects the next item under Sympson-Hetter exposure control.
 *
 * @remarks
 * The item the rule selects is administered with probability equal to its exposure parameter; otherwise it is withheld
 * for the rest of the test and the rule selects again among the remaining eligible items. The last eligible item is not
 * withheld, and an item with an exposure parameter of 1 takes no draw. Withheld items are cleared in `context.eligible`
 * and appended to `withheld`. With `membersOf`, an item is withheld with the items that go with it, such as the rest of
 * its testlet, and the last eligible unit is not withheld.
 *
 * @param membersOf - Returns the pool indices of the items administered or withheld together with an item, itself
 * included.
 * @returns The index of the item to administer, or `null` when the rule scores every eligible item `-Infinity`, as when
 * no item is left.
 * @defaultValue `membersOf` returns the item alone
 * @see {@link https://doi.org/10.1037/10244-014 | Hetter and Sympson (1997)}
 */
export function selectItem<TItem extends PoolControls, TEstimate>(
  rule: ItemSelectionRule<TItem, TEstimate>,
  context: AdaptiveTestContext<TItem, TEstimate>,
  withheld: number[],
  membersOf: (index: number) => readonly number[] = (index) => [index],
): number | null {
  const { pool, eligible, random } = context;
  const scores = new Float64Array(pool.length);
  let remaining = eligible.reduce((sum, flag) => sum + flag, 0);
  for (;;) {
    rule.scores(context, scores);
    const index = highestScore(scores, eligible);
    if (index < 0) return null;
    const exposure = pool[index]?.exposure ?? 1;
    const unit = membersOf(index).filter((member) => eligible[member] === 1);
    if (remaining === unit.length || exposure >= 1 || random.uniform() < exposure) return index;
    for (const member of unit) {
      withheld.push(member);
      eligible[member] = 0;
    }
    remaining -= unit.length;
  }
}

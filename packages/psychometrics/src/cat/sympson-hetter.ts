import { checkUnitInterval, DataError } from "@measurekit/core";
import type { PoolControls } from "./items.js";
import { highestScore, type ItemSelectionRule, type SelectionContext } from "./selection.js";

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
 * and appended to `withheld`.
 *
 * @throws `DataError` when the rule scores every eligible item `-Infinity`.
 * @see {@link https://doi.org/10.1037/10244-014 | Hetter and Sympson (1997)}
 */
export function selectItem<TItem extends PoolControls, TEstimate>(
  rule: ItemSelectionRule<TItem, TEstimate>,
  context: SelectionContext<TItem, TEstimate>,
  withheld: number[],
): number {
  const { pool, eligible, random } = context;
  const scores = new Float64Array(pool.length);
  let remaining = eligible.reduce((sum, flag) => sum + flag, 0);
  for (;;) {
    rule.scores(context, scores);
    const index = highestScore(scores, eligible);
    if (index < 0) throw new DataError("The selection rule scored every eligible item -Infinity.");
    const exposure = pool[index]?.exposure ?? 1;
    if (remaining === 1 || exposure >= 1 || random.uniform() < exposure) return index;
    withheld.push(index);
    eligible[index] = 0;
    remaining--;
  }
}

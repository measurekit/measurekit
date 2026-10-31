import type { Random } from "@measurekit/core";
import type { AbilityEstimate } from "../irt/ability.js";
import type { PoolControls, PoolItem } from "./items.js";

/**
 * An adaptive test after its latest response, as selection and stopping rules see it.
 *
 * @typeParam TItem - The items of the pool.
 * @typeParam TEstimate - The estimate of what the test measures.
 */
export interface AdaptiveTestContext<TItem extends PoolControls = PoolItem, TEstimate = AbilityEstimate> {
  /** The item pool. */
  readonly pool: readonly TItem[];
  /** The pool indices of the administered items, in order. */
  readonly administered: readonly number[];
  /** The responses to the administered items. */
  readonly responses: readonly number[];
  /** The estimate from the responses. */
  readonly estimate: TEstimate;
  /** 1 for each pool item that may be administered next, 0 for the others. */
  readonly eligible: Uint8Array;
}

/**
 * The context of item selection: the test, the eligible items and a source of random numbers.
 *
 * @typeParam TItem - The items of the pool.
 * @typeParam TEstimate - The estimate of what the test measures.
 */
export interface SelectionContext<TItem extends PoolControls = PoolItem, TEstimate = AbilityEstimate>
  extends AdaptiveTestContext<TItem, TEstimate> {
  /** The source of random numbers, for rules that select at random. */
  readonly random: Random;
}

/**
 * A rule that selects the next item of an adaptive test by scoring the eligible items. The test administers the
 * eligible item with the highest score, the first in pool order on ties. Built-in rules implement this interface as
 * a custom rule does.
 *
 * @typeParam TItem - The items of the pool.
 * @typeParam TEstimate - The estimate of what the test measures.
 */
export interface ItemSelectionRule<TItem extends PoolControls = PoolItem, TEstimate = AbilityEstimate> {
  /**
   * Writes a score for each eligible item to `out`, which has one entry per pool item; the entries of other items are
   * ignored. An item scored `-Infinity` is not selected. Rules that select at random draw from `context.random`.
   */
  scores(context: SelectionContext<TItem, TEstimate>, out: Float64Array): void;
}

/** Maximum Fisher information (MFI): scores each item by its information at the current estimate, catR's `MFI`. */
export const maximumInformation: ItemSelectionRule = {
  scores({ pool, eligible, estimate }, out) {
    pool.forEach((item, index) => {
      if (eligible[index] === 1) out[index] = item.model.information(item.parameters, estimate.theta);
    });
  },
};

/** Returns the eligible item with the highest score, the first on ties, or -1 when every score is `-Infinity`. */
export function highestScore(scores: Float64Array, eligible: Uint8Array): number {
  let best = -1;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (let index = 0; index < scores.length; index++) {
    const score = scores[index] ?? Number.NEGATIVE_INFINITY;
    if (eligible[index] === 1 && score > bestScore) {
      best = index;
      bestScore = score;
    }
  }
  return best;
}

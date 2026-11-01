import { checkChoice, DataError, describe, isNonEmptyString } from "@measurekit/core";
import type { ContentBalancing } from "./content-balancing.js";
import type { PoolControls } from "./items.js";
import type { ItemSelectionRule, SelectionContext } from "./selection.js";

/**
 * How the items of a testlet are ordered once the testlet is selected: `"in-order"`, in pool order, as a stimulus is
 * shown with its questions in the order they are listed; `"adaptive"`, each next item the one the selection rule scores
 * highest among the testlet's remaining items at the current estimate.
 */
export type WithinTestlet = "in-order" | "adaptive";

const WITHIN_TESTLET: readonly WithinTestlet[] = ["in-order", "adaptive"];

/** The testlets of a pool: each item's testlet and each testlet's items in pool order. */
export interface Testlets {
  /** The testlet of each pool item, -1 for an item in none. */
  readonly of: Int32Array;
  /** The pool indices of each testlet's items, in pool order. */
  readonly members: readonly (readonly number[])[];
}

/**
 * Checks the option that orders the items of a testlet.
 *
 * @throws `DataError` unless it is `"in-order"` or `"adaptive"`.
 */
export function checkWithinTestlet(withinTestlet: unknown): asserts withinTestlet is WithinTestlet {
  checkChoice(withinTestlet, "The option withinTestlet", WITHIN_TESTLET);
}

/**
 * Returns the testlets of a pool, or `null` when no item has one.
 *
 * @throws `DataError` when an item's testlet is not a non-empty string.
 */
export function poolTestlets(pool: readonly PoolControls[]): Testlets | null {
  const indices = new Map<string, number>();
  const members: number[][] = [];
  const of = Int32Array.from(pool, ({ id, testlet }, index) => {
    if (testlet === undefined) return -1;
    if (!isNonEmptyString(testlet)) {
      throw new DataError(`The testlet of item "${id}" must be a non-empty string, found ${describe(testlet)}.`);
    }
    let testletIndex = indices.get(testlet);
    if (testletIndex === undefined) {
      testletIndex = members.length;
      indices.set(testlet, testletIndex);
      members.push([]);
    }
    members[testletIndex]?.push(index);
    return testletIndex;
  });
  return members.length === 0 ? null : { of, members };
}

/**
 * Checks that the items of each testlet share one target area, as content balancing chooses the area of a testlet as a
 * whole.
 *
 * @throws `DataError` when a testlet has items in two target areas.
 */
export function checkTestletAreas(
  pool: readonly PoolControls[],
  testlets: Testlets,
  balancing: ContentBalancing,
): void {
  const { areas, itemAreas } = balancing;
  for (const [first = -1, ...rest] of testlets.members) {
    const other = rest.find((index) => itemAreas[index] !== itemAreas[first]);
    if (other !== undefined) {
      throw new DataError(
        `The items of testlet "${pool[first]?.testlet}" are in content areas ${describe(areas[itemAreas[first] ?? -1])} and ${describe(areas[itemAreas[other] ?? -1])}; with content targets they must share one area.`,
      );
    }
  }
}

/**
 * Returns the eligible items of the testlet the last administered item belongs to, in pool order, or `null` when that
 * item is in no testlet or the testlet has none left: the testlet in progress.
 */
export function remainingInTestlet(
  testlets: Testlets,
  administered: readonly number[],
  eligible: Uint8Array,
): number[] | null {
  const last = administered.at(-1);
  const testlet = last === undefined ? -1 : (testlets.of[last] ?? -1);
  if (testlet < 0) return null;
  const remaining = (testlets.members[testlet] ?? []).filter((index) => eligible[index] === 1);
  return remaining.length === 0 ? null : remaining;
}

/**
 * Returns the item to administer next in the testlet in progress: the first of its remaining items in pool order, or
 * with `"adaptive"` the one `rule` scores highest among them, the first on ties and when none scores above `-Infinity`.
 */
export function nextInTestlet<TItem extends PoolControls, TEstimate>(
  rule: ItemSelectionRule<TItem, TEstimate>,
  context: SelectionContext<TItem, TEstimate>,
  remaining: readonly number[],
  withinTestlet: WithinTestlet,
): number {
  const first = remaining[0] ?? -1;
  if (withinTestlet === "in-order") return first;
  const eligible = new Uint8Array(context.pool.length);
  for (const index of remaining) eligible[index] = 1;
  const scores = new Float64Array(context.pool.length);
  rule.scores({ ...context, eligible }, scores);
  return bestOf(scores, remaining, first);
}

/** Returns the member with the highest score above `-Infinity`, the first on ties, or `fallback` without one. */
function bestOf(scores: Float64Array, members: readonly number[], fallback: number): number {
  let best = fallback;
  let bestScore = Number.NEGATIVE_INFINITY;
  for (const index of members) {
    const score = scores[index] ?? Number.NEGATIVE_INFINITY;
    if (score > bestScore) {
      best = index;
      bestScore = score;
    }
  }
  return best;
}

/**
 * Returns a rule that selects testlets as units: it scores each testlet by the mean of the scores `rule` gives its
 * eligible items above `-Infinity`, `-Infinity` when it has none, and gives that score to the testlet's first item to be
 * administered, the item `nextInTestlet` would choose, and `-Infinity` to its other items. Items in no testlet keep their
 * scores.
 */
export function unitSelection<TItem extends PoolControls, TEstimate>(
  rule: ItemSelectionRule<TItem, TEstimate>,
  testlets: Testlets,
  withinTestlet: WithinTestlet,
): ItemSelectionRule<TItem, TEstimate> {
  return {
    scores(context, out) {
      rule.scores(context, out);
      for (const members of testlets.members) {
        const eligible = members.filter((index) => context.eligible[index] === 1);
        if (eligible.length > 0) scoreAsUnit(eligible, out, withinTestlet);
      }
    },
  };
}

/** Gives a testlet's eligible items the unit's score at its first item to be administered and `-Infinity` elsewhere. */
function scoreAsUnit(eligible: readonly number[], out: Float64Array, withinTestlet: WithinTestlet): void {
  const first = eligible[0] ?? -1;
  const scored = eligible.filter((index) => (out[index] ?? Number.NEGATIVE_INFINITY) > Number.NEGATIVE_INFINITY);
  const unit =
    scored.length === 0
      ? Number.NEGATIVE_INFINITY
      : scored.reduce((sum, index) => sum + (out[index] ?? 0), 0) / scored.length;
  const entry = withinTestlet === "in-order" ? first : bestOf(out, eligible, first);
  for (const index of eligible) out[index] = index === entry ? unit : Number.NEGATIVE_INFINITY;
}

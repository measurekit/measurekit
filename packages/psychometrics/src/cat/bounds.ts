import { checkMethod, checkPositiveInteger, DataError } from "@measurekit/core";
import {
  type ContentBounds,
  type ContentConstraint,
  countAreas,
  poolAreas,
  readBounds,
} from "./content-constraints.js";
import type { PoolControls } from "./items.js";
import type { AdaptiveTestContext, ItemSelectionRule } from "./selection.js";
import { poolTestlets, remainingInTestlet, type Testlets } from "./testlets.js";

/** What the filter reads of a pool, computed once for each pool array. */
interface BoundedPool {
  /** The constrained areas of each item, ascending. */
  readonly itemAreas: readonly (readonly number[])[];
  /** The distinct sets of constrained areas that items have, by first appearance in the pool. */
  readonly groups: readonly (readonly number[])[];
  /** The index in `groups` of each item's set. */
  readonly groupOf: Int32Array;
  /** The most areas of one set. */
  readonly widest: number;
  /** The testlets of the pool, or `null` when there are none. */
  readonly testlets: Testlets | null;
}

/** The verdict on a unit, an item or a testlet that has not begun, that breaks a maximum or the length. */
const OUT = 0;
/** The verdict on a unit within the maximums and the length, whose minimums left may not fit in the places left. */
const WITHIN = 1;
/** The verdict on a unit within the maximums and the length, whose minimums left fit in the places left. */
const SAFE = 2;

/** The verdict on a unit and the needs it fills. */
interface Judgment {
  readonly verdict: number;
  readonly filled: number;
}

/**
 * Returns a rule that keeps a test within the bounds of its content areas: it scores like `rule`, except `-Infinity` for
 * each item whose administration would break a maximum, the length, or the reach of a minimum.
 *
 * @remarks
 * A unit is an item, or a testlet that has not begun, whose items are all administered. With `length` the most items
 * of a test, `t` those given, `x` the given in an area and `a` the unit's in it, a unit is kept when
 *
 * - it fits: `t + size ≤ length`;
 * - it respects every maximum: `x + a ≤ maximum`;
 * - the minimums fit in the places left: the items still needed, `max(0, minimum - x - a)` summed over the areas, are
 *   at most `length - t - size`. When they are more, as when items are in two areas, a greedy cover decides: it takes the
 *   eligible items one at a time, each time the kind of item (those with one set of areas) whose areas have the most items
 *   still needed, without breaking a maximum, and the unit is kept when the needs are met within the places left.
 *
 * If no unit passes, the units within the maximums and the length that fill the most needs are kept; if there is none,
 * every item scores `-Infinity`, and the test ends. The testlet in progress passes, as the items that follow are
 * administered without a selection. The filter works on kinds of items, so a step costs little at any pool size.
 *
 * With exclusive areas, a step that must fill a need does, so a pool with enough items in each area meets every
 * minimum. With items in several areas, the cover is a heuristic that can miss a feasible completion. Exposure control
 * and enemies can leave a minimum unreachable; the test then goes on within the maximums. The first selection throws
 * `DataError` for what is certain to fail: an area with fewer pool items than its minimum, a minimum above `length`,
 * and minimums that sum above `length` when no item is in two constrained areas. Areas are counted by the pool array,
 * so a pool must not change after its first use. Use it instead of `contentTargets`, not with it.
 *
 * @param rule - The rule that scores the items that pass.
 * @param constraints - The area, the minimum and the maximum, as integers from 0.
 * @param length - The most items a test may have, a positive integer.
 * @returns The selection rule, which scores like `rule` and `-Infinity` for the items the bounds exclude.
 * @throws `DataError` when `rule` is not a selection rule, the constraints are invalid or on one area twice, or
 * `length` is not a positive integer.
 */
export function contentBounds<TItem extends PoolControls, TEstimate>(
  rule: ItemSelectionRule<TItem, TEstimate>,
  constraints: readonly ContentConstraint[],
  length: number,
): ItemSelectionRule<TItem, TEstimate> {
  checkMethod(rule, "scores", "contentBounds needs a selection rule, such as maximumInformation");
  const bounds = readBounds(constraints);
  checkPositiveInteger(length, "The length of contentBounds");
  const itemAreasOf = poolAreas(bounds.areas);
  const byPool = new WeakMap<readonly TItem[], BoundedPool>();
  const boundedPool = (pool: readonly TItem[]): BoundedPool => {
    let bounded = byPool.get(pool);
    if (bounded === undefined) {
      bounded = describePool(pool, itemAreasOf(pool));
      checkFeasible(bounds, length, bounded);
      byPool.set(pool, bounded);
    }
    return bounded;
  };
  return {
    scores(context, out) {
      rule.scores(context, out);
      const bounded = boundedPool(context.pool);
      const { testlets } = bounded;
      if (testlets !== null && remainingInTestlet(testlets, context.administered, context.eligible) !== null) return;
      filter(bounds, length, bounded, context, out);
    },
  };
}

function describePool(pool: readonly PoolControls[], itemAreas: readonly (readonly number[])[]): BoundedPool {
  const kinds = new Map<string, number>();
  const groups: (readonly number[])[] = [];
  const groupOf = Int32Array.from(itemAreas, (areas) => {
    const key = areas.join(",");
    let group = kinds.get(key);
    if (group === undefined) {
      group = groups.length;
      kinds.set(key, group);
      groups.push(areas);
    }
    return group;
  });
  return {
    itemAreas,
    groups,
    groupOf,
    widest: groups.reduce((most, areas) => Math.max(most, areas.length), 0),
    testlets: poolTestlets(pool),
  };
}

/** Throws for bounds that no test of the pool can meet, whatever the items given. */
function checkFeasible({ areas, minimums }: ContentBounds, length: number, { itemAreas, groups }: BoundedPool): void {
  const sizes = new Int32Array(areas.length);
  for (const list of itemAreas) for (const area of list) sizes[area] = (sizes[area] ?? 0) + 1;
  let total = 0;
  areas.forEach((area, k) => {
    const minimum = minimums[k] ?? 0;
    if ((sizes[k] ?? 0) < minimum) {
      throw new DataError(
        `The pool has ${sizes[k]} items in content area "${area}", fewer than its minimum ${minimum}; add items or lower the minimum.`,
      );
    }
    if (minimum > length) {
      throw new DataError(
        `The minimum ${minimum} of content area "${area}" is above the test length ${length}; lower the minimum or raise the length.`,
      );
    }
    total += minimum;
  });
  if (total > length && groups.every((list) => list.length <= 1)) {
    throw new DataError(
      `The minimums of the content areas sum to ${total}, above the test length ${length}, and no item is in two of the areas; lower a minimum or raise the length.`,
    );
  }
}

/** The numbers of a step that every unit is judged by. */
interface Step {
  readonly bounds: ContentBounds;
  readonly length: number;
  readonly bounded: BoundedPool;
  /** The items given in each area. */
  readonly given: Int32Array;
  /** The items still needed in each area, `max(0, minimum - given)`. */
  readonly needs: Int32Array;
  /** The items still needed in all. */
  readonly needed: number;
  /** The items given. */
  readonly count: number;
  /** The eligible items of each group. */
  readonly available: Int32Array;
}

function startStep(
  bounds: ContentBounds,
  length: number,
  bounded: BoundedPool,
  { administered, eligible }: AdaptiveTestContext<PoolControls, unknown>,
): Step {
  const given = countAreas(administered, bounded.itemAreas, bounds.areas.length);
  const needs = Int32Array.from(given, (count, k) => Math.max(0, (bounds.minimums[k] ?? 0) - count));
  const available = new Int32Array(bounded.groups.length);
  eligible.forEach((flag, index) => {
    const group = bounded.groupOf[index] ?? 0;
    if (flag === 1) available[group] = (available[group] ?? 0) + 1;
  });
  return {
    bounds,
    length,
    bounded,
    given,
    needs,
    needed: needs.reduce((sum, need) => sum + need, 0),
    count: administered.length,
    available,
  };
}

function filter<TItem extends PoolControls, TEstimate>(
  bounds: ContentBounds,
  length: number,
  bounded: BoundedPool,
  context: AdaptiveTestContext<TItem, TEstimate>,
  out: Float64Array,
): void {
  const { eligible } = context;
  const step = startStep(bounds, length, bounded, context);
  const verdicts = new Uint8Array(out.length);
  const filled = new Int32Array(out.length);
  const record = (members: readonly number[], { verdict, filled: needsFilled }: Judgment): void => {
    for (const member of members) {
      verdicts[member] = verdict;
      filled[member] = needsFilled;
    }
  };
  // Items alone with one set of areas are judged alike, so each set is judged once.
  const ofGroup: (Judgment | undefined)[] = [];
  const { testlets, groupOf } = bounded;
  eligible.forEach((flag, index) => {
    if (flag !== 1 || (testlets?.of[index] ?? -1) >= 0 || out[index] === Number.NEGATIVE_INFINITY) return;
    const group = groupOf[index] ?? 0;
    const judgment = ofGroup[group] ?? judge(step, [index]);
    ofGroup[group] = judgment;
    verdicts[index] = judgment.verdict;
    filled[index] = judgment.filled;
  });
  const judgeTestlet = testletJudge(step);
  for (const all of testlets?.members ?? []) {
    const members = all.filter((member) => eligible[member] === 1);
    if (members.some((member) => (out[member] ?? Number.NEGATIVE_INFINITY) > Number.NEGATIVE_INFINITY)) {
      record(members, judgeTestlet(members));
    }
  }
  const kept = keepRule(verdicts, filled, eligible, out);
  eligible.forEach((flag, index) => {
    if (flag === 1 && !kept(index)) out[index] = Number.NEGATIVE_INFINITY;
  });
}

/** Returns the judgment of a testlet's eligible items as a unit, remembering those it made for the same groups of items. */
function testletJudge(step: Step): (members: readonly number[]) => Judgment {
  const { groupOf } = step.bounded;
  const judgments = new Map<string, Judgment>();
  return (members) => {
    const key = members.map((member) => groupOf[member]).join(",");
    const known = judgments.get(key);
    if (known !== undefined) return known;
    const judgment = judge(step, members);
    judgments.set(key, judgment);
    return judgment;
  };
}

/**
 * Returns whether an item is kept: its unit is safe; else, where no unit is, it is within the maximums and fills the most
 * needs of those that are.
 */
function keepRule(
  verdicts: Uint8Array,
  filled: Int32Array,
  eligible: Uint8Array,
  out: Float64Array,
): (index: number) => boolean {
  let safe = false;
  let most = -1;
  eligible.forEach((flag, index) => {
    if (flag !== 1 || out[index] === Number.NEGATIVE_INFINITY) return;
    safe = safe || verdicts[index] === SAFE;
    if (verdicts[index] !== OUT) most = Math.max(most, filled[index] ?? 0);
  });
  return (index) => (safe ? verdicts[index] === SAFE : verdicts[index] !== OUT && (filled[index] ?? 0) === most);
}

/** Returns the number of the unit's items in each area. */
function areasTaken({ bounds, bounded }: Step, members: readonly number[]): Int32Array {
  const taken = new Int32Array(bounds.areas.length);
  for (const member of members)
    for (const area of bounded.itemAreas[member] ?? []) taken[area] = (taken[area] ?? 0) + 1;
  return taken;
}

/** Judges a unit: its verdict, and the needs it fills. */
function judge(step: Step, members: readonly number[]): Judgment {
  const { bounds, length, needs, needed, count, given } = step;
  const taken = areasTaken(step, members);
  const fits = count + members.length <= length;
  if (!fits || taken.some((taking, k) => (given[k] ?? 0) + taking > (bounds.maximums[k] ?? 0))) {
    return { verdict: OUT, filled: 0 };
  }
  const filled = taken.reduce((sum, taking, k) => sum + Math.min(taking, needs[k] ?? 0), 0);
  const places = length - count - members.length;
  const reaches = needed - filled <= places || coverReaches(step, members, taken, places);
  return { verdict: reaches ? SAFE : WITHIN, filled };
}

/** What a cover has left: the eligible items of each group, the items still needed and the room under the maximums. */
interface Cover {
  readonly remaining: Int32Array;
  readonly need: Int32Array;
  readonly room: Int32Array;
  outstanding: number;
}

function startCover(step: Step, members: readonly number[], taken: Int32Array): Cover {
  const { bounds, bounded, given, needs, available } = step;
  const remaining = available.slice();
  for (const member of members) {
    const group = bounded.groupOf[member] ?? 0;
    remaining[group] = (remaining[group] ?? 0) - 1;
  }
  const need = new Int32Array(needs.length);
  const room = new Int32Array(needs.length);
  let outstanding = 0;
  for (let area = 0; area < needs.length; area++) {
    need[area] = Math.max(0, (needs[area] ?? 0) - (taken[area] ?? 0));
    room[area] = (bounds.maximums[area] ?? 0) - (given[area] ?? 0) - (taken[area] ?? 0);
    outstanding += need[area] ?? 0;
  }
  return { remaining, need, room, outstanding };
}

/**
 * Returns the group a place of the cover takes: of the groups with an eligible item left and room under every maximum,
 * the one whose areas have the most items still needed in all, the first on ties; -1 without one that needs any.
 */
function bestGroup(groups: readonly (readonly number[])[], { remaining, need, room }: Cover): number {
  let best = -1;
  let bestGain = 0;
  for (let group = 0; group < groups.length; group++) {
    const areas = groups[group] ?? [];
    let gain = 0;
    let open = (remaining[group] ?? 0) > 0;
    for (let i = 0; open && i < areas.length; i++) {
      const area = areas[i] ?? 0;
      open = (room[area] ?? 0) >= 1;
      gain += need[area] ?? 0;
    }
    if (open && gain > bestGain) {
      best = group;
      bestGain = gain;
    }
  }
  return best;
}

/**
 * Returns whether the needs of a cover can still be met in the places left: no area needs more items than there are
 * places, and the places fill at most as many needs as the largest group has areas.
 */
function mayReach({ need, outstanding }: Cover, widest: number, places: number): boolean {
  return outstanding <= places * widest && need.every((count) => count <= places);
}

/**
 * Returns whether a greedy cover of the needs left after a unit fits in the places left: each place takes the best group
 * of {@link bestGroup}. Weighing an area by the items it still needs, not by one, takes the areas that few places remain
 * for first.
 */
function coverReaches(step: Step, members: readonly number[], taken: Int32Array, places: number): boolean {
  const { groups, widest } = step.bounded;
  const cover = startCover(step, members, taken);
  if (!mayReach(cover, widest, places)) return false;
  for (let place = 0; place < places && cover.outstanding > 0; place++) {
    const best = bestGroup(groups, cover);
    if (best < 0) return false;
    cover.remaining[best] = (cover.remaining[best] ?? 0) - 1;
    for (const area of groups[best] ?? []) {
      cover.room[area] = (cover.room[area] ?? 0) - 1;
      const need = cover.need[area] ?? 0;
      cover.need[area] = Math.max(0, need - 1);
      cover.outstanding -= Math.min(1, need);
    }
  }
  return cover.outstanding === 0;
}

import { checkPositive, checkRecord, DataError, describe, type Random } from "@measurekit/core";
import type { PoolItem } from "./items.js";

/** The content areas of a pool with their target proportions, and each item's area. */
export interface ContentBalancing {
  /** The target proportion of each area, summing to 1. */
  readonly targets: Float64Array;
  /** The index of each pool item's area. */
  readonly itemAreas: Int32Array;
}

/**
 * Returns the content balancing of a pool: the target proportions, normalized to sum to 1, and each item's area.
 *
 * @throws `DataError` when the targets are not an object or empty, a target is not a positive finite number, or when an
 * item has no content area among the targets.
 */
export function contentBalancing(
  pool: readonly PoolItem[],
  targets: Readonly<Record<string, number>>,
): ContentBalancing {
  checkRecord(targets, "The option contentTargets must be an object of target proportions by area");
  const areas = Object.keys(targets);
  if (areas.length === 0) throw new DataError("Content balancing needs a target proportion for at least one area.");
  const proportions = Float64Array.from(areas, (area) => {
    const target = targets[area] ?? 0;
    checkPositive(target, `The target of content area "${area}"`);
    return target;
  });
  const sum = proportions.reduce((total, target) => total + target, 0);
  const itemAreas = Int32Array.from(pool, (item) => {
    const area = item.contentArea === undefined ? -1 : areas.indexOf(item.contentArea);
    if (area < 0) {
      throw new DataError(
        `Item "${item.id}" needs a content area among the targets (${areas.join(", ")}), found ${describe(item.contentArea)}.`,
      );
    }
    return area;
  });
  return { targets: proportions.map((target) => target / sum), itemAreas };
}

/**
 * Restricts the eligible items to one content area by Kingsbury and Zara's rule, as catR's `cbControl`: an area
 * without administered items if there is one, otherwise the area whose administered proportion falls furthest below
 * its target; ties at random. Only areas with eligible items are chosen.
 */
export function restrictToArea(
  { targets, itemAreas }: ContentBalancing,
  administered: readonly number[],
  eligible: Uint8Array,
  random: Random,
): void {
  const counts = new Float64Array(targets.length);
  for (const index of administered) {
    const area = itemAreas[index] ?? 0;
    counts[area] = (counts[area] ?? 0) + 1;
  }
  const available = new Uint8Array(targets.length);
  eligible.forEach((flag, index) => {
    if (flag === 1) available[itemAreas[index] ?? 0] = 1;
  });
  const areas = Array.from(targets.keys()).filter((area) => available[area] === 1);
  const unused = areas.filter((area) => counts[area] === 0);
  const candidates = unused.length > 0 ? unused : furthestBelowTarget(areas, targets, counts, administered.length);
  const area = candidates.length === 1 ? candidates[0] : candidates[Math.floor(random.uniform() * candidates.length)];
  eligible.forEach((_, index) => {
    if (itemAreas[index] !== area) eligible[index] = 0;
  });
}

/** Returns the areas whose administered proportion falls furthest below their target. */
function furthestBelowTarget(
  areas: readonly number[],
  targets: Float64Array,
  counts: Float64Array,
  administered: number,
): number[] {
  const gaps = areas.map((area) => (targets[area] ?? 0) - (counts[area] ?? 0) / administered);
  const largest = gaps.reduce((most, gap) => Math.max(most, gap), Number.NEGATIVE_INFINITY);
  return areas.filter((_, i) => gaps[i] === largest);
}

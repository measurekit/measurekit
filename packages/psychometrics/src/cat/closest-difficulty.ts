import { DataError, isRecord } from "@measurekit/core";
import type { PoolItem } from "./items.js";
import type { ItemSelectionRule } from "./selection.js";

/**
 * Selection by the closest difficulty (bOpt), catR's `bOpt`: scores each item by minus the distance between its
 * difficulty and the current estimate.
 *
 * @remarks
 * The difficulty is the item's `b`: for a logistic item its location, for an item with thresholds `b` their mean, and
 * for a rating scale item `b` plus the mean of its `steps`, the mean of its thresholds. catR takes dichotomous items
 * only. The rule needs no information, so it is cheap; for the 1PL and 2PL it is maximum information.
 *
 * @throws `DataError` from `scores` when an eligible item has no difficulty `b`, such as an item of the nominal model.
 * @see {@link https://doi.org/10.1177/01466219922031338 | Chang and Ying (1999)}
 */
export const closestDifficulty: ItemSelectionRule = {
  scores({ pool, eligible, estimate }, out) {
    pool.forEach((item, index) => {
      if (eligible[index] === 1) out[index] = -Math.abs(difficultyOf(item) - estimate.theta);
    });
  },
};

/**
 * Returns the mean threshold of an item: `b`, or the mean of `b`, plus the mean of `steps` where the item has them.
 *
 * @throws `DataError` when the item's parameters have no finite `b` or list of them.
 */
function difficultyOf(item: PoolItem): number {
  const { b, steps } = isRecord(item.parameters) ? item.parameters : {};
  const location = typeof b === "number" ? b : Array.isArray(b) ? meanOf(b) : Number.NaN;
  const offset = Array.isArray(steps) ? meanOf(steps) : 0;
  const difficulty = location + offset;
  if (!Number.isFinite(difficulty)) {
    throw new DataError(
      `Selection by the closest difficulty needs items with a difficulty b; item "${item.id}" of the ${item.model.name} has none.`,
    );
  }
  return difficulty;
}

/** Returns the mean of a list, `NaN` when it is empty or holds a value that is not a number. */
function meanOf(values: readonly unknown[]): number {
  let sum = 0;
  for (const value of values) sum += typeof value === "number" ? value : Number.NaN;
  return values.length === 0 ? Number.NaN : sum / values.length;
}

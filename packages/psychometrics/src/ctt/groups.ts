import { DataError, describe } from "@measurekit/core";

/**
 * Kelley's share of persons in the upper and lower groups.
 *
 * @see {@link https://doi.org/10.1037/h0057123 | Kelley (1939)}
 */
const KELLEY_FRACTION = 0.27;

/** Options of statistics that compare the upper and lower score groups. */
export interface GroupOptions {
  /**
   * The share of persons in each of the upper and lower groups, more than 0 and at most 0.5.
   *
   * @defaultValue `0.27`
   * @see {@link https://doi.org/10.1037/h0057123 | Kelley (1939)}
   */
  readonly groupFraction?: number;
}

/** Weights of persons in the upper and lower score groups. */
export interface ScoreGroups {
  /** Weight of each person in the upper group, from 0 to 1. */
  readonly upper: Float64Array;
  /** Weight of each person in the lower group, from 0 to 1. */
  readonly lower: Float64Array;
}

/**
 * Returns the upper and lower groups of persons by total score. Each group holds `groupFraction` times the number of
 * persons. Persons tied at a group's boundary share the remaining places equally, as fractions of a person, so the
 * groups do not depend on the order of persons.
 */
export function scoreGroups(totals: Float64Array, options: GroupOptions): ScoreGroups {
  const { groupFraction = KELLEY_FRACTION } = options;
  if (!(typeof groupFraction === "number" && groupFraction > 0 && groupFraction <= 0.5)) {
    throw new DataError(
      `The option groupFraction must be a number above 0 and at most 0.5, found ${describe(groupFraction)}.`,
    );
  }
  const size = groupFraction * totals.length;
  return { upper: groupWeights(totals, size, -1), lower: groupWeights(totals, size, 1) };
}

/** Returns the weighted mean of `values` in a group, or `null` when the group is empty. */
export function groupMean(weights: Float64Array, values: Float64Array): number | null {
  let sum = 0;
  let total = 0;
  for (let i = 0; i < weights.length; i++) {
    sum += (weights[i] ?? 0) * (values[i] ?? 0);
    total += weights[i] ?? 0;
  }
  return total > 0 ? sum / total : null;
}

/** Fills a group of `size` persons from the lowest totals (`direction` 1) or the highest (`direction` -1). */
function groupWeights(totals: Float64Array, size: number, direction: 1 | -1): Float64Array {
  const order = Array.from(totals.keys()).sort((a, b) => direction * ((totals[a] ?? 0) - (totals[b] ?? 0)));
  const weights = new Float64Array(totals.length);
  let remaining = size;
  let start = 0;
  while (start < order.length && remaining > 0) {
    let end = start + 1;
    while (end < order.length && totals[order[end] ?? 0] === totals[order[start] ?? 0]) end++;
    const share = Math.min(remaining, end - start) / (end - start);
    for (let k = start; k < end; k++) weights[order[k] ?? 0] = share;
    remaining -= end - start;
    start = end;
  }
  return weights;
}

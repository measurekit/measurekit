import {
  checkArray,
  checkFinite,
  checkIntegerFrom,
  checkPositive,
  checkRecord,
  DataError,
  describe,
  isNonEmptyString,
} from "@measurekit/core";
import type { PoolControls } from "./items.js";

/** A content constraint of a form or a test: the number of its items in a content area. */
export interface ContentConstraint {
  /** The content area, as an entry of the pool items' `contentAreas`. */
  readonly area: string;
  /** The fewest items of the area. */
  readonly minimum: number;
  /** The most items of the area. */
  readonly maximum: number;
  /**
   * The weight of a deviation of one item from the bounds, in form assembly; in `maximumPriorityIndex`, the weight of
   * the area's priority.
   *
   * @defaultValue `1`
   */
  readonly weight?: number;
}

/** The bounds of a test on the number of its items in each of several content areas, by area. */
export interface ContentBounds {
  /** The areas, in the order of the constraints. */
  readonly areas: readonly string[];
  /** The fewest items of each area. */
  readonly minimums: Int32Array;
  /** The most items of each area. */
  readonly maximums: Int32Array;
  /** The weight of each area. */
  readonly weights: Float64Array;
}

/**
 * Reads a content constraint, with its default weight.
 *
 * @throws `DataError` unless it is an object with a string area, finite bounds with the minimum at most the maximum, and
 * a weight that is a positive number.
 */
function readConstraint(constraint: ContentConstraint): Required<ContentConstraint> {
  checkRecord(constraint, "A content constraint must be an object { area, minimum, maximum }");
  const { area, minimum, maximum, weight = 1 } = constraint;
  if (typeof area !== "string") {
    throw new DataError(`A content constraint needs a string area, found ${describe(area)}.`);
  }
  checkFinite(minimum, `The minimum of the constraint on "${area}"`);
  checkFinite(maximum, `The maximum of the constraint on "${area}"`);
  if (!(minimum <= maximum)) {
    throw new DataError(
      `The constraint on "${area}" needs the minimum at most the maximum, found ${minimum} and ${maximum}.`,
    );
  }
  checkPositive(weight, `The weight of the constraint on "${area}"`);
  return { area, minimum, maximum, weight };
}

/**
 * Reads content constraints whose areas are distinct.
 *
 * @throws `DataError` when they are not an array of constraints, or two are on one area.
 */
export function readConstraints(constraints: readonly ContentConstraint[]): readonly Required<ContentConstraint>[] {
  checkArray(constraints, "The content constraints must be an array of { area, minimum, maximum }");
  const read = constraints.map(readConstraint);
  const seen = new Set<string>();
  for (const { area } of read) {
    if (seen.has(area)) throw new DataError(`Content area "${area}" has two constraints; give it one.`);
    seen.add(area);
  }
  return read;
}

/**
 * Reads the bounds of a test, whose counts are integers from 0.
 *
 * @throws `DataError` when the constraints are not valid for {@link readConstraints}, or a bound is not an integer from 0.
 */
export function readBounds(constraints: readonly ContentConstraint[]): ContentBounds {
  const read = readConstraints(constraints);
  for (const { area, minimum, maximum } of read) {
    checkIntegerFrom(minimum, `The minimum of the constraint on "${area}"`, 0);
    checkIntegerFrom(maximum, `The maximum of the constraint on "${area}"`, 0);
  }
  return {
    areas: read.map(({ area }) => area),
    minimums: Int32Array.from(read, ({ minimum }) => minimum),
    maximums: Int32Array.from(read, ({ maximum }) => maximum),
    weights: Float64Array.from(read, ({ weight }) => weight),
  };
}

/** Describes the content areas of an item in a message. */
export function describeAreas(areas: unknown): string {
  return Array.isArray(areas) ? `[${areas.map(describe).join(", ")}]` : describe(areas);
}

/**
 * Returns a function that gives, for each item of a pool, the indices in `areas` of the areas it is in, ascending: the
 * entries of its `contentAreas` that are among `areas`. It reads each pool array once and keeps the result, so a pool
 * must not change after its first use.
 *
 * @throws `DataError` from the returned function when an item's `contentAreas` is not an array of distinct non-empty
 * strings.
 */
export function poolAreas(areas: readonly string[]): (pool: readonly PoolControls[]) => readonly (readonly number[])[] {
  const position = new Map(areas.map((area, k) => [area, k]));
  const byPool = new WeakMap<readonly PoolControls[], readonly (readonly number[])[]>();
  return (pool) => {
    const known = byPool.get(pool);
    if (known !== undefined) return known;
    const itemAreas = pool.map(({ id, contentAreas }) => {
      if (contentAreas === undefined) return [];
      if (
        !(Array.isArray(contentAreas) && contentAreas.every(isNonEmptyString)) ||
        new Set(contentAreas).size !== contentAreas.length
      ) {
        throw new DataError(
          `The contentAreas of item "${id}" must be an array of distinct non-empty strings, found ${describeAreas(contentAreas)}.`,
        );
      }
      return contentAreas.flatMap((area) => position.get(area) ?? []).sort((i, j) => i - j);
    });
    byPool.set(pool, itemAreas);
    return itemAreas;
  };
}

/** Returns the number of the administered items in each area. */
export function countAreas(
  administered: readonly number[],
  itemAreas: readonly (readonly number[])[],
  areaCount: number,
): Int32Array {
  const counts = new Int32Array(areaCount);
  for (const index of administered) for (const area of itemAreas[index] ?? []) counts[area] = (counts[area] ?? 0) + 1;
  return counts;
}

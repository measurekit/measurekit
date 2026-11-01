import { DataError, describe, isRecord, type ThresholdItem } from "@measurekit/core";
import type { ModelParameter } from "./model.js";

/**
 * Returns the estimated parameters of an item of a threshold model with `categoryCount` categories: `slope`, b1, ...
 */
export function thresholdParameterList(slope: ModelParameter, categoryCount: number): ModelParameter[] {
  return [slope, ...Array.from({ length: categoryCount - 1 }, (_, j) => ({ name: `b${j + 1}` }))];
}

/**
 * Returns the parameter vector a, b1, ..., b(K-1).
 *
 * @throws `DataError` unless `a` is finite and `b` holds at least one threshold, all finite.
 */
export function thresholdVector(model: string, parameters: ThresholdItem): Float64Array {
  const { a, b }: Partial<ThresholdItem> = isRecord(parameters) ? parameters : {};
  if (!(typeof a === "number" && Number.isFinite(a) && Array.isArray(b) && b.length > 0 && b.every(Number.isFinite))) {
    throw new DataError(
      `A ${model} item needs a finite a and at least one finite threshold b, found a = ${describe(a)}, b = ${formatList(b)}.`,
    );
  }
  return Float64Array.from([a, ...b]);
}

/** Returns the threshold parameters of a vector a, b1, ..., b(K-1). */
export function thresholdsFromVector(vector: Float64Array): ThresholdItem {
  return { a: vector[0] ?? 0, b: Array.from(vector.subarray(1)) };
}

/** Returns the share of persons in categories k and above, for k = 1 to K - 1, from the share in each category. */
export function sharesAbove(proportions: ArrayLike<number>): number[] {
  const shares: number[] = [];
  let above = 1;
  for (let k = 1; k < proportions.length; k++) {
    above -= proportions[k - 1] ?? 0;
    shares.push(above);
  }
  return shares;
}

/** Formats a list of numbers for a message, or describes any other value. */
export function formatList(values: unknown): string {
  return Array.isArray(values) ? `[${values.map(describe).join(", ")}]` : describe(values);
}

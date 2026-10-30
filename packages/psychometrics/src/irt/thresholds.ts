import { DataError, describe, isRecord } from "@measurekit/core";
import type { ModelParameter } from "./model.js";

/**
 * Parameters of a polytomous item with a slope and one threshold per boundary between adjacent categories, in mirt's
 * IRT parameterization: the items of the graded response model and the (generalized) partial credit model. An item
 * with K categories has K - 1 thresholds.
 */
export interface ThresholdParameters {
  /** Discrimination: the slope. */
  readonly a: number;
  /** The thresholds b1 to b(K-1), one per boundary between adjacent categories. */
  readonly b: readonly number[];
}

/** Returns the estimated parameters of an item of a threshold model with `categoryCount` categories: `slope`, b1, ... */
export function thresholdParameterList(slope: ModelParameter, categoryCount: number): ModelParameter[] {
  return [slope, ...Array.from({ length: categoryCount - 1 }, (_, j) => ({ name: `b${j + 1}` }))];
}

/**
 * Returns the parameter vector a, b1, ..., b(K-1).
 *
 * @throws `DataError` unless `a` is finite and `b` holds at least one threshold, all finite.
 */
export function thresholdVector(model: string, parameters: ThresholdParameters): Float64Array {
  const { a, b }: Partial<ThresholdParameters> = isRecord(parameters) ? parameters : {};
  if (!(typeof a === "number" && Number.isFinite(a) && Array.isArray(b) && b.length > 0 && b.every(Number.isFinite))) {
    throw new DataError(
      `A ${model} item needs a finite a and at least one finite threshold b, found a = ${describe(a)}, b = ${formatList(b)}.`,
    );
  }
  return Float64Array.from([a, ...b]);
}

/** Returns the threshold parameters of a vector a, b1, ..., b(K-1). */
export function thresholdsFromVector(vector: Float64Array): ThresholdParameters {
  return { a: vector[0] ?? 0, b: Array.from(vector.subarray(1)) };
}

/** Formats a list of numbers for a message, or describes any other value. */
export function formatList(values: unknown): string {
  return Array.isArray(values) ? `[${values.map(describe).join(", ")}]` : describe(values);
}

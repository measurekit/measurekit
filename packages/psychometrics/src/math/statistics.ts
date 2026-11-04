import { mean } from "@measurekit/core";

/**
 * Returns the sample covariance (denominator n - 1) of at least two pairs of values, as (n Σxy - Σx Σy) / (n (n - 1)).
 * For scores, which are integers, the sums are exact, so the result does not depend on the order of the pairs.
 */
export function covariance(x: Float64Array, y: Float64Array): number {
  const n = x.length;
  let sumX = 0;
  let sumY = 0;
  let sumProducts = 0;
  for (let i = 0; i < n; i++) {
    const xi = x[i] ?? 0;
    const yi = y[i] ?? 0;
    sumX += xi;
    sumY += yi;
    sumProducts += xi * yi;
  }
  return (n * sumProducts - sumX * sumY) / (n * (n - 1));
}

/** Returns the largest absolute change from `previous` to `current`, over the length of `current`. */
export function largestChange(previous: Float64Array, current: Float64Array): number {
  let largest = 0;
  for (let j = 0; j < current.length; j++) {
    largest = Math.max(largest, Math.abs((current[j] ?? 0) - (previous[j] ?? 0)));
  }
  return largest;
}

/** Returns the integer nearest to x, the even one of two at a half, as R's `round`. */
export function roundHalfToEven(x: number): number {
  const below = Math.floor(x);
  const fraction = x - below;
  return fraction > 0.5 || (fraction === 0.5 && below % 2 !== 0) ? below + 1 : below;
}

/** Returns the sample variance (denominator n - 1) of at least two values. */
export function variance(values: Float64Array): number {
  return covariance(values, values);
}

/**
 * Returns the sample standard deviation (denominator n - 1) of at least two values by two passes, which keeps the
 * precision of values that are not integers.
 */
export function standardDeviation(values: Float64Array): number {
  const center = mean(values);
  let squares = 0;
  for (const value of values) squares += (value - center) ** 2;
  return Math.sqrt(squares / (values.length - 1));
}

/** Returns the Pearson correlation of paired values, or `null` when there are fewer than two or either is constant. */
export function correlation(x: Float64Array, y: Float64Array): number | null {
  if (x.length < 2) return null;
  const varianceX = variance(x);
  const varianceY = variance(y);
  if (varianceX === 0 || varianceY === 0) return null;
  return covariance(x, y) / Math.sqrt(varianceX * varianceY);
}

/**
 * Returns the `probability` quantile of values sorted in ascending order, by linear interpolation between order
 * statistics: R's default, type 7. The values are at least one, and `probability` is from 0 to 1.
 */
export function sampleQuantile(sorted: Float64Array, probability: number): number {
  const position = (sorted.length - 1) * probability;
  const below = Math.floor(position);
  const lower = sorted[below] ?? 0;
  const upper = sorted[Math.min(below + 1, sorted.length - 1)] ?? lower;
  return lower + (position - below) * (upper - lower);
}

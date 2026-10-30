/** Returns the dot product of two vectors of the same length. */
export function dot(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] ?? 0) * (b[i] ?? 0);
  return sum;
}

/** Returns the largest absolute value, 0 for no values. */
export function maxAbs(values: Float64Array): number {
  let largest = 0;
  for (const value of values) largest = Math.max(largest, Math.abs(value));
  return largest;
}

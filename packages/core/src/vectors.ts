/** Returns the dot product of two vectors of the same length. */
export function dot(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] ?? 0) * (b[i] ?? 0);
  return sum;
}

/** Writes the product of the square matrix `matrix`, by rows, and a vector of as many entries into `out`. */
export function multiplyVector(matrix: Float64Array, vector: Float64Array, out: Float64Array): void {
  const size = vector.length;
  for (let i = 0; i < size; i++) {
    let sum = 0;
    for (let j = 0; j < size; j++) sum += (matrix[i * size + j] ?? 0) * (vector[j] ?? 0);
    out[i] = sum;
  }
}

/** Returns the largest absolute value, 0 for no values. */
export function maxAbs(values: Float64Array): number {
  let largest = 0;
  for (const value of values) largest = Math.max(largest, Math.abs(value));
  return largest;
}

/** Returns the largest absolute change from `previous` to `current`, over the length of `current`. */
export function largestChange(previous: Float64Array, current: Float64Array): number {
  let largest = 0;
  for (let j = 0; j < current.length; j++) {
    largest = Math.max(largest, Math.abs((current[j] ?? 0) - (previous[j] ?? 0)));
  }
  return largest;
}

/** Returns the mean of at least one value. */
export function mean(values: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += values[i] ?? 0;
  return sum / values.length;
}

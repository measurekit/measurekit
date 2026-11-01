/**
 * Returns the dot product of two vectors of the same length.
 *
 * @param a - The first vector.
 * @param b - The second vector.
 * @returns The sum of the products of the entries.
 */
export function dot(a: Float64Array, b: Float64Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] ?? 0) * (b[i] ?? 0);
  return sum;
}

/**
 * Writes the product of the square matrix `matrix`, by rows, and a vector of as many entries into `out`.
 *
 * @param matrix - The square matrix, by rows.
 * @param vector - The vector, with as many entries as `matrix` has rows.
 * @param out - Receives the product, one entry for each row.
 */
export function multiplyVector(matrix: Float64Array, vector: Float64Array, out: Float64Array): void {
  const size = vector.length;
  for (let i = 0; i < size; i++) {
    let sum = 0;
    for (let j = 0; j < size; j++) sum += (matrix[i * size + j] ?? 0) * (vector[j] ?? 0);
    out[i] = sum;
  }
}

/**
 * Returns the largest absolute value, 0 for no values.
 *
 * @param values - The values.
 * @returns The largest absolute value.
 */
export function maxAbs(values: Float64Array): number {
  let largest = 0;
  for (const value of values) largest = Math.max(largest, Math.abs(value));
  return largest;
}

/**
 * Returns the largest absolute change from `previous` to `current`, over the length of `current`.
 *
 * @param previous - The values before.
 * @param current - The values after, of the length that is compared.
 * @returns The largest absolute difference of an entry.
 */
export function largestChange(previous: Float64Array, current: Float64Array): number {
  let largest = 0;
  for (let j = 0; j < current.length; j++) {
    largest = Math.max(largest, Math.abs((current[j] ?? 0) - (previous[j] ?? 0)));
  }
  return largest;
}

/**
 * Returns the mean of at least one value.
 *
 * @param values - The values.
 * @returns The sum of the values over their number.
 */
export function mean(values: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < values.length; i++) sum += values[i] ?? 0;
  return sum / values.length;
}

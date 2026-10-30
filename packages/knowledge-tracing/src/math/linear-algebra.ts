/**
 * Solves A x = b for a symmetric positive definite `size` by `size` matrix A, stored by rows, by its Cholesky
 * factorization, and writes x into `out`. Returns `false`, leaving `out` unspecified, when A is not positive definite
 * to working precision.
 */
export function choleskySolve(a: Float64Array, size: number, b: Float64Array, out: Float64Array): boolean {
  const lower = cholesky(a, size);
  if (lower === null) return false;
  // L y = b, then Lᵀ x = y.
  for (let i = 0; i < size; i++) {
    let sum = b[i] ?? 0;
    for (let k = 0; k < i; k++) sum -= (lower[i * size + k] ?? 0) * (out[k] ?? 0);
    out[i] = sum / (lower[i * size + i] ?? 1);
  }
  for (let i = size - 1; i >= 0; i--) {
    let sum = out[i] ?? 0;
    for (let k = i + 1; k < size; k++) sum -= (lower[k * size + i] ?? 0) * (out[k] ?? 0);
    out[i] = sum / (lower[i * size + i] ?? 1);
  }
  return true;
}

/** Returns the lower triangular L with L Lᵀ = A, by rows, or `null` when A is not positive definite. */
function cholesky(a: Float64Array, size: number): Float64Array | null {
  const lower = new Float64Array(size * size);
  for (let j = 0; j < size; j++) {
    const diagonal = a[j * size + j] ?? 0;
    const pivot = diagonal - rowProduct(lower, size, j, j, j);
    if (!(pivot > size * Number.EPSILON * Math.abs(diagonal))) return null;
    const root = Math.sqrt(pivot);
    lower[j * size + j] = root;
    for (let i = j + 1; i < size; i++) {
      lower[i * size + j] = ((a[i * size + j] ?? 0) - rowProduct(lower, size, i, j, j)) / root;
    }
  }
  return lower;
}

/** Returns the sum over k below `count` of m[i][k] m[j][k]. */
function rowProduct(m: Float64Array, size: number, i: number, j: number, count: number): number {
  let sum = 0;
  for (let k = 0; k < count; k++) sum += (m[i * size + k] ?? 0) * (m[j * size + k] ?? 0);
  return sum;
}

/** Returns the sum of squares of the values. */
export function squares(values: Float64Array): number {
  let sum = 0;
  for (const value of values) sum += value * value;
  return sum;
}

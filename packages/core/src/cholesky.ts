/**
 * Returns the lower triangular L with L Lᵀ = A of a symmetric `size` by `size` matrix A, stored by rows, or `null`
 * when A is not positive definite to working precision: when a pivot is at most `size` ε |A_jj|.
 */
export function cholesky(a: Float64Array, size: number): Float64Array | null {
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

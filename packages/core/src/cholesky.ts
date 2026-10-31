/** The rows of a column that are factored together, so that their sums, which are independent, overlap in the processor. */
const BLOCK = 4;

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
    let i = j + 1;
    for (; i + BLOCK <= size; i += BLOCK) factorRows(a, lower, size, i, j, root);
    for (; i < size; i++) {
      lower[i * size + j] = ((a[i * size + j] ?? 0) - rowProduct(lower, size, i, j, j)) / root;
    }
  }
  return lower;
}

/**
 * Writes L[i + r][j] for the BLOCK rows from `i`, each the sum over k below `j` of L[i + r][k] L[j][k] taken in order
 * as {@link rowProduct} takes it, from one pass over row j.
 */
function factorRows(a: Float64Array, lower: Float64Array, size: number, i: number, j: number, root: number): void {
  const row0 = i * size;
  const row1 = row0 + size;
  const row2 = row1 + size;
  const row3 = row2 + size;
  const pivotRow = j * size;
  let sum0 = 0;
  let sum1 = 0;
  let sum2 = 0;
  let sum3 = 0;
  for (let k = 0; k < j; k++) {
    const factor = lower[pivotRow + k] ?? 0;
    sum0 += (lower[row0 + k] ?? 0) * factor;
    sum1 += (lower[row1 + k] ?? 0) * factor;
    sum2 += (lower[row2 + k] ?? 0) * factor;
    sum3 += (lower[row3 + k] ?? 0) * factor;
  }
  lower[row0 + j] = ((a[row0 + j] ?? 0) - sum0) / root;
  lower[row1 + j] = ((a[row1 + j] ?? 0) - sum1) / root;
  lower[row2 + j] = ((a[row2 + j] ?? 0) - sum2) / root;
  lower[row3 + j] = ((a[row3 + j] ?? 0) - sum3) / root;
}

/** Returns the sum over k below `count` of m[i][k] m[j][k]. */
function rowProduct(m: Float64Array, size: number, i: number, j: number, count: number): number {
  let sum = 0;
  for (let k = 0; k < count; k++) sum += (m[i * size + k] ?? 0) * (m[j * size + k] ?? 0);
  return sum;
}

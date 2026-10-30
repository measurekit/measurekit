import { cholesky } from "@measurekit/core";

/**
 * Returns the inverse of the `size` by `size` matrix `a`, stored by rows, or `null` when it is singular to working
 * precision. Uses Gauss-Jordan elimination with partial pivoting; meant for the small matrices of item parameters.
 */
export function invert(a: Float64Array, size: number): Float64Array | null {
  const work = Float64Array.from(a);
  const inverse = new Float64Array(size * size);
  for (let i = 0; i < size; i++) inverse[i * size + i] = 1;
  const scale = work.reduce((largest, value) => Math.max(largest, Math.abs(value)), 0);
  const tolerance = size * Number.EPSILON * scale;
  for (let column = 0; column < size; column++) {
    const pivot = pivotRow(work, size, column);
    const pivotValue = work[pivot * size + column] ?? 0;
    if (!(Math.abs(pivotValue) > tolerance)) return null;
    swapRows(work, size, pivot, column);
    swapRows(inverse, size, pivot, column);
    scaleRow(work, size, column, 1 / pivotValue);
    scaleRow(inverse, size, column, 1 / pivotValue);
    for (let row = 0; row < size; row++) {
      const factor = work[row * size + column] ?? 0;
      if (row === column || factor === 0) continue;
      subtractRow(work, size, row, column, factor);
      subtractRow(inverse, size, row, column, factor);
    }
  }
  return inverse;
}

function pivotRow(work: Float64Array, size: number, column: number): number {
  let best = column;
  for (let row = column + 1; row < size; row++) {
    if (Math.abs(work[row * size + column] ?? 0) > Math.abs(work[best * size + column] ?? 0)) best = row;
  }
  return best;
}

function swapRows(m: Float64Array, size: number, a: number, b: number): void {
  if (a === b) return;
  for (let j = 0; j < size; j++) {
    const value = m[a * size + j] ?? 0;
    m[a * size + j] = m[b * size + j] ?? 0;
    m[b * size + j] = value;
  }
}

function scaleRow(m: Float64Array, size: number, row: number, factor: number): void {
  for (let j = 0; j < size; j++) m[row * size + j] = (m[row * size + j] ?? 0) * factor;
}

/** Subtracts `factor` times row `source` from row `target`. */
function subtractRow(m: Float64Array, size: number, target: number, source: number, factor: number): void {
  for (let j = 0; j < size; j++)
    m[target * size + j] = (m[target * size + j] ?? 0) - factor * (m[source * size + j] ?? 0);
}

/**
 * Returns the inverse of the symmetric positive definite `size` by `size` matrix `a`, stored by rows, or `null` when it
 * is not positive definite to working precision. Uses the Cholesky factor L of a: the inverse is L⁻ᵀ L⁻¹, about half
 * the work of {@link invert}.
 */
export function invertPositiveDefinite(a: Float64Array, size: number): Float64Array | null {
  const factor = cholesky(a, size);
  return factor === null ? null : crossProductOfColumns(invertedTranspose(factor, size), size);
}

/**
 * Returns M = L⁻¹ of a lower triangular L, by forward substitution, stored transposed, by columns of M, so that the
 * columns are contiguous.
 */
function invertedTranspose(factor: Float64Array, size: number): Float64Array {
  const transposed = new Float64Array(size * size);
  for (let j = 0; j < size; j++) {
    const columnJ = j * size;
    transposed[columnJ + j] = 1 / (factor[j * size + j] ?? 1);
    for (let i = j + 1; i < size; i++) {
      let sum = 0;
      for (let k = j; k < i; k++) sum -= (factor[i * size + k] ?? 0) * (transposed[columnJ + k] ?? 0);
      transposed[columnJ + i] = sum / (factor[i * size + i] ?? 1);
    }
  }
  return transposed;
}

/** Returns Mᵀ M of a lower triangular M stored by columns: entry (i, j) sums M[k][i] M[k][j] over k ≥ max(i, j). */
function crossProductOfColumns(transposed: Float64Array, size: number): Float64Array {
  const product = new Float64Array(size * size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = 0;
      for (let k = i; k < size; k++) sum += (transposed[i * size + k] ?? 0) * (transposed[j * size + k] ?? 0);
      product[i * size + j] = sum;
      product[j * size + i] = sum;
    }
  }
  return product;
}

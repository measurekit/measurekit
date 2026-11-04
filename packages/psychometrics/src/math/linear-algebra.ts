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
 * Solves L X = B in place for the `columns` columns of B, by rows, given the lower triangular `size` by `size` L: each
 * row needs the rows above it, so that the columns of B are worked on in one pass over its memory.
 */
export function forwardSolve(factor: Float64Array, size: number, right: Float64Array, columns: number): void {
  for (let i = 0; i < size; i++) {
    for (let k = 0; k < i; k++) subtractRow(right, columns, i, k, factor[i * size + k] ?? 0);
    scaleRow(right, columns, i, 1 / (factor[i * size + i] ?? 1));
  }
}

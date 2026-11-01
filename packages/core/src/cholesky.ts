/** The columns of L that are factored together. */
const COLUMNS = 3;
/** The rows of a block of columns that are factored together: with {@link COLUMNS}, twelve sums that stay in registers. */
const ROWS = 4;

/**
 * Returns the lower triangular L with L Lᵀ = A of a symmetric `size` by `size` matrix A, stored by rows, or `null`
 * when A is not positive definite to working precision: when a pivot is at most `size` ε |A_jj|.
 *
 * Each entry is A_ij less the sum over k of L_ik L_jk, taken in order of k, so that the blocking below, which only shares
 * the reads of the sums that are independent, leaves every result as an unblocked factorization gives it.
 *
 * @param a - The symmetric matrix, by rows.
 * @param size - The number of rows and columns of `a`.
 * @returns The factor L by rows, with zeros above the diagonal; `null` when `a` is not positive definite.
 */
export function cholesky(a: Float64Array, size: number): Float64Array | null {
  const lower = new Float64Array(size * size);
  for (let first = 0; first < size; first += COLUMNS) {
    const end = Math.min(first + COLUMNS, size);
    for (let column = first; column < end; column++) {
      if (!factorPivot(a, lower, size, column)) return null;
      solveRows(a, lower, size, column, column + 1, end);
    }
    let row = end;
    for (; row + ROWS <= size && end - first === COLUMNS; row += ROWS) factorBlock(a, lower, size, row, first);
    for (let column = first; column < end; column++) solveRows(a, lower, size, column, row, size);
  }
  return lower;
}

/** Writes L[column][column], and returns `false` when the pivot is too small for A to be positive definite. */
function factorPivot(a: Float64Array, lower: Float64Array, size: number, column: number): boolean {
  const diagonal = a[column * size + column] ?? 0;
  const pivot = diagonal - rowProduct(lower, size, column, column, column);
  if (!(pivot > size * Number.EPSILON * Math.abs(diagonal))) return false;
  lower[column * size + column] = Math.sqrt(pivot);
  return true;
}

/** Writes L[i][column] for each row i from `from` below `to`, given the entries of L to its left and its pivot. */
function solveRows(a: Float64Array, lower: Float64Array, size: number, column: number, from: number, to: number): void {
  const root = lower[column * size + column] ?? 0;
  for (let i = from; i < to; i++) {
    lower[i * size + column] = ((a[i * size + column] ?? 0) - rowProduct(lower, size, i, column, column)) / root;
  }
}

/**
 * Writes L[i + r][first + c] for the ROWS rows from `i` and the COLUMNS columns from `first`, given the pivots and the
 * rows `first` to `first + COLUMNS` of L. Each sum over k below `first` is taken in order as {@link rowProduct} takes it,
 * and its terms from `first` on follow as the entries to their left are known, so that the twelve sums, which are
 * independent, overlap in the processor and each read of L serves three or four of them.
 */
function factorBlock(a: Float64Array, lower: Float64Array, size: number, i: number, first: number): void {
  const row0 = i * size;
  const row1 = row0 + size;
  const row2 = row1 + size;
  const row3 = row2 + size;
  const pivotRow0 = first * size;
  const pivotRow1 = pivotRow0 + size;
  const pivotRow2 = pivotRow1 + size;
  let sum00 = 0;
  let sum01 = 0;
  let sum02 = 0;
  let sum10 = 0;
  let sum11 = 0;
  let sum12 = 0;
  let sum20 = 0;
  let sum21 = 0;
  let sum22 = 0;
  let sum30 = 0;
  let sum31 = 0;
  let sum32 = 0;
  for (let k = 0; k < first; k++) {
    const factor0 = lower[pivotRow0 + k] ?? 0;
    const factor1 = lower[pivotRow1 + k] ?? 0;
    const factor2 = lower[pivotRow2 + k] ?? 0;
    const x0 = lower[row0 + k] ?? 0;
    const x1 = lower[row1 + k] ?? 0;
    const x2 = lower[row2 + k] ?? 0;
    const x3 = lower[row3 + k] ?? 0;
    sum00 += x0 * factor0;
    sum01 += x0 * factor1;
    sum02 += x0 * factor2;
    sum10 += x1 * factor0;
    sum11 += x1 * factor1;
    sum12 += x1 * factor2;
    sum20 += x2 * factor0;
    sum21 += x2 * factor1;
    sum22 += x2 * factor2;
    sum30 += x3 * factor0;
    sum31 += x3 * factor1;
    sum32 += x3 * factor2;
  }
  finishRow(a, lower, row0, first, size, sum00, sum01, sum02);
  finishRow(a, lower, row1, first, size, sum10, sum11, sum12);
  finishRow(a, lower, row2, first, size, sum20, sum21, sum22);
  finishRow(a, lower, row3, first, size, sum30, sum31, sum32);
}

/**
 * Writes L[i][first + c] for the COLUMNS columns from `first` of the row that starts at `row`, from its sums over k below
 * `first`: each column takes its terms from `first` on, in order, from the entries to its left.
 */
function finishRow(
  a: Float64Array,
  lower: Float64Array,
  row: number,
  first: number,
  size: number,
  sum0: number,
  sum1: number,
  sum2: number,
): void {
  const pivotRow1 = (first + 1) * size;
  const pivotRow2 = pivotRow1 + size;
  const entry0 = ((a[row + first] ?? 0) - sum0) / (lower[first * size + first] ?? 0);
  const entry1 =
    ((a[row + first + 1] ?? 0) - (sum1 + entry0 * (lower[pivotRow1 + first] ?? 0))) /
    (lower[pivotRow1 + first + 1] ?? 0);
  const terms2 = sum2 + entry0 * (lower[pivotRow2 + first] ?? 0) + entry1 * (lower[pivotRow2 + first + 1] ?? 0);
  lower[row + first] = entry0;
  lower[row + first + 1] = entry1;
  lower[row + first + 2] = ((a[row + first + 2] ?? 0) - terms2) / (lower[pivotRow2 + first + 2] ?? 0);
}

/** Returns the sum over k below `count` of m[i][k] m[j][k]. */
function rowProduct(m: Float64Array, size: number, i: number, j: number, count: number): number {
  let sum = 0;
  for (let k = 0; k < count; k++) sum += (m[i * size + k] ?? 0) * (m[j * size + k] ?? 0);
  return sum;
}

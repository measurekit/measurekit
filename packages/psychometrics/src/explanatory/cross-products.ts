/**
 * Returns XᵀW^kX of a design matrix, symmetric and by rows, for the diagonal matrix W of `weights` and k = `power`: XᵀX
 * without weights.
 *
 * @param design - X, by row, then column.
 */
export function designCrossProducts(
  design: Float64Array,
  rows: number,
  columns: number,
  weights?: Float64Array,
  power = 1,
): Float64Array {
  const matrix = new Float64Array(columns * columns);
  for (let i = 0; i < rows; i++) {
    const w = weights === undefined ? 1 : (weights[i] ?? 0) ** power;
    for (let j = 0; j < columns; j++) {
      const x = w * (design[i * columns + j] ?? 0);
      for (let k = 0; k <= j; k++) {
        matrix[j * columns + k] = (matrix[j * columns + k] ?? 0) + x * (design[i * columns + k] ?? 0);
      }
    }
  }
  return symmetrize(matrix, columns);
}

/** Copies the lower triangle of a square matrix, by rows, to its upper triangle. */
function symmetrize(matrix: Float64Array, size: number): Float64Array {
  for (let j = 0; j < size; j++) {
    for (let k = 0; k < j; k++) matrix[k * size + j] = matrix[j * size + k] ?? 0;
  }
  return matrix;
}

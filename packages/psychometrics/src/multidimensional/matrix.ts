/** Writes the product of the `rows` by `inner` matrix `a` and the `inner` by `columns` matrix `b`, by rows, into `out`. */
export function multiply(
  a: Float64Array,
  b: Float64Array,
  rows: number,
  inner: number,
  columns: number,
  out: Float64Array,
): void {
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < columns; j++) {
      let sum = 0;
      for (let k = 0; k < inner; k++) sum += (a[i * inner + k] ?? 0) * (b[k * columns + j] ?? 0);
      out[i * columns + j] = sum;
    }
  }
}

/** Returns the square matrix of `size` rows averaged with its transpose. */
export function symmetrized(matrix: Float64Array, size: number): Float64Array {
  const result = new Float64Array(size * size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++)
      result[i * size + j] = ((matrix[i * size + j] ?? 0) + (matrix[j * size + i] ?? 0)) / 2;
  }
  return result;
}

/** Returns the transpose of the `rows` by `columns` matrix `a`. */
export function transpose(a: Float64Array, rows: number, columns: number): Float64Array {
  const result = new Float64Array(a.length);
  for (let i = 0; i < rows; i++) for (let j = 0; j < columns; j++) result[j * rows + i] = a[i * columns + j] ?? 0;
  return result;
}

/** Returns the identity matrix of `size` rows, by rows. */
export function identity(size: number): Float64Array {
  const matrix = new Float64Array(size * size);
  for (let i = 0; i < size; i++) matrix[i * size + i] = 1;
  return matrix;
}

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

/** Writes the product of the `size` by `size` matrix `matrix`, by rows, and a vector of `size` entries into `out`. */
export function multiplyVector(matrix: Float64Array, vector: Float64Array, out: Float64Array): void {
  const size = vector.length;
  for (let i = 0; i < size; i++) {
    let sum = 0;
    for (let j = 0; j < size; j++) sum += (matrix[i * size + j] ?? 0) * (vector[j] ?? 0);
    out[i] = sum;
  }
}

/** Returns tr(a b) of two `size` by `size` matrices. */
export function traceOfProduct(a: Float64Array, b: Float64Array, size: number): number {
  let sum = 0;
  for (let i = 0; i < size; i++) for (let k = 0; k < size; k++) sum += (a[i * size + k] ?? 0) * (b[k * size + i] ?? 0);
  return sum;
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

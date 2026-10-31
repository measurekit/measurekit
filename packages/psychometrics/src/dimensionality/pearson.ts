/**
 * Returns the Pearson correlations of columns of equal length as a symmetric matrix by rows, each pair over the rows
 * where both are numbers, as R's `cor(use = "pairwise.complete.obs")`; NaN where a pair has fewer than two such rows or
 * a column has no variance among them. A column is NaN where missing.
 */
export function pearsonMatrix(columns: readonly Float64Array[]): Float64Array {
  const size = columns.length;
  const result = new Float64Array(size * size).fill(Number.NaN);
  for (let i = 0; i < size; i++) {
    for (let j = i; j < size; j++) {
      const value = i === j ? 1 : pairwise(columns[i] ?? new Float64Array(), columns[j] ?? new Float64Array());
      result[i * size + j] = value;
      result[j * size + i] = value;
    }
  }
  return result;
}

/** Returns the correlation of two columns over the rows where both are numbers, by two passes, NaN if undefined. */
function pairwise(x: Float64Array, y: Float64Array): number {
  let count = 0;
  let sumX = 0;
  let sumY = 0;
  for (let row = 0; row < x.length; row++) {
    const a = x[row] ?? Number.NaN;
    const b = y[row] ?? Number.NaN;
    if (Number.isNaN(a) || Number.isNaN(b)) continue;
    count++;
    sumX += a;
    sumY += b;
  }
  if (count < 2) return Number.NaN;
  const meanX = sumX / count;
  const meanY = sumY / count;
  let squaresX = 0;
  let squaresY = 0;
  let products = 0;
  for (let row = 0; row < x.length; row++) {
    const a = (x[row] ?? Number.NaN) - meanX;
    const b = (y[row] ?? Number.NaN) - meanY;
    if (Number.isNaN(a) || Number.isNaN(b)) continue;
    squaresX += a * a;
    squaresY += b * b;
    products += a * b;
  }
  return squaresX > 0 && squaresY > 0 ? products / Math.sqrt(squaresX * squaresY) : Number.NaN;
}

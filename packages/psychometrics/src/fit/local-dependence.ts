import type { ResponseMatrix } from "../data/response-matrix.js";
import { largestCategoryCount } from "../irt/items.js";
import type { IrtItem } from "../irt/model.js";
import { checkItems, checkPersonThetas, scoreMoments } from "./inputs.js";

/**
 * Returns Yen's Q3 for every pair of items.
 *
 * @remarks
 * The correlation over persons of the residuals x - E(x | θ) of the two items, at the persons' thetas, as mirt's
 * `residuals(type = "Q3")` computes it. Under local independence it is near -1 / (k - 1) for k items; large values flag
 * dependent pairs. Each pair uses the persons who answered both items. The result is a symmetric matrix in item order
 * with 1 on the diagonal, `null` for a pair with fewer than two such persons or a residual without variance.
 *
 * @throws `DataError` when the matrix, the items or the thetas are invalid, or a response is not a category of its
 * item.
 */
export function yenQ3(
  matrix: ResponseMatrix,
  items: readonly IrtItem[],
  thetas: readonly number[],
): (number | null)[][] {
  checkItems(matrix, items);
  checkPersonThetas(matrix, thetas);
  const itemCount = items.length;
  const residuals = residualTable(matrix, items, thetas);
  // The correlations of the pairs i < j, by i, then j; NaN where undefined.
  const upper = new Float64Array(itemCount * itemCount);
  for (let i = 0; i < itemCount; i++) {
    for (let j = i + 1; j < itemCount; j++) upper[i * itemCount + j] = pairwiseCorrelation(residuals, itemCount, i, j);
  }
  return items.map((_, i) =>
    items.map((__, j) => {
      if (i === j) return 1;
      const value = upper[Math.min(i, j) * itemCount + Math.max(i, j)] ?? Number.NaN;
      return Number.isNaN(value) ? null : value;
    }),
  );
}

/** Returns the residuals x - E(x | θ) by person, then item, NaN where a response is missing. */
function residualTable(matrix: ResponseMatrix, items: readonly IrtItem[], thetas: readonly number[]): Float64Array {
  const itemCount = items.length;
  const residuals = new Float64Array(matrix.responses.length * itemCount).fill(Number.NaN);
  const probabilities = new Float64Array(largestCategoryCount(items));
  const moments = new Float64Array(3);
  matrix.responses.forEach((row, person) => {
    row.forEach((response, column) => {
      const item = items[column];
      if (response === null || item === undefined) return;
      scoreMoments(item, thetas[person] ?? 0, probabilities, moments);
      residuals[person * itemCount + column] = response - (moments[0] ?? 0);
    });
  });
  return residuals;
}

/**
 * Returns the correlation of columns i and j over the rows where both are numbers, by two passes, as R's `cor`; NaN
 * with fewer than two such rows or a column without variance.
 */
function pairwiseCorrelation(residuals: Float64Array, itemCount: number, i: number, j: number): number {
  let count = 0;
  let sumI = 0;
  let sumJ = 0;
  for (let index = 0; index < residuals.length; index += itemCount) {
    const x = residuals[index + i] ?? Number.NaN;
    const y = residuals[index + j] ?? Number.NaN;
    if (Number.isNaN(x) || Number.isNaN(y)) continue;
    count++;
    sumI += x;
    sumJ += y;
  }
  if (count < 2) return Number.NaN;
  const meanI = sumI / count;
  const meanJ = sumJ / count;
  let squaresI = 0;
  let squaresJ = 0;
  let products = 0;
  for (let index = 0; index < residuals.length; index += itemCount) {
    const x = (residuals[index + i] ?? Number.NaN) - meanI;
    const y = (residuals[index + j] ?? Number.NaN) - meanJ;
    if (Number.isNaN(x) || Number.isNaN(y)) continue;
    squaresI += x * x;
    squaresJ += y * y;
    products += x * y;
  }
  return squaresI > 0 && squaresJ > 0 ? products / Math.sqrt(squaresI * squaresJ) : Number.NaN;
}

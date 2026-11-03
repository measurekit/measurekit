import { DataError } from "@measurekit/core";

/**
 * The most items of an analysis that holds a matrix of every pair of items, such as their correlations: at 1,000 items
 * the worst of them, an exploratory factor analysis, adds about 150 MB to the peak memory and its eigendecompositions
 * take minutes, while each matrix grows with the square of the items and its decompositions with their cube.
 */
const MAX_MATRIX_ITEMS = 1000;

/**
 * Checks that an analysis that holds a matrix of every pair of items has at most 1,000 items.
 *
 * @param items - The number of items.
 * @param analysis - Names the analysis in the error message.
 * @throws `DataError` when there are more.
 */
export function checkMatrixItems(items: number, analysis: string): void {
  if (items > MAX_MATRIX_ITEMS) {
    throw new DataError(
      `${analysis} holds a matrix of every pair of items, so it takes at most ${MAX_MATRIX_ITEMS} items, found ${items}.`,
    );
  }
}

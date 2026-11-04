import type { ResponseMatrix } from "../data/internal.js";
import { type IrtItem, largestCategoryCount } from "../irt/internal.js";
import { checkItems, checkPersonThetas, isExtreme, scoreMoments } from "./inputs.js";
import { addResponse, type MeanSquareFit, meanSquares, SUM_COUNT } from "./mean-squares.js";

/** The residual fit of one item. */
export interface ItemFit extends MeanSquareFit {
  /** The item id. */
  readonly id: string;
}

/**
 * Returns the infit and outfit mean squares of every item, with their standardized forms, from the residuals of the
 * responses at the persons' thetas, as eRm's `itemfit` computes them.
 *
 * @remarks
 * The items are usually Rasch or partial credit items and the thetas their maximum likelihood estimates. Persons with
 * extreme scores, every answered response in its item's lowest category or every one in its highest, are left out, and
 * missing responses are skipped. A statistic is `null` for an item without other responses.
 *
 * @param matrix - The responses of the persons.
 * @param items - The calibrated items, in the order of the matrix's items.
 * @param thetas - The estimate of theta of each person, in the order of the matrix.
 * @returns For each item, the infit and outfit mean squares and their standardized forms.
 * @throws `DataError` when the matrix, the items or the thetas are invalid, or a response is not a category of its
 * item.
 *
 * @example
 * ```ts
 * import { buildResponseMatrix } from "@measurekit/psychometrics/data";
 * import { itemFit } from "@measurekit/psychometrics/fit";
 * import { oneParameterLogistic } from "@measurekit/psychometrics/irt";
 *
 * const matrix = buildResponseMatrix([
 *   [1, 0, 1, 1],
 *   [0, 0, 1, 0],
 *   [1, 1, 1, 1],
 *   [0, 1, 0, 0],
 *   [1, 1, 0, 1],
 *   [0, 0, 0, 1],
 * ]);
 * const items = [-1, -0.3, 0.3, 1].map((b) => ({ model: oneParameterLogistic, parameters: { a: 1, b, c: 0, d: 1 } }));
 * const thetas = [0.5, -1, 1.2, -0.7, 0.6, -0.5];
 * itemFit(matrix, items, thetas)[0]; // { id: "item1", infitMeanSquare: 1.01, outfitMeanSquare: 0.885, ... }
 * ```
 */
export function itemFit(
  matrix: ResponseMatrix,
  items: readonly IrtItem[],
  thetas: readonly number[],
): readonly ItemFit[] {
  checkItems(matrix, items);
  checkPersonThetas(matrix, thetas);
  const sums = new Float64Array(items.length * SUM_COUNT);
  const probabilities = new Float64Array(largestCategoryCount(items));
  const moments = new Float64Array(3);
  matrix.responses.forEach((row, person) => {
    if (isExtreme(items, row)) return;
    const theta = thetas[person] ?? 0;
    row.forEach((response, column) => {
      const item = items[column];
      if (response === null || item === undefined) return;
      scoreMoments(item, theta, probabilities, moments);
      addResponse(sums, column * SUM_COUNT, response - (moments[0] ?? 0), moments);
    });
  });
  return matrix.itemIds.map((id, column) => ({ id, ...meanSquares(sums, column * SUM_COUNT) }));
}

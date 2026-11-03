import { normalQuantile } from "@measurekit/core";
import { checkMatrixItems, checkResponseMatrix, type ResponseMatrix } from "../data/internal.js";

/** A rank for a missing response. */
export const MISSING_RANK = -1;

/** An item of a response matrix coded for correlations. */
export interface CodedItem {
  /** The id of the item. */
  readonly id: string;
  /** The response of each person, NaN where missing. */
  readonly values: Float64Array;
  /** The rank of each person's response among the item's observed categories, from 0, or {@link MISSING_RANK}. */
  readonly ranks: Int32Array;
  /** The number of observed categories. */
  readonly levels: number;
  /**
   * The Φ⁻¹ of the cumulative proportion of each observed category but the last, over all the item's responses: one
   * fewer than `levels`.
   */
  readonly thresholds: Float64Array;
}

/**
 * Checks a response matrix for an analysis of the pairs of its items and returns its items coded by observed category.
 *
 * @param analysis - Names the analysis in the error message.
 * @throws `DataError` when the matrix is invalid or has more than 1,000 items.
 */
export function codeItems(matrix: ResponseMatrix, analysis: string): CodedItem[] {
  checkResponseMatrix(matrix);
  checkMatrixItems(matrix.itemIds.length, analysis);
  const persons = matrix.responses.length;
  return matrix.itemIds.map((id, column) => {
    const values = new Float64Array(persons);
    const counts = new Map<number, number>();
    matrix.responses.forEach((row, person) => {
      const response = row[column] ?? null;
      values[person] = response ?? Number.NaN;
      if (response !== null) counts.set(response, (counts.get(response) ?? 0) + 1);
    });
    const categories = [...counts.keys()].sort((a, b) => a - b);
    const rankOf = new Map(categories.map((category, rank) => [category, rank]));
    const ranks = Int32Array.from(values, (value) =>
      Number.isNaN(value) ? MISSING_RANK : (rankOf.get(value) ?? MISSING_RANK),
    );
    const responses = [...counts.values()].reduce((sum, count) => sum + count, 0);
    let cumulative = 0;
    const thresholds = new Float64Array(Math.max(categories.length - 1, 0));
    for (let rank = 0; rank < thresholds.length; rank++) {
      cumulative += counts.get(categories[rank] ?? 0) ?? 0;
      thresholds[rank] = normalQuantile(cumulative / responses);
    }
    return { id, values, ranks, levels: categories.length, thresholds };
  });
}

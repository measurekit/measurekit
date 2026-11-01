import { DataError } from "@measurekit/core";
import type { ResponseMatrix } from "../data/response-matrix.js";
import { type CodedItem, codeItems } from "./coded-items.js";
import { pearsonMatrix } from "./pearson.js";
import { polychoricMatrix } from "./polychoric.js";
import { smooth } from "./smoothing.js";

/** The correlation of items: Pearson's for scores treated as continuous, or the polychoric correlation. */
export type ItemCorrelation = "pearson" | "polychoric";

/** The choices of {@link ItemCorrelation}. */
export const ITEM_CORRELATIONS = ["pearson", "polychoric"] as const;

/** The fewest items of an analysis of dimensions: one factor is not identified by two. */
const MIN_ITEMS = 3;

/** The coded items of a response matrix and their correlations, smoothed when they are not positive definite. */
export interface ItemCorrelations {
  readonly items: readonly CodedItem[];
  /** The correlations by rows, symmetric, 1 on the diagonal, positive definite. */
  readonly correlations: Float64Array;
  /** Whether the correlations of the data were not positive definite and were smoothed. */
  readonly smoothed: boolean;
}

/** Returns the correlations of coded items as a symmetric matrix by rows, NaN where undefined. */
export function correlateItems(
  items: readonly CodedItem[],
  correlation: ItemCorrelation,
  correction: number,
): Float64Array {
  return correlation === "pearson"
    ? pearsonMatrix(items.map((item) => item.values))
    : polychoricMatrix(items, correction);
}

/**
 * Checks that there are at least three items and each has two observed categories.
 *
 * @throws `DataError` naming `analysis` when not.
 */
export function checkItems(items: readonly CodedItem[], analysis: string): void {
  if (items.length < MIN_ITEMS) {
    throw new DataError(
      `${analysis} needs at least ${MIN_ITEMS} items, as one factor is not identified by two, found ${items.length}.`,
    );
  }
  for (const item of items) {
    if (item.levels < 2) {
      throw new DataError(
        `${analysis} needs items with at least two observed categories; item "${item.id}" has ${item.levels}.`,
      );
    }
  }
}

/**
 * Returns the correlations after checking that every pair has one.
 *
 * @throws `DataError` naming the first pair without a correlation and `what`.
 */
export function checkedMatrix(correlations: Float64Array, items: readonly CodedItem[], what: string): Float64Array {
  const size = items.length;
  for (let i = 0; i < size; i++) {
    for (let j = i + 1; j < size; j++) {
      if (Number.isNaN(correlations[i * size + j])) {
        throw new DataError(
          `${what} gives no correlation of items "${items[i]?.id}" and "${items[j]?.id}": too few persons answered both.`,
        );
      }
    }
  }
  return correlations;
}

/**
 * Codes the items of a response matrix, checks them and returns their correlations, smoothed if they are not positive
 * definite, for an analysis that needs a matrix that is.
 *
 * @throws `DataError` when the matrix is invalid, has fewer than three items or an item with fewer than two observed
 * categories, or a pair of items has too few persons for a correlation.
 */
export function positiveDefiniteCorrelations(
  matrix: ResponseMatrix,
  correlation: ItemCorrelation,
  correction: number,
  analysis: string,
): ItemCorrelations {
  const items = codeItems(matrix);
  checkItems(items, analysis);
  const observed = checkedMatrix(correlateItems(items, correlation, correction), items, "The data");
  const { correlations, smoothed } = smooth(observed, items.length);
  return { items, correlations, smoothed };
}

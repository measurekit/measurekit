import { DataError } from "@measurekit/core";
import { checkResponseMatrix, type ResponseMatrix } from "../data/response-matrix.js";
import { covariance } from "../math/statistics.js";

/** The most categories an item may have: the mokken package handles scores from 0 to 9. */
const MAX_CATEGORIES = 10;

/** The scores of the persons who answered every item, as Mokken scale analysis takes them. */
export interface MokkenData {
  /** The item ids. */
  readonly itemIds: readonly string[];
  /** The number of persons who answered every item. */
  readonly persons: number;
  /** The number of items. */
  readonly items: number;
  /** The number of categories: the highest score of any item, once the lowest score of all is taken as 0, plus 1. */
  readonly categories: number;
  /** The scores by person, then item. */
  readonly scores: Uint8Array;
}

/**
 * Returns the scores of a response matrix's complete cases, with the lowest score of all items taken as 0.
 *
 * @param subject - Names the analysis in error messages, such as `Scalability`.
 * @param itemsNeeded - The fewest items the analysis needs.
 * @throws `DataError` when the matrix is invalid, fewer than two persons answered every item, there are fewer items
 * than the analysis needs, or scores span more than 10 categories.
 */
export function mokkenData(matrix: ResponseMatrix, subject: string, itemsNeeded = 2): MokkenData {
  checkResponseMatrix(matrix);
  const items = matrix.itemIds.length;
  if (items < itemsNeeded) {
    throw new DataError(`${subject} needs at least ${itemsNeeded} items, found ${items}.`);
  }
  const complete = matrix.responses.filter((row) => !row.includes(null));
  if (complete.length < 2) {
    throw new DataError(`${subject} needs at least 2 persons who answered every item, found ${complete.length}.`);
  }
  let lowest = Number.POSITIVE_INFINITY;
  let highest = 0;
  for (const row of complete) {
    for (const response of row) {
      lowest = Math.min(lowest, response ?? 0);
      highest = Math.max(highest, response ?? 0);
    }
  }
  const categories = highest - lowest + 1;
  if (categories > MAX_CATEGORIES) {
    throw new DataError(
      `${subject} needs scores from at most ${MAX_CATEGORIES} categories, as the mokken package does, found ${categories}.`,
    );
  }
  const scores = new Uint8Array(complete.length * items);
  complete.forEach((row, person) => {
    row.forEach((response, item) => {
      scores[person * items + item] = (response ?? 0) - lowest;
    });
  });
  return { itemIds: matrix.itemIds, persons: complete.length, items, categories, scores };
}

/** Returns the score of a person on an item. */
export function scoreOf(data: MokkenData, person: number, item: number): number {
  return data.scores[person * data.items + item] ?? 0;
}

/** Returns an item's scores, one for each person. */
export function columnOf(data: MokkenData, item: number): Float64Array {
  return Float64Array.from({ length: data.persons }, (_, person) => scoreOf(data, person, item));
}

/** The covariances of every pair of items, and the largest they can be given the items' score distributions. */
export interface Covariances {
  /** The covariances (denominator n - 1) by item, then item; the diagonal holds the variances. */
  readonly observed: Float64Array;
  /** The covariances of the items' scores each sorted, the most that the distributions allow, by item, then item. */
  readonly maximum: Float64Array;
}

/** Returns the covariances of the items and of their sorted scores. */
export function covariances(data: MokkenData): Covariances {
  const { items } = data;
  const columns = Array.from({ length: items }, (_, item) => columnOf(data, item));
  const sorted = columns.map((column) => Float64Array.from(column).sort());
  const observed = new Float64Array(items * items);
  const maximum = new Float64Array(items * items);
  for (let i = 0; i < items; i++) {
    for (let j = i; j < items; j++) {
      const together = covariance(columns[i] ?? new Float64Array(), columns[j] ?? new Float64Array());
      const most = covariance(sorted[i] ?? new Float64Array(), sorted[j] ?? new Float64Array());
      observed[i * items + j] = observed[j * items + i] = together;
      maximum[i * items + j] = maximum[j * items + i] = most;
    }
  }
  return { observed, maximum };
}

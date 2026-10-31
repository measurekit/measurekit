import { checkNonNegative, checkOptions } from "@measurekit/core";
import type { ResponseMatrix } from "../data/response-matrix.js";
import { type CodedItem, codeItems, MISSING_RANK } from "./coded-items.js";
import { createPairEstimator, type PairEstimator } from "./pair-correlation.js";

/** Options of {@link polychoricCorrelations}. */
export interface PolychoricOptions {
  /**
   * Added to each empty cell of the table of a pair of items with two categories each, when the table has one, as
   * psych's `correct`; 0 adds nothing. A table of items with more categories is not adjusted.
   *
   * @defaultValue `0.5`
   */
  readonly correction?: number;
}

/** The polychoric correlations of the items of a response matrix, with their thresholds. */
export interface PolychoricCorrelations {
  /** The item ids, in the matrix's order. */
  readonly itemIds: readonly string[];
  /**
   * The correlations, symmetric, by item: 1 on the diagonal. An item with fewer than two observed categories has `null`
   * in its row and column, the diagonal included, and so does a pair of items that no person answered both.
   */
  readonly correlations: readonly (readonly (number | null)[])[];
  /**
   * Each item's thresholds, ascending: one fewer than its observed categories, Φ⁻¹ of the cumulative proportion of its
   * responses up to each category but the last. Categories nobody chose are skipped. Empty for an item with fewer than
   * two observed categories.
   */
  readonly thresholds: readonly (readonly number[])[];
}

const DEFAULT_CORRECTION = 0.5;

/**
 * Returns the polychoric correlations of the items of a response matrix: the correlation of the normal variables
 * underlying each pair of items, tetrachoric for items with two categories.
 *
 * @remarks
 * Each item is taken to be a normal variable cut at thresholds into its categories. The thresholds are Φ⁻¹ of the
 * item's cumulative proportions over all the persons who answered it. For each pair, in two steps, ρ maximizes the
 * log-likelihood Σ n_kl ln π_kl(ρ) of the table of the persons who answered both, π_kl the probability of cell (k, l)
 * under the bivariate normal distribution with correlation ρ, by Newton's method with the exact derivatives safeguarded
 * by bisection. A table with an empty cell and no association gives -1 or 1, where the likelihood increases to the
 * boundary; the correction of `correction` avoids it for items with two categories. Categories are the observed
 * responses in order, so responses may skip numbers. Responses are integers from 0, which `responseMatrix` checks.
 *
 * @throws `DataError` when the matrix is invalid or `correction` is not a non-negative number.
 * @see {@link https://doi.org/10.1007/BF02296207 | Olsson (1979)}
 */
export function polychoricCorrelations(
  matrix: ResponseMatrix,
  options: PolychoricOptions = {},
): PolychoricCorrelations {
  checkOptions(options);
  const { correction = DEFAULT_CORRECTION } = options;
  checkNonNegative(correction, "The option correction");
  const items = codeItems(matrix);
  const size = items.length;
  const values = polychoricMatrix(items, correction);
  return {
    itemIds: matrix.itemIds,
    correlations: items.map((_, i) =>
      items.map((__, j) => {
        const value = values[i * size + j] ?? Number.NaN;
        return Number.isNaN(value) ? null : value;
      }),
    ),
    thresholds: items.map((item) => Array.from(item.thresholds)),
  };
}

/**
 * Returns the polychoric correlations of coded items as a symmetric matrix by rows, NaN where undefined.
 *
 * @remarks
 * The items' `ranks` are read at each call, so a caller can permute them in place.
 */
export function polychoricMatrix(items: readonly CodedItem[], correction: number): Float64Array {
  const size = items.length;
  const result = new Float64Array(size * size).fill(Number.NaN);
  const largest = items.reduce((most, item) => Math.max(most, item.levels), 1);
  const estimator = createPairEstimator(largest, largest);
  const counts = new Float64Array(largest * largest);
  for (let i = 0; i < size; i++) {
    const first = items[i];
    if (first === undefined || first.levels < 2) continue;
    result[i * size + i] = 1;
    for (let j = i + 1; j < size; j++) {
      const second = items[j];
      if (second === undefined || second.levels < 2) continue;
      const value = pairCorrelation(first, second, correction, counts, estimator);
      result[i * size + j] = value;
      result[j * size + i] = value;
    }
  }
  return result;
}

/** Returns the correlation of two items with at least two categories each, NaN when no person answered both. */
function pairCorrelation(
  x: CodedItem,
  y: CodedItem,
  correction: number,
  counts: Float64Array,
  estimator: PairEstimator,
): number {
  const cells = x.levels * y.levels;
  counts.fill(0, 0, cells);
  let persons = 0;
  for (let person = 0; person < x.ranks.length; person++) {
    const row = x.ranks[person] ?? MISSING_RANK;
    const column = y.ranks[person] ?? MISSING_RANK;
    if (row === MISSING_RANK || column === MISSING_RANK) continue;
    counts[row * y.levels + column] = (counts[row * y.levels + column] ?? 0) + 1;
    persons++;
  }
  if (persons === 0) return Number.NaN;
  if (cells === 4 && correction > 0) {
    for (let cell = 0; cell < 4; cell++) if (counts[cell] === 0) counts[cell] = correction;
  }
  return estimator.estimate({
    counts,
    rows: x.levels,
    columns: y.levels,
    rowThresholds: x.thresholds,
    columnThresholds: y.thresholds,
  });
}

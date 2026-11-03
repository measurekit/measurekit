import { checkChoice, checkNonNegative, checkOptions, checkPositiveInteger, DataError } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { symmetricEigen } from "../math/internal.js";
import { ITEM_CORRELATIONS, type ItemCorrelation, positiveDefiniteCorrelations } from "./item-correlations.js";
import { DEFAULT_CORRECTION, type PolychoricOptions } from "./polychoric.js";

/** Options of {@link minimumAveragePartial}. */
export interface MinimumAveragePartialOptions extends PolychoricOptions {
  /**
   * The correlation of the items: `"pearson"` for scores treated as continuous, over the persons who answered both
   * items of each pair, or `"polychoric"` for item responses.
   *
   * @defaultValue `"pearson"`
   */
  readonly correlation?: ItemCorrelation;
  /**
   * The most components to remove, from 1 to the number of items less 1.
   *
   * @defaultValue the number of items less 1
   */
  readonly maxFactors?: number;
}

/** The result of Velicer's minimum average partial test. */
export interface MinimumAveragePartial {
  /** The mean squared partial correlation of the items after removing 1, 2, ... components, by the number removed. */
  readonly map: readonly number[];
  /** The number of components with the lowest mean squared partial correlation, from 1: the first when tied. */
  readonly factors: number;
  /** Whether the correlations of the items were not positive definite and were smoothed. */
  readonly smoothed: boolean;
}

/**
 * Returns Velicer's minimum average partial test for the number of factors of the items of a response matrix.
 *
 * @remarks
 * The principal components of the correlations are removed one at a time, and after each the partial correlations of
 * the items, the correlations left over scaled to unit variances, are squared and averaged over the pairs. While the
 * components remove the common variance, the average falls; when they start removing the specific variance of single
 * items, it rises, and the number of components at its minimum is the number of factors. Like psych's `VSS`, from
 * which `map` is the `map` it returns, the test starts at one component, so it never answers none. The correlations
 * are smoothed when they are not positive definite.
 *
 * @param data - The responses of the persons.
 * @param options - The correlation, the most factors to test and the correction for continuity.
 * @returns The average squared partial correlation for each number of factors, the number of factors that minimizes it
 *   and whether the correlations were smoothed.
 * @throws `DataError` when the matrix is invalid, has fewer than three items or more than 1,000, or an item with fewer
 * than two observed categories, a pair of items has too few persons for a correlation, an item has more than 8 observed
 * categories for polychoric correlations, an option is invalid, or the components leave an item no variance, so that
 * its partial correlations are undefined, as for an item uncorrelated with every other.
 * @see {@link https://doi.org/10.1007/BF02293557 | Velicer (1976)}
 */
export function minimumAveragePartial(
  data: ResponseMatrix,
  options: MinimumAveragePartialOptions = {},
): MinimumAveragePartial {
  checkOptions(options);
  const { correlation = "pearson", correction = DEFAULT_CORRECTION } = options;
  checkChoice(correlation, "The option correlation", ITEM_CORRELATIONS);
  checkNonNegative(correction, "The option correction");
  const { items, correlations, smoothed } = positiveDefiniteCorrelations(
    data,
    correlation,
    correction,
    "The minimum average partial test",
  );
  const size = items.length;
  const { maxFactors = size - 1 } = options;
  checkPositiveInteger(maxFactors, "The option maxFactors");
  if (maxFactors > size - 1) {
    throw new DataError(
      `The option maxFactors must be at most the number of items less 1, ${size - 1}, found ${maxFactors}.`,
    );
  }
  const partials = averagePartials(correlations, size, maxFactors);
  const map = partials.filter((value) => value !== null);
  if (map.length < partials.length) {
    throw new DataError(
      `The minimum average partial test is undefined after ${partials.indexOf(null) + 1} components: they leave an item no variance, as when it is uncorrelated with every other item.`,
    );
  }
  return { map, factors: map.indexOf(map.reduce((least, value) => Math.min(least, value))) + 1, smoothed };
}

/**
 * Returns, for 1 to `count` principal components removed from the correlation matrix `r` (`size` rows, by rows, positive
 * definite), the mean over the pairs of items of the squared partial correlation; `null` once the components leave an
 * item no variance.
 */
export function averagePartials(r: Float64Array, size: number, count: number): (number | null)[] {
  const { values, vectors } = symmetricEigen(r, size);
  const partial = Float64Array.from(r);
  return Array.from({ length: count }, (_, k) => {
    const weight = values[k] ?? 0;
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        partial[i * size + j] =
          (partial[i * size + j] ?? 0) - weight * (vectors[k * size + i] ?? 0) * (vectors[k * size + j] ?? 0);
      }
    }
    return meanSquaredPartial(partial, size);
  });
}

/**
 * Returns the mean squared off-diagonal entry of a covariance matrix scaled to unit variances; `null` when a variance is
 * 0, which leaves the correlations of its variable undefined.
 */
function meanSquaredPartial(covariance: Float64Array, size: number): number | null {
  for (let i = 0; i < size; i++) if (!((covariance[i * size + i] ?? 0) > 0)) return null;
  let sum = 0;
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      if (i === j) continue;
      const entry = covariance[i * size + j] ?? 0;
      sum += (entry * entry) / ((covariance[i * size + i] ?? 1) * (covariance[j * size + j] ?? 1));
    }
  }
  return sum / (size * (size - 1));
}

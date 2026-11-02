import { DataError } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { covariance, minresLoadings, variance } from "../math/internal.js";
import { itemScores } from "./complete-cases.js";
import { reliabilityCases } from "./reliability.js";

/** McDonald's omega of a test, with the loadings of its one-factor model. */
export interface McDonaldOmega {
  /** Omega total: 1 - Σ (1 - λ²) / Σ r, over the item correlations r. */
  readonly omega: number;
  /** Each item's loading λ on the factor, in item order; their sum is positive. */
  readonly loadings: readonly number[];
  /** Whether Newton's method reached the minimum of the residuals. */
  readonly converged: boolean;
  /** Newton iterations run. */
  readonly iterations: number;
}

/**
 * Returns McDonald's omega total of a one-factor model, as psych's `omega(nfactors = 1)`.
 *
 * @remarks
 * The coefficient is 1 - Σ (1 - λ_i²) / Σ_ij r_ij, from the Pearson correlations r of the items, the diagonal included,
 * and the minres loadings λ, which minimize the squared off-diagonal residuals Σ_(i<j) (r_ij - λ_i λ_j)², by Newton's
 * method from the centroid loadings. Uses the persons who answered every item. A loading above 1, a Heywood case, shows
 * that one factor does not fit.
 *
 * @param matrix - The responses of the persons.
 * @returns Omega, the loadings of the items and whether the factor analysis converged.
 * @throws `DataError` when there are fewer than three items, fewer than two persons answered every item, an item's
 * scores have no variance, or the correlations sum to 0, as when the standardized scores of every person sum to the
 * same total.
 * @see {@link https://doi.org/10.4324/9781410601087 | McDonald (1999)}
 */
export function mcdonaldOmega(matrix: ResponseMatrix): McDonaldOmega {
  const cases = reliabilityCases(matrix, "Omega");
  const k = cases.itemCount;
  if (k < 3) throw new DataError("Omega needs at least three items, as one factor is not identified by two.");
  const scores = matrix.itemIds.map((_, item) => itemScores(cases, [item]));
  const variances = scores.map((x, item) => {
    const value = variance(x);
    if (value === 0) throw new DataError(`Omega needs items with variance; item "${matrix.itemIds[item]}" has none.`);
    return value;
  });
  // The Pearson correlations as `correlation` computes them, with each item's variance computed once; covariance(x, y)
  // and covariance(y, x) take the same operations, so one triangle gives both.
  const r = new Float64Array(k * k);
  scores.forEach((x, i) => {
    for (let j = i; j < k; j++) {
      const value = covariance(x, scores[j] ?? x) / Math.sqrt((variances[i] ?? 0) * (variances[j] ?? 0));
      r[i * k + j] = value;
      r[j * k + i] = value;
    }
  });
  // Σ_ij r_ij, omega's denominator, is the variance of the sum of the standardized scores.
  const total = r.reduce((sum, value) => sum + value, 0);
  if (!(total > 0)) {
    throw new DataError(
      "Omega is undefined when the item correlations sum to 0: the standardized total does not vary.",
    );
  }
  const { loadings, converged, iterations } = minresLoadings(r, k);
  const uniqueness = loadings.reduce((sum, loading) => sum + 1 - loading * loading, 0);
  return { omega: 1 - uniqueness / total, loadings: Array.from(loadings), converged, iterations };
}

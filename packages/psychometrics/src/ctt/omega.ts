import { DataError } from "@measurekit/core";
import type { ResponseMatrix } from "../data/response-matrix.js";
import { newton } from "../math/optimize.js";
import { covariance, variance } from "../math/statistics.js";
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

/** Newton's method converges quadratically: a step below 1e-8 leaves loadings exact to rounding. */
const LOADING_STEP_TOLERANCE = 1e-8;
/** The most Newton iterations of the minimum-residual loadings. */
const MAX_ITERATIONS = 100;

/**
 * Returns McDonald's omega total of a one-factor model, as psych's `omega(nfactors = 1)`.
 *
 * @remarks
 * The coefficient is 1 - Σ (1 - λ_i²) / Σ_ij r_ij, from the Pearson correlations r of the items, the diagonal included,
 * and the minres loadings λ, which minimize the squared off-diagonal residuals Σ_(i<j) (r_ij - λ_i λ_j)², by Newton's
 * method from the centroid loadings. Uses the persons who answered every item. A loading above 1, a Heywood case, shows
 * that one factor does not fit.
 *
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
  const start = Array.from(
    { length: k },
    (_, i) => r.subarray(i * k, (i + 1) * k).reduce((sum, value) => sum + value, 0) / Math.sqrt(total),
  );
  const result = newton(
    (loadings, gradient, hessian) => offDiagonalResiduals(r, k, loadings, gradient, hessian),
    start,
    {
      tolerance: 0,
      stepTolerance: LOADING_STEP_TOLERANCE,
      maxIterations: MAX_ITERATIONS,
    },
  );
  const sign = result.x.reduce((sum, value) => sum + value, 0) < 0 ? -1 : 1;
  const loadings = Array.from(result.x, (value) => sign * value);
  const uniqueness = loadings.reduce((sum, loading) => sum + 1 - loading * loading, 0);
  return { omega: 1 - uniqueness / total, loadings, converged: result.converged, iterations: result.iterations };
}

/**
 * Returns Σ_(i<j) (r_ij - λ_i λ_j)² and writes its gradient, -2 Σ_(j≠i) (r_ij - λ_i λ_j) λ_j, and Hessian: 2 Σ_(j≠i)
 * λ_j² on the diagonal, 4 λ_i λ_j - 2 r_ij off it.
 */
function offDiagonalResiduals(
  r: Float64Array,
  k: number,
  loadings: Float64Array,
  gradient: Float64Array,
  hessian: Float64Array,
): number {
  let value = 0;
  let squares = 0;
  for (const loading of loadings) squares += loading * loading;
  for (let i = 0; i < k; i++) {
    const li = loadings[i] ?? 0;
    let slope = 0;
    for (let j = 0; j < k; j++) {
      const lj = loadings[j] ?? 0;
      if (j === i) {
        hessian[i * k + i] = 2 * (squares - li * li);
        continue;
      }
      const residual = (r[i * k + j] ?? 0) - li * lj;
      if (j > i) value += residual * residual;
      slope -= 2 * residual * lj;
      hessian[i * k + j] = 4 * li * lj - 2 * (r[i * k + j] ?? 0);
    }
    gradient[i] = slope;
  }
  return value;
}

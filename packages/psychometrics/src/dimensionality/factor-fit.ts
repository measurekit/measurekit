import { chiSquareUpperTail, cholesky, invertPositiveDefinite } from "@measurekit/core";
import { rmseaInterval } from "../math/internal.js";

/** The fit of exploratory factors to the correlations of the items, as psych's `fa` reports it. */
export interface FactorFit {
  /** The likelihood ratio statistic with Bartlett's correction, psych's `STATISTIC`. */
  readonly chiSquare: number;
  /** The correlations less the loadings, plus the rotations that leave the model as it is. */
  readonly degreesOfFreedom: number;
  /** P(χ² > chiSquare); `null` with 0 degrees of freedom, and so are the RMSEA, its interval and the TLI. */
  readonly significance: number | null;
  /** The root mean square error of approximation. */
  readonly rmsea: number | null;
  /** The 90% interval of the RMSEA. */
  readonly rmseaInterval: readonly [lower: number, upper: number] | null;
  /** The Tucker-Lewis index; it can exceed 1. */
  readonly tli: number | null;
  /** chiSquare less the degrees of freedom times the log of the persons. */
  readonly bic: number;
  /** The root mean square of the residual correlations over the pairs of items, psych's `rms`. */
  readonly srmsr: number;
}

/** The level of the RMSEA's interval: psych's `fa` computes it at `alpha = 0.1`. */
const RMSEA_LEVEL = 0.9;

/**
 * Returns the fit of a factor solution to a correlation matrix, as psych's `fa.stats`, or `null` when the correlations
 * of the model are not positive definite, as when a communality beyond 1 leaves an item a negative uniqueness.
 *
 * @remarks
 * The model's correlations are Λ Φ Λᵀ off the diagonal and 1 on it. The statistic is (N - 1 - (2p + 5) / 6 - 2m / 3) F,
 * with F = tr(M⁻¹ R) - log det(M⁻¹ R) - p, the likelihood ratio's discrepancy, at least 0, for p items, m factors and N
 * persons; the degrees of freedom are p (p - 1) / 2 - p m + m (m - 1) / 2. The RMSEA is √max(χ² / (df N) - 1 / (N - 1),
 * 0) and its interval √(λ / ((N - 1) df)) at the noncentralities that put χ² at the 5th and 95th percentiles. The TLI
 * compares F / df with the independence model's -log det(R) / (p (p - 1) / 2), each against 1 / (N - 1 - (2p + 5) / 6 -
 * 2m / 3). The residuals of a pair weigh as many as the persons who answered it, and their mean square divides by the
 * harmonic mean of those counts over every pair and item, as psych's `np.obs`.
 *
 * @param r - The correlations of the items, positive definite, by rows.
 * @param size - The number of items.
 * @param model - Λ Φ Λᵀ, by rows.
 * @param factors - The number of factors.
 * @param persons - The persons of the data, psych's `n.obs`.
 * @param counts - The persons who answered each pair of items, by rows; the diagonal those who answered each item.
 */
export function factorFit(
  r: Float64Array,
  size: number,
  model: Float64Array,
  factors: number,
  persons: number,
  counts: Float64Array,
): FactorFit | null {
  const implied = Float64Array.from(model, (value, entry) => (entry % (size + 1) === 0 ? 1 : value));
  const factor = cholesky(implied, size);
  const inverse = invertPositiveDefinite(implied, size);
  const correlations = cholesky(r, size);
  if (factor === null || inverse === null || correlations === null) return null;
  let trace = 0;
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) trace += (inverse[i * size + j] ?? 0) * (r[j * size + i] ?? 0);
  }
  const discrepancy = trace - (logDeterminant(correlations, size) - logDeterminant(factor, size)) - size;
  const multiplier = persons - 1 - (2 * size + 5) / 6 - (2 * factors) / 3;
  const chiSquare = Math.max(discrepancy * multiplier, 0);
  const degreesOfFreedom = (size * (size - 1)) / 2 - size * factors + (factors * (factors - 1)) / 2;
  const srmsr = residualRoot(r, implied, size, counts);
  const bic = chiSquare - degreesOfFreedom * Math.log(persons);
  if (degreesOfFreedom === 0) {
    return { chiSquare, degreesOfFreedom, significance: null, rmsea: null, rmseaInterval: null, tli: null, bic, srmsr };
  }
  const independence = (-2 * logDeterminant(correlations, size)) / (size * (size - 1));
  const tli = (independence - discrepancy / degreesOfFreedom) / (independence - 1 / multiplier);
  return {
    chiSquare,
    degreesOfFreedom,
    significance: chiSquareUpperTail(chiSquare, degreesOfFreedom),
    rmsea: Math.sqrt(Math.max(chiSquare / (degreesOfFreedom * persons) - 1 / (persons - 1), 0)),
    rmseaInterval: rmseaInterval(chiSquare, degreesOfFreedom, persons - 1, RMSEA_LEVEL),
    tli: Number.isFinite(tli) ? tli : null,
    bic,
    srmsr,
  };
}

/** Returns log det A from the Cholesky factor L of A, by rows: twice the sum of the logs of L's diagonal. */
function logDeterminant(factor: Float64Array, size: number): number {
  let sum = 0;
  for (let i = 0; i < size; i++) sum += Math.log(factor[i * size + i] ?? 1);
  return 2 * sum;
}

/** Returns √(Σ_(i≠j) n_ij (r_ij - m_ij)² / (n̄ p (p - 1))), n̄ the harmonic mean of the counts. */
function residualRoot(r: Float64Array, implied: Float64Array, size: number, counts: Float64Array): number {
  let weighted = 0;
  let reciprocals = 0;
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      const count = counts[i * size + j] ?? 0;
      reciprocals += 1 / count;
      if (i !== j) weighted += count * ((r[i * size + j] ?? 0) - (implied[i * size + j] ?? 0)) ** 2;
    }
  }
  const harmonic = (size * size) / reciprocals;
  return Math.sqrt(weighted / (harmonic * size * (size - 1)));
}

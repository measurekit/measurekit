import { checkOptions, checkPositiveInteger, DataError, type FieldNames } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { type ExploratoryFactorOptions, exploratoryFactors } from "./exploratory-factors.js";
import { MIN_GROUP_FACTORS, type SchmidLeiman, transformation } from "./schmid-leiman.js";

/** Options of {@link omegaHierarchical}: those of `exploratoryFactorAnalysis`, its rotation always oblimin. */
export interface OmegaHierarchicalOptions extends Omit<ExploratoryFactorOptions, "factors" | "rotation"> {
  /**
   * The number of group factors, at least 3.
   *
   * @defaultValue `3`
   */
  readonly factors?: number;
}

const OMEGA_HIERARCHICAL_OPTIONS: FieldNames<OmegaHierarchicalOptions> = {
  factors: true,
  tolerance: true,
  maxIterations: true,
  correlation: true,
  extraction: true,
  correction: true,
};

/** McDonald's omega hierarchical and omega total of a test, with the loadings they come from. */
export interface OmegaHierarchical {
  /** Omega hierarchical: (Σ λ_g)² over the variance of the total of the standardized items, the sum of their correlations. */
  readonly hierarchical: number;
  /** Omega total: 1 - Σ (1 - h²) over the same sum, with h² the communalities of the group factors. */
  readonly total: number;
  /** The general and group loadings of the Schmid-Leiman transformation. */
  readonly loadings: SchmidLeiman;
  /** Whether the factor analysis, its rotation and the fit of the general factor converged. */
  readonly converged: boolean;
}

const DEFAULT_FACTORS = 3;

/**
 * Returns McDonald's omega hierarchical, the share of the variance of the total score that a general factor accounts
 * for, and omega total, from the Schmid-Leiman transformation of an oblique factor analysis, as psych's `omega`.
 *
 * @remarks
 * The items are factored by `exploratoryFactorAnalysis` with oblimin, and `schmidLeiman` gives each item's general
 * loading λ_g. With R the correlations of the items, omega hierarchical is (Σ λ_g)² / Σ R and omega total
 * 1 - Σ (1 - h²) / Σ R, the omega of `mcdonaldOmega` with the communalities of the group factors. Items count as scored:
 * a reversed item lowers both, so reverse it first. At most omega total, omega hierarchical reaches it when the group
 * factors add nothing.
 *
 * @param matrix - The responses of the persons.
 * @param options - The number of group factors and the options of the factor analysis.
 * @returns Omega hierarchical, omega total, the loadings and whether the fits converged.
 * @throws `DataError` as `exploratoryFactorAnalysis` does, and when `factors` is below 3.
 * @see {@link https://doi.org/10.4324/9781410601087 | McDonald (1999)}
 */
export function omegaHierarchical(matrix: ResponseMatrix, options: OmegaHierarchicalOptions = {}): OmegaHierarchical {
  checkOptions(options, OMEGA_HIERARCHICAL_OPTIONS);
  const { factors = DEFAULT_FACTORS } = options;
  checkPositiveInteger(factors, "The option factors");
  if (factors < MIN_GROUP_FACTORS) {
    throw new DataError(
      `Omega hierarchical needs at least ${MIN_GROUP_FACTORS} group factors, as two do not identify the general factor, found ${factors}.`,
    );
  }
  const solved = exploratoryFactors(matrix, { ...options, factors, rotation: "oblimin" }, "Omega hierarchical");
  const loadings = transformation(solved.factors);
  // Σ R is the variance of the total of the standardized items, positive for a positive definite R.
  const variance = solved.correlations.reduce((sum, value) => sum + value, 0);
  const general = loadings.general.reduce((sum, loading) => sum + loading, 0);
  const uniqueness = loadings.communalities.reduce((sum, communality) => sum + 1 - communality, 0);
  return {
    hierarchical: (general * general) / variance,
    total: 1 - uniqueness / variance,
    loadings,
    converged: solved.factors.converged && loadings.converged,
  };
}

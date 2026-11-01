import { cholesky, dot } from "@measurekit/core";
import { addOuter, assemble, createSystem, type FlatDirection, type Problem, type System } from "./arrow-system.js";

/** The standard errors of a logistic regression's coefficients and group intercepts. */
export interface StandardErrors {
  /** One per coefficient. */
  readonly coefficients: Float64Array;
  /** One per group intercept; empty when the groups are not asked for. */
  readonly groups: Float64Array;
}

/** Options of {@link standardErrors}. */
export interface StandardErrorOptions {
  /** The penalty on the group intercepts at the estimates: 0 for fixed ones, 1 / σ² for random ones. */
  readonly groupPenalty: number;
  /** The flat directions of fixed group intercepts; see `GroupIntercepts.flats`. */
  readonly flats: readonly FlatDirection[];
  /** Whether to give the standard errors of the group intercepts. */
  readonly groups: boolean;
}

/** The Newton system at the estimates, with the inverse of its Schur complement and the variances being built. */
interface Moments {
  readonly problem: Problem;
  readonly system: System;
  /** The inverse of the Schur complement with the flat directions added. */
  readonly inverse: Float64Array;
  /** Updated in place: the variance of each coefficient, and of each group's intercept when asked for. */
  readonly variances: Float64Array;
  readonly groupVariances: Float64Array;
}

/**
 * Returns the standard errors at the estimates, the square roots of the diagonal of the inverse Hessian of the
 * log-likelihood, or `null` when the information is singular: a coefficient without a row that uses it, or features that
 * are linear combinations of each other.
 *
 * @remarks
 * The Hessian has the arrow form of the Newton system, so its inverse for the coefficients is the inverse of the Schur
 * complement of the groups, and a group's variance adds its own 1 / a_g and its coupling with the coefficients. Along a
 * flat direction the estimates are fixed by a mean of 0 in the groups of each set, as the fit reports them; the
 * covariance of those estimates is the pseudo-inverse's, moved along each direction to that constraint. The
 * variance of a row is the binomial p (1 - p) whatever the outcome, as `glm` has it.
 */
export function standardErrors(
  problem: Problem,
  beta: Float64Array,
  intercepts: Float64Array,
  options: StandardErrorOptions,
): StandardErrors | null {
  const { groupPenalty, flats, groups } = options;
  const p = beta.length;
  if (hasEmptyColumn(problem)) return null;
  const system = createSystem(p, intercepts.length);
  assemble(problem, beta, intercepts, 0, groupPenalty, system);
  for (const { coefficients } of flats) addOuter(system.schur, coefficients, 1 / dot(coefficients, coefficients));
  const inverse = inverseOf(system.schur, p);
  if (inverse === null) return null;
  const variances = Float64Array.from({ length: p }, (_, j) => inverse[j * p + j] ?? 0);
  const groupVariances = new Float64Array(groups ? intercepts.length : 0);
  const moments: Moments = { problem, system, inverse, variances, groupVariances };
  for (const flat of flats) constrain(moments, flat);
  if (groups) freeGroups(moments, flats);
  return { coefficients: variances.map(root), groups: groupVariances.map(root) };
}

/** Returns the square root of a variance, which rounding may leave a hair below 0. */
function root(variance: number): number {
  return Math.sqrt(Math.max(0, variance));
}

/** Returns whether a column of the design has no entry: nothing estimates its coefficient. */
function hasEmptyColumn({ design }: Problem): boolean {
  const used = new Uint8Array(design.columnCount);
  for (const column of design.columns) used[column] = 1;
  return used.includes(0);
}

/** Sets the variances of the groups that no flat direction covers: their own 1 / a_g and their coupling. */
function freeGroups(moments: Moments, flats: readonly FlatDirection[]): void {
  const { problem, system, inverse, variances, groupVariances } = moments;
  const covered = new Uint8Array(groupVariances.length);
  for (const { groups } of flats) for (const g of groups) covered[g] = 1;
  for (let g = 0; g < groupVariances.length; g++) {
    if (covered[g] === 1) continue;
    const a = system.diagonal[g] ?? 1;
    groupVariances[g] = 1 / a + coupling(problem, inverse, variances.length, g, 1 / a);
  }
}

/**
 * Replaces the variances of the coefficients and groups of a flat direction by those of the estimates with a mean of 0
 * in its groups.
 *
 * @remarks
 * The estimates are the pseudo-inverse's β₀ with f·β₀ = 0, moved along the direction by the mean ū of the groups'
 * intercepts: β = β₀ - ū f. A group's intercept given β₀ is -b_g·β₀ + ε_g, with b_g = B_g / a_g and ε_g independent of
 * β₀ with variance 1 / a_g, so ū = -b̄·β₀ + ε̄ and the variances follow from the covariance S⁺ of β₀.
 */
function constrain(moments: Moments, flat: FlatDirection): void {
  const { problem, system, inverse, variances, groupVariances } = moments;
  const { coefficients: f, groups } = flat;
  const norm = dot(f, f);
  const { mean, noise } = groupMean(moments, groups);
  const projected = pseudoProduct(inverse, f, norm, mean);
  const meanVariance = dot(mean, projected);
  const p = variances.length;
  for (let j = 0; j < p; j++) {
    const fj = f[j] ?? 0;
    if (fj === 0) continue;
    variances[j] =
      (inverse[j * p + j] ?? 0) - (fj * fj) / norm + 2 * fj * (projected[j] ?? 0) + fj * fj * (meanVariance + noise);
  }
  if (groupVariances.length === 0) return;
  for (const g of groups) {
    const a = system.diagonal[g] ?? 1;
    const centered = coupling(problem, inverse, p, g, 1 / a, { f, norm });
    const alongMean = sparseDot(problem, g, projected) / a;
    groupVariances[g] = centered - 2 * alongMean + meanVariance + 1 / a - 2 / (groups.length * a) + noise;
  }
}

/**
 * Returns the mean b̄ of b_g = B_g / a_g over the groups of a flat direction, and the variance of the mean of their
 * independent errors: Σ 1 / a_g over the square of their number.
 */
function groupMean(moments: Moments, groups: Int32Array): { mean: Float64Array; noise: number } {
  const { problem, system: newton, variances } = moments;
  const mean = new Float64Array(variances.length);
  let noise = 0;
  for (const g of groups) {
    const a = newton.diagonal[g] ?? 1;
    noise += 1 / a;
    for (let i = problem.touchedStarts[g] ?? 0; i < (problem.touchedStarts[g + 1] ?? 0); i++) {
      const column = problem.touchedColumns[i] ?? 0;
      mean[column] = (mean[column] ?? 0) + (problem.touchedSums[i] ?? 0) / (a * groups.length);
    }
  }
  return { mean, noise: noise / (groups.length * groups.length) };
}

/** Returns S⁺ x for the mean x: the inverse of the regularized complement, less the flat direction's own part. */
function pseudoProduct(inverse: Float64Array, f: Float64Array, norm: number, x: Float64Array): Float64Array {
  const p = x.length;
  const nonzero: number[] = [];
  x.forEach((value, i) => {
    if (value !== 0) nonzero.push(i);
  });
  const along = dot(f, x) / norm;
  const product = new Float64Array(p);
  for (let i = 0; i < p; i++) {
    let sum = 0;
    for (const j of nonzero) sum += (inverse[i * p + j] ?? 0) * (x[j] ?? 0);
    product[i] = sum - (f[i] ?? 0) * along;
  }
  return product;
}

/** Returns Σ B_g[i] y[i] over the columns that group g touches. */
function sparseDot(problem: Problem, g: number, y: Float64Array): number {
  let sum = 0;
  for (let i = problem.touchedStarts[g] ?? 0; i < (problem.touchedStarts[g + 1] ?? 0); i++) {
    sum += (problem.touchedSums[i] ?? 0) * (y[problem.touchedColumns[i] ?? 0] ?? 0);
  }
  return sum;
}

/**
 * Returns b_gᵀ S⁺ b_g for b_g = scale B_g, with S⁺ the inverse less the flat direction's part when one is given: the
 * variance that a group's intercept gets from the coefficients' uncertainty.
 */
function coupling(
  problem: Problem,
  inverse: Float64Array,
  p: number,
  g: number,
  scale: number,
  flat?: { readonly f: Float64Array; readonly norm: number },
): number {
  const start = problem.touchedStarts[g] ?? 0;
  const end = problem.touchedStarts[g + 1] ?? 0;
  let sum = 0;
  let along = 0;
  for (let i = start; i < end; i++) {
    const ci = problem.touchedColumns[i] ?? 0;
    const bi = (problem.touchedSums[i] ?? 0) * scale;
    along += bi * (flat?.f[ci] ?? 0);
    for (let j = start; j < end; j++) {
      sum += bi * (problem.touchedSums[j] ?? 0) * scale * (inverse[ci * p + (problem.touchedColumns[j] ?? 0)] ?? 0);
    }
  }
  return flat === undefined ? sum : sum - (along * along) / flat.norm;
}

/**
 * Returns the inverse of a symmetric positive definite matrix stored by rows, from its Cholesky factor, or `null` when it
 * is not positive definite to working precision.
 */
function inverseOf(matrix: Float64Array, size: number): Float64Array | null {
  const lower = cholesky(matrix, size);
  return lower === null ? null : gram(invertLower(lower, size), size);
}

/** Returns the inverse of a lower triangular matrix stored by rows, solved by column: L x = e_j for x below j. */
function invertLower(lower: Float64Array, size: number): Float64Array {
  const inverse = new Float64Array(size * size);
  const column = new Float64Array(size);
  for (let j = 0; j < size; j++) {
    for (let i = j; i < size; i++) {
      let sum = i === j ? 1 : 0;
      for (let k = j; k < i; k++) sum -= (lower[i * size + k] ?? 0) * (column[k] ?? 0);
      column[i] = sum / (lower[i * size + i] ?? 1);
      inverse[i * size + j] = column[i] ?? 0;
    }
  }
  return inverse;
}

/** Returns Mᵀ M of a lower triangular matrix stored by rows: each row k adds its outer product. */
function gram(lower: Float64Array, size: number): Float64Array {
  const product = new Float64Array(size * size);
  for (let k = 0; k < size; k++) addRowOuter(product, lower, size, k);
  for (let i = 0; i < size; i++) for (let j = 0; j < i; j++) product[j * size + i] = product[i * size + j] ?? 0;
  return product;
}

/** Adds the outer product of row k of a lower triangular matrix with itself to the lower triangle of `product`. */
function addRowOuter(product: Float64Array, lower: Float64Array, size: number, k: number): void {
  for (let i = 0; i <= k; i++) {
    const factor = lower[k * size + i] ?? 0;
    if (factor === 0) continue;
    for (let j = 0; j <= i; j++) {
      product[i * size + j] = (product[i * size + j] ?? 0) + factor * (lower[k * size + j] ?? 0);
    }
  }
}

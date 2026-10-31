import { symmetricEigen } from "../math/eigen.js";
import { minresLoadings } from "../math/minres.js";

/** The eigenvalues of the correlation matrix of items and of its reduced matrix. */
export interface CorrelationEigenvalues {
  /** The eigenvalues of the correlation matrix, the principal components', largest first. */
  readonly components: Float64Array;
  /** The eigenvalues of the matrix with the communalities of the one-factor minres solution, λ², on its diagonal. */
  readonly factors: Float64Array;
}

/**
 * The largest loading of the one-factor solution: psych's `fa` keeps each uniqueness at least 0.005, which keeps the
 * minimum residual search of a matrix that one factor does not fit, such as one of random data, from running off to a
 * Heywood case.
 */
const MAX_LOADING = Math.sqrt(1 - 0.005);

/**
 * Returns the eigenvalues of a correlation matrix and of the same with its diagonal replaced by the communalities of
 * the one-factor minimum residual solution, as psych's `fa.parallel` takes them for components and factors.
 *
 * @remarks
 * The matrix is symmetric, by rows, with 1 on the diagonal and no NaN, and is used as it is, also when it is not
 * positive definite, so that negative eigenvalues are reported.
 */
export function correlationEigenvalues(r: Float64Array, size: number): CorrelationEigenvalues {
  const components = symmetricEigen(r, size).values;
  const { loadings } = minresLoadings(r, size, MAX_LOADING);
  const reduced = Float64Array.from(r);
  loadings.forEach((loading, i) => {
    reduced[i * size + i] = loading * loading;
  });
  return { components, factors: symmetricEigen(reduced, size).values };
}

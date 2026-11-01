import { checkOptions, checkPositive, checkPositiveInteger, DataError } from "@measurekit/core";
import { symmetricEigen } from "../math/eigen.js";

/** Options of {@link smoothCorrelations}. */
export interface SmoothingOptions {
  /**
   * The eigenvalues below this are raised to 100 times it.
   *
   * @defaultValue `1e-12`
   */
  readonly tolerance?: number;
}

/** psych's `eig.tol`: the eigenvalues of a smoothed matrix are at least 100 times it. */
const DEFAULT_TOLERANCE = 1e-12;
/** How much the tolerance is multiplied to raise an eigenvalue below it: psych's. */
const RAISE = 100;
/** The asymmetry or distance of a diagonal entry from 1 that a correlation matrix may have. */
const MATRIX_TOLERANCE = 1e-8;

/**
 * Returns a correlation matrix that is positive definite: the matrix itself when it is, and otherwise its eigenvalues
 * below `tolerance` raised, rescaled to sum to the number of variables, and the matrix rebuilt and rescaled to a
 * correlation matrix.
 *
 * @remarks
 * As psych's `cor.smooth`, which smooths a matrix whose least eigenvalue is below the machine epsilon. Polychoric
 * correlations come from pairs of items estimated apart and can fail to be positive definite, which factor analysis
 * and the minimum average partial test need.
 *
 * @param correlations - The symmetric matrix by rows, 1 on its diagonal, with `size` rows.
 * @throws `DataError` when `size` is not a positive integer, `correlations` is not a `Float64Array` of `size` squared
 * finite numbers, symmetric with 1 on the diagonal, or `tolerance` is not positive.
 * @see {@link https://doi.org/10.1177/014662168801200305 | Bock, Gibbons and Muraki (1988)}
 */
export function smoothCorrelations(
  correlations: Float64Array,
  size: number,
  options: SmoothingOptions = {},
): Float64Array {
  checkOptions(options);
  const { tolerance = DEFAULT_TOLERANCE } = options;
  checkPositive(tolerance, "The option tolerance");
  checkMatrix(correlations, size);
  return smooth(correlations, size, tolerance).correlations;
}

/**
 * Smooths a correlation matrix as {@link smoothCorrelations} does and reports whether it had to, for the functions that
 * need a matrix that is positive definite. The matrix is valid.
 */
export function smooth(
  correlations: Float64Array,
  size: number,
  tolerance = DEFAULT_TOLERANCE,
): { correlations: Float64Array; smoothed: boolean } {
  const { values, vectors } = symmetricEigen(correlations, size);
  if (!((values[size - 1] ?? 0) < Number.EPSILON)) {
    return { correlations: Float64Array.from(correlations), smoothed: false };
  }
  const raised = values.map((value) => (value < tolerance ? RAISE * tolerance : value));
  const scale = size / raised.reduce((sum, value) => sum + value, 0);
  return { correlations: toCorrelations(rebuild(raised, vectors, size, scale), size), smoothed: true };
}

/** Returns Σ_k scale · values[k] · v_k v_kᵀ over the eigenvectors, stored by rows. */
function rebuild(values: Float64Array, vectors: Float64Array, size: number, scale: number): Float64Array {
  const matrix = new Float64Array(size * size);
  for (let k = 0; k < size; k++) {
    const weight = scale * (values[k] ?? 0);
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        matrix[i * size + j] =
          (matrix[i * size + j] ?? 0) + weight * (vectors[k * size + i] ?? 0) * (vectors[k * size + j] ?? 0);
      }
    }
  }
  return matrix;
}

/** Returns the correlation matrix of a covariance matrix, with 1 on the diagonal exactly. */
function toCorrelations(covariance: Float64Array, size: number): Float64Array {
  const result = new Float64Array(size * size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      const scale = Math.sqrt((covariance[i * size + i] ?? 1) * (covariance[j * size + j] ?? 1));
      result[i * size + j] = i === j ? 1 : (covariance[i * size + j] ?? 0) / scale;
    }
  }
  return result;
}

/**
 * Checks the matrix of correlations.
 *
 * @throws `DataError` when it is not what {@link smoothCorrelations} takes.
 */
function checkMatrix(correlations: unknown, size: unknown): asserts correlations is Float64Array {
  checkPositiveInteger(size, "The size of the correlation matrix");
  if (!(correlations instanceof Float64Array) || correlations.length !== size * size) {
    throw new DataError(`The correlations must be a Float64Array of ${size * size} numbers, ${size} rows of ${size}.`);
  }
  for (let i = 0; i < size; i++) {
    for (let j = i; j < size; j++) {
      const value = correlations[i * size + j] ?? Number.NaN;
      const mirror = correlations[j * size + i] ?? Number.NaN;
      const valid =
        Number.isFinite(value) &&
        Math.abs(value - mirror) <= MATRIX_TOLERANCE &&
        (i !== j || Math.abs(value - 1) <= MATRIX_TOLERANCE);
      if (!valid) {
        throw new DataError(
          `The correlations must be finite, symmetric and 1 on the diagonal; row ${i + 1}, column ${j + 1} is ${value} and its mirror is ${mirror}.`,
        );
      }
    }
  }
}

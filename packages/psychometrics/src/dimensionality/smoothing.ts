import { checkOptions, checkPositive, type FieldNames } from "@measurekit/core";
import { checkMatrixItems } from "../data/internal.js";
import { checkCorrelationMatrix, symmetricEigen } from "../math/internal.js";

/** Options of {@link smoothCorrelations}. */
export interface SmoothingOptions {
  /**
   * The eigenvalues below this are raised to 100 times it.
   *
   * @defaultValue `1e-12`
   */
  readonly tolerance?: number;
}

const SMOOTHING_OPTIONS: FieldNames<SmoothingOptions> = { tolerance: true };

/** psych's `eig.tol`: the eigenvalues of a smoothed matrix are at least 100 times it. */
const DEFAULT_TOLERANCE = 1e-12;
/** How much the tolerance is multiplied to raise an eigenvalue below it: psych's. */
const RAISE = 100;

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
 * @param correlations - The symmetric matrix by rows, 1 on its diagonal, as `polychoricCorrelations` returns it.
 * @param options - The smallest eigenvalue that is kept.
 * @returns The smoothed matrix, by rows.
 * @throws `DataError` when `correlations` is not a square array of finite numbers from -1 to 1, symmetric with 1 on
 * the diagonal, or of more than 1,000 rows, or `tolerance` is not positive.
 * @see {@link https://doi.org/10.1177/014662168801200305 | Bock, Gibbons and Muraki (1988)}
 */
export function smoothCorrelations(
  correlations: readonly (readonly number[])[],
  options: SmoothingOptions = {},
): readonly (readonly number[])[] {
  checkOptions(options, SMOOTHING_OPTIONS);
  const { tolerance = DEFAULT_TOLERANCE } = options;
  checkPositive(tolerance, "The option tolerance");
  checkCorrelationMatrix(correlations, "the correlations");
  const size = correlations.length;
  // The decomposition takes the cube of the rows: about 3 s at the limit of the analyses of pairs of items.
  checkMatrixItems(size, "Smoothing");
  const smoothed = smooth(Float64Array.from(correlations.flat()), size, tolerance).correlations;
  return Array.from({ length: size }, (_, i) => Array.from(smoothed.subarray(i * size, (i + 1) * size)));
}

/**
 * Smooths a flat correlation matrix as {@link smoothCorrelations} does and reports whether it had to, for the functions that
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

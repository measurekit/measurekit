import { symmetricEigen } from "./eigen.js";
import { invertPositiveDefinite } from "./linear-algebra.js";

/** The relative size below which an eigenvalue counts as 0: the square root of the machine epsilon, as R's `MASS::ginv`. */
const RANK_TOLERANCE = Math.sqrt(Number.EPSILON);

/**
 * Returns the Moore-Penrose inverse of the symmetric positive semidefinite `size` by `size` matrix `a`, stored by rows:
 * the eigenvalues that are at most `sqrt(ε)` times the largest count as 0, as `MASS::ginv` does.
 *
 * @remarks
 * A matrix that no eigenvalue of is that small has the inverse as its pseudo-inverse, which a Cholesky factor gives in
 * about a tenth of the work of the eigendecomposition. It is certain that none is when the trace of the matrix, which is
 * at least its largest eigenvalue, times the trace of its inverse, which is at least the reciprocal of its smallest,
 * is below the reciprocal of the tolerance. The matrix must be finite and symmetric, which the function does not check.
 * The eigendecomposition is by the cyclic Jacobi method, O(size³) per sweep.
 */
export function pseudoInverse(a: Float64Array, size: number): Float64Array {
  const inverse = invertPositiveDefinite(a, size);
  if (inverse !== null && trace(a, size) * trace(inverse, size) < 1 / RANK_TOLERANCE) return inverse;
  return truncatedInverse(a, size);
}

/** Returns the inverse over the eigenvalues above the threshold of the rank, and with a finite reciprocal. */
function truncatedInverse(a: Float64Array, size: number): Float64Array {
  const { values, vectors } = symmetricEigen(a, size);
  const threshold = Math.max(RANK_TOLERANCE * (values[0] ?? 0), 0);
  const inverse = new Float64Array(size * size);
  values.forEach((value, k) => {
    if (!(value > threshold) || !Number.isFinite(1 / value)) return;
    const vector = vectors.subarray(k * size, (k + 1) * size);
    for (let i = 0; i < size; i++) {
      const scaled = (vector[i] ?? 0) / value;
      for (let j = 0; j < size; j++) inverse[i * size + j] = (inverse[i * size + j] ?? 0) + scaled * (vector[j] ?? 0);
    }
  });
  return inverse;
}

function trace(a: Float64Array, size: number): number {
  let sum = 0;
  for (let i = 0; i < size; i++) sum += a[i * size + i] ?? 0;
  return sum;
}

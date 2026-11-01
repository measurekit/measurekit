import { dot } from "@measurekit/core";

/** The eigenvalues and eigenvectors of a symmetric matrix. */
export interface SymmetricEigen {
  /** The eigenvalues, largest first. */
  readonly values: Float64Array;
  /**
   * The unit eigenvectors, stored by rows: row k, `vectors[k * size + i]`, belongs to `values[k]`. The sign of an
   * eigenvector is arbitrary.
   */
  readonly vectors: Float64Array;
  /** Sweeps run. */
  readonly iterations: number;
  /** Whether the off-diagonal part fell to `tolerance` times the matrix's norm within `maxIterations` sweeps. */
  readonly converged: boolean;
}

/** Stopping rules of {@link symmetricEigen}. */
export interface EigenOptions {
  /**
   * The Frobenius norm of the off-diagonal entries, relative to that of the matrix, at which the matrix counts as
   * diagonal.
   *
   * @defaultValue `1e-14`
   */
  readonly tolerance?: number;
  /**
   * The most sweeps, each rotating every pair of rows and columns once.
   *
   * @defaultValue `50`
   */
  readonly maxIterations?: number;
}

const DEFAULT_TOLERANCE = 1e-14;
const DEFAULT_MAX_ITERATIONS = 50;

/**
 * Returns the eigendecomposition of the symmetric `size` by `size` matrix `a`, stored by rows, by the cyclic Jacobi
 * method.
 *
 * @remarks
 * Each rotation zeroes one off-diagonal pair, and a sweep rotates every pair; the sweeps converge quadratically, in
 * about 6 to 10 for matrices of tens of rows, and give eigenvalues and eigenvectors accurate to rounding, small
 * eigenvalues included. The matrix must be finite and symmetric, which the function does not check; only the upper
 * triangle is read to start the rotations. It suits the small matrices of correlations and item loadings, in
 * O(size³) per sweep.
 *
 * @see {@link https://doi.org/10.1515/crll.1846.30.51 | Jacobi (1846)}
 * @see {@link https://doi.org/10.1007/BF02165223 | Rutishauser (1966)}
 */
export function symmetricEigen(a: Float64Array, size: number, options: EigenOptions = {}): SymmetricEigen {
  const { tolerance = DEFAULT_TOLERANCE, maxIterations = DEFAULT_MAX_ITERATIONS } = options;
  const work = Float64Array.from(a);
  const rotated = new Float64Array(size * size);
  for (let i = 0; i < size; i++) rotated[i * size + i] = 1;
  const threshold = tolerance * Math.sqrt(dot(work, work));
  let iterations = 0;
  while (offDiagonalNorm(work, size) > threshold && iterations < maxIterations) {
    for (let p = 0; p < size - 1; p++) {
      for (let q = p + 1; q < size; q++) rotate(work, rotated, size, p, q);
    }
    iterations++;
  }
  const order = Array.from({ length: size }, (_, k) => k).sort(
    (x, y) => (work[y * size + y] ?? 0) - (work[x * size + x] ?? 0) || x - y,
  );
  const values = Float64Array.from(order, (k) => work[k * size + k] ?? 0);
  const vectors = new Float64Array(size * size);
  order.forEach((k, row) => {
    vectors.set(rotated.subarray(k * size, (k + 1) * size), row * size);
  });
  return { values, vectors, iterations, converged: offDiagonalNorm(work, size) <= threshold };
}

/** Returns the Frobenius norm of the entries of `a` off the diagonal. */
function offDiagonalNorm(a: Float64Array, size: number): number {
  let sum = 0;
  for (let p = 0; p < size - 1; p++) {
    for (let q = p + 1; q < size; q++) sum += (a[p * size + q] ?? 0) ** 2;
  }
  return Math.sqrt(2 * sum);
}

/**
 * Applies the Jacobi rotation that zeroes entry (p, q) of the symmetric `a` to `a` and to the eigenvectors `v`, whose
 * rows p and q turn with it. The tangent t of the angle is the smaller root of t² + 2 t θ - 1 = 0, θ = (a_qq - a_pp) /
 * (2 a_pq), so that the rotation is at most 45 degrees.
 */
function rotate(a: Float64Array, v: Float64Array, size: number, p: number, q: number): void {
  const apq = a[p * size + q] ?? 0;
  if (apq === 0) return;
  const app = a[p * size + p] ?? 0;
  const aqq = a[q * size + q] ?? 0;
  const theta = (aqq - app) / (2 * apq);
  const t = (theta < 0 ? -1 : 1) / (Math.abs(theta) + Math.hypot(theta, 1));
  const c = 1 / Math.sqrt(t * t + 1);
  const s = t * c;
  a[p * size + p] = app - t * apq;
  a[q * size + q] = aqq + t * apq;
  a[p * size + q] = 0;
  a[q * size + p] = 0;
  for (let r = 0; r < size; r++) {
    if (r === p || r === q) continue;
    const arp = a[r * size + p] ?? 0;
    const arq = a[r * size + q] ?? 0;
    const rp = c * arp - s * arq;
    const rq = s * arp + c * arq;
    a[r * size + p] = rp;
    a[p * size + r] = rp;
    a[r * size + q] = rq;
    a[q * size + r] = rq;
  }
  for (let k = 0; k < size; k++) {
    const vp = v[p * size + k] ?? 0;
    const vq = v[q * size + k] ?? 0;
    v[p * size + k] = c * vp - s * vq;
    v[q * size + k] = s * vp + c * vq;
  }
}

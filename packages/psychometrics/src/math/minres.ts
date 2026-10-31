import { ARMIJO, backtrack, dot } from "@measurekit/core";
import { symmetricEigen } from "./eigen.js";
import { invertPositiveDefinite } from "./linear-algebra.js";

/** The one-factor minimum residual solution of a correlation matrix. */
export interface MinresLoadings {
  /** The loadings λ, one per variable; their sum is not negative. */
  readonly loadings: Float64Array;
  /** Whether the search reached the minimum of the residuals within the bounds. */
  readonly converged: boolean;
  /** Newton iterations run. */
  readonly iterations: number;
}

/** Newton's method converges quadratically: a step below 1e-8 leaves loadings exact to rounding. */
const LOADING_STEP_TOLERANCE = 1e-8;
/** The most Newton iterations of the minimum-residual loadings. */
const MAX_ITERATIONS = 100;

/**
 * Returns the loadings λ of the one-factor model of a correlation matrix that minimize the squared off-diagonal
 * residuals Σ_(i<j) (r_ij - λ_i λ_j)², the minimum residual (minres) solution, with |λ| at most `maxLoading`.
 *
 * @remarks
 * The search is projected Newton: from the centroid loadings, the row sums of r over the square root of their total,
 * or from the leading eigenvector scaled to its eigenvalue when the total of r is not positive, each step solves the
 * Newton equations for the loadings not held at a bound and a backtracking line search keeps the projection onto the
 * bounds. A loading at a bound whose gradient points outward stays there. Without a bound, a matrix that one factor
 * does not fit can have loadings above 1, a Heywood case, or no minimum at all, where the loadings grow without limit;
 * a bound, such as psych's 0.995 for the communality, keeps the search finite. The matrix is symmetric, by rows,
 * with 1 on the diagonal, and the function assumes it valid; `maxLoading` is positive, or `Infinity`.
 *
 * @param maxLoading - The largest absolute loading; `Infinity` for none.
 */
export function minresLoadings(r: Float64Array, size: number, maxLoading = Number.POSITIVE_INFINITY): MinresLoadings {
  const x = Float64Array.from(startingLoadings(r, size), (value) => clamp(value, maxLoading));
  const trial = new Float64Array(size);
  const gradient = new Float64Array(size);
  const hessian = new Float64Array(size * size);
  const direction = new Float64Array(size);
  for (let iteration = 1; iteration <= MAX_ITERATIONS; iteration++) {
    const value = offDiagonalResiduals(r, size, x, gradient, hessian);
    newtonDirection(gradient, hessian, size, freeLoadings(x, gradient, maxLoading), direction);
    let moved = 0;
    // The decrease is measured along the projected step, which a bound shortens. A step below the tolerance is taken
    // in full: it is below the rounding of the loadings, so the decrease it makes is not visible in the value.
    const step = backtrack(
      (length) => {
        moved = project(x, direction, length, maxLoading, trial);
        if (moved < LOADING_STEP_TOLERANCE) return Number.NEGATIVE_INFINITY;
        return residualSum(r, size, trial) - ARMIJO * (dot(gradient, trial) - dot(gradient, x));
      },
      value,
      0,
      { decrease: 0 },
    );
    x.set(trial);
    const converged = moved < LOADING_STEP_TOLERANCE;
    if (converged || step === null) return { loadings: signed(x), converged, iterations: iteration };
  }
  return { loadings: signed(x), converged: false, iterations: MAX_ITERATIONS };
}

function clamp(value: number, bound: number): number {
  return Math.max(-bound, Math.min(bound, value));
}

/** Returns the indices of the loadings not at a bound: one there stays while steepest descent points outward. */
function freeLoadings(x: Float64Array, gradient: Float64Array, bound: number): number[] {
  return Array.from(x.keys()).filter(
    (i) => !(Math.abs(x[i] ?? 0) >= bound && (gradient[i] ?? 0) * Math.sign(x[i] ?? 0) < 0),
  );
}

/** Writes x + step * direction, clamped to the bounds, to `out`, and returns the largest change of a loading. */
function project(x: Float64Array, direction: Float64Array, step: number, bound: number, out: Float64Array): number {
  let moved = 0;
  for (let i = 0; i < x.length; i++) {
    out[i] = clamp((x[i] ?? 0) + step * (direction[i] ?? 0), bound);
    moved = Math.max(moved, Math.abs((out[i] ?? 0) - (x[i] ?? 0)));
  }
  return moved;
}

/** Returns the loadings with the sign that makes their sum not negative. */
function signed(x: Float64Array): Float64Array {
  const sign = x.reduce((sum, value) => sum + value, 0) < 0 ? -1 : 1;
  return x.map((value) => sign * value);
}

/**
 * Writes to `direction` the Newton step -H⁻¹ g of the loadings in `free`, zero for the others, or the steepest descent
 * -g where the Hessian of those loadings is not positive definite.
 */
function newtonDirection(
  gradient: Float64Array,
  hessian: Float64Array,
  size: number,
  free: readonly number[],
  direction: Float64Array,
): void {
  direction.fill(0);
  const count = free.length;
  const reduced = new Float64Array(count * count);
  const slope = new Float64Array(count);
  free.forEach((i, a) => {
    slope[a] = gradient[i] ?? 0;
    free.forEach((j, b) => {
      reduced[a * count + b] = hessian[i * size + j] ?? 0;
    });
  });
  const inverse = invertPositiveDefinite(reduced, count);
  const step = new Float64Array(count);
  if (inverse !== null) {
    for (let a = 0; a < count; a++) step[a] = -dot(inverse.subarray(a * count, (a + 1) * count), slope);
  }
  if (inverse === null || dot(step, slope) >= 0) {
    for (let a = 0; a < count; a++) step[a] = -(slope[a] ?? 0);
  }
  free.forEach((i, a) => {
    direction[i] = step[a] ?? 0;
  });
}

function startingLoadings(r: Float64Array, size: number): number[] {
  const total = r.reduce((sum, value) => sum + value, 0);
  if (total > 0) {
    return Array.from(
      { length: size },
      (_, i) => r.subarray(i * size, (i + 1) * size).reduce((sum, value) => sum + value, 0) / Math.sqrt(total),
    );
  }
  const { values, vectors } = symmetricEigen(r, size);
  const scale = Math.sqrt(Math.max(values[0] ?? 0, 0));
  return Array.from({ length: size }, (_, i) => scale * (vectors[i] ?? 0));
}

/** Returns Σ_(i<j) (r_ij - λ_i λ_j)². */
function residualSum(r: Float64Array, size: number, loadings: Float64Array): number {
  let value = 0;
  for (let i = 0; i < size; i++) {
    for (let j = i + 1; j < size; j++) value += ((r[i * size + j] ?? 0) - (loadings[i] ?? 0) * (loadings[j] ?? 0)) ** 2;
  }
  return value;
}

/**
 * Returns Σ_(i<j) (r_ij - λ_i λ_j)² and writes its gradient, -2 Σ_(j≠i) (r_ij - λ_i λ_j) λ_j, and Hessian: 2 Σ_(j≠i)
 * λ_j² on the diagonal, 4 λ_i λ_j - 2 r_ij off it.
 */
function offDiagonalResiduals(
  r: Float64Array,
  size: number,
  loadings: Float64Array,
  gradient: Float64Array,
  hessian: Float64Array,
): number {
  let value = 0;
  let squares = 0;
  for (const loading of loadings) squares += loading * loading;
  for (let i = 0; i < size; i++) {
    const li = loadings[i] ?? 0;
    let slope = 0;
    for (let j = 0; j < size; j++) {
      const lj = loadings[j] ?? 0;
      if (j === i) {
        hessian[i * size + i] = 2 * (squares - li * li);
        continue;
      }
      const residual = (r[i * size + j] ?? 0) - li * lj;
      if (j > i) value += residual * residual;
      slope -= 2 * residual * lj;
      hessian[i * size + j] = 4 * li * lj - 2 * (r[i * size + j] ?? 0);
    }
    gradient[i] = slope;
  }
  return value;
}

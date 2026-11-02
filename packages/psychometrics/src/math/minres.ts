import { ARMIJO, backtrack, dot } from "@measurekit/core";
import { symmetricEigen } from "./eigen.js";
import { invertPositiveDefinite } from "./linear-algebra.js";
import { identity } from "./matrix.js";

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

function clamp(value: number, bound: number, lower = -bound): number {
  return Math.max(lower, Math.min(bound, value));
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
  const step = descent(reduced, slope);
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

/** The minimum residual solution of a correlation matrix for one or more factors. */
export interface Minres {
  /**
   * The loadings, items by factors, by rows, on the principal axes of the reduced correlation matrix: the factors are
   * in the order of its eigenvalues, and each factor's loadings sum to a number that is not negative.
   */
  readonly loadings: Float64Array;
  /** The communalities, the squared loadings of each item summed over the factors. */
  readonly communalities: Float64Array;
  /** Whether the largest change of a communality in an iteration fell to the tolerance. */
  readonly converged: boolean;
  /** Iterations run. */
  readonly iterations: number;
}

/** Stopping rules of {@link minres}. */
export interface MinresOptions {
  /**
   * The largest change of a communality in an iteration at which the solution counts as found.
   *
   * @defaultValue `1e-6`
   */
  readonly tolerance?: number;
  /**
   * The most iterations.
   *
   * @defaultValue `100`
   */
  readonly maxIterations?: number;
}

const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_MAX_ITERATIONS = 100;
/** The least uniqueness of an item, so that a communality is at most 0.995: the bound of psych's `fa`. */
const MIN_UNIQUENESS = 0.005;
/** A rise of the residuals that is rounding, relative to their value. */
const RESIDUAL_ROUNDING = 1e-12;
/** The ridge added to the approximate Hessian, relative to its mean diagonal, which it may leave singular. */
const RIDGE = 1e-10;

/** The fit of the factors at some communalities, as an objective of the communalities measures it. */
interface CommunalityFit {
  /** The objective, which the communalities that solve the extraction minimize. */
  readonly value: number;
  /** The derivative of `value` with respect to each communality. */
  readonly gradient: Float64Array;
  /** Returns the Hessian of `value` at a perfect fit for the communalities in `free`, by rows, with a small ridge. */
  readonly hessian: (free: readonly number[]) => Float64Array;
  /**
   * Returns the loadings, items by factors, by rows, on the principal axes of the fit: each factor's loadings sum to a
   * number that is not negative.
   */
  readonly loadings: () => Float64Array;
}

/** An objective of the communalities of a factor extraction: the fit of the factors at each set of communalities. */
export type CommunalityObjective = (communalities: Float64Array) => CommunalityFit;

/**
 * Returns the minimum residual (minres) solution of a correlation matrix for `factors` factors: the communalities for
 * which the best rank-`factors` fit of the matrix, with them on its diagonal, leaves the least squared residuals.
 *
 * @remarks
 * The loadings are the leading eigenvectors of the reduced matrix, scaled by the roots of their eigenvalues, as the
 * unweighted least squares method of Jöreskog and psych's `fa` define the solution. The objective is half the sum of the
 * squared eigenvalues that the fit leaves out, with a negative eigenvalue among the leading ones counting in full, and
 * its Hessian at a perfect fit is the squares of the entries of the projector onto the eigenvectors left out. The search
 * is {@link minimizeCommunalities}, from the squared multiple correlations, or from the squared largest correlation of
 * each item when the matrix is not positive definite. The residuals of a communality at a bound are those of psych's,
 * whose loadings there can have a communality beyond the bound; `minresLoadings` instead bounds the loadings of one
 * factor. The matrix is symmetric, by rows, with 1 on the diagonal, and `factors` is a positive integer below `size`;
 * the function assumes both valid.
 *
 * @see {@link https://doi.org/10.1007/BF02289658 | Jöreskog (1967)}
 */
export function minres(r: Float64Array, size: number, factors: number, options: MinresOptions = {}): Minres {
  return minimizeCommunalities(
    residualObjective(r, size, factors),
    startingCommunalities(r, size),
    size,
    factors,
    options,
  );
}

/**
 * Returns the communalities from 0 to 0.995 that minimize an objective, with the loadings of its fit there.
 *
 * @remarks
 * Each iteration takes a Newton step on the communalities that are not at a bound, with the Hessian of the objective at
 * a perfect fit, and a backtracking line search that keeps the projection onto the bounds. A communality at a bound
 * whose gradient points outward stays there. The search stops when the largest change of a communality falls below the
 * tolerance. `start` holds one communality per item, which the bounds clamp.
 *
 * @param objective - The fit of the factors at each set of communalities.
 * @param start - The communalities the search starts from, one per item.
 * @param size - The number of items.
 * @param factors - The number of factors.
 * @param options - The stopping rules.
 */
export function minimizeCommunalities(
  objective: CommunalityObjective,
  start: ArrayLike<number>,
  size: number,
  factors: number,
  options: MinresOptions = {},
): Minres {
  const { tolerance = DEFAULT_TOLERANCE, maxIterations = DEFAULT_MAX_ITERATIONS } = options;
  const upper = 1 - MIN_UNIQUENESS;
  const communalities = Float64Array.from(start, (value) => clamp(value, upper, 0));
  const trial = new Float64Array(size);
  let current = objective(communalities);
  let converged = false;
  let iterations = 0;
  while (!converged && iterations < maxIterations) {
    iterations++;
    const direction = communalityStep(current, communalities, upper);
    let moved = 0;
    let latest = current;
    const step = backtrack(
      (length) => {
        moved = 0;
        for (let i = 0; i < size; i++) {
          trial[i] = clamp((communalities[i] ?? 0) + length * (direction[i] ?? 0), upper, 0);
          moved = Math.max(moved, Math.abs((trial[i] ?? 0) - (communalities[i] ?? 0)));
        }
        latest = objective(trial);
        // A move below the tolerance is accepted whatever it does to the value: it is below what the solution is
        // sought to, and the value changes by less than its rounding.
        if (moved < tolerance) return Number.NEGATIVE_INFINITY;
        return latest.value - ARMIJO * (dot(current.gradient, trial) - dot(current.gradient, communalities));
      },
      current.value,
      0,
      { decrease: 0, slack: RESIDUAL_ROUNDING * Math.abs(current.value) },
    );
    if (step === null) break;
    communalities.set(trial);
    current = latest;
    converged = moved < tolerance;
  }
  const loadings = current.loadings();
  const fitted = Float64Array.from({ length: size }, (_, i) =>
    loadings.subarray(i * factors, (i + 1) * factors).reduce((sum, value) => sum + value * value, 0),
  );
  return { loadings, communalities: fitted, converged, iterations };
}

/** Returns minres's objective of the communalities of `factors` factors of the correlation matrix `r`. */
function residualObjective(r: Float64Array, size: number, factors: number): CommunalityObjective {
  return (communalities) => {
    const reduced = Float64Array.from(r);
    for (let i = 0; i < size; i++) reduced[i * size + i] = communalities[i] ?? 0;
    const { values, vectors } = symmetricEigen(reduced, size);
    const gradient = new Float64Array(size);
    let value = 0;
    for (let k = 0; k < size; k++) {
      const excess = k < factors ? Math.min(values[k] ?? 0, 0) : (values[k] ?? 0);
      value += 0.5 * excess * excess;
      for (let i = 0; i < size; i++) gradient[i] = (gradient[i] ?? 0) + excess * (vectors[k * size + i] ?? 0) ** 2;
    }
    return {
      value,
      gradient,
      hessian: (free) => projectorHessian(vectors, size, factors, free),
      loadings: () =>
        principalAxes(
          values,
          vectors,
          size,
          factors,
          Float64Array.from({ length: size }, () => 1),
        ),
    };
  };
}

/**
 * Returns the loadings on the leading principal axes of a matrix, eigenvector k scaled by the root of `extent[k]` and
 * entry i by `scale[i]`, each factor's sum made not negative.
 *
 * @param values - The extent of each factor, such as an eigenvalue; a negative one counts as 0.
 * @param vectors - The eigenvectors by rows, row k of factor k.
 */
export function principalAxes(
  values: Float64Array,
  vectors: Float64Array,
  size: number,
  factors: number,
  scale: Float64Array,
): Float64Array {
  const loadings = new Float64Array(size * factors);
  for (let k = 0; k < factors; k++) {
    const root = Math.sqrt(Math.max(values[k] ?? 0, 0));
    let sum = 0;
    for (let i = 0; i < size; i++) sum += (scale[i] ?? 0) * root * (vectors[k * size + i] ?? 0);
    const sign = sum < 0 ? -1 : 1;
    for (let i = 0; i < size; i++)
      loadings[i * factors + k] = sign * (scale[i] ?? 0) * root * (vectors[k * size + i] ?? 0);
  }
  return loadings;
}

/**
 * Returns the Newton step of the communalities, zero for those at a bound that the gradient pushes outside, or the
 * steepest descent where the Hessian is not positive definite.
 */
function communalityStep(fit: CommunalityFit, communalities: Float64Array, upper: number): Float64Array {
  const free = Array.from(communalities.keys()).filter((i) => !pushedOut(communalities[i], fit.gradient[i], upper));
  const slope = Float64Array.from(free, (i) => fit.gradient[i] ?? 0);
  const step = descent(fit.hessian(free), slope);
  const direction = new Float64Array(communalities.length);
  free.forEach((i, a) => {
    direction[i] = step[a] ?? 0;
  });
  return direction;
}

/** Returns whether a communality is at a bound that its gradient pushes it beyond. */
function pushedOut(communality = 0, gradient = 0, upper = 1): boolean {
  return (communality <= 0 && gradient > 0) || (communality >= upper && gradient < 0);
}

/**
 * Returns the Hessian at a perfect fit of an objective of the communalities in `free`, with a small ridge: the squares
 * of the entries of the projector onto the eigenvectors that the fit leaves out, each row and column i times `scale[i]`.
 *
 * @param vectors - The eigenvectors by rows, row k of the eigenvalue k, the `factors` leading ones first.
 * @param scale - The factor of each row and column; 1 when not given.
 */
export function projectorHessian(
  vectors: Float64Array,
  size: number,
  factors: number,
  free: readonly number[],
  scale?: Float64Array,
): Float64Array {
  const projector = identity(size);
  for (let k = 0; k < factors; k++) {
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        projector[i * size + j] =
          (projector[i * size + j] ?? 0) - (vectors[k * size + i] ?? 0) * (vectors[k * size + j] ?? 0);
      }
    }
  }
  const count = free.length;
  const hessian = new Float64Array(count * count);
  free.forEach((i, a) => {
    free.forEach((j, b) => {
      hessian[a * count + b] = (projector[i * size + j] ?? 0) ** 2 * (scale?.[i] ?? 1) * (scale?.[j] ?? 1);
    });
  });
  let trace = 0;
  for (let a = 0; a < count; a++) trace += hessian[a * count + a] ?? 0;
  for (let a = 0; a < count; a++) hessian[a * count + a] = (hessian[a * count + a] ?? 0) + (RIDGE * trace) / count;
  return hessian;
}

/** Returns -H⁻¹ g, or -g where H is not positive definite or the step does not descend. */
function descent(hessian: Float64Array, slope: Float64Array): Float64Array {
  const count = slope.length;
  const inverse = invertPositiveDefinite(hessian, count);
  const step = new Float64Array(count);
  if (inverse !== null) {
    for (let a = 0; a < count; a++) step[a] = -dot(inverse.subarray(a * count, (a + 1) * count), slope);
  }
  if (inverse === null || dot(step, slope) >= 0) {
    for (let a = 0; a < count; a++) step[a] = -(slope[a] ?? 0);
  }
  return step;
}

/** Returns the squared multiple correlations, or where the matrix is not positive definite the squared largest. */
export function startingCommunalities(r: Float64Array, size: number): number[] {
  const inverse = invertPositiveDefinite(r, size);
  return Array.from({ length: size }, (_, i) => {
    if (inverse !== null) return 1 - 1 / (inverse[i * size + i] ?? 1);
    let largest = 0;
    for (let j = 0; j < size; j++) if (j !== i) largest = Math.max(largest, Math.abs(r[i * size + j] ?? 0));
    return largest * largest;
  });
}

import { type DescentPoint, type GradientFunction, stepAlong } from "./descent.js";
import { dot, maxAbs, multiplyVector } from "./vectors.js";

/** Stopping rules of a minimization. */
export interface OptimizerOptions {
  /**
   * The largest absolute gradient component at which the minimization has converged.
   *
   * @defaultValue `1e-8`
   */
  readonly tolerance?: number;
  /**
   * The most iterations to run.
   *
   * @defaultValue `100`
   */
  readonly maxIterations?: number;
}

/** The outcome of a minimization. */
export interface OptimizationResult {
  /** The point reached. */
  readonly x: Float64Array;
  /** The function value at `x`. */
  readonly value: number;
  /** Iterations run. */
  readonly iterations: number;
  /**
   * Whether the minimization converged: the gradient at `x` is within the tolerance or, for BFGS, no step from steepest
   * descent makes progress, so that `x` is a minimum to the precision of the function.
   */
  readonly converged: boolean;
}

const DEFAULT_TOLERANCE = 1e-8;
const DEFAULT_MAX_ITERATIONS = 100;

/**
 * Minimizes a differentiable function with the BFGS quasi-Newton method and a backtracking line search.
 *
 * @remarks
 * The inverse Hessian approximation starts as the identity, is scaled before its first update, and restarts from the
 * identity when it stops giving a descent direction or a step that makes progress. When steepest descent makes no
 * progress either, the point is a minimum to the precision of the function and the minimization has converged, as in
 * R's `optim`.
 *
 * @param f - The function to minimize.
 * @param start - The point to start from.
 * @param options - The stopping rules.
 * @see {@link https://doi.org/10.1007/978-0-387-40065-5 | Nocedal and Wright (2006), eq. 6.20}
 */
export function bfgs(
  f: GradientFunction,
  start: ArrayLike<number>,
  options: OptimizerOptions = {},
): OptimizationResult {
  const { tolerance = DEFAULT_TOLERANCE, maxIterations = DEFAULT_MAX_ITERATIONS } = options;
  const size = start.length;
  const approximation = inverseHessianOf(size);
  const direction = new Float64Array(size);
  let point: DescentPoint = { x: Float64Array.from(start), gradient: new Float64Array(size), value: 0 };
  point.value = f(point.x, point.gradient);
  let trial: DescentPoint = { x: new Float64Array(size), gradient: new Float64Array(size), value: 0 };
  for (let iteration = 0; iteration < maxIterations; iteration++) {
    if (maxAbs(point.gradient) <= tolerance) {
      return { x: point.x, value: point.value, iterations: iteration, converged: true };
    }
    approximation.direction(point.gradient, direction);
    // Near a minimum, rounding of the value hides the decrease a gradient above the tolerance promises, and the Armijo
    // condition accepts steps that change nothing.
    if (!(stepAlong(f, point, direction, trial) && progressed(point, trial))) {
      if (approximation.fresh()) return { x: point.x, value: point.value, iterations: iteration, converged: true };
      approximation.reset();
      continue;
    }
    approximation.update(point, trial);
    [point, trial] = [trial, point];
  }
  return {
    x: point.x,
    value: point.value,
    iterations: maxIterations,
    converged: maxAbs(point.gradient) <= tolerance,
  };
}

/** An approximation of the inverse Hessian of a descent: the identity, until the first step gives a curvature to scale it. */
interface InverseHessian {
  /** Writes the quasi-Newton direction for `gradient`, or its negative where that is not a descent direction. */
  direction(gradient: Float64Array, out: Float64Array): void;
  /** Learns from an accepted step, if it has positive curvature. */
  update(from: DescentPoint, to: DescentPoint): void;
  /** Returns the approximation to the identity. */
  reset(): void;
  /** Whether it is the identity, as at the start and after a reset. */
  fresh(): boolean;
}

function inverseHessianOf(size: number): InverseHessian {
  const matrix = new Float64Array(size * size);
  const product = new Float64Array(size);
  const step = new Float64Array(size);
  const gradientChange = new Float64Array(size);
  let isFresh = true;
  const reset = (): void => {
    matrix.fill(0);
    for (let i = 0; i < size; i++) matrix[i * size + i] = 1;
    isFresh = true;
  };
  reset();
  return {
    direction(gradient, out) {
      multiplyVector(matrix, gradient, out);
      for (let i = 0; i < size; i++) out[i] = -(out[i] ?? 0);
      if (dot(gradient, out) >= 0) {
        reset();
        for (let i = 0; i < size; i++) out[i] = -(gradient[i] ?? 0);
      }
    },
    update(from, to) {
      for (let i = 0; i < size; i++) {
        step[i] = (to.x[i] ?? 0) - (from.x[i] ?? 0);
        gradientChange[i] = (to.gradient[i] ?? 0) - (from.gradient[i] ?? 0);
      }
      const curvature = dot(step, gradientChange);
      if (!(curvature > 0)) return;
      if (isFresh) {
        const scale = curvature / dot(gradientChange, gradientChange);
        for (let i = 0; i < size; i++) matrix[i * size + i] = scale;
        isFresh = false;
      }
      updateInverseHessian(matrix, step, gradientChange, curvature, product);
    },
    reset,
    fresh: () => isFresh,
  };
}

/** Returns whether a step lowered the value or the largest gradient component. */
function progressed(point: DescentPoint, trial: DescentPoint): boolean {
  return trial.value < point.value || maxAbs(trial.gradient) < maxAbs(point.gradient);
}

/** Applies the BFGS update to an inverse Hessian approximation, given a step with positive `curvature`. */
function updateInverseHessian(
  inverseHessian: Float64Array,
  step: Float64Array,
  gradientChange: Float64Array,
  curvature: number,
  product: Float64Array,
): void {
  const size = step.length;
  multiplyVector(inverseHessian, gradientChange, product);
  const rho = 1 / curvature;
  const factor = rho + rho * rho * dot(gradientChange, product);
  for (let i = 0; i < size; i++) {
    const si = step[i] ?? 0;
    const pi = product[i] ?? 0;
    for (let j = 0; j < size; j++) {
      const sj = step[j] ?? 0;
      const change = factor * si * sj - rho * (pi * sj + si * (product[j] ?? 0));
      inverseHessian[i * size + j] = (inverseHessian[i * size + j] ?? 0) + change;
    }
  }
}

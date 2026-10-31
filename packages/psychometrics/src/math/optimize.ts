import { backtrack, dot, maxAbs } from "@measurekit/core";
import { invert } from "./linear-algebra.js";

/** A function to minimize: it returns its value at `x` and writes its gradient into `gradient`. */
export type GradientFunction = (x: Float64Array, gradient: Float64Array) => number;

/**
 * A function to minimize by Newton's method: it returns its value at `x` and writes its gradient and its Hessian, by
 * rows, into the arrays given.
 */
export type HessianFunction = (x: Float64Array, gradient: Float64Array, hessian: Float64Array) => number;

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
  /**
   * When the search direction changes no coordinate by more than this, the minimization takes it in full, without the
   * line search, and has converged.
   *
   * @remarks
   * It is meant for Newton's method, which converges quadratically, so that its next step would be below the rounding
   * of the coordinates, and where rounding of the value hides the decrease of a small step.
   */
  readonly stepTolerance?: number;
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
 * Minimizes a twice-differentiable function with Newton's method and a backtracking line search. Where the Hessian
 * is singular or not positive definite, the step follows the negative gradient.
 */
export function newton(
  f: HessianFunction,
  start: ArrayLike<number>,
  options: OptimizerOptions = {},
): OptimizationResult {
  const size = start.length;
  const hessian = new Float64Array(size * size);
  return descend(
    (x, gradient) => f(x, gradient, hessian),
    start,
    options,
    (gradient, direction) => {
      const inverse = invert(hessian, size);
      if (inverse !== null) {
        multiply(inverse, gradient, direction);
        negate(direction);
      }
      if (inverse === null || dot(gradient, direction) >= 0) steepest(gradient, direction);
    },
  );
}

/**
 * Minimizes a differentiable function with the BFGS quasi-Newton method and a backtracking line search.
 *
 * @remarks
 * The inverse Hessian approximation starts as the identity, is scaled before its first update, and restarts from the
 * identity when it stops giving a descent direction or a step that makes progress. When steepest descent makes no
 * progress either, the point is a minimum to the precision of the function and the minimization has converged, as in
 * R's `optim`.
 *
 * @see {@link https://doi.org/10.1007/978-0-387-40065-5 | Nocedal and Wright (2006), eq. 6.20}
 */
export function bfgs(
  f: GradientFunction,
  start: ArrayLike<number>,
  options: OptimizerOptions = {},
): OptimizationResult {
  const size = start.length;
  const inverseHessian = new Float64Array(size * size);
  const product = new Float64Array(size);
  const step = new Float64Array(size);
  const gradientChange = new Float64Array(size);
  let fresh = true;
  const reset = (): void => {
    inverseHessian.fill(0);
    for (let i = 0; i < size; i++) inverseHessian[i * size + i] = 1;
    fresh = true;
  };
  reset();
  return descend(
    f,
    start,
    options,
    (gradient, direction) => {
      multiply(inverseHessian, gradient, direction);
      negate(direction);
      if (dot(gradient, direction) >= 0) {
        reset();
        steepest(gradient, direction);
      }
    },
    (from, to) => {
      for (let i = 0; i < size; i++) {
        step[i] = (to.x[i] ?? 0) - (from.x[i] ?? 0);
        gradientChange[i] = (to.gradient[i] ?? 0) - (from.gradient[i] ?? 0);
      }
      const curvature = dot(step, gradientChange);
      if (!(curvature > 0)) return;
      if (fresh) {
        const scale = curvature / dot(gradientChange, gradientChange);
        for (let i = 0; i < size; i++) inverseHessian[i * size + i] = scale;
        fresh = false;
      }
      updateInverseHessian(inverseHessian, step, gradientChange, curvature, product);
    },
    () => {
      if (fresh) return false;
      reset();
      return true;
    },
  );
}

/**
 * Runs a line-search descent: `direction` writes a descent direction for the current gradient, and `update`, if
 * given, sees each accepted step, from one point to the next. `restart`, if given, is called when no step makes
 * progress, and returns whether to try again with a new direction; if not, the descent has converged.
 */
function descend(
  f: GradientFunction,
  start: ArrayLike<number>,
  options: OptimizerOptions,
  direction: (gradient: Float64Array, out: Float64Array) => void,
  update?: (from: Point, to: Point) => void,
  restart?: () => boolean,
): OptimizationResult {
  const { tolerance = DEFAULT_TOLERANCE, maxIterations = DEFAULT_MAX_ITERATIONS, stepTolerance } = options;
  const size = start.length;
  let point: Point = { x: Float64Array.from(start), gradient: new Float64Array(size), value: 0 };
  point.value = f(point.x, point.gradient);
  let trial: Point = { x: new Float64Array(size), gradient: new Float64Array(size), value: 0 };
  const search = new Float64Array(size);
  for (let iteration = 0; iteration < maxIterations; iteration++) {
    if (maxAbs(point.gradient) <= tolerance) {
      return { x: point.x, value: point.value, iterations: iteration, converged: true };
    }
    direction(point.gradient, search);
    if (stepTolerance !== undefined && maxAbs(search) <= stepTolerance) return fullStep(f, point, search, iteration);
    const accepted = lineSearch(f, point, search, trial);
    // Near a minimum, rounding of the value hides the decrease a gradient above the tolerance promises, and the Armijo
    // condition accepts steps that change nothing.
    if (restart !== undefined && !(accepted && progressed(point, trial))) {
      if (restart()) continue;
      return { x: point.x, value: point.value, iterations: iteration, converged: true };
    }
    if (!accepted) return { x: point.x, value: point.value, iterations: iteration, converged: false };
    update?.(point, trial);
    [point, trial] = [trial, point];
  }
  return {
    x: point.x,
    value: point.value,
    iterations: maxIterations,
    converged: maxAbs(point.gradient) <= tolerance,
  };
}

/** Takes the step along `search` from `point` in full and returns the converged result there. */
function fullStep(f: GradientFunction, point: Point, search: Float64Array, iteration: number): OptimizationResult {
  const x = point.x.map((value, i) => value + (search[i] ?? 0));
  return { x, value: f(x, new Float64Array(x.length)), iterations: iteration + 1, converged: true };
}

/** A point of a descent, with the function's gradient and value there. */
interface Point {
  readonly x: Float64Array;
  readonly gradient: Float64Array;
  value: number;
}

/**
 * Halves the step along `search` from `point` until the value decreases enough (the Armijo condition), leaving the
 * accepted point in `trial`. Returns whether a step was accepted.
 */
function lineSearch(f: GradientFunction, point: Point, search: Float64Array, trial: Point): boolean {
  const accepted = backtrack(
    (length) => {
      for (let i = 0; i < search.length; i++) trial.x[i] = (point.x[i] ?? 0) + length * (search[i] ?? 0);
      trial.value = f(trial.x, trial.gradient);
      return trial.value;
    },
    point.value,
    dot(point.gradient, search),
  );
  return accepted !== null;
}

/** Returns whether a step lowered the value or the largest gradient component. */
function progressed(point: Point, trial: Point): boolean {
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
  multiply(inverseHessian, gradientChange, product);
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

function negate(values: Float64Array): void {
  for (let i = 0; i < values.length; i++) values[i] = -(values[i] ?? 0);
}

function steepest(gradient: Float64Array, direction: Float64Array): void {
  for (let i = 0; i < gradient.length; i++) direction[i] = -(gradient[i] ?? 0);
}

function multiply(matrix: Float64Array, vector: Float64Array, out: Float64Array): void {
  const size = vector.length;
  for (let i = 0; i < size; i++) {
    let sum = 0;
    for (let j = 0; j < size; j++) sum += (matrix[i * size + j] ?? 0) * (vector[j] ?? 0);
    out[i] = sum;
  }
}

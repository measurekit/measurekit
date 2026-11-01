import {
  type DescentPoint,
  dot,
  type GradientFunction,
  maxAbs,
  multiplyVector,
  type OptimizationResult,
  type OptimizerOptions,
  stepAlong,
} from "@measurekit/core";
import { invert } from "./linear-algebra.js";

/**
 * A function to minimize by Newton's method: it returns its value at `x` and writes its gradient and its Hessian, by
 * rows, into the arrays given.
 */
export type HessianFunction = (x: Float64Array, gradient: Float64Array, hessian: Float64Array) => number;

/** Stopping rules of Newton's method. */
export interface NewtonOptions extends OptimizerOptions {
  /**
   * When the search direction changes no coordinate by more than this, the minimization takes it in full, without the
   * line search, and has converged.
   *
   * @remarks
   * Newton's method converges quadratically, so that its next step would be below the rounding of the coordinates, and
   * rounding of the value hides the decrease of a small step.
   *
   * @defaultValue none: every step goes through the line search
   */
  readonly stepTolerance?: number;
}

const DEFAULT_TOLERANCE = 1e-8;
const DEFAULT_MAX_ITERATIONS = 100;

/**
 * Minimizes a twice-differentiable function with Newton's method and a backtracking line search. Where the Hessian
 * is singular or not positive definite, the step follows the negative gradient.
 */
export function newton(f: HessianFunction, start: ArrayLike<number>, options: NewtonOptions = {}): OptimizationResult {
  const { tolerance = DEFAULT_TOLERANCE, maxIterations = DEFAULT_MAX_ITERATIONS, stepTolerance } = options;
  const size = start.length;
  const hessian = new Float64Array(size * size);
  const objective: GradientFunction = (x, gradient) => f(x, gradient, hessian);
  let point: DescentPoint = { x: Float64Array.from(start), gradient: new Float64Array(size), value: 0 };
  point.value = objective(point.x, point.gradient);
  let trial: DescentPoint = { x: new Float64Array(size), gradient: new Float64Array(size), value: 0 };
  const search = new Float64Array(size);
  for (let iteration = 0; iteration < maxIterations; iteration++) {
    if (maxAbs(point.gradient) <= tolerance) {
      return { x: point.x, value: point.value, iterations: iteration, converged: true };
    }
    direction(hessian, point.gradient, search);
    if (stepTolerance !== undefined && maxAbs(search) <= stepTolerance) {
      return fullStep(objective, point, search, iteration);
    }
    if (!stepAlong(objective, point, search, trial)) {
      return { x: point.x, value: point.value, iterations: iteration, converged: false };
    }
    [point, trial] = [trial, point];
  }
  return {
    x: point.x,
    value: point.value,
    iterations: maxIterations,
    converged: maxAbs(point.gradient) <= tolerance,
  };
}

/** Writes the Newton direction, -H⁻¹ g, into `out`, or the negative gradient where H is not positive definite. */
function direction(hessian: Float64Array, gradient: Float64Array, out: Float64Array): void {
  const inverse = invert(hessian, gradient.length);
  if (inverse !== null) {
    multiplyVector(inverse, gradient, out);
    for (let i = 0; i < out.length; i++) out[i] = -(out[i] ?? 0);
  }
  if (inverse === null || dot(gradient, out) >= 0) {
    for (let i = 0; i < out.length; i++) out[i] = -(gradient[i] ?? 0);
  }
}

/** Takes the step along `search` from `point` in full and returns the converged result there. */
function fullStep(
  f: GradientFunction,
  point: DescentPoint,
  search: Float64Array,
  iteration: number,
): OptimizationResult {
  const x = point.x.map((value, i) => value + (search[i] ?? 0));
  return { x, value: f(x, new Float64Array(x.length)), iterations: iteration + 1, converged: true };
}

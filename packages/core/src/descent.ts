import { backtrack } from "./line-search.js";
import { dot } from "./vectors.js";

/** A function to minimize: it returns its value at `x` and writes its gradient into `gradient`. */
export type GradientFunction = (x: Float64Array, gradient: Float64Array) => number;

/** A point of a descent, with the function's gradient and value there. */
export interface DescentPoint {
  /** The coordinates. */
  readonly x: Float64Array;
  /** The gradient of the function at `x`. */
  readonly gradient: Float64Array;
  /** The function's value at `x`. */
  value: number;
}

/**
 * Halves the step along `search` from `point` until the value decreases enough (the Armijo condition), leaving the
 * accepted point in `trial`. Returns whether a step was accepted.
 *
 * @param f - The function to minimize.
 * @param point - The point the step starts from, with its gradient and value.
 * @param search - The direction of the step, a descent direction at `point`.
 * @param trial - Receives the point of each length tried, and so the accepted point.
 */
export function stepAlong(
  f: GradientFunction,
  point: DescentPoint,
  search: Float64Array,
  trial: DescentPoint,
): boolean {
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

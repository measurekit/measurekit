import { backtrack } from "./line-search.js";
import { dot, maxAbs } from "./vectors.js";

/** A function to minimize: it returns its value at `x` and writes its gradient into `gradient`. */
export type GradientFunction = (x: Float64Array, gradient: Float64Array) => number;

/**
 * A function to minimize given as its value and, apart, its gradient, for a gradient that costs more than the value,
 * such as one by finite differences: a descent then evaluates the value alone at the points its line search tries, and
 * the gradient at the point it accepts, as R's `optim` calls `fn` and `gr`.
 */
export interface ValueAndGradient {
  /** Returns the function's value at `x`. */
  readonly value: (x: Float64Array) => number;
  /** Writes the function's gradient at `x` into `gradient`. */
  readonly gradient: (x: Float64Array, gradient: Float64Array) => void;
}

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
 * @remarks
 * A {@link GradientFunction} is evaluated whole at every length tried, and rejects a length by returning a value that
 * is not finite. A {@link ValueAndGradient} gives its value alone at every length tried and its gradient at the length
 * the value accepts, which a gradient that is not finite rejects in turn, so that the step goes on halving.
 *
 * @param f - The function to minimize, with its gradient together or apart.
 * @param point - The point the step starts from, with its gradient and value.
 * @param search - The direction of the step, a descent direction at `point`.
 * @param trial - Receives the point of each length tried, and so the accepted point.
 * @returns Whether a length of the step was accepted; `trial` then holds the accepted point.
 */
export function stepAlong(
  f: GradientFunction | ValueAndGradient,
  point: DescentPoint,
  search: Float64Array,
  trial: DescentPoint,
): boolean {
  const whole = typeof f === "function";
  const accepted = backtrack(
    (length) => {
      for (let i = 0; i < search.length; i++) trial.x[i] = (point.x[i] ?? 0) + length * (search[i] ?? 0);
      trial.value = whole ? f(trial.x, trial.gradient) : f.value(trial.x);
      return trial.value;
    },
    point.value,
    dot(point.gradient, search),
    whole
      ? {}
      : {
          accept: () => {
            f.gradient(trial.x, trial.gradient);
            return Number.isFinite(maxAbs(trial.gradient));
          },
        },
  );
  return accepted !== null;
}

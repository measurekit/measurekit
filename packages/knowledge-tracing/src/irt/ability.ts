import { indexOfMax } from "../math/extremes.js";
import { type Jet, quinticDerivative, quinticHermite, quinticMaximum } from "../math/quintic.js";
import type { IrtState } from "./state.js";

/** An ability estimate. */
export interface Ability {
  /** The estimate of θ. */
  readonly theta: number;
  /**
   * Its standard error: the posterior standard deviation for EAP, 1 / √(-ℓ''(θ̂)) of the log-posterior ℓ for MAP, or
   * `Infinity` where ℓ is not curved down there.
   */
  readonly standardError: number;
}

/** Returns Σ w_q f(q) / Σ w_q. */
export function average(weights: ArrayLike<number>, f: (q: number) => number): number {
  let total = 0;
  let sum = 0;
  for (let q = 0; q < weights.length; q++) {
    const weight = weights[q] ?? 0;
    total += weight;
    sum += weight * f(q);
  }
  return sum / total;
}

/** Returns the posterior mean of θ on the grid and the posterior standard deviation. */
export function posteriorMean(state: IrtState, grid: readonly number[]): Ability {
  const weights = state.logDensity.map(Math.exp);
  const theta = average(weights, (q) => grid[q] ?? 0);
  const variance = average(weights, (q) => ((grid[q] ?? 0) - theta) ** 2);
  return { theta, standardError: Math.sqrt(variance) };
}

/**
 * Returns the posterior mode: the highest grid point locates it, and the quintic that matches the log-posterior and
 * its first two derivatives at the ends of the interval where the slope changes sign gives it between points. A mode
 * beyond the grid's ends is reported at the end.
 */
export function posteriorMode(state: IrtState, grid: readonly number[]): Ability {
  const { logDensity, slope, curvature } = state;
  const top = indexOfMax(logDensity);
  const left = (slope[top] ?? 0) > 0 ? top : top - 1;
  if (left < 0 || left >= grid.length - 1) {
    return { theta: grid[top] ?? 0, standardError: modeError(curvature[top] ?? 0) };
  }
  const start = grid[left] ?? 0;
  const h = (grid[left + 1] ?? 0) - start;
  const jet = (q: number): Jet => [logDensity[q] ?? 0, h * (slope[q] ?? 0), h * h * (curvature[q] ?? 0)];
  const quintic = quinticHermite(jet(left), jet(left + 1));
  const t = quinticMaximum(quintic);
  return { theta: start + t * h, standardError: modeError(quinticDerivative(quintic, t, 2) / (h * h)) };
}

/** Returns 1 / √(-ℓ''), or `Infinity` where the log-posterior is not curved down, as the data then fix no error. */
function modeError(curvature: number): number {
  return curvature < 0 ? 1 / Math.sqrt(-curvature) : Number.POSITIVE_INFINITY;
}

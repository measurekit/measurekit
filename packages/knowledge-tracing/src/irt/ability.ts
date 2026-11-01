import { indexOfMax } from "../math/extremes.js";
import { type Jet, quinticDerivative, quinticHermite, quinticMaximum } from "../math/quintic.js";

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

/** The first and second derivatives of the log-posterior at a grid point, by its index. */
export type Derivatives = (q: number) => readonly [slope: number, curvature: number];

/**
 * Fourth-order differences of the first derivative over 12 h: at an end and one point in, in order from the end, and
 * centered, from two points before to two after.
 */
const END_SLOPE = [-25, 48, -36, 16, -3];
const NEAR_END_SLOPE = [-3, -10, 18, -6, 1];
const CENTRAL_SLOPE = [1, -8, 0, 8, -1];
/** Fourth-order differences of the second derivative over 12 h², in the same order. */
const END_CURVATURE = [45, -154, 214, -156, 61, -10];
const NEAR_END_CURVATURE = [10, -15, -4, 14, -6, 1];
const CENTRAL_CURVATURE = [-1, 16, -30, 16, -1];
/** The divisor of the stencils, times h or h². */
const STENCIL_DIVISOR = 12;

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
export function posteriorMean(logDensity: readonly number[], grid: readonly number[]): Ability {
  const weights = logDensity.map(Math.exp);
  const theta = average(weights, (q) => grid[q] ?? 0);
  const variance = average(weights, (q) => ((grid[q] ?? 0) - theta) ** 2);
  return { theta, standardError: Math.sqrt(variance) };
}

/**
 * Returns the posterior mode: the highest grid point locates it, and the quintic that matches the log-posterior and
 * its first two derivatives at the ends of the interval where the slope changes sign gives it between points. A mode
 * beyond the grid's ends is reported at the end.
 */
export function posteriorMode(
  logDensity: readonly number[],
  grid: readonly number[],
  derivatives: Derivatives,
): Ability {
  const top = indexOfMax(logDensity);
  const [topSlope, topCurvature] = derivatives(top);
  const left = topSlope > 0 ? top : top - 1;
  if (left < 0 || left >= grid.length - 1) {
    return { theta: grid[top] ?? 0, standardError: modeError(topCurvature) };
  }
  const start = grid[left] ?? 0;
  const h = (grid[left + 1] ?? 0) - start;
  const jet = (q: number): Jet => {
    const [slope, curvature] = derivatives(q);
    return [logDensity[q] ?? 0, h * slope, h * h * curvature];
  };
  const quintic = quinticHermite(jet(left), jet(left + 1));
  const t = quinticMaximum(quintic);
  return { theta: start + t * h, standardError: modeError(quinticDerivative(quintic, t, 2) / (h * h)) };
}

/**
 * Returns the derivatives of a log-density on a grid of spacing h, of at least 6 points, by fourth-order differences:
 * centered inside, and one-sided at the two points at each end, which the far end mirrors with the first derivative's
 * sign flipped.
 */
export function differences(logDensity: readonly number[], h: number): Derivatives {
  const last = logDensity.length - 1;
  const stencil = (coefficients: readonly number[], start: number, direction: 1 | -1): number => {
    let sum = 0;
    for (let i = 0; i < coefficients.length; i++)
      sum += (coefficients[i] ?? 0) * (logDensity[start + direction * i] ?? 0);
    return sum;
  };
  return (q) => {
    if (q >= 2 && q <= last - 2) {
      return [
        stencil(CENTRAL_SLOPE, q - 2, 1) / (STENCIL_DIVISOR * h),
        stencil(CENTRAL_CURVATURE, q - 2, 1) / (STENCIL_DIVISOR * h * h),
      ];
    }
    const direction = q < 2 ? 1 : -1;
    const atEnd = q === 0 || q === last;
    const start = direction === 1 ? 0 : last;
    return [
      (direction * stencil(atEnd ? END_SLOPE : NEAR_END_SLOPE, start, direction)) / (STENCIL_DIVISOR * h),
      stencil(atEnd ? END_CURVATURE : NEAR_END_CURVATURE, start, direction) / (STENCIL_DIVISOR * h * h),
    ];
  };
}

/** Returns 1 / √(-ℓ''), or `Infinity` where the log-posterior is not curved down, as the data then fix no error. */
function modeError(curvature: number): number {
  return curvature < 0 ? 1 / Math.sqrt(-curvature) : Number.POSITIVE_INFINITY;
}

/** A function's value and its first two derivatives at a point, with t scaled to the interval [0, 1]. */
export type Jet = readonly [value: number, first: number, second: number];

/**
 * Returns the coefficients c_0 to c_5 of the quintic p(t) = Σ c_i tⁱ that matches the jets at t = 0 and t = 1:
 * Hermite interpolation, exact for polynomials of degree 5.
 */
export function quinticHermite(start: Jet, end: Jet): Float64Array {
  const [c0, c1, curvature] = start;
  const c2 = curvature / 2;
  const value = end[0] - c0 - c1 - c2;
  const first = end[1] - c1 - 2 * c2;
  const second = end[2] - 2 * c2;
  return Float64Array.of(
    c0,
    c1,
    c2,
    10 * value - 4 * first + second / 2,
    -15 * value + 7 * first - second,
    6 * value - 3 * first + second / 2,
  );
}

/** Returns p'(t), or p''(t) with `order` 2, of the quintic with coefficients `c`. */
export function quinticDerivative(c: Float64Array, t: number, order: 1 | 2 = 1): number {
  const [, c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0] = c;
  return order === 1
    ? c1 + t * (2 * c2 + t * (3 * c3 + t * (4 * c4 + t * 5 * c5)))
    : 2 * c2 + t * (6 * c3 + t * (12 * c4 + t * 20 * c5));
}

/** The most Newton and bisection steps to the maximum of a quintic: bisection alone halves the bracket 50 times. */
const MAX_ITERATIONS = 100;

/** The width of the bracket, and the size of the slope, below which the maximum is found. */
const TOLERANCE = 1e-15;

/**
 * Returns the t in [0, 1] where p' falls through 0, for a quintic with p'(0) > 0 ≥ p'(1): Newton's method, kept inside
 * the bracket where p' changes sign by bisection.
 */
export function quinticMaximum(c: Float64Array): number {
  let [low, high] = [0, 1];
  let t = 0.5;
  for (let iteration = 0; iteration < MAX_ITERATIONS && high - low > TOLERANCE; iteration++) {
    const slope = quinticDerivative(c, t);
    if (Math.abs(slope) < TOLERANCE) break;
    if (slope > 0) low = t;
    else high = t;
    const next = t - slope / quinticDerivative(c, t, 2);
    t = next >= low && next <= high ? next : (low + high) / 2;
  }
  return t;
}

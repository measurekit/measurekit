import { type Continuized, continuize, densitySlopeSign, kernelDensity } from "./kernel-continuization.js";

/** The upper end of the search for a bandwidth, kequate's for the Gaussian kernel. */
const MAX_BANDWIDTH = 4;
/** Where the second term of the penalty looks at the density: a quarter of a score on each side, kequate's `wpen`. */
const PENALTY_OFFSET = 0.25;
/** The tolerance of the search, R's `.Machine$double.eps^0.5`, as kequate passes it to `optimize`. */
const SEARCH_TOLERANCE = Math.sqrt(Number.EPSILON);
/** A minimum this close to the upper end, 25 times the search's tolerance there, lies on the bound. */
const BOUND_DISTANCE = 1e-6;

/** A bandwidth chosen by the penalty, and whether the minimum lies inside the search's range. */
export interface BandwidthChoice {
  readonly bandwidth: number;
  readonly converged: boolean;
}

/**
 * Returns the bandwidth that minimizes the penalty PEN₁(h) + K PEN₂(h) of a distribution.
 *
 * @remarks
 * PEN₁ = Σ_j (r_j - f_h(x_j))² compares the density with the probabilities at the scores, and PEN₂ counts the scores
 * where the density's slope changes sign between x_j - 1/4 and x_j + 1/4. As kequate, Brent's search over (0, 4]
 * minimizes PEN₁ and, when K > 0, then minimizes the whole penalty above that minimum.
 */
export function penaltyBandwidth(
  probabilities: Float64Array,
  moments: [number, number],
  penaltyWeight: number,
): BandwidthChoice {
  const first = brentMinimum((h) => penalty(continuize(probabilities, moments, h), 0), 0, MAX_BANDWIDTH);
  const bandwidth =
    penaltyWeight === 0
      ? first
      : brentMinimum((h) => penalty(continuize(probabilities, moments, h), penaltyWeight), first, MAX_BANDWIDTH);
  return { bandwidth, converged: MAX_BANDWIDTH - bandwidth > BOUND_DISTANCE };
}

/** Returns PEN₁ + K PEN₂ of a continuized distribution. */
function penalty(c: Continuized, penaltyWeight: number): number {
  let squares = 0;
  let turns = 0;
  c.probabilities.forEach((p, x) => {
    const difference = p - kernelDensity(c, x);
    squares += difference * difference;
    if (penaltyWeight > 0) {
      const left = densitySlopeSign(c, x - PENALTY_OFFSET);
      const right = densitySlopeSign(c, x + PENALTY_OFFSET);
      if (left * right < 0) turns++;
    }
  });
  return squares + penaltyWeight * turns;
}

/** The squared inverse of the golden ratio, (3 - √5) / 2. */
const GOLDEN_SECTION = (3 - Math.sqrt(5)) / 2;
/** √ε, the relative tolerance of Brent's search. */
const RELATIVE_TOLERANCE = Math.sqrt(Number.EPSILON);

/**
 * The state of Brent's search: the bracket [a, b], the best point x, the second best w and the previous w, v, with
 * their values, and the last two steps d and e.
 */
interface BrentState {
  a: number;
  b: number;
  x: number;
  w: number;
  v: number;
  fx: number;
  fw: number;
  fv: number;
  d: number;
  e: number;
}

/**
 * Returns the x in (lower, upper) where f is least, by Brent's golden-section search with parabolic interpolation, as
 * R's `optimize` (Brent_fmin), so that a function with several local minima gives the minimum R finds.
 *
 * @see {@link https://stat.ethz.ch/R-manual/R-devel/library/stats/html/optimize.html | R's optimize}
 */
function brentMinimum(f: (x: number) => number, lower: number, upper: number): number {
  const start = lower + GOLDEN_SECTION * (upper - lower);
  const value = f(start);
  const state: BrentState = {
    a: lower,
    b: upper,
    x: start,
    w: start,
    v: start,
    fx: value,
    fw: value,
    fv: value,
    d: 0,
    e: 0,
  };
  for (;;) {
    const middle = (state.a + state.b) / 2;
    const tolerance = RELATIVE_TOLERANCE * Math.abs(state.x) + SEARCH_TOLERANCE / 3;
    if (Math.abs(state.x - middle) <= 2 * tolerance - (state.b - state.a) / 2) return state.x;
    const step = nextStep(state, middle, tolerance);
    const u = state.x + (Math.abs(step) >= tolerance ? step : Math.sign(step || -1) * tolerance);
    update(state, u, f(u));
  }
}

/** Returns the next step from x, by a parabola through x, w and v when it falls well inside the bracket, else golden. */
function nextStep(state: BrentState, middle: number, tolerance: number): number {
  const { a, b, x, w, v, fx, fw, fv } = state;
  let p = 0;
  let q = 0;
  let previous = 0;
  if (Math.abs(state.e) > tolerance) {
    const r = (x - w) * (fx - fv);
    q = (x - v) * (fx - fw);
    p = (x - v) * q - (x - w) * r;
    q = (q - r) * 2;
    if (q > 0) p = -p;
    else q = -q;
    previous = state.e;
    state.e = state.d;
  }
  if (Math.abs(p) >= Math.abs(q * 0.5 * previous) || p <= q * (a - x) || p >= q * (b - x)) {
    state.e = x < middle ? b - x : a - x;
    state.d = GOLDEN_SECTION * state.e;
    return state.d;
  }
  state.d = p / q;
  const u = x + state.d;
  // f is not evaluated within twice the tolerance of the ends of the bracket.
  if (u - a < 2 * tolerance || b - u < 2 * tolerance) state.d = x >= middle ? -tolerance : tolerance;
  return state.d;
}

/** Moves the bracket and the best points to take in u and its value. */
function update(state: BrentState, u: number, fu: number): void {
  if (fu <= state.fx) {
    if (u < state.x) state.b = state.x;
    else state.a = state.x;
    Object.assign(state, { v: state.w, fv: state.fw, w: state.x, fw: state.fx, x: u, fx: fu });
    return;
  }
  if (u < state.x) state.a = u;
  else state.b = u;
  if (fu <= state.fw || state.w === state.x) {
    Object.assign(state, { v: state.w, fv: state.fw, w: u, fw: fu });
  } else if (fu <= state.fv || state.v === state.x || state.v === state.w) {
    Object.assign(state, { v: u, fv: fu });
  }
}

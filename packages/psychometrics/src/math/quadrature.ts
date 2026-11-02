import { normalDensity } from "./normal.js";

/**
 * Nodes and weights of a quadrature rule: the integral of `f` is approximated by the sum of `weights[i] * f(nodes[i])`.
 */
export interface Quadrature {
  /** Nodes, in ascending order. */
  readonly nodes: Float64Array;
  /** Weights, one per node. */
  readonly weights: Float64Array;
}

const PI_TO_MINUS_QUARTER = 0.7511255444649425;
/** The rectangular rule spans [-RECTANGULAR_LIMIT, RECTANGULAR_LIMIT], mirt's default `theta_lim`. */
const RECTANGULAR_LIMIT = 6;
/**
 * The Newton step at which a root of the Hermite polynomial is exact to double precision (Numerical Recipes, gauher).
 */
const TOLERANCE = 1e-14;
/** The most Newton steps per root, Numerical Recipes' MAXIT. */
const MAX_ITERATIONS = 10;

/**
 * Returns the `count`-point Gauss-Hermite rule for the standard normal density: the sum of `weights[i] * f(nodes[i])`
 * approximates the expectation of `f(X)` for X ~ N(0, 1), exactly for polynomials up to degree 2 * count - 1. The
 * weights sum to 1.
 */
export function gaussHermite(count: number): Quadrature {
  const nodes = new Float64Array(count);
  const weights = new Float64Array(count);
  // Roots of the Hermite polynomial for the weight exp(-x^2) by Newton's method from asymptotic starting values
  // (Numerical Recipes, gauher), largest first, then rescaled to the standard normal density.
  const roots: number[] = [];
  let z = 0;
  for (let i = 0; i < Math.floor((count + 1) / 2); i++) {
    z = startingValue(i, z, count, roots);
    let derivative = 0;
    for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
      const [value, slope] = hermite(z, count);
      derivative = slope;
      const step = value / slope;
      z -= step;
      if (Math.abs(step) <= TOLERANCE) break;
    }
    roots.push(z);
    const weight = 2 / (derivative * derivative) / Math.sqrt(Math.PI);
    nodes[i] = -Math.SQRT2 * z;
    nodes[count - 1 - i] = Math.SQRT2 * z;
    weights[i] = weight;
    weights[count - 1 - i] = weight;
  }
  return { nodes, weights };
}

function startingValue(i: number, previous: number, count: number, roots: readonly number[]): number {
  const largest = roots[0] ?? 0;
  const second = roots[1] ?? 0;
  switch (i) {
    case 0:
      return Math.sqrt(2 * count + 1) - 1.85575 * (2 * count + 1) ** -0.16667;
    case 1:
      return previous - (1.14 * count ** 0.426) / previous;
    case 2:
      return 1.86 * previous - 0.86 * largest;
    case 3:
      return 1.91 * previous - 0.91 * second;
    default:
      return 2 * previous - (roots[i - 2] ?? 0);
  }
}

/** Returns the orthonormal Hermite polynomial of degree `count` at `z` and its derivative. */
function hermite(z: number, count: number): [number, number] {
  let p1 = PI_TO_MINUS_QUARTER;
  let p2 = 0;
  for (let j = 1; j <= count; j++) {
    const p3 = p2;
    p2 = p1;
    p1 = z * Math.sqrt(2 / j) * p2 - Math.sqrt((j - 1) / j) * p3;
  }
  return [p1, Math.sqrt(2 * count) * p2];
}

/**
 * Returns the `count`-point rectangular rule for the standard normal density, `count` at least 2: equally spaced nodes
 * from -6 to 6 with weights proportional to the density, summing to 1. mirt integrates with this rule by default.
 */
export function rectangular(count: number): Quadrature {
  const nodes = new Float64Array(count);
  const weights = new Float64Array(count);
  let sum = 0;
  for (let i = 0; i < count; i++) {
    const node = -RECTANGULAR_LIMIT + (2 * RECTANGULAR_LIMIT * i) / (count - 1);
    const weight = normalDensity(node);
    nodes[i] = node;
    weights[i] = weight;
    sum += weight;
  }
  for (let i = 0; i < count; i++) weights[i] = (weights[i] ?? 0) / sum;
  return { nodes, weights };
}

// The 21-point Gauss-Kronrod rule on [-1, 1] and its embedded 10-point Gauss rule, QUADPACK's qk21 rounded to doubles:
// the Kronrod nodes from the outermost to the center, their weights, and the Gauss weights of the odd-numbered nodes.
const KRONROD_NODES = [
  0.9956571630258081, 0.9739065285171717, 0.9301574913557082, 0.8650633666889845, 0.7808177265864169,
  0.6794095682990244, 0.5627571346686047, 0.4333953941292472, 0.2943928627014602, 0.14887433898163122, 0,
];
const KRONROD_WEIGHTS = [
  0.011694638867371874, 0.032558162307964725, 0.054755896574351995, 0.07503967481091996, 0.0931254545836976,
  0.10938715880229764, 0.12349197626206584, 0.13470921731147334, 0.14277593857706009, 0.14773910490133849,
  0.1494455540029169,
];
const GAUSS_WEIGHTS = [
  0.06667134430868814, 0.1494513491505806, 0.21908636251598204, 0.26926671930999635, 0.29552422471475287,
];
/** The most intervals of an adaptive integral: QUADPACK's `limit` that R's `integrate` passes, ten times over. */
const MAX_INTERVALS = 1000;

/** An interval of an adaptive integral with its estimate and the estimate's error. */
interface Interval {
  readonly lower: number;
  readonly upper: number;
  readonly value: number;
  readonly error: number;
}

/** Returns the 21-point Kronrod estimate of the integral of `f` over an interval, its error |K21 - G10| beside it. */
function kronrod(f: (x: number) => number, lower: number, upper: number): Interval {
  const center = (lower + upper) / 2;
  const half = (upper - lower) / 2;
  const middle = f(center);
  let value = (KRONROD_WEIGHTS[10] ?? 0) * middle;
  let gauss = 0;
  for (let i = 0; i < 10; i++) {
    const offset = half * (KRONROD_NODES[i] ?? 0);
    const pair = f(center - offset) + f(center + offset);
    value += (KRONROD_WEIGHTS[i] ?? 0) * pair;
    if (i % 2 === 1) gauss += (GAUSS_WEIGHTS[(i - 1) / 2] ?? 0) * pair;
  }
  return { lower, upper, value: value * half, error: Math.abs((value - gauss) * half) };
}

/**
 * Returns the integral of `f` from `lower` to `upper` by adaptive Gauss-Kronrod quadrature, as QUADPACK's qag with
 * the 21-point rule: the interval of the largest error is halved until the errors, each |K21 - G10|, sum to at most
 * `relativeTolerance` times the integral's magnitude, an interval can no longer be halved in floating point, or there
 * are 1,000 intervals.
 *
 * @remarks
 * `f` must be finite on the open interval; it is not evaluated at the ends. An integrable singularity at an end takes
 * many halvings, so a caller removes it by a change of variables first.
 */
export function adaptiveIntegral(
  f: (x: number) => number,
  lower: number,
  upper: number,
  relativeTolerance: number,
): number {
  const intervals = [kronrod(f, lower, upper)];
  let value = intervals[0]?.value ?? 0;
  let error = intervals[0]?.error ?? 0;
  while (error > relativeTolerance * Math.abs(value) && intervals.length < MAX_INTERVALS) {
    let worst = 0;
    intervals.forEach((interval, i) => {
      if (interval.error > (intervals[worst]?.error ?? 0)) worst = i;
    });
    const halved = intervals[worst];
    if (halved === undefined) break;
    const middle = (halved.lower + halved.upper) / 2;
    if (!(halved.lower < middle && middle < halved.upper) || halved.error === 0) break;
    const left = kronrod(f, halved.lower, middle);
    const right = kronrod(f, middle, halved.upper);
    intervals.splice(worst, 1, left, right);
    value += left.value + right.value - halved.value;
    error += left.error + right.error - halved.error;
  }
  return intervals.reduce((sum, interval) => sum + interval.value, 0);
}

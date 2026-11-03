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

/** The rectangular rule spans [-RECTANGULAR_LIMIT, RECTANGULAR_LIMIT], mirt's default `theta_lim`. */
const RECTANGULAR_LIMIT = 6;
/**
 * The magnitude at which the recurrence of the orthonormal polynomials divides its two latest terms by itself: at the
 * outer nodes of a rule of several hundred points they would overflow double precision.
 */
const RESCALE = 1e100;

/**
 * Returns the `count`-point Gauss-Hermite rule for the standard normal density: the sum of `weights[i] * f(nodes[i])`
 * approximates the expectation of `f(X)` for X ~ N(0, 1), exactly for polynomials up to degree 2 * count - 1. The
 * weights sum to 1.
 *
 * @remarks
 * The nodes are the eigenvalues of the Jacobi matrix of the probabilists' Hermite polynomials, 0 on the diagonal and
 * √k beside it (Golub and Welsch). Each is isolated by bisection on the Sturm count of the eigenvalues below a point,
 * which finds the k-th eigenvalue whatever the number of points, and a Newton step on the polynomial polishes it. The
 * weight of a node x is 1 / (count φ(x)²), with φ the orthonormal polynomial of degree count - 1.
 *
 * @see {@link https://doi.org/10.1090/S0025-5718-69-99647-1 | Golub and Welsch (1969)}
 * @see {@link https://doi.org/10.1007/BF02162154 | Barth, Martin and Wilkinson (1967)}
 */
export function gaussHermite(count: number): Quadrature {
  const nodes = new Float64Array(count);
  const weights = new Float64Array(count);
  for (let i = 0; i < Math.floor((count + 1) / 2); i++) {
    const start = jacobiEigenvalue(count - 1 - i, count);
    const [atStart, belowStart] = orthonormalHermite(start, count);
    const node = start - atStart / (Math.sqrt(count) * belowStart);
    const [, below, logScale] = orthonormalHermite(node, count);
    const weight = Math.exp(-Math.log(count) - 2 * (Math.log(Math.abs(below)) + logScale));
    nodes[count - 1 - i] = node;
    nodes[i] = -node;
    weights[count - 1 - i] = weight;
    weights[i] = weight;
  }
  if (count % 2 === 1) nodes[(count - 1) / 2] = 0;
  return { nodes, weights };
}

/**
 * Returns the eigenvalue of index `index`, from the smallest at 0, of the Jacobi matrix of the probabilists' Hermite
 * polynomials of degree `count`, by bisection to the rounding of the matrix's norm. Its eigenvalues lie within
 * ±2 √count, Gershgorin's bound.
 */
function jacobiEigenvalue(index: number, count: number): number {
  let lower = -2 * Math.sqrt(count);
  let upper = 2 * Math.sqrt(count);
  const width = 4 * Number.EPSILON * upper;
  for (;;) {
    const middle = (lower + upper) / 2;
    if (!(upper - lower > width && lower < middle && middle < upper)) return middle;
    if (eigenvaluesBelow(middle, count) > index) upper = middle;
    else lower = middle;
  }
}

/**
 * Returns the number of eigenvalues below `x` of the Jacobi matrix: the negative pivots of the LDLᵀ factorization of
 * the matrix less x times the identity, Sturm's count for a symmetric tridiagonal matrix.
 */
function eigenvaluesBelow(x: number, count: number): number {
  let below = 0;
  let pivot = 1;
  for (let k = 0; k < count; k++) {
    pivot = -x - (k === 0 ? 0 : k / pivot);
    // A pivot of exactly 0 is moved off it by rounding, as in the bisection of Barth, Martin and Wilkinson.
    if (pivot === 0) pivot = -Number.EPSILON * (1 + Math.abs(x));
    if (pivot < 0) below++;
  }
  return below;
}

/**
 * Returns the orthonormal probabilists' Hermite polynomials of degree `count` and `count - 1` at x, both divided by
 * e to the power of the third entry, which keeps them finite.
 */
function orthonormalHermite(x: number, count: number): readonly [number, number, number] {
  let previous = 0;
  let current = 1;
  let logScale = 0;
  for (let k = 0; k < count; k++) {
    const next = (x * current - Math.sqrt(k) * previous) / Math.sqrt(k + 1);
    previous = current;
    current = next;
    if (Math.abs(current) > RESCALE) {
      current /= RESCALE;
      previous /= RESCALE;
      logScale += Math.log(RESCALE);
    }
  }
  return [current, previous, logScale];
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

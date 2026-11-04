import { normalDensity } from "@measurekit/core";

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
 * Returns the number of eigenvalues below `x` of the Jacobi matrix: the negative pivots of the factorization L D Lᵀ of
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

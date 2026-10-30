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

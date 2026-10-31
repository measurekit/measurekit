import { logGamma } from "../math/chi-square.js";

/** The relative size of a term at which the continued fraction stops: below double precision. */
const EPSILON = 1e-16;
/** The most terms of the continued fraction, ten times Numerical Recipes' limit. */
const MAX_TERMS = 1000;
/** The smallest magnitude Lentz's algorithm lets a denominator take. */
const TINY = 1e-300;

/**
 * Returns the probability that Student's t variable with `df` degrees of freedom is at most `t`: through the
 * regularized incomplete beta function, P(T > |t|) = I_x(df / 2, 1/2) / 2 with x = df / (df + t²) (Numerical Recipes,
 * betai). Assumes `df` ≥ 1.
 */
export function studentLowerTail(t: number, df: number): number {
  if (t === 0) return 0.5;
  const x = df / (df + t * t);
  const tail = 0.5 * regularizedBeta(x, df / 2, 0.5);
  return t > 0 ? 1 - tail : tail;
}

/** Returns the regularized incomplete beta function I_x(a, b) for x from 0 to 1, by its continued fraction. */
function regularizedBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;
  const logScale = logGamma(a + b) - logGamma(a) - logGamma(b) + a * Math.log(x) + b * Math.log(1 - x);
  // The fraction converges quickly below the mean of the beta distribution; above it the symmetry I_x(a, b) = 1 -
  // I_(1-x)(b, a) is used.
  return x < (a + 1) / (a + b + 2)
    ? (Math.exp(logScale) * betaFraction(x, a, b)) / a
    : 1 - (Math.exp(logScale) * betaFraction(1 - x, b, a)) / b;
}

/** Returns the continued fraction of the incomplete beta function, with Lentz's method. */
function betaFraction(x: number, a: number, b: number): number {
  let c = 1;
  let d = 1 - ((a + b) * x) / (a + 1);
  if (Math.abs(d) < TINY) d = TINY;
  d = 1 / d;
  let fraction = d;
  for (let m = 1; m < MAX_TERMS; m++) {
    const even = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m));
    d = 1 + even * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + even / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    fraction *= d * c;
    const odd = (-(a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1));
    d = 1 + odd * d;
    if (Math.abs(d) < TINY) d = TINY;
    c = 1 + odd / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const delta = d * c;
    fraction *= delta;
    if (Math.abs(delta - 1) < EPSILON) break;
  }
  return fraction;
}

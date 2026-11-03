import { continuedFraction } from "./continued-fraction.js";

// Coefficients of the Lanczos approximation with g = 7 and 9 terms.
const LANCZOS = [
  0.9999999999998099, 676.5203681218851, -1259.1392167224028, 771.3234287776531, -176.6150291621406, 12.507343278686905,
  -0.13857109526572012, 9.984369578019572e-6, 1.5056327351493116e-7,
];
const LANCZOS_G = 7;
const HALF_LOG_TWO_PI = 0.9189385332046727;
/** The relative size of a term at which the series stops: below double precision. */
const EPSILON = 1e-16;
/** The most terms of the series beyond those {@link TERMS_PER_ROOT} gives: ten times Numerical Recipes' limit. */
const MAX_TERMS = 1000;
/** Near z = a the terms fall as exp(-n² / 2a), so that they pass below {@link EPSILON} by n = 9 √a. */
const TERMS_PER_ROOT = 9;

/**
 * Returns ln Γ(x) for x ≥ 0.5 by the Lanczos approximation, accurate to about 1e-15.
 *
 * @param x - A number of at least 0.5.
 * @returns ln Γ(x).
 */
export function logGamma(x: number): number {
  let sum = LANCZOS[0] ?? 0;
  for (let i = 1; i < LANCZOS.length; i++) sum += (LANCZOS[i] ?? 0) / (x - 1 + i);
  const t = x - 0.5 + LANCZOS_G;
  return HALF_LOG_TWO_PI + (x - 0.5) * Math.log(t) - t + Math.log(sum);
}

/**
 * Returns the probability that a chi-square variable with `df` degrees of freedom is at least `x`: the regularized
 * upper incomplete gamma function Q(df / 2, x / 2), from its series below df / 2 + 1 and its continued fraction above
 * (Numerical Recipes, gammq).
 *
 * @param x - The statistic, at least 0.
 * @param df - The degrees of freedom, at least 1.
 * @returns The upper tail probability, from 0 to 1.
 * @see {@link https://numerical.recipes | Press et al. (2007), section 6.2}
 */
export function chiSquareUpperTail(x: number, df: number): number {
  const a = df / 2;
  const z = x / 2;
  if (z === 0) return 1;
  // An infinite statistic, as a likelihood ratio against a fit of probability 0 gives, has no tail; its log scale
  // would be ∞ - ∞.
  if (z === Number.POSITIVE_INFINITY) return 0;
  const logScale = a * Math.log(z) - z - logGamma(a);
  if (z < a + 1) return 1 - lowerSeries(a, z, logScale);
  const fraction = continuedFraction(
    (n) => -n * (n - a),
    (n) => z + 1 - a + 2 * n,
  );
  return Math.exp(logScale) * fraction;
}

/** Returns the regularized lower incomplete gamma function P(a, z) by its series. */
function lowerSeries(a: number, z: number, logScale: number): number {
  let term = 1 / a;
  let sum = term;
  const terms = MAX_TERMS + TERMS_PER_ROOT * Math.sqrt(a);
  for (let n = 1; n < terms && Math.abs(term) > Math.abs(sum) * EPSILON; n++) {
    term *= z / (a + n);
    sum += term;
  }
  return sum * Math.exp(logScale);
}

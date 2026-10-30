// Coefficients of the Lanczos approximation with g = 7 and 9 terms.
const LANCZOS = [
  0.9999999999998099, 676.5203681218851, -1259.1392167224028, 771.3234287776531, -176.6150291621406, 12.507343278686905,
  -0.13857109526572012, 9.984369578019572e-6, 1.5056327351493116e-7,
];
const LANCZOS_G = 7;
const HALF_LOG_TWO_PI = 0.9189385332046727;
/** The relative size of a term at which the series and the continued fraction stop: below double precision. */
const EPSILON = 1e-16;
/** The most terms of the series and the continued fraction, ten times Numerical Recipes' limit. */
const MAX_TERMS = 1000;
/** The smallest magnitude Lentz's algorithm lets a denominator take. */
const TINY = 1e-300;

/** Returns ln Γ(x) for x ≥ 0.5 by the Lanczos approximation, accurate to about 1e-15. */
function logGamma(x: number): number {
  let sum = LANCZOS[0] ?? 0;
  for (let i = 1; i < LANCZOS.length; i++) sum += (LANCZOS[i] ?? 0) / (x - 1 + i);
  const t = x - 0.5 + LANCZOS_G;
  return HALF_LOG_TWO_PI + (x - 0.5) * Math.log(t) - t + Math.log(sum);
}

/**
 * Returns the probability that a chi-square variable with `df` degrees of freedom is at least `x`: the regularized
 * upper incomplete gamma function Q(df / 2, x / 2), from its series below df / 2 + 1 and its continued fraction above
 * (Numerical Recipes, gammq). Assumes `x` ≥ 0 and `df` ≥ 1.
 */
export function chiSquareUpperTail(x: number, df: number): number {
  const a = df / 2;
  const z = x / 2;
  if (z === 0) return 1;
  const logScale = a * Math.log(z) - z - logGamma(a);
  return z < a + 1 ? 1 - lowerSeries(a, z, logScale) : upperFraction(a, z, logScale);
}

/** Returns the regularized lower incomplete gamma function P(a, z) by its series. */
function lowerSeries(a: number, z: number, logScale: number): number {
  let term = 1 / a;
  let sum = term;
  for (let n = 1; n < MAX_TERMS && Math.abs(term) > Math.abs(sum) * EPSILON; n++) {
    term *= z / (a + n);
    sum += term;
  }
  return sum * Math.exp(logScale);
}

/** Returns the regularized upper incomplete gamma function Q(a, z) by its continued fraction, with Lentz's method. */
function upperFraction(a: number, z: number, logScale: number): number {
  let b = z + 1 - a;
  let c = 1 / TINY;
  let d = 1 / b;
  let fraction = d;
  for (let n = 1; n < MAX_TERMS; n++) {
    const an = -n * (n - a);
    b += 2;
    d = an * d + b;
    if (Math.abs(d) < TINY) d = TINY;
    c = b + an / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const delta = d * c;
    fraction *= delta;
    if (Math.abs(delta - 1) <= EPSILON) break;
  }
  return Math.exp(logScale) * fraction;
}

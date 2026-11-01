import { logGamma } from "./chi-square.js";
import { continuedFraction } from "./continued-fraction.js";

/**
 * Returns the probability that Student's t variable with `df` degrees of freedom is at most `t`: through the
 * regularized incomplete beta function, P(T > |t|) = I_x(df / 2, 1/2) / 2 with x = df / (df + t²) (Numerical Recipes,
 * betai).
 *
 * @param t - The statistic.
 * @param df - The degrees of freedom, at least 1.
 * @see {@link https://numerical.recipes | Press et al. (2007), section 6.4}
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

/**
 * Returns the continued fraction of the incomplete beta function, 1 / (1 + d₁ / (1 + d₂ / ...)): d₂ₘ is its even term
 * and d₂ₘ₊₁ its odd term.
 */
function betaFraction(x: number, a: number, b: number): number {
  return continuedFraction(
    (n) => {
      const m = Math.floor(n / 2);
      if (n % 2 === 0) return (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m));
      // The first term has the factor a in its numerator and its denominator. Cancelled, it is rounded once, which
      // matters where 1 + d₁ is small, for x near 1.
      if (m === 0) return (-(a + b) * x) / (a + 1);
      return (-(a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1));
    },
    () => 1,
  );
}

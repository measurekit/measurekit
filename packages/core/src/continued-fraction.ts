/** The relative size of a term at which a continued fraction stops: below double precision. */
const EPSILON = 1e-16;
/** The most terms: ten times Numerical Recipes' limit of 100 steps, which count two terms each for the beta fraction. */
const MAX_TERMS = 2000;
/** The smallest magnitude Lentz's algorithm lets a denominator take. */
const TINY = 1e-300;

/**
 * Returns 1 / (b₀ + a₁ / (b₁ + a₂ / (b₂ + ...))) by the modified Lentz method, the fraction of the incomplete gamma and
 * beta functions.
 *
 * @remarks
 * The fraction stops when two terms in a row change it by a relative 1e-16 at most, as the two halves of a step of the
 * beta fraction do, or after 2,000 terms, and then returns what it has. `b₀` must not be 0.
 *
 * @param numerator - Returns aₙ for n from 1.
 * @param denominator - Returns bₙ for n from 0.
 * @see {@link https://numerical.recipes | Press et al. (2007), section 5.2}
 */
export function continuedFraction(numerator: (n: number) => number, denominator: (n: number) => number): number {
  let c = 1 / TINY;
  let d = 1 / denominator(0);
  let fraction = d;
  let settled = 0;
  for (let n = 1; n < MAX_TERMS; n++) {
    const a = numerator(n);
    const b = denominator(n);
    d = a * d + b;
    if (Math.abs(d) < TINY) d = TINY;
    c = b + a / c;
    if (Math.abs(c) < TINY) c = TINY;
    d = 1 / d;
    const delta = d * c;
    fraction *= delta;
    settled = Math.abs(delta - 1) <= EPSILON ? settled + 1 : 0;
    if (settled === 2) break;
  }
  return fraction;
}

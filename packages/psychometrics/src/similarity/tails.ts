import { logGamma } from "@measurekit/core";

/**
 * Returns P(low ≤ X ≤ high) for X binomial with `trials` trials of probability `p`.
 *
 * @remarks
 * The terms are summed outward from the largest one in the range, whose logarithm comes from ln Γ, so that a tail of
 * 1e-300 keeps its digits and no sum is 1 less a tail, which R's `1 - pbinom(m - 1, n, p)` rounds to 0 below about
 * 1e-16.
 */
export function binomialRange(low: number, high: number, trials: number, p: number): number {
  const from = Math.max(low, 0);
  const to = Math.min(high, trials);
  if (from > to) return 0;
  if (p <= 0) return from === 0 ? 1 : 0;
  if (p >= 1) return to === trials ? 1 : 0;
  const start = Math.min(Math.max(Math.floor((trials + 1) * p), from), to);
  const odds = p / (1 - p);
  const peak = Math.exp(
    logGamma(trials + 1) -
      logGamma(start + 1) -
      logGamma(trials - start + 1) +
      start * Math.log(p) +
      (trials - start) * Math.log1p(-p),
  );
  let sum = peak;
  let term = peak;
  for (let k = start; k < to && term > 0; k++) {
    term *= ((trials - k) / (k + 1)) * odds;
    sum += term;
  }
  term = peak;
  for (let k = start; k > from && term > 0; k--) {
    term *= k / ((trials - k + 1) * odds);
    sum += term;
  }
  return Math.min(sum, 1);
}

/**
 * Returns P(low ≤ X ≤ high) for X Poisson of mean `mean`, summed outward from the largest term in the range, as
 * {@link binomialRange} does.
 */
export function poissonRange(low: number, high: number, mean: number): number {
  const from = Math.max(low, 0);
  if (from > high) return 0;
  if (mean <= 0) return from === 0 ? 1 : 0;
  const start = Math.min(Math.max(Math.floor(mean), from), high);
  const peak = Math.exp(start * Math.log(mean) - mean - logGamma(start + 1));
  let sum = peak;
  let term = peak;
  for (let k = start; k < high && term > 0; k++) {
    term *= mean / (k + 1);
    sum += term;
  }
  term = peak;
  for (let k = start; k > from && term > 0; k--) {
    term *= k / mean;
    sum += term;
  }
  return Math.min(sum, 1);
}

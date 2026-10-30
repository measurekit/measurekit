import type { BktGuessSlip, BktLearning } from "./parameters.js";

/** Returns P(correct) of an attempt, given P(L): P(L) (1 - P(S)) + (1 - P(L)) P(G). */
export function correctness(mastery: number, { guess, slip }: BktGuessSlip): number {
  return mastery * (1 - slip) + (1 - mastery) * guess;
}

/**
 * Returns P(L) at the next attempt: P(L) conditioned on the outcome, then moved by the learn and forget rates. Returns
 * `null` when the outcome has probability 0, as a correct answer has with P(L) = 0 and P(G) = 0.
 */
export function nextMastery(
  mastery: number,
  outcome: number,
  { guess, slip }: BktGuessSlip,
  { learn, forget }: BktLearning,
): number | null {
  const known = mastery * (outcome === 1 ? 1 - slip : slip);
  const unknown = (1 - mastery) * (outcome === 1 ? guess : 1 - guess);
  const total = known + unknown;
  if (!(total > 0)) return null;
  // The two terms are shares of 1, which rounding can push past it.
  return Math.min(1, (unknown / total) * learn + (known / total) * (1 - forget));
}

/**
 * Returns the number of further attempts after which P(L), as expected before their outcomes, reaches `threshold`.
 *
 * @remarks
 * The least n with limit + (P(L) - limit) (1 - P(T) - P(F))ⁿ ≥ threshold, where limit = P(T) / (P(T) + P(F)). Before an
 * outcome is seen, the expected P(L) after it is P(L) moved by the learn and forget rates alone. Returns 0 when P(L)
 * has reached the threshold, and `Infinity` when it never will or the number exceeds the largest double.
 */
export function opportunitiesToMastery(mastery: number, { learn, forget }: BktLearning, threshold: number): number {
  if (mastery >= threshold) return 0;
  if (learn === 0) return Number.POSITIVE_INFINITY;
  const limit = learn / (learn + forget);
  const ratio = 1 - learn - forget;
  // With P(T) + P(F) ≥ 1, one attempt reaches or passes the limit, and later ones only close in on it.
  if (ratio <= 0) return limit + (mastery - limit) * ratio >= threshold ? 1 : Number.POSITIVE_INFINITY;
  if (limit <= threshold) return Number.POSITIVE_INFINITY;
  // ln(1 - P(T) - P(F)) by log1p: 1 - P(T) - P(F) loses the rates' digits and rounds to 1 below about 1e-16.
  const logRatio = Math.log1p(-(learn + forget));
  const after = (n: number): number => limit + (mastery - limit) * Math.exp(n * logRatio);
  const n = Math.max(1, Math.ceil(Math.log((limit - threshold) / (limit - mastery)) / logRatio));
  // The logarithms round by less than an attempt; the expected P(L) itself decides between neighbors.
  if (after(n) < threshold) return n + 1;
  return n > 1 && after(n - 1) >= threshold ? n - 1 : n;
}

import { checkFinite, checkRecord, DataError } from "@measurekit/core";
import { checkScoreReliability, type ScoreReliability } from "./score-reliability.js";

/** The scores of two occasions that decide the reliability of their difference. */
export interface DifferenceReliabilityInput {
  /** The scores of the first occasion. */
  readonly pre: ScoreReliability;
  /** The scores of the second occasion. */
  readonly post: ScoreReliability;
  /** The correlation of the pre and post scores, from -1 to 1. */
  readonly correlation: number;
}

/**
 * Returns the reliability of the difference between two scores, which Lord's formula gives from the reliabilities,
 * standard deviations and correlation of the scores, or `null` when the difference has no variance.
 *
 * @remarks
 * With standard deviations s1 and s2, reliabilities r1 and r2 and correlation rho, the reliability is
 * (s1² r1 + s2² r2 - 2 rho s1 s2) / (s1² + s2² - 2 rho s1 s2). With equal standard deviations and reliabilities r it is
 * (r - rho) / (1 - rho): it falls as the scores correlate more, is 0 where they correlate as highly as they are
 * reliable, and is negative above that. A negative value says the difference measures nothing, and it is returned as
 * computed. The difference has no variance, and the result is `null`, only when the standard deviations are equal and
 * the correlation is 1.
 *
 * Valid input: positive standard deviations, reliabilities from 0 to 1 and a correlation from -1 to 1.
 *
 * @param input - The standard deviation and reliability of each occasion, and their correlation.
 * @returns The reliability of the difference, at most 1; `null` where it is undefined.
 *
 * @example
 * ```ts
 * import { differenceReliability } from "@measurekit/psychometrics/change";
 *
 * differenceReliability({
 *   pre: { sd: 10, reliability: 0.9 },
 *   post: { sd: 12, reliability: 0.85 },
 *   correlation: 0.5,
 * }); // 0.7451612903225805
 * ```
 * @throws `DataError` when the input is not an object `{ pre, post, correlation }` of valid values.
 * @see {@link https://doi.org/10.1177/001316445601600401 | Lord (1956)}
 * @see {@link https://doi.org/10.1037/h0029382 | Cronbach and Furby (1970)}
 */
export function differenceReliability(input: DifferenceReliabilityInput): number | null {
  checkRecord(input, "The input must be an object { pre, post, correlation }");
  const { pre, post, correlation } = input;
  checkScoreReliability(pre, "pre scores");
  checkScoreReliability(post, "post scores");
  checkFinite(correlation, "The correlation");
  if (Math.abs(correlation) > 1) {
    throw new DataError(`The correlation must be a number from -1 to 1, found ${correlation}.`);
  }
  // The variance of the difference as a sum of non-negative terms, which is 0 exactly when it should be: the expanded
  // form loses all its digits for equal deviations and a correlation near 1.
  const variance = (pre.sd - post.sd) ** 2 + 2 * (1 - correlation) * pre.sd * post.sd;
  if (variance === 0) return null;
  return (
    (pre.sd ** 2 * pre.reliability + post.sd ** 2 * post.reliability - 2 * correlation * pre.sd * post.sd) / variance
  );
}

import { checkPairedScores } from "./paired-scores.js";

/**
 * Returns each person's gain: the post score minus the pre score.
 *
 * @remarks
 * Valid input: the pre and post scores of the same persons, in the same order, at least one person, and finite
 * scores. A positive gain is an increase. The reliability of a gain falls as the pre and post scores correlate more, and
 * a gain depends on the pre score ({@link residualGain} removes that dependence): see `differenceReliability`.
 *
 * @param pre - The scores of the first occasion.
 * @param post - The scores of the second occasion, one for each person.
 * @returns The gain of each person.
 *
 * @example
 * ```ts
 * import { simpleGain } from "@measurekit/psychometrics/change";
 *
 * simpleGain([12, 20, 31], [18, 19, 40]); // => [6, -1, 9]
 * ```
 * @throws `DataError` when the scores are not arrays of the same length with at least one finite number each.
 * @see {@link https://doi.org/10.1037/h0029382 | Cronbach and Furby (1970)}
 */
export function simpleGain(pre: readonly number[], post: readonly number[]): readonly number[] {
  checkPairedScores(pre, post, 1);
  return post.map((score, person) => score - (pre[person] ?? 0));
}

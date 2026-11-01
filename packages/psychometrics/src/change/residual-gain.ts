import { DataError, mean } from "@measurekit/core";
import { checkPairedScores } from "./paired-scores.js";

/**
 * Returns each person's residual gain: the post score minus the post score that least squares predicts from the pre
 * score.
 *
 * @remarks
 * The residuals of the regression of the post scores on the pre scores, as `residuals(lm(post ~ pre))` of R. They sum
 * to 0 and are uncorrelated with the pre scores, so a residual gain says how far a person is above or below the
 * persons who started alike. It is relative to the group: the same person has another residual gain in another group.
 *
 * Valid input: the pre and post scores of the same persons, in the same order, at least two persons, finite scores,
 * and pre scores that are not all equal, as the regression has no slope then.
 *
 * @throws `DataError` when the scores are not arrays of the same length with at least two finite numbers each, or the
 * pre scores are all equal.
 * @see {@link https://doi.org/10.1037/h0029382 | Cronbach and Furby (1970)}
 */
export function residualGain(pre: readonly number[], post: readonly number[]): readonly number[] {
  checkPairedScores(pre, post, 2);
  const preMean = mean(pre);
  const postMean = mean(post);
  let preSquares = 0;
  let products = 0;
  pre.forEach((score, person) => {
    const centered = score - preMean;
    preSquares += centered * centered;
    products += centered * ((post[person] ?? 0) - postMean);
  });
  if (preSquares === 0) {
    throw new DataError(`The pre scores are all ${preMean}, so the post scores cannot be regressed on them.`);
  }
  const slope = products / preSquares;
  return post.map((score, person) => score - postMean - slope * ((pre[person] ?? 0) - preMean));
}

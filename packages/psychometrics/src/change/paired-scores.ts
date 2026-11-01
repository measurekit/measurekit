import { checkArray, checkFinite, DataError } from "@measurekit/core";

/**
 * Checks the scores of the same persons on two occasions: arrays of the same length, with at least `minimum` persons,
 * each score a finite number.
 *
 * @throws `DataError` when they are not.
 */
export function checkPairedScores(pre: readonly number[], post: readonly number[], minimum: number): void {
  checkArray(pre, "The pre scores must be an array of scores");
  checkArray(post, "The post scores must be an array of scores");
  if (pre.length !== post.length) {
    throw new DataError(
      `The pre and post scores must come in pairs, found ${pre.length} pre scores and ${post.length} post scores.`,
    );
  }
  if (pre.length < minimum) {
    throw new DataError(
      `The scores need at least ${minimum} ${minimum === 1 ? "person" : "persons"}, found ${pre.length}.`,
    );
  }
  pre.forEach((score, person) => {
    checkFinite(score, `The pre score of person ${person + 1}`);
    checkFinite(post[person], `The post score of person ${person + 1}`);
  });
}

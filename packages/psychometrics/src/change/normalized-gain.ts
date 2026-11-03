import { checkFinite, checkOptions, DataError } from "@measurekit/core";
import { checkPairedScores } from "./paired-scores.js";

/** Options of {@link normalizedGain}. */
export interface NormalizedGainOptions {
  /** The highest possible score of the test, which no score exceeds. */
  readonly maximum: number;
}

/** The normalized gain of a group and of each of its persons. */
export interface NormalizedGain {
  /** The gain of the group's means, ⟨g⟩ of Hake; `null` when everyone's pre score is the maximum. */
  readonly group: number | null;
  /** Each person's gain; `null` for a person whose pre score is the maximum. */
  readonly persons: readonly (number | null)[];
}

/**
 * Returns the normalized gain of a group, and of each person: the share of the possible gain that was made.
 *
 * @remarks
 * The group's gain is ⟨g⟩ = (mean post - mean pre) / (maximum - mean pre), from the means of the persons, as Hake
 * defines it: it is not the mean of the persons' gains. A person's gain is (post - pre) / (maximum - pre), which is 1
 * at the maximum, negative for a decline and undefined for a person who started at the maximum and so could not gain:
 * that person is `null`. The group's gain is `null` only when everyone started at the maximum.
 *
 * Valid input: the pre and post scores of the same persons, in the same order, at least one person, finite scores
 * that do not exceed the maximum.
 *
 * @param pre - The scores of the first occasion.
 * @param post - The scores of the second occasion, one for each person.
 * @param options - The highest possible score.
 * @returns The gain of the group and of each person; `null` where a gain is undefined.
 *
 * @example
 * ```ts
 * import { normalizedGain } from "@measurekit/psychometrics/change";
 *
 * normalizedGain([10, 30, 50], [30, 40, 50], { maximum: 50 }); // { group: 0.5, persons: [0.5, 0.5, null] }
 * ```
 * @throws `DataError` when the scores are not arrays of the same length with at least one finite number each, the
 * options are not an object, the maximum is not a finite number, or a score exceeds it.
 * @see {@link https://doi.org/10.1119/1.18809 | Hake (1998)}
 */
export function normalizedGain(
  pre: readonly number[],
  post: readonly number[],
  options: NormalizedGainOptions,
): NormalizedGain {
  checkPairedScores(pre, post, 1);
  checkOptions(options);
  const { maximum } = options;
  checkFinite(maximum, "The option maximum");
  pre.forEach((score, person) => {
    checkAtMost(score, maximum, `The pre score of person ${person + 1}`);
    checkAtMost(post[person] ?? 0, maximum, `The post score of person ${person + 1}`);
  });
  let gained = 0;
  let possible = 0;
  const persons = pre.map((score, person) => {
    const change = (post[person] ?? 0) - score;
    gained += change;
    possible += maximum - score;
    return score === maximum ? null : change / (maximum - score);
  });
  // (mean post - mean pre) / (maximum - mean pre) as sums over the persons: each maximum - pre is above 0 unless the
  // person started at the maximum, so the denominator is 0 only when everyone did, where the mean of pre scores just
  // below the maximum can round to it and leave 0 / 0.
  return { group: possible > 0 ? gained / possible : null, persons };
}

/** Checks that a score does not exceed the maximum. */
function checkAtMost(score: number, maximum: number, label: string): void {
  if (score > maximum) throw new DataError(`${label} is ${score}, above the maximum ${maximum}.`);
}

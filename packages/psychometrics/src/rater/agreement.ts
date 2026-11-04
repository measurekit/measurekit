import { checkNonNegative, checkOptions, DataError, describe, type FieldNames } from "@measurekit/core";
import { completeRatings } from "./ratings.js";

/** Options of {@link adjacentAgreement}. */
export interface AdjacentAgreementOptions {
  /**
   * The widest distance, in categories, between a subject's highest and lowest rating that counts as agreement: a
   * non-negative integer. With 0 the agreement is exact.
   *
   * @defaultValue `1`
   */
  readonly width?: number;
}

const ADJACENT_AGREEMENT_OPTIONS: FieldNames<AdjacentAgreementOptions> = { width: true };

const DEFAULT_WIDTH = 1;

/**
 * Returns the share of subjects on which every rater gave the same rating.
 *
 * @remarks
 * As irr's `agree` with `tolerance = 0`, as a proportion rather than a percentage. Uses the subjects every rater rated.
 * Ratings are categories, integers from 0. With two raters it is Cohen's p_o; with more, every rater must agree.
 *
 * @param ratings - The ratings, one row for each subject and one entry for each rater; `null` when missing.
 * @returns The share of subjects, from 0 to 1.
 * @throws `DataError` when there are fewer than two raters, a rating is not a category or `null`, or no subject was
 * rated by every rater.
 */
export function exactAgreement(ratings: readonly (readonly (number | null)[])[]): number {
  return agreement(ratings, 0, "Exact agreement");
}

/**
 * Returns the share of subjects whose ratings all lie within `width` categories of each other: the highest rating less
 * the lowest is at most `width`.
 *
 * @remarks
 * As irr's `agree` with `tolerance = width`, as a proportion rather than a percentage; with a width of 1, the adjacent
 * agreement used to evaluate automated scoring. Uses the subjects every rater rated. Ratings are categories, integers
 * from 0. With two raters, a subject agrees when the raters differ by at most `width`; with more, when no two raters
 * do by more.
 *
 * @param ratings - The ratings, one row for each subject and one entry for each rater; `null` when missing.
 * @param options - The width, in categories.
 * @returns The share of subjects, from 0 to 1.
 * @throws `DataError` when `width` is not a non-negative integer, there are fewer than two raters, a rating is not a
 * category or `null`, or no subject was rated by every rater.
 */
export function adjacentAgreement(
  ratings: readonly (readonly (number | null)[])[],
  options: AdjacentAgreementOptions = {},
): number {
  checkOptions(options, ADJACENT_AGREEMENT_OPTIONS);
  const { width = DEFAULT_WIDTH } = options;
  checkNonNegative(width, "The option width");
  if (!Number.isSafeInteger(width)) {
    throw new DataError(`The option width must be a non-negative integer, found ${describe(width)}.`);
  }
  return agreement(ratings, width, "Adjacent agreement");
}

/** Returns the share of the subjects rated by every rater whose ratings span at most `width` categories. */
function agreement(ratings: readonly (readonly (number | null)[])[], width: number, statistic: string): number {
  const rows = completeRatings(ratings, statistic, { categories: true, minRaters: 2, minSubjects: 1 });
  let agreeing = 0;
  for (const row of rows) {
    let lowest = Number.POSITIVE_INFINITY;
    let highest = Number.NEGATIVE_INFINITY;
    for (const rating of row) {
      lowest = Math.min(lowest, rating);
      highest = Math.max(highest, rating);
    }
    if (highest - lowest <= width) agreeing++;
  }
  return agreeing / rows.length;
}

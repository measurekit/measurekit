import { DataError, describe } from "@measurekit/core";

/**
 * Checks one subject's row of ratings.
 *
 * @throws `DataError` when it is not an array of one rating per rater, each valid or `null`.
 */
function checkRow(row: readonly (number | null)[], subject: number, raters: number, categories: boolean): void {
  if (!(Array.isArray(row) && row.length === raters)) {
    throw new DataError(`Subject ${subject + 1} needs ${raters} ratings, one per rater.`);
  }
  for (const [rater, rating] of row.entries()) {
    const valid = categories ? Number.isSafeInteger(rating) && (rating ?? -1) >= 0 : Number.isFinite(rating);
    if (rating !== null && !valid) {
      const kind = categories ? "a category, an integer from 0," : "a finite number";
      throw new DataError(
        `The rating of subject ${subject + 1} by rater ${rater + 1} must be ${kind} or null, found ${describe(rating)}.`,
      );
    }
  }
}

/**
 * Checks ratings, one row per subject with one rating per rater, `null` when missing, and returns the rows of the
 * subjects that every rater rated.
 *
 * @param statistic - Names the statistic in error messages.
 * @param categories - Whether ratings are categories, integers from 0, rather than any finite numbers.
 * @throws `DataError` when the rows are not arrays of the same length, with at least `minRaters` raters, a rating is
 * not valid, or fewer than `minSubjects` subjects were rated by every rater.
 */
export function completeRatings(
  ratings: readonly (readonly (number | null)[])[],
  statistic: string,
  options: {
    readonly categories: boolean;
    readonly minRaters: number;
    readonly maxRaters?: number;
    readonly minSubjects: number;
  },
): number[][] {
  const { categories, minRaters, maxRaters = Number.POSITIVE_INFINITY, minSubjects } = options;
  // Ratings from JSON may not be an array; checking a copy keeps the parameter's type.
  const input: unknown = ratings;
  if (!Array.isArray(input)) throw new DataError(`${statistic} needs an array of ratings, one row per subject.`);
  const raters = Array.isArray(ratings[0]) ? ratings[0].length : 0;
  if (raters < minRaters || raters > maxRaters) {
    const expected = maxRaters === minRaters ? `${minRaters}` : `at least ${minRaters}`;
    throw new DataError(`${statistic} needs ${expected} raters, found ${raters}.`);
  }
  const complete: number[][] = [];
  ratings.forEach((row, subject) => {
    checkRow(row, subject, raters, categories);
    if (!row.includes(null)) complete.push(row.map((rating) => rating ?? 0));
  });
  if (complete.length < minSubjects) {
    throw new DataError(
      `${statistic} needs at least ${minSubjects} subjects rated by every rater, found ${complete.length}.`,
    );
  }
  return complete;
}

/**
 * Checks ratings by two raters and returns their columns over the subjects both rated: the ratings of the first rater
 * and of the second.
 *
 * @param statistic - Names the statistic in error messages.
 * @throws `DataError` when the ratings do not have two raters, a rating is not a finite number or `null`, or fewer than
 * two subjects were rated by both.
 */
export function pairOfRaters(
  ratings: readonly (readonly (number | null)[])[],
  statistic: string,
): { readonly first: Float64Array; readonly second: Float64Array } {
  const rows = completeRatings(ratings, statistic, { categories: false, minRaters: 2, maxRaters: 2, minSubjects: 2 });
  return {
    first: Float64Array.from(rows, ([rating = 0]) => rating),
    second: Float64Array.from(rows, ([, rating = 0]) => rating),
  };
}

/**
 * Returns whether every rating equals the first. The variance of constant ratings is 0 only when the sums are exact,
 * which they are for whole numbers but not for 0.7 repeated, so a constant rater is found by comparing the ratings.
 */
export function isConstant(ratings: Float64Array): boolean {
  return ratings.every((rating) => rating === ratings[0]);
}

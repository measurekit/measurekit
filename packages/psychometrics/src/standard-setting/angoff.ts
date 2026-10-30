import { checkArray, checkNonNegative, DataError, mean } from "@measurekit/core";
import { standardDeviation } from "../math/statistics.js";

/** A modified Angoff cut score with its spread across the panelists. */
export interface AngoffCutScore {
  /** The cut score on the total score scale: the mean of the panelists' cut scores. */
  readonly cutScore: number;
  /** Each panelist's cut score, in panelist order: the sum of the panelist's ratings. */
  readonly panelistCutScores: readonly number[];
  /** The standard deviation of the panelists' cut scores (denominator n - 1); `null` with one panelist. */
  readonly sd: number | null;
  /** The standard error of the cut score, sd / √n for n panelists; `null` with one panelist. */
  readonly standardError: number | null;
}

/**
 * Returns the modified Angoff cut score: each panelist rates each item with the expected score of a borderline person,
 * for a dichotomous item the probability of a correct response; a panelist's cut score is the sum of the panelist's
 * ratings, and the panel's cut score is their mean.
 *
 * @param ratings - Ratings by panelist, then item.
 * @throws `DataError` without panelists or items, when the panelists rate different numbers of items, or a rating is
 * not a non-negative number.
 * @see {@link https://openlibrary.org/works/OL3700878W | Angoff (1971)}
 */
export function angoffCutScore(ratings: readonly (readonly number[])[]): AngoffCutScore {
  checkArray(ratings, "Angoff ratings must be an array of rows, one per panelist");
  const itemCount = ratings[0]?.length ?? 0;
  if (itemCount === 0) throw new DataError("An Angoff cut score needs ratings of at least one item by one panelist.");
  const cutScores = ratings.map((row, panelist) => {
    checkArray(row, `The ratings of panelist ${panelist + 1} must be an array`);
    if (row.length !== itemCount) {
      throw new DataError(`Panelist ${panelist + 1} has ${row.length} ratings, panelist 1 has ${itemCount}.`);
    }
    return row.reduce((sum, rating, item) => {
      checkNonNegative(rating, `The rating of item ${item + 1} by panelist ${panelist + 1}`);
      return sum + rating;
    }, 0);
  });
  const values = Float64Array.from(cutScores);
  const sd = values.length >= 2 ? standardDeviation(values) : null;
  return {
    cutScore: mean(values),
    panelistCutScores: cutScores,
    sd,
    standardError: sd === null ? null : sd / Math.sqrt(values.length),
  };
}

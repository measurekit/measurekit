import { completeRatings } from "./ratings.js";

/**
 * The intraclass correlations of ratings in the forms of Shrout and Fleiss and McGraw and Wong: of a single rater's
 * ratings and of the mean of the raters'. A coefficient is `null` where its denominator is 0.
 *
 * @see {@link https://doi.org/10.1037/0033-2909.86.2.420 | Shrout and Fleiss (1979)}
 * @see {@link https://doi.org/10.1037/1082-989X.1.1.30 | McGraw and Wong (1996)}
 */
export interface IntraclassCorrelation {
  /** ICC(1): one-way random effects, a single rater. */
  readonly oneWay: number | null;
  /** ICC(A,1), Shrout and Fleiss' ICC(2,1): two-way, absolute agreement, a single rater. */
  readonly agreement: number | null;
  /** ICC(C,1), Shrout and Fleiss' ICC(3,1): two-way, consistency, a single rater. */
  readonly consistency: number | null;
  /** ICC(k): one-way random effects, the mean of k raters. */
  readonly oneWayAverage: number | null;
  /** ICC(A,k): two-way, absolute agreement, the mean of k raters. */
  readonly agreementAverage: number | null;
  /** ICC(C,k): two-way, consistency, the mean of k raters. */
  readonly consistencyAverage: number | null;
}

/**
 * Returns the intraclass correlations of ratings, one row per subject with one rating per rater, from the mean squares
 * of subjects MS_R, raters MS_C, the residual MS_E and within subjects MS_W, as irr's `icc` and psych's `ICC`.
 *
 * @remarks
 * ICC(1) = (MS_R - MS_W) / (MS_R + (k - 1) MS_W), ICC(A,1) = (MS_R - MS_E) / (MS_R + (k - 1) MS_E + k (MS_C - MS_E) /
 * n), ICC(C,1) = (MS_R - MS_E) / (MS_R + (k - 1) MS_E), and for the mean of k raters (MS_R - MS_W) / MS_R, (MS_R -
 * MS_E) / (MS_R + (MS_C - MS_E) / n) and (MS_R - MS_E) / MS_R. Ratings are finite numbers. Uses the subjects every
 * rater rated.
 *
 * @throws `DataError` when there are fewer than two raters, a rating is not a finite number or `null`, or fewer
 * than two subjects were rated by every rater.
 */
export function intraclassCorrelation(ratings: readonly (readonly (number | null)[])[]): IntraclassCorrelation {
  const rows = completeRatings(ratings, "The intraclass correlation", {
    categories: false,
    minRaters: 2,
    minSubjects: 2,
  });
  const n = rows.length;
  const k = rows[0]?.length ?? 0;
  let grand = 0;
  for (const row of rows) for (const rating of row) grand += rating;
  grand /= n * k;
  const raterMeans = new Float64Array(k);
  for (const row of rows) {
    for (const [j, rating] of row.entries()) raterMeans[j] = (raterMeans[j] ?? 0) + rating / n;
  }
  let subjects = 0;
  let within = 0;
  let total = 0;
  for (const row of rows) {
    const mean = row.reduce((sum, rating) => sum + rating, 0) / k;
    subjects += (mean - grand) ** 2;
    for (const rating of row) {
      within += (rating - mean) ** 2;
      total += (rating - grand) ** 2;
    }
  }
  let raters = 0;
  for (const mean of raterMeans) raters += (mean - grand) ** 2;
  const msr = (k * subjects) / (n - 1);
  const msw = within / (n * (k - 1));
  const msc = (n * raters) / (k - 1);
  const mse = (total - k * subjects - n * raters) / ((n - 1) * (k - 1));
  const ratio = (numerator: number, denominator: number): number | null =>
    denominator === 0 ? null : numerator / denominator;
  return {
    oneWay: ratio(msr - msw, msr + (k - 1) * msw),
    agreement: ratio(msr - mse, msr + (k - 1) * mse + (k * (msc - mse)) / n),
    consistency: ratio(msr - mse, msr + (k - 1) * mse),
    oneWayAverage: ratio(msr - msw, msr),
    agreementAverage: ratio(msr - mse, msr + (msc - mse) / n),
    consistencyAverage: ratio(msr - mse, msr),
  };
}

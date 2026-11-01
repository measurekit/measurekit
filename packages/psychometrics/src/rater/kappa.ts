import { checkChoice, checkOptions, DataError, type KappaWeights, kappaOfPairs } from "@measurekit/core";
import { completeRatings } from "./ratings.js";

/** Options of {@link cohenKappa}. */
export interface CohenKappaOptions {
  /**
   * The weights of partial agreement.
   *
   * @defaultValue `"none"`
   */
  readonly weights?: KappaWeights;
}

const WEIGHTS: readonly KappaWeights[] = ["none", "linear", "quadratic"];

/**
 * Returns Cohen's kappa of two raters, (p_o - p_e) / (1 - p_e), with p_o the observed and p_e the chance agreement from
 * the raters' margins, and with weights its weighted form; quadratic weights give the quadratic weighted kappa (QWK).
 *
 * @remarks
 * As irr's `kappa2`, with `weight = "equal"` for linear and `"squared"` for quadratic weights. Uses the subjects both
 * raters rated. Ratings are categories, integers from 0; weights use their values.
 *
 * @param ratings - The ratings of two raters, one row for each subject; `null` when missing.
 * @param options - The weights of the disagreements.
 * @returns Kappa, at most 1.
 * @throws `DataError` when the ratings do not have two raters, a rating is not a category or `null`, no subject was
 * rated by both, the weights are unknown, or the chance agreement is 1, as when every rating is one category.
 * @see {@link https://doi.org/10.1177/001316446002000104 | Cohen (1960)}
 * @see {@link https://doi.org/10.1037/h0026256 | Cohen (1968)}
 */
export function cohenKappa(ratings: readonly (readonly (number | null)[])[], options: CohenKappaOptions = {}): number {
  checkOptions(options);
  const { weights = "none" } = options;
  checkChoice(weights, "The option weights", WEIGHTS);
  const rows = completeRatings(ratings, "Cohen's kappa", {
    categories: true,
    minRaters: 2,
    maxRaters: 2,
    minSubjects: 1,
  });
  const first = new Float64Array(rows.length);
  const second = new Float64Array(rows.length);
  rows.forEach(([x = 0, y = 0], subject) => {
    first[subject] = x;
    second[subject] = y;
  });
  const kappa = kappaOfPairs(first, second, weights);
  if (kappa === null) throw undefinedKappa("Cohen's kappa");
  return kappa;
}

/**
 * Returns Fleiss' kappa of two or more raters, (P̄ - P_e) / (1 - P_e): P̄ the mean over subjects of the share of
 * agreeing pairs of raters, Σ_k n_k (n_k - 1) / (m (m - 1)) for m raters of whom n_k chose category k, and P_e = Σ_k
 * p_k² from the share p_k of all ratings in category k. As irr's `kappam.fleiss`. Uses the subjects every rater rated.
 *
 * @param ratings - The ratings, one row for each subject and one entry for each rater; `null` when missing.
 * @returns Kappa, at most 1.
 * @throws `DataError` when there are fewer than two raters, a rating is not a category or `null`, no subject was rated
 * by every rater, or P_e is 1, as when every rating is one category.
 * @see {@link https://doi.org/10.1037/h0031619 | Fleiss (1971)}
 */
export function fleissKappa(ratings: readonly (readonly (number | null)[])[]): number {
  const rows = completeRatings(ratings, "Fleiss' kappa", { categories: true, minRaters: 2, minSubjects: 1 });
  const raters = rows[0]?.length ?? 0;
  const totals = new Map<number, number>();
  let agreement = 0;
  for (const row of rows) {
    const counts = new Map<number, number>();
    for (const rating of row) counts.set(rating, (counts.get(rating) ?? 0) + 1);
    for (const [category, count] of counts) {
      agreement += (count * (count - 1)) / (raters * (raters - 1));
      totals.set(category, (totals.get(category) ?? 0) + count);
    }
  }
  let chance = 0;
  for (const count of totals.values()) chance += (count / (rows.length * raters)) ** 2;
  if (!(chance < 1)) throw undefinedKappa("Fleiss' kappa");
  return (agreement / rows.length - chance) / (1 - chance);
}

function undefinedKappa(statistic: string): DataError {
  return new DataError(`${statistic} is undefined when the chance agreement is 1.`);
}

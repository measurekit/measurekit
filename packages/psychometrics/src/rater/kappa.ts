import { checkChoice, checkOptions, DataError } from "@measurekit/core";
import { completeRatings } from "./ratings.js";

/**
 * The weights of Cohen's kappa: `"none"`, agreement only; `"linear"`, 1 - |i - j| / (K - 1); `"quadratic"`,
 * 1 - (i - j)² / (K - 1)², for ratings i and j over the range K - 1 of the categories used.
 */
export type KappaWeights = "none" | "linear" | "quadratic";

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
  let lowest = Number.POSITIVE_INFINITY;
  let highest = Number.NEGATIVE_INFINITY;
  for (const row of rows) {
    for (const rating of row) {
      lowest = Math.min(lowest, rating);
      highest = Math.max(highest, rating);
    }
  }
  const range = highest - lowest;
  const agreement = (x: number, y: number): number => {
    if (weights === "none" || range === 0) return x === y ? 1 : 0;
    const distance = Math.abs(x - y) / range;
    return weights === "linear" ? 1 - distance : 1 - distance * distance;
  };
  const size = range + 1;
  const first = new Float64Array(size);
  const second = new Float64Array(size);
  let observed = 0;
  for (const [x = 0, y = 0] of rows) {
    observed += agreement(x, y);
    first[x - lowest] = (first[x - lowest] ?? 0) + 1;
    second[y - lowest] = (second[y - lowest] ?? 0) + 1;
  }
  const n = rows.length;
  let chance = 0;
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) chance += ((first[i] ?? 0) * (second[j] ?? 0) * agreement(i, j)) / (n * n);
  }
  return kappa(observed / n, chance, "Cohen's kappa");
}

/**
 * Returns Fleiss' kappa of two or more raters, (P̄ - P_e) / (1 - P_e): P̄ the mean over subjects of the share of
 * agreeing pairs of raters, Σ_k n_k (n_k - 1) / (m (m - 1)) for m raters of whom n_k chose category k, and P_e = Σ_k
 * p_k² from the share p_k of all ratings in category k. As irr's `kappam.fleiss`. Uses the subjects every rater rated.
 *
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
  return kappa(agreement / rows.length, chance, "Fleiss' kappa");
}

function kappa(observed: number, chance: number, statistic: string): number {
  if (!(chance < 1)) throw new DataError(`${statistic} is undefined when the chance agreement is 1.`);
  return (observed - chance) / (1 - chance);
}

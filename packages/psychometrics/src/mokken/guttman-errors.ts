import type { ResponseMatrix } from "../data/internal.js";
import { sampleQuantile } from "../math/internal.js";
import { type MokkenData, mokkenData, scoreOf } from "./scores.js";
import { stepPairErrors } from "./weights.js";

/** Tukey's multiple of the interquartile range above the third quartile that marks an outlier. */
const FENCE_MULTIPLE = 1.5;

/** The Guttman errors of each person, step 3 of a Mokken scale analysis, and the persons they mark as outliers. */
export interface GuttmanErrors {
  /**
   * G+ of each person, in the matrix's order: the Guttman errors over the item steps, which on 0/1 items is G; `null`
   * for a person with a missing response.
   */
  readonly errors: readonly (number | null)[];
  /** Tukey's upper fence of G+, Q3 + 1.5 IQR, over the persons who answered every item. */
  readonly upperFence: number;
  /** The indices of the persons whose G+ is above the fence, in the matrix's order. */
  readonly outliers: readonly number[];
}

/**
 * Returns the number of Guttman errors G+ of each person and the persons above Tukey's upper fence, as the mokken
 * package's `check.errors`.
 *
 * @remarks
 * Each item has a step for every score above the lowest, which a person passes by scoring at least that high. The steps
 * of all items are put in the order of how many persons pass them, the most first, and a person's G+ counts the pairs of
 * a step passed and a step failed that comes before it. Steps that as many persons pass take every order, equally
 * likely, and the count is its mean over them, a multiple of 1/2: the sum over the pairs of items of the errors of
 * their scores that `scalability` weighs. On 0/1 items G+ is G. The upper fence is Q3 + 1.5 (Q3 - Q1), with quartiles
 * by linear interpolation, R's default, and an outlier has G+ above it, the rule of Sijtsma and van der Ark's step 3.
 * The steps' order and the fence use the persons who answered every item; the others have no count, as they have fewer
 * steps to err on. Scores are integers, the lowest of all items taken as 0, spanning at most 10 categories.
 *
 * @param matrix - The responses of the persons.
 * @returns G+ of each person, the upper fence and the indices of the outliers.
 * @throws `DataError` when the matrix is invalid, has fewer than two items, fewer than two persons answered every item,
 * or the scores span more than 10 categories.
 * @see {@link https://doi.org/10.1111/bmsp.12078 | Sijtsma and van der Ark (2017)}
 */
export function guttmanErrors(matrix: ResponseMatrix): GuttmanErrors {
  const totals = personErrors(mokkenData(matrix, "Guttman errors"));
  const sorted = Float64Array.from(totals).sort();
  const lower = sampleQuantile(sorted, 0.25);
  const upper = sampleQuantile(sorted, 0.75);
  const upperFence = upper + FENCE_MULTIPLE * (upper - lower);
  let complete = 0;
  const errors = matrix.responses.map((row) => (row.includes(null) ? null : (totals[complete++] ?? 0)));
  const outliers: number[] = [];
  errors.forEach((count, person) => {
    if (count !== null && count > upperFence) outliers.push(person);
  });
  return { errors, upperFence, outliers };
}

/** Returns G+ of each person of the data: the errors of their scores summed over the pairs of items. */
function personErrors(data: MokkenData): Float64Array {
  const { items, persons, categories } = data;
  const counts = Array.from({ length: items }, (_, item) => {
    const tally = new Float64Array(categories);
    for (let person = 0; person < persons; person++) {
      const score = scoreOf(data, person, item);
      tally[score] = (tally[score] ?? 0) + 1;
    }
    return tally;
  });
  const totals = new Float64Array(persons);
  for (let i = 0; i < items; i++) {
    for (let j = i + 1; j < items; j++) {
      const errors = stepPairErrors(
        counts[i] ?? new Float64Array(categories),
        counts[j] ?? new Float64Array(categories),
      );
      for (let person = 0; person < persons; person++) {
        const pair = scoreOf(data, person, i) * categories + scoreOf(data, person, j);
        totals[person] = (totals[person] ?? 0) + (errors[pair] ?? 0);
      }
    }
  }
  // The mean over the orders of tied steps is a multiple of 1/2; rounding to it drops the sums' rounding errors.
  return totals.map((total) => Math.round(2 * total) / 2);
}

import { DataError, describe, mean } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { invertPositiveDefinite } from "../math/linear-algebra.js";
import { multiply } from "../math/matrix.js";
import { covariance, standardDeviation } from "../math/statistics.js";
import { type SectionStatistics, sectionCases, sectionStatistics } from "./sections.js";

/** One section of {@link AugmentedSubscores}. */
export interface AugmentedSection {
  /** The section's name. */
  readonly section: string;
  /** The reliability of the observed subscore: coefficient alpha of the section's items. */
  readonly reliability: number;
  /** The reliability of the augmented subscore, the proportion of its variance that is true-score variance. */
  readonly augmentedReliability: number;
  /** The mean of the subscores, observed and augmented alike. */
  readonly mean: number;
  /** The standard deviation (denominator n - 1) of the augmented subscores. */
  readonly standardDeviation: number;
}

/** Subscores of the persons that borrow strength from the other sections. */
export interface AugmentedSubscores {
  /** The sections in the order given. */
  readonly sections: readonly AugmentedSection[];
  /** The indices in the response matrix of the persons scored: those who answered every item of the sections. */
  readonly persons: readonly number[];
  /** The augmented subscores by person, then section. */
  readonly scores: readonly (readonly number[])[];
  /**
   * The weights B by section, then section, with which the deviations of the observed subscores from their means
   * give the augmented subscores: row j of B times the deviations is the deviation of section j's augmented
   * subscore.
   */
  readonly weights: readonly (readonly number[])[];
}

/**
 * Returns each person's augmented subscores: the estimate of each section's true score from the observed subscores of
 * all sections, which lends the precision of the whole test to sections of few items.
 *
 * @remarks
 * With x the observed subscores and S their covariance matrix, the covariance matrix of the true subscores is S with
 * each variance replaced by ρ_j S_jj, where ρ_j is coefficient alpha of section j's items, and the errors of different
 * sections are uncorrelated. The augmented subscores are x̄ + B (x - x̄) with B = S_true S⁻¹, the multivariate
 * regression of the true subscores on the observed ones, as `subscore.Wainer` of the subscore package computes them.
 * The reliability of augmented subscore j is [B B S_true]_jj / [B S_true]_jj, as `subscore.Wainer` computes it. A
 * section that correlates with the others gains reliability from them; sections that do not correlate keep their own.
 *
 * When the sections correlate more strongly than their reliabilities allow, a disattenuated correlation above 1, the
 * covariance matrix of the true subscores is not positive definite, as it is in the example data of the subscore
 * package, and the formula can give a reliability outside 0 to 1: the sections then do not measure different things.
 *
 * The statistics use the persons who answered every item of the sections, as `subscoreValue` does. A section needs at
 * least two items, for alpha, and a coefficient alpha above 0; there must be at least two sections, and their
 * observed subscores must not be linearly dependent.
 *
 * @param matrix - The responses of the persons.
 * @param sections - The ids of the items of each section by the section's name, such as `{ reading: ["r1", "r2"],
 *   listening: ["l1", "l2"] }`. A section's ids are items of the matrix, and no item is in two sections.
 * @returns The sections with their reliabilities before and after augmenting, the persons with their augmented
 *   subscores, and the weights of the sections.
 * @throws `DataError` when the matrix is invalid, the sections are not an object of arrays of item ids, there are fewer
 * than two sections, a section has fewer than two items, an id is not an item of the matrix or is in two places, fewer
 * than two persons answered every item of the sections, a section has a coefficient alpha that is not above 0, or the
 * covariance matrix of the observed subscores is singular.
 * @see {@link https://doi.org/10.4324/9781410604729 | Wainer et al. (2001), in Thissen and Wainer, Test Scoring}
 * @see {@link https://doi.org/10.1111/j.1745-3992.2011.00208.x | Sinharay, Puhan and Haberman (2011)}
 */
export function augmentedSubscores(
  matrix: ResponseMatrix,
  sections: Readonly<Record<string, readonly string[]>>,
): AugmentedSubscores {
  const { cases, names, indices } = sectionCases(matrix, sections, "Augmented subscores");
  const statistics = indices.map((section) => sectionStatistics(cases, section));
  statistics.forEach(({ alpha }, i) => {
    if (!(alpha > 0)) {
      throw new DataError(
        `Section "${names[i]}" needs a coefficient alpha above 0 to be augmented, found ${describe(alpha)}; a section without true-score variance has nothing to estimate.`,
      );
    }
  });
  const size = names.length;
  const observed = covarianceMatrix(statistics);
  const inverse = invertPositiveDefinite(observed, size);
  if (inverse === null) {
    throw new DataError(
      "Augmented subscores need observed subscores that are not linearly dependent, with more persons than sections.",
    );
  }
  const truth = Float64Array.from(observed);
  statistics.forEach(({ alpha }, i) => {
    truth[i * size + i] = alpha * (observed[i * size + i] ?? 0);
  });
  const weights = new Float64Array(size * size);
  multiply(truth, inverse, size, size, size, weights);
  const reliabilities = augmentedReliabilities(weights, truth, size);
  const means = statistics.map(({ scores }) => mean(scores));
  const scores = augmentedScores(statistics, means, weights);
  return {
    sections: names.map((section, j) => ({
      section,
      reliability: statistics[j]?.alpha ?? 0,
      augmentedReliability: reliabilities[j] ?? 0,
      mean: means[j] ?? 0,
      standardDeviation: standardDeviation(Float64Array.from(scores, (row) => row[j] ?? 0)),
    })),
    persons: cases.persons,
    scores,
    weights: Array.from({ length: size }, (_, j) => Array.from(weights.subarray(j * size, (j + 1) * size))),
  };
}

/** Returns the covariance matrix of the observed subscores, by rows. */
function covarianceMatrix(statistics: readonly SectionStatistics[]): Float64Array {
  const size = statistics.length;
  const result = new Float64Array(size * size);
  for (let i = 0; i < size; i++) {
    for (let j = i; j < size; j++) {
      const together = covariance(
        statistics[i]?.scores ?? new Float64Array(),
        statistics[j]?.scores ?? new Float64Array(),
      );
      result[i * size + j] = result[j * size + i] = together;
    }
  }
  return result;
}

/** Returns [B B S_true]_jj / [B S_true]_jj for every section j, with `weights` B and `truth` S_true. */
function augmentedReliabilities(weights: Float64Array, truth: Float64Array, size: number): Float64Array {
  const explained = new Float64Array(size * size);
  multiply(weights, truth, size, size, size, explained);
  const reached = new Float64Array(size * size);
  multiply(weights, explained, size, size, size, reached);
  return Float64Array.from({ length: size }, (_, j) => (reached[j * size + j] ?? 0) / (explained[j * size + j] ?? 1));
}

/** Returns x̄ + B (x - x̄) for every person. */
function augmentedScores(
  statistics: readonly SectionStatistics[],
  means: readonly number[],
  weights: Float64Array,
): number[][] {
  const size = statistics.length;
  const persons = statistics[0]?.scores.length ?? 0;
  return Array.from({ length: persons }, (_, person) => {
    const deviations = statistics.map(({ scores }, k) => (scores[person] ?? 0) - (means[k] ?? 0));
    return Array.from({ length: size }, (_, j) => {
      let estimate = means[j] ?? 0;
      for (let k = 0; k < size; k++) estimate += (weights[j * size + k] ?? 0) * (deviations[k] ?? 0);
      return estimate;
    });
  });
}

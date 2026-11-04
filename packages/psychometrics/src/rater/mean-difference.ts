import { checkFields, checkNonNegative, checkOptions, checkRecord, type FieldNames, mean } from "@measurekit/core";
import { standardDeviation } from "../math/internal.js";
import { isConstant, pairOfRaters } from "./ratings.js";

/** The standard deviation of each rater's ratings over a whole population, such as every subject of a test. */
export interface PopulationDeviations {
  /** The standard deviation of the first rater's ratings. */
  readonly first: number;
  /** The standard deviation of the second rater's ratings. */
  readonly second: number;
}

const POPULATION_DEVIATION_FIELDS: FieldNames<PopulationDeviations> = { first: true, second: true };

/** Options of {@link standardizedMeanDifference}. */
export interface StandardizedMeanDifferenceOptions {
  /**
   * The standard deviation of each rater's ratings over the whole population, for the SMD of a subgroup: its mean
   * difference over √((first² + second²) / 2), as Williamson, Xi and Breyer recommend and RSMTool's
   * `method = "williamson"` computes.
   *
   * @defaultValue the pooled standard deviation of the subjects given
   */
  readonly population?: PopulationDeviations;
}

const STANDARDIZED_MEAN_DIFFERENCE_OPTIONS: FieldNames<StandardizedMeanDifferenceOptions> = { population: true };

/**
 * Returns the standardized mean difference between two raters: the first rater's mean rating less the second's, over
 * the pooled standard deviation √((s₁² + s₂²) / 2).
 *
 * @remarks
 * The measure of rater leniency used to compare automated with human scores, a difference of the means in standard
 * deviations: positive when the first rater rates higher on average, so `[machine, human]` gives RSMTool's sign. With the
 * subjects rated by both raters in the same number for both, the pooled standard deviation is that of Cohen's d, so the
 * value is effsize's `cohen.d` of the two raters' ratings as independent samples. Uses the subjects both raters rated,
 * with variances on n - 1. With `population`, the standard deviations are the population's, so that the SMDs of
 * subgroups share one scale. Ratings are finite numbers, not only categories. It is `null` when the standard deviation
 * is 0: both raters' ratings constant, or both of the population's deviations 0.
 *
 * @param ratings - The ratings of two raters, one row for each subject; `null` when missing.
 * @param options - The population's standard deviations, for a subgroup.
 * @returns The difference of the means over the pooled standard deviation; `null` where it is undefined.
 * @throws `DataError` when the ratings do not have two raters, a rating is not a finite number or `null`, fewer than
 * two subjects were rated by both, or the population's deviations are not non-negative numbers.
 * @see {@link https://doi.org/10.1111/j.1745-3992.2011.00223.x | Williamson, Xi and Breyer (2012)}
 */
export function standardizedMeanDifference(
  ratings: readonly (readonly (number | null)[])[],
  options: StandardizedMeanDifferenceOptions = {},
): number | null {
  checkOptions(options, STANDARDIZED_MEAN_DIFFERENCE_OPTIONS);
  const { population } = options;
  if (population !== undefined) {
    checkRecord(population, "The option population must be an object { first, second }");
    checkFields(population, POPULATION_DEVIATION_FIELDS, "The option population");
    checkNonNegative(population.first, "The population's standard deviation of the first rater");
    checkNonNegative(population.second, "The population's standard deviation of the second rater");
  }
  const { first, second } = pairOfRaters(ratings, "The standardized mean difference");
  const difference = mean(first) - mean(second);
  if (population !== undefined) {
    const pooled = Math.sqrt((population.first ** 2 + population.second ** 2) / 2);
    return pooled > 0 ? difference / pooled : null;
  }
  if (isConstant(first) && isConstant(second)) return null;
  const pooled = Math.sqrt((standardDeviation(first) ** 2 + standardDeviation(second) ** 2) / 2);
  return difference / pooled;
}

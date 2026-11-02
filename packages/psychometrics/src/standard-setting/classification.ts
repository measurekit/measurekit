import { checkArray, checkFinite, checkPositive, checkRecord, DataError } from "@measurekit/core";
import type { AbilityEstimate } from "../irt/internal.js";
import { normalCdf } from "../math/internal.js";

/** A person's classification by cut scores on theta. */
export interface PersonClassification {
  /** The category of the estimate: the number of cuts at or below it. */
  readonly category: number;
  /**
   * The probability of each category, from below the first cut to above the last, with theta ~ N(θ̂, SE²): with one
   * cut, the probabilities of failing and of passing.
   */
  readonly probabilities: readonly number[];
}

/** Rudner's classification accuracy and consistency of a sample. */
export interface ClassificationAccuracy {
  /** The mean over persons of the probability of their estimated category. */
  readonly accuracy: number;
  /** The mean over persons of Σ_k P_k², the probability that two independent measurements give the same category. */
  readonly consistency: number;
  /** Each person's category and probabilities, in the order of the estimates. */
  readonly persons: readonly PersonClassification[];
}

/**
 * Returns Rudner's classification accuracy and consistency of persons classified by cut scores on theta, as cacIRT's
 * `class.Rud` with estimates and standard errors; with several cuts, its simultaneous values.
 *
 * @remarks
 * Each person's theta is taken as normal around the estimate with its standard error, which gives the probability of
 * each category. A person at a cut is in the category above it.
 *
 * @param estimates - The estimates of theta with their standard errors.
 * @param cuts - The cut scores on theta, in increasing order.
 * @returns The accuracy and consistency of the classification, and the classification of each person.
 * @throws `DataError` without estimates or cuts, when an estimate is not finite or a standard error not positive and
 * finite, or when the cuts are not finite and increasing.
 * @see {@link https://doi.org/10.7275/an9m-2035 | Rudner (2001)}
 * @see {@link https://doi.org/10.7275/56a5-6b14 | Rudner (2005)}
 */
export function classificationAccuracy(
  estimates: readonly AbilityEstimate[],
  cuts: readonly number[],
): ClassificationAccuracy {
  checkArray(estimates, "Classification accuracy needs an array of estimates");
  checkArray(cuts, "Classification accuracy needs an array of cuts");
  if (estimates.length === 0) throw new DataError("Classification accuracy needs at least one estimate.");
  if (cuts.length === 0) throw new DataError("Classification accuracy needs at least one cut.");
  cuts.forEach((cut, k) => {
    checkFinite(cut, `Cut ${k + 1}`);
    if (k > 0 && !(cut > (cuts[k - 1] ?? 0))) throw new DataError(`The cuts must increase; cut ${k + 1} is ${cut}.`);
  });
  let accuracy = 0;
  let consistency = 0;
  const persons = estimates.map((estimate, index) => {
    checkRecord(estimate, `Estimate ${index + 1} must be an object { theta, standardError }`);
    const { theta, standardError } = estimate;
    checkFinite(theta, `The theta of estimate ${index + 1}`);
    checkPositive(standardError, `The standard error of estimate ${index + 1}`);
    const bounds = [Number.NEGATIVE_INFINITY, ...cuts, Number.POSITIVE_INFINITY].map(
      (cut) => (cut - theta) / standardError,
    );
    const probabilities = cuts.map((_, k) => categoryProbability(bounds[k] ?? 0, bounds[k + 1] ?? 0));
    probabilities.push(categoryProbability(bounds[cuts.length] ?? 0, Number.POSITIVE_INFINITY));
    const category = cuts.filter((cut) => cut <= theta).length;
    accuracy += probabilities[category] ?? 0;
    consistency += probabilities.reduce((sum, probability) => sum + probability * probability, 0);
    return { category, probabilities };
  });
  return { accuracy: accuracy / estimates.length, consistency: consistency / estimates.length, persons };
}

/** Returns Φ(upper) - Φ(lower), from the tail on the side of the interval so that no precision cancels. */
function categoryProbability(lower: number, upper: number): number {
  return lower > 0 ? normalCdf(-lower) - normalCdf(-upper) : normalCdf(upper) - normalCdf(lower);
}

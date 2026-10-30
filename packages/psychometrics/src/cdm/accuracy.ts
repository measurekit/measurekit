import { checkChoice, checkOptions, DataError } from "@measurekit/core";
import type { ResponseMatrix } from "../data/response-matrix.js";
import type { CdmCalibration } from "./calibration.js";
import { attributeProfiles } from "./persons.js";

/** Options of {@link cdmClassificationAccuracy}. */
export interface CdmAccuracyOptions {
  /**
   * The classification of each person: `"map"`, the profile of largest posterior, or `"eap"`.
   *
   * @defaultValue `"map"`
   */
  readonly classification?: "map" | "eap";
}

const CLASSIFICATIONS: readonly ("map" | "eap")[] = ["map", "eap"];

/** The classification accuracy and consistency of a calibrated cognitive diagnosis model, as GDINA's `CA`. */
export interface CdmClassificationAccuracy {
  /** τ: Σ_c π_c τ_c, the probability that a person is classified in the true profile. */
  readonly accuracy: number;
  /** τ_c: per profile, the posterior share of its persons classified in it, in the calibration's profile order. */
  readonly profileAccuracy: readonly number[];
  /** τ_k: per attribute, the mean of p_k m + (1 - p_k)(1 - m), m the classification and p_k the mastery probability. */
  readonly attributeAccuracy: readonly number[];
  /** γ: the mean over persons of Σ_c post(c)², the probability that two measurements give the same profile. */
  readonly consistency: number;
  /** γ_k: per attribute, the mean of p_k² + (1 - p_k)². */
  readonly attributeConsistency: readonly number[];
}

/**
 * Returns the classification accuracy and consistency of the persons of a matrix under a calibrated cognitive diagnosis
 * model, as GDINA's `CA`. τ_c weights each person's posterior of c by whether the person is classified in c, over the
 * persons' posterior of c. Uses the persons with responses.
 *
 * @throws `DataError` when the matrix is invalid, its items are not the calibration's, a response is neither 0 nor 1,
 * no person has a response, or the classification is unknown.
 * @see {@link https://doi.org/10.7282/t3w95d95 | Iaconangelo (2017)}
 * @see {@link https://doi.org/10.1111/jedm.12096 | Wang et al. (2015)}
 */
export function cdmClassificationAccuracy(
  matrix: ResponseMatrix,
  calibration: CdmCalibration,
  options: CdmAccuracyOptions = {},
): CdmClassificationAccuracy {
  checkOptions(options);
  const { classification = "map" } = options;
  checkChoice(classification, "The option classification", CLASSIFICATIONS);
  const profiles = attributeProfiles(matrix, calibration);
  const persons = profiles.filter((_, i) => matrix.responses[i]?.some((response) => response !== null));
  if (persons.length === 0) throw new DataError("Classification accuracy needs at least one person with a response.");
  const keys = calibration.profiles.map(({ profile }) => profile.join(""));
  const profileCount = keys.length;
  const posteriorTotals = new Float64Array(profileCount);
  const classified = new Float64Array(profileCount);
  let consistency = 0;
  const attributeCount = calibration.attributes.length;
  const attributeAccuracy = new Float64Array(attributeCount);
  const attributeConsistency = new Float64Array(attributeCount);
  for (const person of persons) {
    const assigned = person[classification];
    const c = keys.indexOf(assigned.join(""));
    person.posterior.forEach((p, profile) => {
      posteriorTotals[profile] = (posteriorTotals[profile] ?? 0) + p;
      consistency += p * p;
    });
    classified[c] = (classified[c] ?? 0) + (person.posterior[c] ?? 0);
    person.mastery.forEach((p, k) => {
      const m = assigned[k] ?? 0;
      attributeAccuracy[k] = (attributeAccuracy[k] ?? 0) + p * m + (1 - p) * (1 - m);
      attributeConsistency[k] = (attributeConsistency[k] ?? 0) + p * p + (1 - p) * (1 - p);
    });
  }
  const profileAccuracy = Array.from(
    classified,
    (value, c) => value / Math.max(posteriorTotals[c] ?? 0, Number.EPSILON),
  );
  const n = persons.length;
  return {
    accuracy: profileAccuracy.reduce((sum, tau, c) => sum + tau * (calibration.profiles[c]?.proportion ?? 0), 0),
    profileAccuracy,
    attributeAccuracy: Array.from(attributeAccuracy, (sum) => sum / n),
    consistency: consistency / n,
    attributeConsistency: Array.from(attributeConsistency, (sum) => sum / n),
  };
}

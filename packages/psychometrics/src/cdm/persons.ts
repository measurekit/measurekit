import type { ResponseMatrix } from "../data/response-matrix.js";
import { type CdmCalibration, calibratedModel, forEachPosterior } from "./calibration.js";

/** A person's attribute mastery from a calibrated cognitive diagnosis model. */
export interface AttributeProfile {
  /** The person's id. */
  readonly id: string;
  /** The posterior probability of mastering each attribute, in the calibration's attribute order. */
  readonly mastery: readonly number[];
  /** The profile of largest posterior probability (MAP), the first in profile order on ties. */
  readonly map: readonly number[];
  /** Each attribute mastered when its posterior probability is above 0.5 (EAP). */
  readonly eap: readonly number[];
  /** The posterior probability of each profile, in the order of the calibration's profiles. */
  readonly posterior: readonly number[];
}

/**
 * Returns each person's posterior over the profiles of a calibrated cognitive diagnosis model, the posterior
 * probability of mastering each attribute, and the MAP and EAP profiles, as GDINA's `personparm`. A person without
 * responses takes the calibrated proportions.
 *
 * @throws `DataError` when the matrix is invalid, its items are not the calibration's in its order, or a response is
 * neither 0 nor 1.
 */
export function attributeProfiles(matrix: ResponseMatrix, calibration: CdmCalibration): AttributeProfile[] {
  const { data, design, state } = calibratedModel(matrix, calibration);
  const posteriors: (readonly number[])[] = matrix.personIds.map(() => Array.from(state.proportions));
  forEachPosterior(data, design, state, (person, posterior) => {
    posteriors[person] = Array.from(posterior);
  });
  return matrix.personIds.map((id, person) => {
    const posterior = posteriors[person] ?? [];
    const mastery = design.attributes.map((_, k) =>
      posterior.reduce((sum, p, c) => sum + p * (design.profiles[c]?.[k] ?? 0), 0),
    );
    const best = posterior.reduce((top, p, c) => (p > (posterior[top] ?? 0) ? c : top), 0);
    return {
      id,
      mastery,
      map: design.profiles[best] ?? [],
      eap: mastery.map((p) => (p > 0.5 ? 1 : 0)),
      posterior,
    };
  });
}

import type { ResponseMatrix } from "../data/response-matrix.js";
import { type CdmCalibration, calibratedModel } from "./calibration.js";
import { forEachPosterior } from "./expectation.js";

/** A posterior over the profiles of a cognitive diagnosis model with the estimates from it. */
export interface ProfileEstimate {
  /** The posterior probability of each profile, in the order of the calibration's profiles. */
  readonly posterior: readonly number[];
  /** The posterior probability of mastering each attribute, in the calibration's attribute order. */
  readonly mastery: readonly number[];
  /** The profile of largest posterior probability (MAP), the first in profile order on ties. */
  readonly map: readonly number[];
  /** Each attribute mastered when its posterior probability is above 0.5 (EAP). */
  readonly eap: readonly number[];
}

/** A person's attribute mastery from a calibrated cognitive diagnosis model. */
export interface AttributeProfile extends ProfileEstimate {
  /** The person's id. */
  readonly id: string;
}

/** Returns the attribute mastery, the MAP and EAP profiles of a posterior over `profiles`, in their order. */
export function profileEstimate(
  posterior: readonly number[],
  profiles: readonly (readonly number[])[],
): ProfileEstimate {
  const mastery: number[] = new Array<number>(profiles[0]?.length ?? 0).fill(0);
  let best = 0;
  for (let c = 0; c < posterior.length; c++) {
    const p = posterior[c] ?? 0;
    const profile = profiles[c];
    if (profile !== undefined)
      for (let k = 0; k < mastery.length; k++) mastery[k] = (mastery[k] ?? 0) + p * (profile[k] ?? 0);
    if (p > (posterior[best] ?? 0)) best = c;
  }
  return { posterior, mastery, map: profiles[best] ?? [], eap: mastery.map((p) => (p > 0.5 ? 1 : 0)) };
}

/** Returns the values as an array: a loop is several times faster than `Array.from` on a typed array. */
function copyOf(values: Float64Array): number[] {
  const copy = new Array<number>(values.length);
  for (let c = 0; c < values.length; c++) copy[c] = values[c] ?? 0;
  return copy;
}

/**
 * Returns each person's posterior over the profiles of a calibrated cognitive diagnosis model, the posterior
 * probability of mastering each attribute, and the MAP and EAP profiles, as GDINA's `personparm`. A person without
 * responses takes the calibrated proportions.
 *
 * @remarks
 * A success probability of 0 or 1, as a calibration from JSON may hold, makes a response impossible; the response's
 * probability enters the likelihood as `Number.MIN_VALUE` instead, so that responses no profile can give still have a
 * posterior, on the profiles they contradict least.
 *
 * @throws `DataError` when the matrix is invalid, its items are not the calibration's in its order, or a response is
 * neither 0 nor 1.
 */
export function attributeProfiles(matrix: ResponseMatrix, calibration: CdmCalibration): AttributeProfile[] {
  const { data, design, state } = calibratedModel(matrix, calibration);
  const estimates: (ProfileEstimate | undefined)[] = [];
  forEachPosterior(data, design, state, (person, posterior) => {
    estimates[person] = profileEstimate(copyOf(posterior), design.profiles);
  });
  return matrix.personIds.map((id, person) => ({
    id,
    ...(estimates[person] ?? profileEstimate(Array.from(state.proportions), design.profiles)),
  }));
}

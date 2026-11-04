import type { ResponseMatrix } from "../data/internal.js";
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
  const { mastery, map, eap } = estimate(posterior, profiles, tableOf(profiles));
  return { posterior, mastery, map, eap };
}

/** Returns the profiles in one array, by profile and then attribute, which the sums over the profiles read in order. */
export function tableOf(profiles: readonly (readonly number[])[]): Float64Array {
  const attributeCount = profiles[0]?.length ?? 0;
  const table = new Float64Array(profiles.length * attributeCount);
  for (let c = 0; c < profiles.length; c++) {
    const profile = profiles[c] ?? [];
    for (let k = 0; k < attributeCount; k++) table[c * attributeCount + k] = profile[k] ?? 0;
  }
  return table;
}

/**
 * Returns the estimates of a posterior over `profiles`, whose attributes `table` holds as {@link tableOf} returns them.
 */
export function estimate(
  posterior: ArrayLike<number>,
  profiles: readonly (readonly number[])[],
  table: Float64Array,
): Omit<ProfileEstimate, "posterior"> {
  const attributeCount = profiles[0]?.length ?? 0;
  const mastery = new Array<number>(attributeCount);
  const eap = new Array<number>(attributeCount);
  for (let k = 0; k < attributeCount; k++) {
    let sum = 0;
    for (let c = 0; c < posterior.length; c++) sum += (posterior[c] ?? 0) * (table[c * attributeCount + k] ?? 0);
    mastery[k] = sum;
    eap[k] = sum > 0.5 ? 1 : 0;
  }
  let best = 0;
  let largest = posterior[0] ?? 0;
  for (let c = 1; c < posterior.length; c++) {
    const p = posterior[c] ?? 0;
    if (p > largest) {
      best = c;
      largest = p;
    }
  }
  return { mastery, map: profiles[best] ?? [], eap };
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
 * @param matrix - The responses of the persons.
 * @param calibration - The calibrated model.
 * @returns For each person, the posterior over the profiles and the estimates that follow from it.
 * @throws `DataError` when the matrix is invalid, its items are not the calibration's in its order, or a response is
 * neither 0 nor 1.
 */
export function attributeProfiles(matrix: ResponseMatrix, calibration: CdmCalibration): readonly AttributeProfile[] {
  const { data, design, state } = calibratedModel(matrix, calibration);
  const table = tableOf(design.profiles);
  const persons = new Array<AttributeProfile>(matrix.personIds.length);
  forEachPosterior(data, design, state, (person, posterior) => {
    const { mastery, map, eap } = estimate(posterior, design.profiles, table);
    persons[person] = { id: matrix.personIds[person] ?? "", posterior: copyOf(posterior), mastery, map, eap };
  });
  for (let person = 0; person < persons.length; person++) {
    if (persons[person] !== undefined) continue;
    const { mastery, map, eap } = estimate(state.proportions, design.profiles, table);
    persons[person] = { id: matrix.personIds[person] ?? "", posterior: copyOf(state.proportions), mastery, map, eap };
  }
  return persons;
}

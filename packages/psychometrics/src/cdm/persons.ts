import { checkBoolean, checkOptions } from "@measurekit/core";
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
export interface AttributeProfile extends Omit<ProfileEstimate, "posterior"> {
  /** The person's id. */
  readonly id: string;
}

/** An {@link AttributeProfile} with the person's posterior over the profiles, which the option `posterior` asks for. */
export interface AttributeProfileWithPosterior extends AttributeProfile, ProfileEstimate {}

/** Options of {@link attributeProfiles}. */
export interface AttributeProfilesOptions {
  /**
   * Whether each person's result holds the posterior probability of every profile: 2^K numbers a person, 260 KB at 15
   * attributes.
   *
   * @defaultValue `false`
   */
  readonly posterior?: boolean;
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
 * Returns each person's posterior probability of mastering each attribute, and the MAP and EAP profiles, as GDINA's
 * `personparm`, and with the option `posterior` the posterior over the profiles they come from. A person without
 * responses takes the calibrated proportions.
 *
 * @remarks
 * A success probability of 0 or 1, as a calibration from JSON may hold, makes a response impossible; the response's
 * probability enters the likelihood as `Number.MIN_VALUE` instead, so that responses no profile can give still have a
 * posterior, on the profiles they contradict least.
 *
 * @param matrix - The responses of the persons.
 * @param calibration - The calibrated model.
 * @param options - Whether to return the posterior over the profiles.
 * @returns For each person, the estimates that follow from the posterior, and with `posterior: true` the posterior.
 * @throws `DataError` when the matrix is invalid, its items are not the calibration's in its order, a response is
 * neither 0 nor 1, or an option is invalid.
 */
export function attributeProfiles(
  matrix: ResponseMatrix,
  calibration: CdmCalibration,
  options: AttributeProfilesOptions & { readonly posterior: true },
): readonly AttributeProfileWithPosterior[];
export function attributeProfiles(
  matrix: ResponseMatrix,
  calibration: CdmCalibration,
  options?: AttributeProfilesOptions,
): readonly AttributeProfile[];
export function attributeProfiles(
  matrix: ResponseMatrix,
  calibration: CdmCalibration,
  options: AttributeProfilesOptions = {},
): readonly AttributeProfile[] {
  checkOptions(options);
  const { posterior: withPosterior = false } = options;
  checkBoolean(withPosterior, "The option posterior");
  const { data, design, state } = calibratedModel(matrix, calibration);
  const table = tableOf(design.profiles);
  const profileOf = (person: number, posterior: Float64Array): AttributeProfile | AttributeProfileWithPosterior => {
    const { mastery, map, eap } = estimate(posterior, design.profiles, table);
    const id = matrix.personIds[person] ?? "";
    return withPosterior ? { id, posterior: copyOf(posterior), mastery, map, eap } : { id, mastery, map, eap };
  };
  const persons = new Array<AttributeProfile | AttributeProfileWithPosterior>(matrix.personIds.length);
  forEachPosterior(data, design, state, (person, posterior) => {
    persons[person] = profileOf(person, posterior);
  });
  for (let person = 0; person < persons.length; person++) {
    if (persons[person] === undefined) persons[person] = profileOf(person, state.proportions);
  }
  return persons;
}

import { checkArray, checkRecord, checkUnitInterval, DataError, isNonEmptyString } from "@measurekit/core";
import { bitPattern, MAX_ATTRIBUTES, type ProfileProportion, profileIndices, profilesOf } from "../cdm/internal.js";
import { formatList } from "../irt/internal.js";

/** The attributes of a diagnostic test, the profiles over them and the prior probability of each profile. */
export interface ProfileSpace {
  /** The attributes, in the order of the profiles. */
  readonly attributes: readonly string[];
  /** Every profile, in the order of {@link profilesOf}: 1 for each mastered attribute, 0 otherwise. */
  readonly profiles: readonly (readonly number[])[];
  /** The prior probability of each profile, summing to 1. */
  readonly prior: Float64Array;
}

/**
 * Checks the attributes of a diagnostic test and its prior, and returns the profiles with the prior matched to them by
 * pattern and normalized to sum to 1: uniform without a prior.
 *
 * @throws `DataError` when the attributes are not 1 to 20 distinct non-empty strings, or the prior does not list each
 * profile once, as 0s and 1s over the attributes, with a proportion from 0 to 1, the proportions not all 0.
 */
export function profileSpace(
  attributes: readonly string[],
  prior: readonly ProfileProportion[] | undefined,
): ProfileSpace {
  checkArray(attributes, "The option attributes must be an array of attribute names");
  const valid = attributes.length > 0 && attributes.length <= MAX_ATTRIBUTES && attributes.every(isNonEmptyString);
  if (!(valid && new Set(attributes).size === attributes.length)) {
    throw new DataError(
      `The option attributes must list from 1 to ${MAX_ATTRIBUTES} distinct non-empty names, found ${formatList(attributes)}.`,
    );
  }
  const profiles = profilesOf(attributes.length);
  const proportions = prior === undefined ? new Float64Array(profiles.length).fill(1) : matchPrior(prior, attributes);
  const total = proportions.reduce((sum, proportion) => sum + proportion, 0);
  if (!(total > 0)) throw new DataError("The option prior needs profile proportions that are not all 0.");
  return { attributes, profiles, prior: proportions.map((proportion) => proportion / total) };
}

/** Returns the proportions of a prior in the order of {@link profilesOf}, each profile matched by its pattern. */
function matchPrior(prior: readonly ProfileProportion[], attributes: readonly string[]): Float64Array {
  checkArray(prior, "The option prior must be an array of { profile, proportion }");
  const count = attributes.length;
  const indices = profileIndices(count);
  const proportions = new Float64Array(2 ** count).fill(Number.NaN);
  prior.forEach((entry, e) => {
    checkRecord(entry, `Prior entry ${e + 1} must be an object { profile, proportion }`);
    const { profile, proportion } = entry;
    const valid = Array.isArray(profile) && profile.length === count && profile.every((v) => v === 0 || v === 1);
    if (!valid) {
      throw new DataError(
        `Prior entry ${e + 1} needs a 0 or 1 for each of the ${count} attributes, found ${formatList(profile)}.`,
      );
    }
    checkUnitInterval(proportion, `The proportion of prior entry ${e + 1}`);
    const index = indices[bitPattern(profile, profile.keys())] ?? 0;
    if (!Number.isNaN(proportions[index])) {
      throw new DataError(`Prior entry ${e + 1} repeats the profile ${formatList(profile)}.`);
    }
    proportions[index] = proportion;
  });
  if (proportions.some(Number.isNaN)) {
    throw new DataError(
      `The option prior must list each of the ${2 ** count} profiles once, found ${prior.length} entries.`,
    );
  }
  return proportions;
}

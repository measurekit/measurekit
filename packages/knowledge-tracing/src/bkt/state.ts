import { checkRecordById, DataError, describe, isNonEmptyString, isUnitInterval } from "@measurekit/core";
import { checkLearnerState } from "../data/internal.js";

/** A learner's state in a BKT model: plain data, which JSON serializes. */
export interface BktState {
  /** The learner's id. */
  readonly learner: string;
  /**
   * P(L) of each skill the learner has practiced, by skill id: the probability that the learner knows the skill
   * at their next attempt. A skill not practiced has the learner's prior.
   */
  readonly skills: Readonly<Record<string, number>>;
  /** The resource of the latest attempt at each skill, by skill id; absent without pair learning. */
  readonly previous?: Readonly<Record<string, string>>;
}

/** Checks a state, which may come from storage, throwing a `DataError` that names the first problem found. */
export function checkState(state: BktState): void {
  const learner = checkLearnerState(state, "BKT");
  const label = `The BKT state of learner "${learner}"`;
  checkRecordById(state.skills, `${label} needs skills, an object of probabilities by skill id`);
  for (const [skill, mastery] of Object.entries(state.skills)) {
    if (!isUnitInterval(mastery)) {
      throw new DataError(`${label} needs a probability from 0 to 1 for skill "${skill}", found ${describe(mastery)}.`);
    }
  }
  if (state.previous === undefined) return;
  checkRecordById(state.previous, `${label} needs previous, an object of resource ids by skill id`);
  for (const [skill, resource] of Object.entries(state.previous)) {
    if (!isNonEmptyString(resource)) {
      throw new DataError(
        `${label} needs a resource id, a non-empty string, for skill "${skill}", found ${describe(resource)}.`,
      );
    }
  }
}

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

/**
 * Checks what a call reads of a state, which may come from storage: its learner, its records and their entries at the
 * skills named, throwing a `DataError` that names the first problem found. A call checks the entries it reads, so that
 * its cost does not grow with the skills the learner has practiced.
 */
export function checkState(state: BktState, skills: readonly string[]): void {
  const learner = checkLearnerState(state, "BKT");
  const label = `The BKT state of learner "${learner}"`;
  checkRecordById(state.skills, `${label} needs skills, an object of probabilities by skill id`);
  if (state.previous !== undefined) {
    checkRecordById(state.previous, `${label} needs previous, an object of resource ids by skill id`);
  }
  for (const skill of skills) {
    if (Object.hasOwn(state.skills, skill) && !isUnitInterval(state.skills[skill])) {
      throw new DataError(
        `${label} needs a probability from 0 to 1 for skill "${skill}", found ${describe(state.skills[skill])}.`,
      );
    }
    const { previous } = state;
    if (previous !== undefined && Object.hasOwn(previous, skill) && !isNonEmptyString(previous[skill])) {
      throw new DataError(
        `${label} needs a resource id, a non-empty string, for skill "${skill}", found ${describe(previous[skill])}.`,
      );
    }
  }
}

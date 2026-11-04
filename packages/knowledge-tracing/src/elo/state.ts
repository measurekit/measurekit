import { checkRecord, checkRecordById, DataError, describe } from "@measurekit/core";
import { checkLearnerState } from "../data/internal.js";

/** A learner's state in an Elo model: plain data, which JSON serializes. */
export interface EloState {
  /** The learner's id. */
  readonly learner: string;
  /** θ, the learner's global ability on the logit scale. */
  readonly ability: number;
  /** n, the number of events that have updated `ability`. */
  readonly count: number;
  /** The learner's ability in each skill practiced, by skill id; empty unless the model has skill abilities. */
  readonly skills: Readonly<Record<string, SkillAbility>>;
}

/** A learner's ability in one skill, a correction to the global ability for the skill's items. */
export interface SkillAbility {
  /** θ_k on the logit scale. */
  readonly ability: number;
  /** n, the number of events that have updated `ability`. */
  readonly count: number;
}

/** Checks a state, which may come from storage, throwing a `DataError` that names the first problem found. */
export function checkState(state: EloState): void {
  const learner = checkLearnerState(state, "Elo");
  const label = `The Elo state of learner "${learner}"`;
  checkRating(state.ability, state.count, label, "an ability");
  checkRecordById(state.skills, `${label} needs skills, an object of skill abilities by skill id`);
  for (const [skill, rating] of Object.entries(state.skills)) {
    checkRecord(rating, `${label} needs an object for skill "${skill}"`);
    checkRating(rating.ability, rating.count, `${label}, skill "${skill}",`, "an ability");
  }
}

/** Checks a rating and the count of events behind it: `name` is how the message calls the rating. */
export function checkRating(value: number, count: number, label: string, name: string): void {
  if (!Number.isFinite(value)) {
    throw new DataError(`${label} needs ${name}, a finite number, found ${describe(value)}.`);
  }
  if (!(Number.isSafeInteger(count) && count >= 0)) {
    throw new DataError(`${label} needs a count, a non-negative integer, found ${describe(count)}.`);
  }
}

import { DataError, describe, isNonEmptyString, isRecord } from "@measurekit/core";
import type { Attempt } from "./event.js";

/** Checks a learner id, as `initialState` takes it: a non-empty string. */
export function checkLearnerId(learner: unknown): void {
  if (!isNonEmptyString(learner)) {
    throw new DataError(`A learner id must be a non-empty string, found ${describe(learner)}.`);
  }
}

/** Checks that an attempt belongs to the learner whose state it is given with. */
export function checkLearner(learner: string, attempt: Attempt): void {
  if (attempt.learner !== learner) {
    throw new DataError(`An attempt of learner "${attempt.learner}" was given the state of learner "${learner}".`);
  }
}

/**
 * Checks the part every learner state has, an object with a learner id, and returns the id.
 *
 * @param kind - Names the model in the error message, such as `Elo`.
 * @throws `DataError` when the state is not an object or its learner id is not a non-empty string.
 */
export function checkLearnerState(state: unknown, kind: string): string {
  if (!isRecord(state)) {
    throw new DataError(`The ${kind} state must be an object, found ${describe(state)}.`);
  }
  const learner = "learner" in state ? state.learner : undefined;
  if (!isNonEmptyString(learner)) {
    throw new DataError(`The ${kind} state needs a learner id, a non-empty string, found ${describe(learner)}.`);
  }
  return learner;
}

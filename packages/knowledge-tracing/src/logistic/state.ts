import { checkRecord, DataError, describe } from "@measurekit/core";
import type { Attempt } from "../data/event.js";
import { checkLearnerState } from "../data/learner.js";
import { entry } from "../data/record.js";

/** A learner's state in a logistic model: their history with each skill practiced. Plain data, which JSON serializes. */
export interface LogisticState {
  /** The learner's id. */
  readonly learner: string;
  /** The learner's history with each skill practiced, by skill id. */
  readonly skills: Readonly<Record<string, SkillHistory>>;
}

/** What the features need of a learner's history with a skill. */
export interface SkillHistory {
  /** The sum of the outcomes of the attempts. */
  readonly successes: number;
  /** The sum of 1 - outcome over the attempts. */
  readonly failures: number;
  /** The time of the last attempt. */
  readonly last: number;
  /** For each decay d of the model's features, by `String(d)`: successes and failures, each attempt weighing d times the next. */
  readonly decayed: Readonly<Record<string, DecayedCounts>>;
}

/** Counts in which each attempt weighs d times the next. */
export interface DecayedCounts {
  /** Σ dᵃ y over the attempts, a the number of later attempts. */
  readonly successes: number;
  /** Σ dᵃ (1 - y) over the attempts. */
  readonly failures: number;
}

/** Returns the history after an attempt with outcome y at `time`. */
export function nextHistory(
  history: SkillHistory | undefined,
  outcome: number,
  time: number,
  decays: readonly number[],
): SkillHistory {
  return {
    successes: (history?.successes ?? 0) + outcome,
    failures: (history?.failures ?? 0) + (1 - outcome),
    last: time,
    decayed: Object.fromEntries(
      decays.map((decay) => {
        const sums = history === undefined ? undefined : entry(history.decayed, String(decay));
        const successes = decay * (sums?.successes ?? 0) + outcome;
        return [String(decay), { successes, failures: decay * (sums?.failures ?? 0) + (1 - outcome) }];
      }),
    ),
  };
}

/** Checks a state, which may come from storage, throwing a `DataError` that names the first problem found. */
export function checkState(state: LogisticState, decays: readonly number[]): void {
  const learner = checkLearnerState(state, "logistic");
  const label = `The logistic state of learner "${learner}"`;
  checkRecord(state.skills, `${label} needs skills, an object of histories by skill id`);
  for (const [skill, history] of Object.entries(state.skills)) {
    const prefix = `${label}, skill "${skill}",`;
    checkRecord(history, `${prefix} must be an object`);
    checkCount(history.successes, `${prefix} needs successes`);
    checkCount(history.failures, `${prefix} needs failures`);
    if (!Number.isFinite(history.last)) {
      throw new DataError(`${prefix} needs last, the time of the last attempt, found ${describe(history.last)}.`);
    }
    checkRecord(history.decayed, `${prefix} needs decayed, an object of decayed counts by decay`);
    for (const decay of decays) {
      if (entry(history.decayed, String(decay)) === undefined) {
        throw new DataError(`${prefix} lacks the decayed counts of decay ${decay}, which the model's features need.`);
      }
    }
    for (const [decay, sums] of Object.entries(history.decayed)) {
      checkRecord(sums, `${prefix} needs an object of decayed counts for decay ${decay}`);
      checkCount(sums.successes, `${prefix} needs decayed successes for decay ${decay}`);
      checkCount(sums.failures, `${prefix} needs decayed failures for decay ${decay}`);
    }
  }
}

function checkCount(value: number, message: string): void {
  if (!(Number.isFinite(value) && value >= 0)) {
    throw new DataError(`${message}, a non-negative number, found ${describe(value)}.`);
  }
}

/**
 * Checks that an attempt is not earlier than the learner's last attempt at any of its skills, as features count the
 * time since it.
 *
 * @param label - Names the attempt in the error message, such as `The event`.
 */
export function checkOrder(state: LogisticState, attempt: Attempt, label: string): void {
  for (const skill of attempt.skills) {
    const history = entry(state.skills, skill);
    if (history !== undefined && attempt.time < history.last) {
      throw new DataError(
        `${label} at time ${attempt.time} is earlier than the learner's last attempt at skill "${skill}", at time ${history.last}.`,
      );
    }
  }
}

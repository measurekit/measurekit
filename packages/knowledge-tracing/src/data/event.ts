import { checkArray, checkRecord, DataError, describe, isNonEmptyString, isUnitInterval } from "@measurekit/core";

/** A learner's attempt at an item, as known before its outcome: what a model predicts from. */
export interface Attempt {
  /** The learner's id. */
  readonly learner: string;
  /** The item's id. */
  readonly item: string;
  /** The ids of the skills (knowledge components) the item practices; empty when no model in use needs them. */
  readonly skills: readonly string[];
  /** When the attempt happened: any finite number that orders a learner's attempts, such as epoch milliseconds. */
  readonly time: number;
}

/** An attempt with its outcome: one entry of a log. */
export interface Event extends Attempt {
  /** The outcome from 0 to 1: 1 correct, 0 incorrect, and between them partial credit where a model supports it. */
  readonly outcome: number;
}

/**
 * Checks an attempt, throwing a `DataError` that names the first problem found and starts with `label`, such as
 * `"Event 3"`.
 */
export function checkAttempt(attempt: Attempt, label: string): void {
  // Attempts built by hand or loaded from JSON may lack a field or hold another type.
  checkRecord(attempt, `${label} must be an object`);
  checkId(attempt.learner, `${label} needs a learner id`);
  checkId(attempt.item, `${label} needs an item id`);
  checkSkills(attempt.skills, label);
  if (!Number.isFinite(attempt.time)) {
    throw new DataError(`${label} needs a time, a finite number, found ${describe(attempt.time)}.`);
  }
}

/**
 * Checks a list of attempts, such as candidates to rank: an array whose every attempt passes {@link checkAttempt}.
 *
 * @param noun - Names an attempt in the error message, such as `Candidate`, for `Candidate 2` and `The candidates`.
 */
export function checkAttempts(attempts: readonly Attempt[], noun: string): void {
  checkArray(attempts, `The ${noun.toLowerCase()}s must be an array of attempts`);
  attempts.forEach((attempt, index) => {
    checkAttempt(attempt, `${noun} ${index + 1}`);
  });
}

/** Checks an event as {@link checkAttempt} does, and its outcome. */
export function checkEvent(event: Event, label: string): void {
  checkAttempt(event, label);
  const { outcome } = event;
  if (!isUnitInterval(outcome)) {
    throw new DataError(`${label} needs an outcome from 0 to 1, found ${describe(outcome)}.`);
  }
}

/**
 * Checks that an attempt has at least one skill, for models that trace skills.
 *
 * @param model - Names the model in the error message, such as `BKT model`.
 */
export function checkHasSkills(attempt: Attempt, label: string, model: string): void {
  if (attempt.skills.length === 0) throw new DataError(`${label} has no skills; a ${model} needs at least one.`);
}

/** Checks a list of skills: distinct non-empty strings. */
export function checkSkills(skills: readonly string[], label: string): void {
  checkArray(skills, `${label} needs skills, an array of skill ids`);
  const seen = new Set<string>();
  for (const skill of skills) {
    checkId(skill, `${label} needs skill ids`);
    if (seen.has(skill)) throw new DataError(`${label} lists the skill "${skill}" twice.`);
    seen.add(skill);
  }
}

function checkId(id: string, message: string): void {
  if (!isNonEmptyString(id)) {
    throw new DataError(`${message}, a non-empty string, found ${describe(id)}.`);
  }
}

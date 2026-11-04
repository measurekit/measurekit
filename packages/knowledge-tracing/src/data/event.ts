import {
  checkArray,
  checkFinite,
  checkIntegerFrom,
  checkPositive,
  checkRecord,
  checkRecordById,
  DataError,
  describe,
  isNonEmptyString,
  isUnitInterval,
} from "@measurekit/core";

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
  /**
   * The key of the learn and forget rates of this attempt: a lesson, a video or an item.
   *
   * @defaultValue the item
   */
  readonly resource?: string;
  /**
   * Finite numbers by name, known before the outcome, which a model outside the package can read.
   *
   * @defaultValue none
   */
  readonly covariates?: Readonly<Record<string, number>>;
}

/** An attempt with its outcome: one entry of a log. */
export interface Event extends Attempt {
  /**
   * The outcome from 0 to 1: 1 correct, 0 incorrect, and between them partial credit where a model supports it; or
   * `null` for a learning opportunity with nothing observed, such as a lesson or a hinted success.
   */
  readonly outcome: number | null;
  /**
   * The hints taken before the outcome: an integer from 0.
   *
   * @defaultValue none recorded
   */
  readonly hints?: number;
  /**
   * The time spent on the attempt, above 0, in the units of `time`.
   *
   * @defaultValue none recorded
   */
  readonly duration?: number;
  /**
   * Which try at the item this is: an integer from 1.
   *
   * @defaultValue none recorded
   */
  readonly attemptNumber?: number;
}

/** An event whose outcome was observed: the events that models predict and that count in likelihoods. */
export interface ObservedEvent extends Event {
  readonly outcome: number;
}

/** Returns whether an event has an outcome. */
export function isObserved(event: Event): event is ObservedEvent {
  return event.outcome !== null;
}

/** Returns the key of an attempt's learn and forget rates: its resource, or its item without one. */
export function resourceOf(attempt: Attempt): string {
  return attempt.resource ?? attempt.item;
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
  const { resource, covariates } = attempt;
  if (resource !== undefined) checkId(resource, `${label} needs a resource id`);
  if (covariates !== undefined) {
    checkRecordById(covariates, `${label} needs covariates, an object of numbers by name`);
    for (const [name, value] of Object.entries(covariates)) {
      checkFinite(value, `The covariate "${name}" of ${label}`);
    }
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

/** Checks an event as {@link checkAttempt} does, and its outcome and attempt metadata. */
export function checkEvent(event: Event, label: string): void {
  checkAttempt(event, label);
  const { outcome, hints, duration, attemptNumber } = event;
  // Only an explicit null means that nothing was observed: a missing outcome is a mistake.
  if (outcome !== null && !isUnitInterval(outcome)) {
    throw new DataError(`${label} needs an outcome from 0 to 1 or null, found ${describe(outcome)}.`);
  }
  if (hints !== undefined) checkIntegerFrom(hints, `The hints of ${label}`, 0);
  if (duration !== undefined) checkPositive(duration, `The duration of ${label}`);
  if (attemptNumber !== undefined) checkIntegerFrom(attemptNumber, `The attempt number of ${label}`, 1);
}

/**
 * Checks that an attempt has at least one skill, for models that trace skills.
 *
 * @param model - Names the model in the error message, such as `BKT model`.
 */
export function checkHasSkills(attempt: Attempt, label: string, model: string): void {
  if (attempt.skills.length === 0) throw new DataError(`${label} has no skills; a ${model} needs at least one.`);
}

/** Checks that an event with an outcome has at least one skill; an event without one may have none. */
export function checkObservedHasSkills(event: Event, label: string, model: string): void {
  if (isObserved(event)) checkHasSkills(event, label, model);
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

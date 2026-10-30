import {
  checkMethod,
  checkOptions,
  checkPositiveInteger,
  checkUnitInterval,
  DataError,
  describe,
} from "@measurekit/core";
import type { Event } from "../data/event.js";
import { checkLog, type EventLog } from "../data/event-log.js";
import { checkModel, type LearnerModel } from "../model.js";

/** A learner model that defines mastery, such as BKT. */
export type MasteryModel<TState> = LearnerModel<TState> & Required<Pick<LearnerModel<TState>, "mastery">>;

/**
 * The opportunity at which each learner first met a mastery criterion in each skill they practiced, by learner and
 * skill: 1 for the first attempt at the skill, or `null` when the log ends before.
 */
export type MasteryTable = Readonly<Record<string, Readonly<Record<string, number | null>>>>;

const DEFAULT_LENGTH = 3;
const DEFAULT_THRESHOLD = 0.95;

/** Options of {@link streakMastery}. */
export interface StreakOptions {
  /**
   * The number of correct answers in a row that counts as mastery, as the three of ASSISTments' skill builders.
   *
   * @defaultValue `3`
   */
  readonly length?: number;
}

/** Options of {@link thresholdMastery}. */
export interface ThresholdOptions {
  /**
   * The probability of mastery that counts as mastery.
   *
   * @defaultValue `0.95`
   * @see {@link https://doi.org/10.1007/BF01099821 | Corbett and Anderson (1994)}
   */
  readonly threshold?: number;
}

/** Options of {@link opportunitiesToStreak}. */
export interface OpportunitiesToStreakOptions extends StreakOptions {
  /**
   * The learner's current number of correct answers in a row.
   *
   * @defaultValue `0`
   */
  readonly run?: number;
}

/**
 * Returns the opportunity at which each learner first answered `length` attempts at each skill correctly in a row. Only
 * an outcome of 1 is correct; any other ends the run.
 *
 * @throws `DataError` when the log is malformed or the length is not a positive integer.
 */
export function streakMastery(log: EventLog, options: StreakOptions = {}): MasteryTable {
  checkOptions(options);
  const { length = DEFAULT_LENGTH } = options;
  checkPositiveInteger(length, "The length");
  checkLog(log);
  const runs = new Map<string, number>();
  return firstOpportunities(log, (event) => (skill) => {
    const key = `${event.learner}\u0000${skill}`;
    const run = event.outcome === 1 ? (runs.get(key) ?? 0) + 1 : 0;
    runs.set(key, run);
    return run >= length;
  });
}

/**
 * Returns the opportunity at which each learner's probability of mastery by the model first reached the threshold
 * after an attempt at each skill, updating each learner's state from `initialState` through the log.
 *
 * @throws `DataError` when the model is not a learner model that defines mastery, the log is malformed, the threshold
 * is not from 0 to 1, or the model rejects an event.
 */
export function thresholdMastery<TState>(
  model: MasteryModel<TState>,
  log: EventLog,
  options: ThresholdOptions = {},
): MasteryTable {
  const message = "thresholdMastery needs a model that defines mastery, as BKT and KST do";
  checkModel(model, message);
  checkMethod(model, "mastery", message);
  checkOptions(options);
  const { threshold = DEFAULT_THRESHOLD } = options;
  checkUnitInterval(threshold, "The threshold");
  checkLog(log);
  const states = new Map<string, TState>();
  return firstOpportunities(log, (event) => {
    const state = model.update(states.get(event.learner) ?? model.initialState(event.learner), event);
    states.set(event.learner, state);
    return (skill) => model.mastery(state, skill) >= threshold;
  });
}

/**
 * Returns the expected number of further attempts until `length` correct answers in a row, each attempt correct with
 * `probability`: (p⁻ⁿ - p⁻ʳ) / (1 - p) from a run of r, which is n - r at p = 1 and `Infinity` at p = 0.
 *
 * @throws `DataError` when the probability is not from 0 to 1, the length not a positive integer, or the run not an
 * integer from 0 to the length.
 */
export function opportunitiesToStreak(probability: number, options: OpportunitiesToStreakOptions = {}): number {
  checkOptions(options);
  const { length = DEFAULT_LENGTH, run = 0 } = options;
  checkPositiveInteger(length, "The length");
  checkUnitInterval(probability, "The probability");
  if (!(Number.isSafeInteger(run) && run >= 0 && run <= length)) {
    throw new DataError(`The run must be an integer from 0 to the length ${length}, found ${describe(run)}.`);
  }
  if (run === length) return 0;
  if (probability === 1) return length - run;
  if (probability === 0) return Number.POSITIVE_INFINITY;
  // p⁻ʳ (p^(r - n) - 1) rather than p⁻ⁿ - p⁻ʳ, which is Infinity - Infinity where both overflow.
  return (probability ** -run * (probability ** (run - length) - 1)) / (1 - probability);
}

/**
 * Goes through the log: `observe` takes each event and returns whether the criterion is met for each of its
 * skills after it. Records, per learner and skill, the opportunity of the first event after which it is.
 */
function firstOpportunities(log: EventLog, observe: (event: Event) => (skill: string) => boolean): MasteryTable {
  const table = new Map<string, Map<string, number | null>>();
  const counts = new Map<string, number>();
  for (const event of log) {
    let skills = table.get(event.learner);
    if (skills === undefined) {
      skills = new Map();
      table.set(event.learner, skills);
    }
    const met = observe(event);
    for (const skill of event.skills) {
      const key = `${event.learner}\u0000${skill}`;
      const count = (counts.get(key) ?? 0) + 1;
      counts.set(key, count);
      const reached = met(skill);
      if ((skills.get(skill) ?? null) === null) skills.set(skill, reached ? count : null);
    }
  }
  return Object.fromEntries([...table].map(([learner, skills]) => [learner, Object.fromEntries(skills)]));
}

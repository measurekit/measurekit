import {
  checkIntegerFrom,
  checkMethod,
  checkOptions,
  checkPositiveInteger,
  checkUnitInterval,
  DataError,
  describe,
} from "@measurekit/core";
import { checkLog, checkObservedHasSkills, type Event, type EventLog, isObserved } from "../data/internal.js";
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

/** Options of {@link movingAverageMastery}. */
export interface MovingAverageOptions {
  /**
   * The weight d of the average on its past, from 0 to 1: the average of a skill moves by 1 - d of its distance to the
   * outcome. Required, as its values depend on the domain.
   */
  readonly decay: number;
  /** The value of the average that counts as mastery, from 0 to 1. Required, for the same reason. */
  readonly threshold: number;
}

/** Options of {@link windowMastery}. */
export interface WindowOptions {
  /** The number of outcomes of 1 among the last `window` that counts as mastery: an integer from 1 to `window`. Required. */
  readonly correct: number;
  /** The number of the latest outcomes to look at, a positive integer. Required. */
  readonly window: number;
}

/** Options of {@link predictedMastery}. */
export interface PredictedOptions {
  /**
   * The predicted probability of a correct answer that counts as mastery, from 0 to 1. Required: it does not mean what
   * the threshold of {@link thresholdMastery} means, as a prediction of 0.95 is out of reach at a slip above 0.05.
   */
  readonly threshold: number;
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
 * an outcome of 1 is correct; any other outcome ends the run. An event without an outcome is no opportunity and leaves
 * the run as it is.
 *
 * @param log - The log to take the opportunities of.
 * @param options - The length of the streak.
 * @returns The opportunity of each learner at each skill; absent where the criterion was never met.
 * @throws `DataError` when the log is malformed or the length is not a positive integer.
 */
export function streakMastery(log: EventLog, options: StreakOptions = {}): MasteryTable {
  checkOptions(options);
  const { length = DEFAULT_LENGTH } = options;
  checkPositiveInteger(length, "The length");
  checkLog(log);
  const runs = new Map<string, number>();
  return firstOpportunities(log, (event) => (skill, outcome) => {
    const key = `${event.learner}\u0000${skill}`;
    const run = outcome === 1 ? (runs.get(key) ?? 0) + 1 : 0;
    runs.set(key, run);
    return run >= length;
  });
}

/**
 * Returns the opportunity at which each learner's moving average of the outcomes at each skill first reached the
 * threshold: m moves to d m + (1 - d) y at each attempt from m = 0. Outcomes from 0 to 1 count as partial credit. An
 * event without an outcome is no opportunity and leaves the average as it is.
 *
 * @param log - The log to take the opportunities of.
 * @param options - The decay of the average and the threshold it must reach.
 * @returns The opportunity of each learner at each skill; absent where the criterion was never met.
 * @throws `DataError` when the log is malformed, or the decay or the threshold is not from 0 to 1.
 * @see {@link https://doi.org/10.1145/3079628.3079667 | Pelánek and Řihák (2017)}
 */
export function movingAverageMastery(log: EventLog, options: MovingAverageOptions): MasteryTable {
  checkOptions(options);
  const { decay, threshold } = options;
  checkUnitInterval(decay, "The option decay");
  checkUnitInterval(threshold, "The option threshold");
  checkLog(log);
  const averages = new Map<string, number>();
  return firstOpportunities(log, (event) => (skill, outcome) => {
    const key = `${event.learner}\u0000${skill}`;
    const average = decay * (averages.get(key) ?? 0) + (1 - decay) * outcome;
    averages.set(key, average);
    return average >= threshold;
  });
}

/**
 * Returns the opportunity at which each learner first had `correct` outcomes of 1 among their last `window` attempts at
 * each skill, the attempts so far where there are fewer than `window`. Only an outcome of 1 is correct. An event without
 * an outcome is no opportunity and leaves the window as it is.
 *
 * @param log - The log to take the opportunities of.
 * @param options - The number of attempts in the window and of correct outcomes it needs.
 * @returns The opportunity of each learner at each skill; absent where the criterion was never met.
 * @throws `DataError` when the log is malformed, `window` is not a positive integer, or `correct` is not an integer from
 * 1 to `window`.
 * @see {@link https://doi.org/10.1145/3079628.3079667 | Pelánek and Řihák (2017)}
 */
export function windowMastery(log: EventLog, options: WindowOptions): MasteryTable {
  checkOptions(options);
  const { correct, window } = options;
  checkPositiveInteger(window, "The option window");
  checkIntegerFrom(correct, "The option correct", 1);
  if (correct > window) {
    throw new DataError(`The option correct must be at most the window ${window}, found ${correct}.`);
  }
  checkLog(log);
  const recent = new Map<string, number[]>();
  return firstOpportunities(log, (event) => (skill, outcome) => {
    const key = `${event.learner}\u0000${skill}`;
    const latest = [...(recent.get(key) ?? []), outcome === 1 ? 1 : 0].slice(-window);
    recent.set(key, latest);
    return latest.reduce((sum, flag) => sum + flag, 0) >= correct;
  });
}

/** Returns a function that updates the state of an event's learner with the event and returns the new state. */
function tracking<TState>(model: LearnerModel<TState>): (event: Event) => TState {
  const states = new Map<string, TState>();
  return (event) => {
    const state = model.update(states.get(event.learner) ?? model.initialState(event.learner), event);
    states.set(event.learner, state);
    return state;
  };
}

/**
 * Returns the opportunity at which each learner's probability of mastery by the model first reached the threshold
 * after an attempt at each skill, updating each learner's state from `initialState` through the log. An event without
 * an outcome updates the state but is no opportunity, and the criterion is tested only after events with an outcome, so
 * a lesson alone never records mastery.
 *
 * @param model - A learner model that defines mastery.
 * @param log - The log to replay.
 * @param options - The threshold.
 * @returns The opportunity of each learner at each skill; absent where the criterion was never met.
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
  const advance = tracking(model);
  return firstOpportunities(log, (event) => {
    const state = advance(event);
    return (skill) => model.mastery(state, skill) >= threshold;
  });
}

/**
 * Returns the opportunity at which the model's prediction for each learner's attempt, made after the attempt's own
 * outcome, first reached the threshold, for each skill of the attempt. It serves the models that define no mastery,
 * such as logistic models and Elo. An event without an outcome updates the model, is no opportunity and is never
 * predicted, so a lesson alone never records mastery.
 *
 * @param model - The learner model.
 * @param log - The log to replay.
 * @param options - The threshold.
 * @returns The opportunity of each learner at each skill; absent where the criterion was never met.
 * @throws `DataError` when the model is not a learner model, the log is malformed, the threshold is not from 0 to 1, an
 * event with an outcome has no skills, as the table is by skill, or the model rejects an event or an attempt.
 */
export function predictedMastery<TState>(
  model: LearnerModel<TState>,
  log: EventLog,
  options: PredictedOptions,
): MasteryTable {
  checkModel(model, "predictedMastery needs a learner model");
  checkOptions(options);
  const { threshold } = options;
  checkUnitInterval(threshold, "The option threshold");
  checkLog(log);
  log.forEach((event, index) => {
    checkObservedHasSkills(event, `Event ${index + 1}`, "mastery table");
  });
  const advance = tracking(model);
  return firstOpportunities(log, (event) => {
    const state = advance(event);
    const reached = isObserved(event) && model.predict(state, event) >= threshold;
    return () => reached;
  });
}

/**
 * Returns the expected number of further attempts until `length` correct answers in a row, each attempt correct with
 * `probability`: (p⁻ⁿ - p⁻ʳ) / (1 - p) from a run of r, which is n - r at p = 1 and `Infinity` at p = 0.
 *
 * @param probability - The probability that an attempt is correct, from 0 to 1.
 * @param options - The length of the streak and the current run.
 * @returns The expected number of further attempts; `Infinity` at 0.
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
 * Goes through the log: `observe` takes each event and returns whether the criterion is met for each of its skills after
 * it, given the skill and the event's outcome. Records, per learner and skill, the opportunity of the first event with an
 * outcome after which it is; an event without one is observed, so that a model updates, and counts for nothing else.
 */
function firstOpportunities(
  log: EventLog,
  observe: (event: Event) => (skill: string, outcome: number) => boolean,
): MasteryTable {
  const table = new Map<string, Map<string, number | null>>();
  const counts = new Map<string, number>();
  for (const event of log) {
    const met = observe(event);
    if (!isObserved(event)) continue;
    const skills = table.get(event.learner) ?? new Map<string, number | null>();
    table.set(event.learner, skills);
    for (const skill of event.skills) {
      const key = `${event.learner}\u0000${skill}`;
      const count = (counts.get(key) ?? 0) + 1;
      counts.set(key, count);
      const reached = met(skill, event.outcome);
      if ((skills.get(skill) ?? null) === null) skills.set(skill, reached ? count : null);
    }
  }
  return Object.fromEntries([...table].map(([learner, skills]) => [learner, Object.fromEntries(skills)]));
}

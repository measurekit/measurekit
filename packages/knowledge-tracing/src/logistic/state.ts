import { checkArray, checkRecord, DataError, describe, isUnitInterval } from "@measurekit/core";
import { type Attempt, checkLearnerState, type Event, entry, isObserved, put } from "../data/internal.js";

/**
 * A learner's state in a logistic model: their history with each skill practiced. Plain data, which JSON serializes.
 */
export interface LogisticState {
  /** The learner's id. */
  readonly learner: string;
  /** The learner's history with each skill practiced, by skill id: the events with an outcome. */
  readonly skills: Readonly<Record<string, SkillHistory>>;
  /** The learner's exposures to each skill, by skill id: the events without an outcome; absent until the first. */
  readonly exposures?: Readonly<Record<string, ExposureHistory>>;
  /** The time the learner has spent on events so far, the sum of their durations; present with the practice features. */
  readonly practice?: number;
  /** The learner's history with each item practiced, by item id: the events with an outcome; present with terms over items. */
  readonly items?: Readonly<Record<string, SkillHistory>>;
  /** The learner's history over all the events with an outcome; present with terms over the learner. */
  readonly overall?: SkillHistory;
}

/** What the exposure features need of a learner's events without an outcome at a skill. */
export interface ExposureHistory {
  /** The number of exposures. */
  readonly count: number;
  /** The time of the latest exposure. */
  readonly last: number;
  /** For each decay d of the model's features, by `String(d)`: Σ dᵃ over the exposures, a the number of later ones. */
  readonly decayed: Readonly<Record<string, number>>;
}

/** What the features need of a learner's history with a skill. */
export interface SkillHistory {
  /** The sum of the outcomes of the attempts. */
  readonly successes: number;
  /** The sum of 1 - outcome over the attempts. */
  readonly failures: number;
  /** The time of the last attempt. */
  readonly last: number;
  /**
   * For each decay d of the model's features, by `String(d)`: successes and failures, each attempt weighing d times the
   * next.
   */
  readonly decayed: Readonly<Record<string, DecayedCounts>>;
  /** The outcome of the last attempt; present with the features of recency by outcome. */
  readonly previous?: number;
  /** The time of the first attempt; present with the base features. */
  readonly first?: number;
  /** The learner's practice time at the first attempt; present with the practice features. */
  readonly firstPractice?: number;
  /** For each time scale τ of the dash features, by `String(τ)`: the leaky counts after the last attempt. */
  readonly leaky?: Readonly<Record<string, LeakyCounts>>;
  /** The times of the attempts, in order; present with the features that read them, and then pruned to the window. */
  readonly times?: readonly number[];
  /** The outcomes of the attempts at `times`. */
  readonly outcomes?: readonly number[];
}

/** Counts that each attempt adds to and that decay by exp(-Δt / τ) in time. */
export interface LeakyCounts {
  /** The count of attempts: each adds 1. */
  readonly attempts: number;
  /** The count of successes: each adds its outcome. */
  readonly successes: number;
}

/** What the features that read one kind of history need it to track, which it tracks only then. */
export interface HistoryNeeds {
  /** The decays the features of outcomes need running sums for. */
  readonly decays: readonly number[];
  /** The decays the exposure features need running sums for. */
  readonly exposureDecays: readonly number[];
  /** Whether a feature reads exposures. */
  readonly exposures: boolean;
  /** Whether a feature reads the outcome of the last attempt. */
  readonly previous: boolean;
  /** Whether a feature reads the time of the first attempt. */
  readonly first: boolean;
  /** Whether a feature reads the practice time, which every event's `duration` gives. */
  readonly practice: boolean;
  /** The time scales τ of the dash features. */
  readonly leaks: readonly number[];
  /** Whether a feature reads the times of the attempts. */
  readonly times: boolean;
  /** How far back the times are kept: the largest window, or infinity. */
  readonly horizon: number;
}

/** What the model's terms need a state to track, by the history their features read. */
export interface StateNeeds {
  /** The histories with skills, which a state always keeps. */
  readonly skills: HistoryNeeds;
  /** The histories with items; `null` without a term over items, and then a state keeps none. */
  readonly items: HistoryNeeds | null;
  /** The history over all events; `null` without a term over the learner. */
  readonly overall: HistoryNeeds | null;
}

/** Counts in which each attempt weighs d times the next. */
export interface DecayedCounts {
  /** Σ dᵃ y over the attempts, a the number of later attempts. */
  readonly successes: number;
  /** Σ dᵃ (1 - y) over the attempts. */
  readonly failures: number;
}

/** Returns the history after an attempt with outcome y at `time`, when the learner had `practice` before it. */
export function nextHistory(
  history: SkillHistory | undefined,
  outcome: number,
  time: number,
  practice: number,
  needs: HistoryNeeds,
): SkillHistory {
  return {
    successes: (history?.successes ?? 0) + outcome,
    failures: (history?.failures ?? 0) + (1 - outcome),
    last: time,
    decayed: Object.fromEntries(
      needs.decays.map((decay) => {
        const sums = history === undefined ? undefined : entry(history.decayed, String(decay));
        const successes = decay * (sums?.successes ?? 0) + outcome;
        return [String(decay), { successes, failures: decay * (sums?.failures ?? 0) + (1 - outcome) }];
      }),
    ),
    ...(needs.previous && { previous: outcome }),
    ...(needs.first && { first: history?.first ?? time }),
    ...(needs.practice && { firstPractice: history?.firstPractice ?? practice }),
    ...(needs.leaks.length > 0 && { leaky: nextLeaky(history, outcome, time, needs.leaks) }),
    ...(needs.times && nextTimes(history, outcome, time, needs.horizon)),
  };
}

/** Returns the leaky counts of a history at `time`, decayed by exp(-Δt / τ) from its last attempt. */
export function leakyAt(history: SkillHistory, tau: number, time: number): LeakyCounts {
  const counts = entry(history.leaky, String(tau));
  const decay = Math.exp(-(time - history.last) / tau);
  return { attempts: (counts?.attempts ?? 0) * decay, successes: (counts?.successes ?? 0) * decay };
}

/** Returns the leaky counts of each time scale after an attempt: the decayed counts, and the attempt's own. */
function nextLeaky(
  history: SkillHistory | undefined,
  outcome: number,
  time: number,
  taus: readonly number[],
): Record<string, LeakyCounts> {
  return Object.fromEntries(
    taus.map((tau) => {
      const before = history === undefined ? { attempts: 0, successes: 0 } : leakyAt(history, tau, time);
      return [String(tau), { attempts: before.attempts + 1, successes: before.successes + outcome }];
    }),
  );
}

/** Returns the times and outcomes of the attempts with this one added, those before `time - horizon` left out. */
function nextTimes(
  history: SkillHistory | undefined,
  outcome: number,
  time: number,
  horizon: number,
): { times: number[]; outcomes: number[] } {
  const times = [...(history?.times ?? []), time];
  const outcomes = [...(history?.outcomes ?? []), outcome];
  const kept = times.flatMap((at, i) => (time - at <= horizon ? [i] : []));
  return { times: kept.map((i) => times[i] ?? 0), outcomes: kept.map((i) => outcomes[i] ?? 0) };
}

/** Returns the exposures after an event without an outcome at `time`. */
function nextExposure(exposure: ExposureHistory | undefined, time: number, decays: readonly number[]): ExposureHistory {
  return {
    count: (exposure?.count ?? 0) + 1,
    last: time,
    decayed: Object.fromEntries(
      decays.map((decay) => {
        const sum = exposure === undefined ? undefined : entry(exposure.decayed, String(decay));
        return [String(decay), decay * (sum ?? 0) + 1];
      }),
    ),
  };
}

/** Returns whether any history tracks the practice time, which every event's `duration` adds to. */
function tracksPractice(needs: StateNeeds): boolean {
  return needs.skills.practice || (needs.items?.practice ?? false) || (needs.overall?.practice ?? false);
}

/** What an event changes in a state: the histories it replaces, and the practice time after it. */
interface Changes {
  /** The histories at the event's skills after an attempt. */
  readonly skills: readonly (readonly [string, SkillHistory])[];
  /** The exposures to the event's skills after an event without an outcome that a feature reads. */
  readonly exposures: readonly (readonly [string, ExposureHistory])[];
  /** The history at the event's item after an attempt, when a term reads items. */
  readonly item?: readonly [string, SkillHistory];
  /** The history over all attempts after an attempt, when a term reads the learner's. */
  readonly overall?: SkillHistory;
  /** The time the learner has spent, when a feature reads it. */
  readonly practice?: number;
}

/**
 * Returns what an event changes in the state, the state and event checked and the event not earlier than the state's
 * attempts: the histories of an event with an outcome, the exposures of one without when a feature reads them, and the
 * practice time of every event when a feature reads it. Without a feature that reads exposures, an event without an
 * outcome leaves the learner as they were.
 */
function changesOf(state: LogisticState, event: Event, needs: StateNeeds): Changes {
  const practice = state.practice ?? 0;
  const spent = tracksPractice(needs) ? { practice: practice + (event.duration ?? 0) } : {};
  if (isObserved(event)) {
    const next = (history: SkillHistory | undefined, kind: HistoryNeeds): SkillHistory =>
      nextHistory(history, event.outcome, event.time, practice, kind);
    return {
      skills: event.skills.map((skill) => [skill, next(entry(state.skills, skill), needs.skills)]),
      exposures: [],
      ...(needs.items !== null && { item: [event.item, next(entry(state.items, event.item), needs.items)] }),
      ...(needs.overall !== null && { overall: next(state.overall, needs.overall) }),
      ...spent,
    };
  }
  return {
    skills: [],
    exposures: needs.skills.exposures
      ? event.skills.map((skill) => [
          skill,
          nextExposure(entry(state.exposures, skill), event.time, needs.skills.exposureDecays),
        ])
      : [],
    ...spent,
  };
}

/**
 * Returns the state after an event, the state and event checked and the event not earlier than the state's attempts.
 * The state is not changed: the new state has copies of its records.
 */
export function advanceState(state: LogisticState, event: Event, needs: StateNeeds): LogisticState {
  const changes = changesOf(state, event, needs);
  return finishState({
    learner: state.learner,
    skills: { ...state.skills, ...Object.fromEntries(changes.skills) },
    exposures:
      changes.exposures.length > 0
        ? { ...state.exposures, ...Object.fromEntries(changes.exposures) }
        : copy(state.exposures),
    items: changes.item === undefined ? copy(state.items) : { ...state.items, ...Object.fromEntries([changes.item]) },
    overall: changes.overall ?? state.overall,
    practice: changes.practice ?? state.practice,
  });
}

/**
 * A state that a pass over a log advances in place, which the pass owns: no other state shares its records. It avoids
 * the copies of the records that {@link advanceState} makes for each event, which grow with the learner's history.
 */
export interface WorkingState {
  readonly learner: string;
  skills: Record<string, SkillHistory>;
  exposures?: Record<string, ExposureHistory>;
  items?: Record<string, SkillHistory>;
  overall?: SkillHistory;
  practice?: number;
}

/** Returns a learner's working state before any event. */
export function startState(learner: string): WorkingState {
  return { learner, skills: {} };
}

/** Advances a working state in place to the state {@link advanceState} returns for the event. */
export function advanceInPlace(state: WorkingState, event: Event, needs: StateNeeds): void {
  const changes = changesOf(state, event, needs);
  for (const [skill, history] of changes.skills) put(state.skills, skill, history);
  if (changes.exposures.length > 0) {
    state.exposures ??= {};
    for (const [skill, exposure] of changes.exposures) put(state.exposures, skill, exposure);
  }
  if (changes.item !== undefined) {
    state.items ??= {};
    put(state.items, changes.item[0], changes.item[1]);
  }
  if (changes.overall !== undefined) state.overall = changes.overall;
  if (changes.practice !== undefined) state.practice = changes.practice;
}

/** Returns the state a working state has reached, with its fields in the order of every state; the records are shared. */
export function finishState(state: WorkingState | LogisticState): LogisticState {
  const { learner, skills, exposures, items, overall, practice } = state;
  return {
    learner,
    skills,
    ...(exposures !== undefined && { exposures }),
    ...(items !== undefined && { items }),
    ...(overall !== undefined && { overall }),
    ...(practice !== undefined && { practice }),
  };
}

/** Returns a copy of a record of histories, or `undefined` for none. */
function copy<T>(record: Readonly<Record<string, T>> | undefined): Record<string, T> | undefined {
  return record === undefined ? undefined : { ...record };
}

/**
 * Checks that an event has the duration the practice features read, in the units of its time.
 *
 * @param label - Names the event in the error message, such as `Event 3`.
 * @throws `DataError` when a feature reads durations and the event has none.
 */
export function checkDuration(event: Event, label: string, needs: StateNeeds): void {
  if (tracksPractice(needs) && event.duration === undefined) {
    throw new DataError(`${label} needs a duration, which the features basePractice read for the practice time.`);
  }
}

/** What a history that no term reads needs: nothing, but that it is well formed. */
const NO_NEEDS: HistoryNeeds = {
  decays: [],
  exposureDecays: [],
  exposures: false,
  previous: false,
  first: false,
  practice: false,
  leaks: [],
  times: false,
  horizon: 0,
};

/**
 * Checks a state, which may come from storage, throwing a `DataError` that names the first problem found. `needs` are
 * what the model's terms need a state to track.
 */
export function checkState(state: LogisticState, needs: StateNeeds): void {
  const learner = checkLearnerState(state, "logistic");
  const label = `The logistic state of learner "${learner}"`;
  checkRecord(state.skills, `${label} needs skills, an object of histories by skill id`);
  checkHistories(state.skills, `${label}, skill`, needs.skills);
  checkExposures(state, label, needs.skills.exposureDecays);
  if (state.items !== undefined) {
    checkRecord(state.items, `${label} needs items, an object of histories by item id`);
    checkHistories(state.items, `${label}, item`, needs.items ?? NO_NEEDS);
  }
  if (state.overall !== undefined) checkHistory(state.overall, `${label}, overall history,`, needs.overall ?? NO_NEEDS);
  if (state.practice !== undefined) checkCount(state.practice, `${label} needs practice, the time spent`);
}

/** Checks the histories of a record, by the id of what they count over, which `kind` names. */
function checkHistories(histories: Readonly<Record<string, SkillHistory>>, kind: string, needs: HistoryNeeds): void {
  for (const [id, history] of Object.entries(histories)) checkHistory(history, `${kind} "${id}",`, needs);
}

/** Checks one history against what its features read. */
function checkHistory(history: SkillHistory, prefix: string, needs: HistoryNeeds): void {
  checkRecord(history, `${prefix} must be an object`);
  checkCounts(history, prefix, needs.decays);
  checkTracked(history, prefix, needs);
}

/** Checks the counts of a history: its successes, failures, last time and decayed counts. */
function checkCounts(history: SkillHistory, prefix: string, decays: readonly number[]): void {
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

/**
 * Checks the fields of a history that only some features track: each must be there when a feature needs it, and valid
 * when it is there.
 */
function checkTracked(history: SkillHistory, prefix: string, needs: HistoryNeeds): void {
  const missing = (field: string, what: string): DataError =>
    new DataError(`${prefix} lacks ${field}, ${what}, which the model's features need.`);
  if (needs.previous && history.previous === undefined) throw missing("previous", "the outcome of the last attempt");
  if (history.previous !== undefined && !isUnitInterval(history.previous)) {
    throw new DataError(`${prefix} needs previous from 0 to 1, found ${describe(history.previous)}.`);
  }
  for (const [field, needed, value] of [
    ["first", needs.first, history.first],
    ["firstPractice", needs.practice, history.firstPractice],
  ] as const) {
    if (needed && value === undefined) throw missing(field, "a time");
    if (value !== undefined && !Number.isFinite(value)) {
      throw new DataError(`${prefix} needs ${field} to be a finite number, found ${describe(value)}.`);
    }
  }
  checkLeaky(history, prefix, needs.leaks);
  checkTimes(history, prefix, needs.times);
}

function checkLeaky(history: SkillHistory, prefix: string, taus: readonly number[]): void {
  for (const tau of taus) {
    if (entry(history.leaky, String(tau)) === undefined) {
      throw new DataError(`${prefix} lacks the leaky counts of scale ${tau}, which the model's features need.`);
    }
  }
  if (history.leaky === undefined) return;
  checkRecord(history.leaky, `${prefix} needs leaky, an object of counts by scale`);
  for (const [tau, counts] of Object.entries(history.leaky)) {
    checkRecord(counts, `${prefix} needs an object of leaky counts for scale ${tau}`);
    checkCount(counts.attempts, `${prefix} needs leaky attempts for scale ${tau}`);
    checkCount(counts.successes, `${prefix} needs leaky successes for scale ${tau}`);
  }
}

function checkTimes(history: SkillHistory, prefix: string, needed: boolean): void {
  const { times, outcomes } = history;
  if (times === undefined && outcomes === undefined) {
    if (needed)
      throw new DataError(`${prefix} lacks times, the times of the attempts, which the model's features need.`);
    return;
  }
  checkArray(times, `${prefix} needs times, an array of times`);
  checkArray(outcomes, `${prefix} needs outcomes, an array of outcomes`);
  if (times?.length !== outcomes?.length || !times?.every((time) => Number.isFinite(time))) {
    throw new DataError(`${prefix} needs times of finite numbers and one outcome for each.`);
  }
  if (!outcomes?.every(isUnitInterval)) throw new DataError(`${prefix} needs outcomes from 0 to 1.`);
}

/** Checks a state's exposures, which are optional: a state stored before there were any has none. */
function checkExposures(state: LogisticState, label: string, exposureDecays: readonly number[]): void {
  if (state.exposures === undefined) return;
  checkRecord(state.exposures, `${label} needs exposures, an object of exposures by skill id`);
  for (const [skill, exposure] of Object.entries(state.exposures)) {
    const prefix = `${label}, exposures to skill "${skill}",`;
    checkRecord(exposure, `${prefix} must be an object`);
    checkCount(exposure.count, `${prefix} needs count`);
    if (!Number.isFinite(exposure.last)) {
      throw new DataError(`${prefix} needs last, the time of the last exposure, found ${describe(exposure.last)}.`);
    }
    checkRecord(exposure.decayed, `${prefix} needs decayed, an object of decayed counts by decay`);
    for (const decay of exposureDecays) {
      if (entry(exposure.decayed, String(decay)) === undefined) {
        throw new DataError(`${prefix} lacks the decayed count of decay ${decay}, which the model's features need.`);
      }
    }
    for (const [decay, sum] of Object.entries(exposure.decayed)) {
      checkCount(sum, `${prefix} needs a decayed count for decay ${decay}`);
    }
  }
}

function checkCount(value: number, message: string): void {
  if (!(Number.isFinite(value) && value >= 0)) {
    throw new DataError(`${message}, a non-negative number, found ${describe(value)}.`);
  }
}

/**
 * Checks that an attempt is not earlier than the learner's last attempt or exposure at any of its skills, nor, when a
 * term reads them, their last attempt at the item or at any item, as features count the time since it.
 *
 * @param label - Names the attempt in the error message, such as `The event`.
 */
export function checkOrder(state: LogisticState, attempt: Attempt, label: string, needs: StateNeeds): void {
  for (const skill of attempt.skills) {
    const history = entry(state.skills, skill);
    if (history !== undefined && attempt.time < history.last) {
      throw new DataError(
        `${label} at time ${attempt.time} is earlier than the learner's last attempt at skill "${skill}", at time ${history.last}.`,
      );
    }
    const exposure = entry(state.exposures, skill);
    if (exposure !== undefined && attempt.time < exposure.last) {
      throw new DataError(
        `${label} at time ${attempt.time} is earlier than the learner's last exposure to skill "${skill}", at time ${exposure.last}.`,
      );
    }
  }
  const item = needs.items === null ? undefined : entry(state.items, attempt.item);
  if (item !== undefined && attempt.time < item.last) {
    throw new DataError(
      `${label} at time ${attempt.time} is earlier than the learner's last attempt at item "${attempt.item}", at time ${item.last}.`,
    );
  }
  if (needs.overall !== null && state.overall !== undefined && attempt.time < state.overall.last) {
    throw new DataError(
      `${label} at time ${attempt.time} is earlier than the learner's last attempt, at time ${state.overall.last}.`,
    );
  }
}

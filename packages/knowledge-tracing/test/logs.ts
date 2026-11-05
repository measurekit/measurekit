import fc from "fast-check";
import { expect } from "vitest";
import { buildEventLog, type EventLog } from "../src/data/index.js";
import type { LearnerModel } from "../src/index.js";
import { defined } from "./assertions.js";

/** Runs a log through a model online, returning each event's prediction and the learners' final states. */
export function online<TState>(
  model: LearnerModel<TState>,
  log: EventLog,
): { predictions: number[]; states: Map<string, TState> } {
  const states = new Map<string, TState>();
  const predictions = log.map((event) => {
    const state = states.get(event.learner) ?? model.initialState(event.learner);
    const prediction = model.predict(state, event);
    states.set(event.learner, model.update(state, event));
    return prediction;
  });
  return { predictions, states };
}

/**
 * Returns the log of events that a benchmark fixture holds as columns: the learner, item, time and outcome of each event,
 * and its skill when the fixture has them, a skill for each event. An outcome may be `null`.
 */
export function columnsLog(columns: {
  readonly learner: readonly string[];
  readonly item: readonly string[];
  readonly time: readonly number[];
  readonly outcome: readonly (number | null)[];
  readonly skill?: readonly string[];
}): EventLog {
  return buildEventLog(
    columns.learner.map((learner, i) => ({
      learner,
      item: defined(columns.item[i]),
      skills: columns.skill === undefined ? [] : [defined(columns.skill[i])],
      time: defined(columns.time[i]),
      outcome: columns.outcome[i] ?? null,
    })),
  );
}

/**
 * Returns the log with a lesson, an event without an outcome at the same learner, skills and time, after every
 * `every`-th event.
 */
export function withLessons(log: EventLog, every: number): EventLog {
  return log.flatMap((event, index) =>
    index % every === 0
      ? [
          event,
          { learner: event.learner, item: `lesson${index}`, skills: event.skills, time: event.time, outcome: null },
        ]
      : [event],
  );
}

/**
 * Expects a model to be unaffected by events without an outcome: whatever the lessons' spacing, the fit has the same
 * parameters, states, log-likelihood and information criteria as the log without them, and the same predictions at the
 * events with an outcome, which the lessons leave `null`.
 */
export function expectUnaffectedByLessons(model: LearnerModel, log: EventLog): void {
  const plain = model.fit(log);
  fc.assert(
    fc.property(fc.integer({ min: 1, max: 7 }), (every) => {
      const lessons = withLessons(log, every);
      const fit = model.fit(lessons);
      expect(fit.model.parameters).toEqual(plain.model.parameters);
      expect(fit.states).toEqual(plain.states);
      expect(fit.logLikelihood).toBe(plain.logLikelihood);
      expect([fit.aic, fit.bic, fit.converged, fit.iterations]).toEqual([
        plain.aic,
        plain.bic,
        plain.converged,
        plain.iterations,
      ]);
      expect(fit.predictions.filter((prediction) => prediction !== null)).toEqual(plain.predictions);
      expect(fit.predictions.filter((prediction) => prediction === null)).toHaveLength(lessons.length - log.length);
    }),
    { numRuns: 7 },
  );
}

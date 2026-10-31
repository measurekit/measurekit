import { expect } from "vitest";
import { type EventLog, eventLog, type LearnerModel } from "../src/index.js";

/** Expects `actual` within `tolerance` of `expected`, or `null` when `expected` is `null`. */
export function expectClose(actual: number | null, expected: number | null, tolerance: number, label = ""): void {
  if (expected === null) {
    expect(actual, label).toBeNull();
    return;
  }
  expect(actual, label).not.toBeNull();
  expect(Math.abs((actual ?? Number.NaN) - expected), label).toBeLessThanOrEqual(tolerance);
}

/** Expects two lists of numbers to have equal lengths and entries within `tolerance`. */
export function expectAllClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: number,
  label = "",
): void {
  expect(actual.length, label).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expectClose(actual[i] ?? null, expected[i] ?? null, tolerance, `${label}[${i}]`);
  }
}

/** Returns the value, failing the test when it is `undefined` or `null`. */
export function defined<T>(value: T | null | undefined): T {
  if (value === undefined || value === null) throw new Error(`Expected a value, found ${value}.`);
  return value;
}

/** Returns the Pearson correlation of paired values. */
export function correlation(x: readonly number[], y: readonly number[]): number {
  const mean = (values: readonly number[]): number => values.reduce((sum, value) => sum + value, 0) / values.length;
  const mx = mean(x);
  const my = mean(y);
  let xy = 0;
  let xx = 0;
  let yy = 0;
  x.forEach((value, i) => {
    const dx = value - mx;
    const dy = defined(y[i]) - my;
    xy += dx * dy;
    xx += dx * dx;
    yy += dy * dy;
  });
  return xy / Math.sqrt(xx * yy);
}

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
 * and its skill when the fixture has them, a skill for each event.
 */
export function columnsLog(columns: {
  readonly learner: readonly string[];
  readonly item: readonly string[];
  readonly time: readonly number[];
  readonly outcome: readonly number[];
  readonly skill?: readonly string[];
}): EventLog {
  return eventLog(
    columns.learner.map((learner, i) => ({
      learner,
      item: defined(columns.item[i]),
      skills: columns.skill === undefined ? [] : [defined(columns.skill[i])],
      time: defined(columns.time[i]),
      outcome: defined(columns.outcome[i]),
    })),
  );
}

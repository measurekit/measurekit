import { describe, expect, it } from "vitest";
import {
  type Attempt,
  type BktOptions,
  type BktParameters,
  type BktState,
  bkt,
  type EloOptions,
  type EloParameters,
  type EloState,
  type Event,
  elo,
  type IrtOptions,
  type IrtParameters,
  type IrtState,
  irt,
  type KstOptions,
  type KstParameters,
  type KstState,
  kst,
  type LearnerModel,
  type LogisticOptions,
  type LogisticParameters,
  type LogisticState,
  logistic,
} from "../src/index.js";
import stored from "./fixtures/stored-state.json" with { type: "json" };
import { defined } from "./helpers.js";

/** What a build wrote for one model: its parameters and states, and what the model made of them one event later. */
interface Stored<TOptions, TState, TParameters> {
  readonly options: TOptions;
  readonly parameters: TParameters;
  readonly states: Readonly<Record<string, TState>>;
  readonly attempt: Attempt;
  readonly prediction: number;
  readonly answered: { readonly event: Event; readonly state: TState };
  readonly lesson: { readonly event: Event; readonly state: TState };
}

// Parsed rather than imported as typed: each model reads its stored data as its own types have it.
function storedOf<TOptions, TState, TParameters>(
  name: keyof typeof stored.models,
): Stored<TOptions, TState, TParameters> {
  return JSON.parse(JSON.stringify(stored.models[name]));
}

/**
 * Expects a model built from stored parameters to read the stored states: every state predicts, and the first learner's
 * gives the stored prediction and the stored states after an answer and after an event without an outcome.
 */
function expectReadable<TState, TParameters>(
  model: LearnerModel<TState, TParameters>,
  found: Stored<unknown, TState, TParameters>,
): void {
  for (const [learner, state] of Object.entries(found.states)) {
    expect(Number.isFinite(model.predict(state, { ...found.attempt, learner })), learner).toBe(true);
  }
  const state = defined(found.states[found.attempt.learner]);
  expect(model.predict(state, found.attempt)).toBe(found.prediction);
  expect(model.update(state, found.answered.event)).toEqual(found.answered.state);
  expect(model.update(state, found.lesson.event)).toEqual(found.lesson.state);
}

describe("stored states and parameters", () => {
  it("are read by BKT as the build that wrote them did", () => {
    const found = storedOf<BktOptions, BktState, BktParameters>("bkt");
    expectReadable(bkt({ ...found.options, parameters: found.parameters }), found);
  });

  it("are read by Elo as the build that wrote them did", () => {
    const found = storedOf<EloOptions, EloState, EloParameters>("elo");
    expectReadable(elo({ ...found.options, parameters: found.parameters }), found);
  });

  it("are read by online IRT as the build that wrote them did", () => {
    const found = storedOf<IrtOptions, IrtState, IrtParameters>("irt");
    expectReadable(irt({ ...found.options, parameters: found.parameters }), found);
  });

  it("are read by KST as the build that wrote them did", () => {
    const found = storedOf<KstOptions, KstState, KstParameters>("kst");
    expectReadable(kst({ ...found.options, parameters: found.parameters }), found);
  });

  it("are read by a logistic model, exposures included, as the build that wrote them did", () => {
    const found = storedOf<LogisticOptions, LogisticState, LogisticParameters>("logistic");
    expectReadable(logistic({ ...found.options, parameters: found.parameters }), found);
  });
});

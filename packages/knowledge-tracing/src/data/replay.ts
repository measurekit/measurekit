import { type Attempt, type Event, isObserved, type ObservedEvent } from "./event.js";
import type { EventLog } from "./event-log.js";

/** A learner model's steps without their checks, for replaying a log that was checked as a whole. */
export interface Steps<TState> {
  readonly initialState: (learner: string) => TState;
  readonly predict: (state: TState, attempt: Attempt) => number;
  readonly update: (state: TState, event: Event) => TState;
}

/** What a model that learns from outcomes alone does at one event with an outcome. */
export interface PassStep<TState> {
  /** The learner's state after the event. */
  readonly state: TState;
  /** The event's prediction, made before it. */
  readonly prediction: number;
  /** The event's term of the log-likelihood. */
  readonly logLikelihood: number;
}

/** The result of a pass over a log, as a model fitted in one pass reports it. */
export interface Pass<TState> {
  /** Each learner's state after the log, by learner id. */
  readonly states: Record<string, TState>;
  /** Each event's prediction, made before the event; `null` for an event without an outcome. */
  readonly predictions: (number | null)[];
  /** The sum over events with an outcome of y ln p + (1 - y) ln(1 - p), with p the event's prediction. */
  readonly logLikelihood: number;
  readonly aic: null;
  readonly bic: null;
  readonly converged: true;
  readonly iterations: 1;
}

/**
 * Passes over a log once, for a model whose parameters are given or updated with the events: `step` takes each event
 * with an outcome, and `idle` each event without one.
 *
 * @defaultValue `idle` leaves the learner's state as it was
 */
export function onePass<TState>(
  log: EventLog,
  initialState: (learner: string) => TState,
  step: (state: TState, event: ObservedEvent) => PassStep<TState>,
  idle: (state: TState, event: Event) => TState = (state) => state,
): Pass<TState> {
  const states = new Map<string, TState>();
  let logLikelihood = 0;
  const predictions = log.map((event) => {
    const state = states.get(event.learner) ?? initialState(event.learner);
    if (!isObserved(event)) {
      states.set(event.learner, idle(state, event));
      return null;
    }
    const result = step(state, event);
    states.set(event.learner, result.state);
    logLikelihood += result.logLikelihood;
    return result.prediction;
  });
  return {
    states: Object.fromEntries(states),
    predictions,
    logLikelihood,
    aic: null,
    bic: null,
    converged: true,
    iterations: 1,
  };
}

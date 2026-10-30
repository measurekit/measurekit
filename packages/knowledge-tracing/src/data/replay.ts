import { outcomeLogLikelihood } from "../math/logistic.js";
import type { Attempt, Event } from "./event.js";
import type { EventLog } from "./event-log.js";

/** A learner model's steps without their checks, for replaying a log that was checked as a whole. */
export interface Steps<TState> {
  readonly initialState: (learner: string) => TState;
  readonly predict: (state: TState, attempt: Attempt) => number;
  readonly update: (state: TState, event: Event) => TState;
}

/** The result of a replay. */
export interface Replay<TState> {
  /** Each learner's state after the log, by learner id. */
  readonly states: Record<string, TState>;
  /** Each event's prediction, made before the event. */
  readonly predictions: number[];
  /** The sum over events of y ln p + (1 - y) ln(1 - p), with p the event's prediction. */
  readonly logLikelihood: number;
}

/** Replays a log through a model's steps: each event is predicted, then its learner's state is updated. */
export function replay<TState>(log: EventLog, steps: Steps<TState>): Replay<TState> {
  const states = new Map<string, TState>();
  let logLikelihood = 0;
  const predictions = log.map((event) => {
    const state = states.get(event.learner) ?? steps.initialState(event.learner);
    const prediction = steps.predict(state, event);
    states.set(event.learner, steps.update(state, event));
    logLikelihood += outcomeLogLikelihood(prediction, event.outcome);
    return prediction;
  });
  return { states: Object.fromEntries(states), predictions, logLikelihood };
}

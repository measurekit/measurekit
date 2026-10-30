import { DataError, describe, isRecord } from "@measurekit/core";
import type { Attempt, Event } from "./data/event.js";
import type { EventLog } from "./data/event-log.js";

/** The methods of {@link LearnerModel} that every model has. */
const MODEL_METHODS = ["fit", "initialState", "update", "predict"] as const;

/**
 * A learner model: parameters, estimated from a log in batch, and a learner state, updated one event at a time online.
 *
 * @remarks
 * Built-in models and models written outside the package implement this interface alike, and evaluation uses only its
 * members.
 *
 * A model is immutable: `update` returns a new state, and `fit` a new model. States and parameters are plain data that
 * JSON serializes, so they can be stored and resumed exactly.
 *
 * @typeParam TState - The state of one learner.
 * @typeParam TParameters - The parameters the model was built with or estimated.
 */
export interface LearnerModel<TState = unknown, TParameters = unknown> {
  /** The model's parameters, such as item difficulties: plain data, from which the model's factory rebuilds it. */
  readonly parameters: TParameters;
  /**
   * Estimates the parameters from a log, starting from this model's own, and returns the model with the estimates; a
   * model whose parameters are given, as IRT's, returns itself, with the states and predictions of the log.
   *
   * @throws `DataError` when the log is malformed or does not fit the model.
   */
  fit(log: EventLog): ModelFit<TState, TParameters>;
  /**
   * Returns the state of a learner before any event. Learner-specific parameters, such as an individual prior, enter
   * here.
   *
   * @throws `DataError` when `learner` is not a non-empty string.
   */
  initialState(learner: string): TState;
  /**
   * Returns the learner's state after an event. The state given is left unchanged.
   *
   * @throws `DataError` when the state or the event is malformed, or the event belongs to another learner.
   */
  update(state: TState, event: Event): TState;
  /**
   * Returns the probability that the learner answers the attempt correctly: the expected outcome, which for an item
   * scored in several categories is the expected score from 0 to 1. An event is an attempt, so a log's events can be
   * predicted, then passed to `update`.
   *
   * @throws `DataError` when the state or the attempt is malformed, or the attempt belongs to another learner.
   */
  predict(state: TState, attempt: Attempt): number;
  /**
   * Returns the probability that the learner has mastered a skill, for models that define mastery.
   *
   * @throws `DataError` when the state is malformed or the model has no parameters for the skill.
   */
  mastery?(state: TState, skill: string): number;
}

/**
 * The outcome of fitting a model to a log.
 *
 * @typeParam TState - The state of one learner.
 * @typeParam TParameters - The model's parameters.
 */
export interface ModelFit<TState = unknown, TParameters = unknown> {
  /** The model with the estimated parameters. */
  readonly model: LearnerModel<TState, TParameters>;
  /** Each learner's state after their events in the log, by learner id. */
  readonly states: Readonly<Record<string, TState>>;
  /** The prediction of each event of the log, in log order, made before the event's outcome was known. */
  readonly predictions: readonly number[];
  /**
   * The log-likelihood ln L of the log under the fitted model: the sum over events of y ln p + (1 - y) ln(1 - p), with p
   * the event's prediction and y its outcome, or, for a model whose learners are drawn from a distribution, the marginal
   * log-likelihood, each learner's parameters integrated out.
   */
  readonly logLikelihood: number;
  /**
   * Akaike's information criterion, 2k - 2 ln L with k estimated parameters, or `null` for a model without k: fitted in
   * one pass, with a penalty, or with parameters estimated elsewhere.
   */
  readonly aic: number | null;
  /** The Bayesian information criterion, k ln n - 2 ln L with n events, or `null` where `aic` is. */
  readonly bic: number | null;
  /** Whether the estimation converged. */
  readonly converged: boolean;
  /** Iterations run, or passes over the log for a model fitted in one pass. */
  readonly iterations: number;
}

/**
 * Checks that a value is a learner model: an object with the methods of {@link LearnerModel}.
 *
 * @param message - Says what the value should be, such as `Cross-validation needs a learner model, such as bkt()`.
 * @throws `DataError` when it is not.
 */
export function checkModel(model: unknown, message: string): asserts model is LearnerModel {
  const valid = isRecord(model) && MODEL_METHODS.every((method) => typeof model[method] === "function");
  if (!valid) throw new DataError(`${message}, found ${describe(model)}.`);
}

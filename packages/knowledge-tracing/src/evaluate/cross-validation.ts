import { checkRecord, DataError } from "@measurekit/core";
import { type Event, isObserved } from "../data/event.js";
import { checkLog } from "../data/event-log.js";
import type { Fold } from "../data/split.js";
import { checkModel, type LearnerModel } from "../model.js";
import { type Metrics, metrics } from "./metrics.js";
import { observedPairs } from "./observed-pairs.js";

/** A model's predictions of one fold's test log, and their metrics. */
export interface FoldEvaluation {
  /** The prediction of each test event, in the order of the test log: `null` for an event without an outcome. */
  readonly predictions: readonly (number | null)[];
  /** The metrics of the events with an outcome, or `null` when the test log has none. */
  readonly metrics: Metrics | null;
}

/** A model's evaluation on folds. */
export interface CrossValidation {
  /** The evaluation of each fold, in order. */
  readonly folds: readonly FoldEvaluation[];
  /** The metrics of the predictions of all folds together, over the events with an outcome. */
  readonly metrics: Metrics;
}

/**
 * Evaluates a model on folds, such as those of `splitByLearner`, `splitByItem` and `splitByTime`, or a single fold of a
 * training and a test log.
 *
 * @remarks
 * For each fold, the model is fitted to the training log; then every learner of the test log starts from `initialState`
 * and goes through their events of both logs in time order, training events first at equal times. Each test event is
 * predicted before the state is updated with it; training events only update the state. A learner's earlier events thus
 * inform the predictions of their later ones, as they would online. An event without an outcome updates the state and
 * has a `null` prediction, and the metrics use the events with an outcome.
 *
 * @throws `DataError` when the model is not a learner model, there are no folds, a log is malformed, the model
 * rejects the data, or no test log has an event with an outcome.
 */
export function crossValidate<TState>(model: LearnerModel<TState>, folds: readonly Fold[]): CrossValidation {
  checkModel(model, "Cross-validation needs a learner model, such as bkt()");
  if (!Array.isArray(folds) || folds.length === 0) throw new DataError("Cross-validation needs at least one fold.");
  const evaluations = folds.map((fold, index): FoldEvaluation => {
    checkRecord(fold, `Fold ${index + 1} must be an object { train, test } of two logs`);
    checkLog(fold.test);
    const predictions = predictFold(model.fit(fold.train).model, fold);
    const pairs = observedPairs(fold.test, predictions);
    return { predictions, metrics: pairs.outcomes.length === 0 ? null : metrics(pairs.predictions, pairs.outcomes) };
  });
  const pooled = folds.map((fold, index) => observedPairs(fold.test, evaluations[index]?.predictions ?? []));
  const outcomes = pooled.flatMap((pairs) => pairs.outcomes);
  if (outcomes.length === 0) {
    throw new DataError("Cross-validation needs an event with an outcome in a test log; no test log has one.");
  }
  return {
    folds: evaluations,
    metrics: metrics(
      pooled.flatMap((pairs) => pairs.predictions),
      outcomes,
    ),
  };
}

/** Returns the fitted model's predictions of a fold's test events, as {@link crossValidate} describes. */
function predictFold<TState>(fitted: LearnerModel<TState>, fold: Fold): (number | null)[] {
  const testLearners = new Set(fold.test.map((event) => event.learner));
  const training = new Map<string, Event[]>();
  for (const event of fold.train) {
    if (!testLearners.has(event.learner)) continue;
    const history = training.get(event.learner);
    if (history === undefined) training.set(event.learner, [event]);
    else history.push(event);
  }
  const states = new Map<string, TState>();
  const trained = new Map<string, number>();
  return fold.test.map((event) => {
    const { learner } = event;
    let state = states.get(learner) ?? fitted.initialState(learner);
    const history = training.get(learner) ?? [];
    let next = trained.get(learner) ?? 0;
    for (let earlier = history[next]; earlier !== undefined && earlier.time <= event.time; earlier = history[next]) {
      state = fitted.update(state, earlier);
      next++;
    }
    trained.set(learner, next);
    const prediction = isObserved(event) ? fitted.predict(state, event) : null;
    states.set(learner, fitted.update(state, event));
    return prediction;
  });
}

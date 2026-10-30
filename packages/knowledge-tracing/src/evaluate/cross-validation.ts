import { checkRecord, DataError } from "@measurekit/core";
import type { Event } from "../data/event.js";
import { checkLog } from "../data/event-log.js";
import type { Fold } from "../data/split.js";
import { checkModel, type LearnerModel } from "../model.js";
import { type Metrics, metrics } from "./metrics.js";

/** A model's predictions of one fold's test log, and their metrics. */
export interface FoldEvaluation {
  /** The prediction of each test event, in the order of the test log. */
  readonly predictions: readonly number[];
  /** The metrics of the predictions. */
  readonly metrics: Metrics;
}

/** A model's evaluation on folds. */
export interface CrossValidation {
  /** The evaluation of each fold, in order. */
  readonly folds: readonly FoldEvaluation[];
  /** The metrics of the predictions of all folds together. */
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
 * inform the predictions of their later ones, as they would online.
 *
 * @throws `DataError` when the model is not a learner model, there are no folds, a log is malformed, or the model
 * rejects the data.
 */
export function crossValidate<TState>(model: LearnerModel<TState>, folds: readonly Fold[]): CrossValidation {
  checkModel(model, "Cross-validation needs a learner model, such as bkt()");
  if (!Array.isArray(folds) || folds.length === 0) throw new DataError("Cross-validation needs at least one fold.");
  const evaluations = folds.map((fold, index): FoldEvaluation => {
    checkRecord(fold, `Fold ${index + 1} must be an object { train, test } of two logs`);
    checkLog(fold.test);
    const predictions = predictFold(model.fit(fold.train).model, fold);
    return { predictions, metrics: metrics(predictions, outcomesOf(fold.test)) };
  });
  return {
    folds: evaluations,
    metrics: metrics(
      evaluations.flatMap((evaluation) => evaluation.predictions),
      folds.flatMap((fold) => outcomesOf(fold.test)),
    ),
  };
}

/** Returns the fitted model's predictions of a fold's test events, as {@link crossValidate} describes. */
function predictFold<TState>(fitted: LearnerModel<TState>, fold: Fold): number[] {
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
    const prediction = fitted.predict(state, event);
    states.set(learner, fitted.update(state, event));
    return prediction;
  });
}

function outcomesOf(log: readonly Event[]): number[] {
  return log.map((event) => event.outcome);
}

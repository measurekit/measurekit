import { checkChoice, checkOptions, checkRecord, DataError } from "@measurekit/core";
import { checkLog, type Event, type EventLog, type Fold, isObserved } from "../data/internal.js";
import { checkModel, type LearnerModel } from "../model.js";
import { type Metrics, metrics } from "./metrics.js";
import { alignPredictions } from "./observed-pairs.js";

/** What {@link crossValidate} does with a test event of a skill that the training log has no outcome of. */
export type UnseenSkills = "throw" | "skip";

/** Options of {@link crossValidate}. */
export interface CrossValidateOptions {
  /**
   * What happens to a test event with an outcome and a skill that the events with an outcome of its fold's training log
   * lack, as a skill of a later part of a curriculum is under a split by time: `"throw"` predicts it with the model,
   * which starts the skill from its initial value, as Elo and IRT models do, or throws its `DataError`, as BKT and
   * logistic models do; `"skip"` gives the event a `null` prediction and leaves the learner's state as it was.
   *
   * @defaultValue `"throw"`
   */
  readonly unseenSkills?: UnseenSkills;
}

const UNSEEN_SKILLS: readonly UnseenSkills[] = ["throw", "skip"];
const DEFAULT_UNSEEN_SKILLS = "throw";

/** A model's predictions of one fold's test log, and their metrics. */
export interface FoldEvaluation {
  /**
   * The prediction of each test event, in the order of the test log: `null` for an event without an outcome, and for an
   * event that `unseenSkills` skipped.
   */
  readonly predictions: readonly (number | null)[];
  /** The metrics of the events with an outcome and a prediction, or `null` when the test log has none. */
  readonly metrics: Metrics | null;
  /** The number of test events that `unseenSkills` skipped. */
  readonly skipped: number;
}

/** A model's evaluation on folds. */
export interface CrossValidation {
  /** The evaluation of each fold, in order. */
  readonly folds: readonly FoldEvaluation[];
  /** The metrics of the predictions of all folds together, over the events with an outcome and a prediction. */
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
 * has a `null` prediction, and the metrics use the events with an outcome and a prediction.
 *
 * @param model - The model to fit and evaluate.
 * @param folds - The folds, each with a training log and a test log.
 * @param options - What to do with a test skill the training log lacks.
 * @returns The evaluation of each fold and the metrics of all folds together.
 * @throws `DataError` when the model is not a learner model, there are no folds, a log is malformed, an option is
 * invalid, the model rejects the data, or no test log has an event with an outcome and a prediction.
 */
export function crossValidate<TState>(
  model: LearnerModel<TState>,
  folds: readonly Fold[],
  options: CrossValidateOptions = {},
): CrossValidation {
  checkOptions(options);
  const { unseenSkills = DEFAULT_UNSEEN_SKILLS } = options;
  checkModel(model, "Cross-validation needs a learner model, such as bkt()");
  checkChoice(unseenSkills, "The option unseenSkills", UNSEEN_SKILLS);
  if (!Array.isArray(folds) || folds.length === 0) throw new DataError("Cross-validation needs at least one fold.");
  const evaluations = folds.map((fold, index): FoldEvaluation => {
    checkRecord(fold, `Fold ${index + 1} must be an object { train, test } of two logs`);
    checkLog(fold.test);
    const { predictions, skipped } = predictFold(model.fit(fold.train).model, fold, unseenSkills);
    const pairs = alignPredictions(fold.test, predictions);
    const scored = pairs.outcomes.length === 0 ? null : metrics(pairs.predictions, pairs.outcomes);
    return { predictions, metrics: scored, skipped };
  });
  const pooled = folds.map((fold, index) => alignPredictions(fold.test, evaluations[index]?.predictions ?? []));
  const outcomes = pooled.flatMap((pairs) => pairs.outcomes);
  if (outcomes.length === 0) {
    const skipped = evaluations.reduce((sum, evaluation) => sum + evaluation.skipped, 0);
    throw new DataError(
      skipped === 0
        ? "Cross-validation needs an event with an outcome in a test log; no test log has one."
        : `Cross-validation needs a test event with a prediction; unseenSkills skipped all ${skipped} test events with an outcome.`,
    );
  }
  return {
    folds: evaluations,
    metrics: metrics(
      pooled.flatMap((pairs) => pairs.predictions),
      outcomes,
    ),
  };
}

/** Returns the skills of the events of a log that have an outcome. */
function skillsWithOutcome(log: EventLog): Set<string> {
  return new Set(log.filter(isObserved).flatMap((event) => event.skills));
}

/** Returns the fitted model's predictions of a fold's test events, as {@link crossValidate} describes, and the skipped. */
function predictFold<TState>(
  fitted: LearnerModel<TState>,
  fold: Fold,
  unseenSkills: UnseenSkills,
): { predictions: (number | null)[]; skipped: number } {
  const testLearners = new Set(fold.test.map((event) => event.learner));
  const training = new Map<string, Event[]>();
  for (const event of fold.train) {
    if (!testLearners.has(event.learner)) continue;
    const history = training.get(event.learner);
    if (history === undefined) training.set(event.learner, [event]);
    else history.push(event);
  }
  const seen = unseenSkills === "skip" ? skillsWithOutcome(fold.train) : null;
  const states = new Map<string, TState>();
  const trained = new Map<string, number>();
  let skipped = 0;
  const predictions = fold.test.map((event) => {
    const { learner } = event;
    let state = states.get(learner) ?? fitted.initialState(learner);
    const history = training.get(learner) ?? [];
    let next = trained.get(learner) ?? 0;
    for (let earlier = history[next]; earlier !== undefined && earlier.time <= event.time; earlier = history[next]) {
      state = fitted.update(state, earlier);
      next++;
    }
    trained.set(learner, next);
    if (seen !== null && isObserved(event) && event.skills.some((skill) => !seen.has(skill))) {
      states.set(learner, state);
      skipped++;
      return null;
    }
    const prediction = isObserved(event) ? fitted.predict(state, event) : null;
    states.set(learner, fitted.update(state, event));
    return prediction;
  });
  return { predictions, skipped };
}

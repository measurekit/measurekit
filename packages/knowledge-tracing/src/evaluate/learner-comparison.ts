import { checkArray, checkChoice, checkOptions, checkRecord, DataError, studentLowerTail } from "@measurekit/core";
import { checkLog } from "../data/event-log.js";
import type { Fold } from "../data/split.js";
import { outcomeLogLikelihood } from "../math/logistic.js";
import type { CrossValidation } from "./cross-validation.js";
import { checkPredictionList } from "./observed-pairs.js";

/** The loss of one prediction that {@link compareByLearner} averages. */
export type LearnerLoss = "log" | "squared";

/** Options of {@link compareByLearner}. */
export interface CompareByLearnerOptions {
  /**
   * The loss of a prediction: `"log"`, -(y ln p + (1 - y) ln(1 - p)), or `"squared"`, (p - y)². Log loss is infinite for
   * a certain prediction that is wrong, which leaves the test undefined: use `"squared"` then.
   *
   * @defaultValue `"log"`
   */
  readonly loss?: LearnerLoss;
}

/** A paired t test of the mean losses of two models, one pair of means per learner. */
export interface LearnerComparison {
  /** The number of learners with at least one test event that both models predicted. */
  readonly learners: number;
  /** The mean over learners of the first model's loss minus the second's: negative when the first predicts better. */
  readonly meanDifference: number | null;
  /** The standard error of the mean difference, s / √n; `null` with one learner, or an infinite loss. */
  readonly standardError: number | null;
  /** The t statistic, the mean difference over its standard error; `null` where that is undefined. */
  readonly statistic: number | null;
  /** The degrees of freedom of the test: learners - 1. */
  readonly degreesOfFreedom: number;
  /** The two-sided p-value of the test; `null` where `statistic` is. */
  readonly pValue: number | null;
}

const LOSSES: readonly LearnerLoss[] = ["log", "squared"];
const DEFAULT_LOSS = "log";
/**
 * R's rule in `t.test`, which stops below it: a standard error under 10 machine epsilons of the mean means that the
 * differences are constant. Here it includes 0, where t would be 0 / 0.
 */
const ESSENTIALLY_CONSTANT = 10 * Number.EPSILON;

/**
 * Tests whether two models predict differently well: a paired t test of their losses, each averaged over a learner's
 * test events first, so that the learners are the independent units.
 *
 * @remarks
 * The models must be evaluated on the same folds, as {@link compareModels} does. Events of one learner depend on each
 * other, so a test of the AUC over events, which assumes independent ones, overstates its evidence; the learners are
 * independent, and losses add up per event, which the AUC does not. Only events with an outcome that both models
 * predicted take part, and a learner in several test folds, as under a split by item or by time, has the mean over all
 * of their events. The standard error, the statistic and the p-value are `null` for one learner, which has no variance
 * of differences, and the statistic and the p-value for differences that are all equal, which have none to test.
 * A learner with an infinite loss under either model has no difference, and the mean difference, the standard error,
 * the statistic and the p-value are `null`.
 *
 * @see {@link https://jedm.educationaldatamining.org/index.php/JEDM/article/view/JEDM087 | Pelánek (2015)}
 * @param first - The evaluation of the first model on `folds`, as `crossValidate` returns it.
 * @param second - The evaluation of the second model on the same folds.
 * @param folds - The folds both were evaluated on.
 * @throws `DataError` when an evaluation does not belong to the folds, an option is invalid, or no test event with an
 * outcome has a prediction of both models.
 */
export function compareByLearner(
  first: CrossValidation,
  second: CrossValidation,
  folds: readonly Fold[],
  options: CompareByLearnerOptions = {},
): LearnerComparison {
  checkOptions(options);
  const { loss = DEFAULT_LOSS } = options;
  checkChoice(loss, "The option loss", LOSSES);
  checkArray(folds, "The folds must be an array of folds");
  folds.forEach((fold, index) => {
    checkRecord(fold, `Fold ${index + 1} must be an object { train, test } of two logs`);
    checkLog(fold.test);
  });
  const firstPredictions = predictionsOf(first, folds, "first");
  const secondPredictions = predictionsOf(second, folds, "second");
  const means = new Map<string, { first: number; second: number; count: number }>();
  folds.forEach((fold, index) => {
    const one = firstPredictions[index] ?? [];
    const two = secondPredictions[index] ?? [];
    fold.test.forEach((event, position) => {
      const a = one[position];
      const b = two[position];
      if (event.outcome === null || a === null || a === undefined || b === null || b === undefined) return;
      const total = means.get(event.learner) ?? { first: 0, second: 0, count: 0 };
      means.set(event.learner, total);
      total.first += lossOf(loss, a, event.outcome);
      total.second += lossOf(loss, b, event.outcome);
      total.count++;
    });
  });
  if (means.size === 0) {
    throw new DataError("A comparison needs a test event with an outcome that both models predicted; found none.");
  }
  return pairedTest([...means.values()].map((total) => (total.first - total.second) / total.count));
}

/** Returns the predictions of each fold of an evaluation, checked against the folds. */
function predictionsOf(
  evaluation: CrossValidation,
  folds: readonly Fold[],
  which: string,
): readonly (readonly (number | null)[])[] {
  checkRecord(evaluation, `The ${which} evaluation must be an object returned by crossValidate`);
  checkArray(evaluation.folds, `The ${which} evaluation must have an array of folds`);
  if (evaluation.folds.length !== folds.length) {
    throw new DataError(
      `The ${which} evaluation has ${evaluation.folds.length} folds, but ${folds.length} were given.`,
    );
  }
  return evaluation.folds.map((fold, index) => {
    checkRecord(fold, `Fold ${index + 1} of the ${which} evaluation must be an object`);
    checkPredictionList(folds[index]?.test ?? [], fold.predictions);
    return fold.predictions;
  });
}

function lossOf(loss: LearnerLoss, prediction: number, outcome: number): number {
  return loss === "log" ? -outcomeLogLikelihood(prediction, outcome) : (prediction - outcome) ** 2;
}

/** Returns the paired t test of differences, one per learner, as R's `t.test(paired = TRUE)` computes it. */
function pairedTest(differences: readonly number[]): LearnerComparison {
  const learners = differences.length;
  const degreesOfFreedom = learners - 1;
  if (differences.some((difference) => !Number.isFinite(difference))) {
    return { learners, meanDifference: null, standardError: null, statistic: null, degreesOfFreedom, pValue: null };
  }
  const meanDifference = differences.reduce((sum, difference) => sum + difference, 0) / learners;
  if (learners === 1) {
    return { learners, meanDifference, standardError: null, statistic: null, degreesOfFreedom, pValue: null };
  }
  const squares = differences.reduce((sum, difference) => sum + (difference - meanDifference) ** 2, 0);
  const standardError = Math.sqrt(squares / degreesOfFreedom / learners);
  if (standardError <= ESSENTIALLY_CONSTANT * Math.abs(meanDifference)) {
    return { learners, meanDifference, standardError, statistic: null, degreesOfFreedom, pValue: null };
  }
  const statistic = meanDifference / standardError;
  return {
    learners,
    meanDifference,
    standardError,
    statistic,
    degreesOfFreedom,
    pValue: 2 * studentLowerTail(-Math.abs(statistic), degreesOfFreedom),
  };
}

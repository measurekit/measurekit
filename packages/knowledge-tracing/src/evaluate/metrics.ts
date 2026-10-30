import { DataError, mean } from "@measurekit/core";
import { outcomeLogLikelihood } from "../math/logistic.js";
import { checkPredictions, isBinary } from "./predictions.js";

/** The metrics of predictions against outcomes. */
export interface Metrics {
  /** The number of predictions. */
  readonly count: number;
  /** The area under the ROC curve, or `null` unless the outcomes are 0 and 1, both present. */
  readonly auc: number | null;
  /** The root mean square error √(mean (p - y)²). */
  readonly rmse: number;
  /** The mean negative log-likelihood -mean(y ln p + (1 - y) ln(1 - p)), natural logarithm. */
  readonly logLoss: number;
  /** The share of outcomes predicted right, a prediction above 0.5 meaning correct, or `null` unless outcomes are 0 or 1. */
  readonly accuracy: number | null;
}

/**
 * Returns the area under the ROC curve: the probability that a correct outcome has a higher prediction than an
 * incorrect one, ties counting one half (Mann-Whitney). It depends only on the order of the predictions, so it is
 * invariant to increasing transformations of them.
 *
 * @throws `DataError` when the predictions and outcomes are malformed, an outcome is not 0 or 1, or all outcomes are
 * equal.
 */
export function auc(predictions: readonly number[], outcomes: readonly number[]): number {
  checkPredictions(predictions, outcomes);
  if (!isBinary(outcomes)) throw new DataError("AUC needs outcomes of 0 or 1, not partial credit.");
  const area = areaUnderCurve(predictions, outcomes);
  if (area === null) {
    throw new DataError(`AUC needs both correct and incorrect outcomes; all ${outcomes.length} are ${outcomes[0]}.`);
  }
  return area;
}

/**
 * Returns the root mean square error of predictions against outcomes: √(mean (p - y)²).
 *
 * @throws `DataError` when the predictions and outcomes are malformed.
 */
export function rmse(predictions: readonly number[], outcomes: readonly number[]): number {
  checkPredictions(predictions, outcomes);
  return rootMeanSquareError(predictions, outcomes);
}

/**
 * Returns the mean negative log-likelihood of outcomes under predictions, -mean(y ln p + (1 - y) ln(1 - p)) with the
 * natural logarithm. A prediction of 0 or 1 that is wrong makes it `Infinity`.
 *
 * @throws `DataError` when the predictions and outcomes are malformed.
 */
export function logLoss(predictions: readonly number[], outcomes: readonly number[]): number {
  checkPredictions(predictions, outcomes);
  return meanLogLoss(predictions, outcomes);
}

/**
 * Returns the share of outcomes predicted right: a prediction above 0.5 predicts a correct outcome, one of 0.5 or below
 * an incorrect one.
 *
 * @throws `DataError` when the predictions and outcomes are malformed or an outcome is not 0 or 1.
 */
export function accuracy(predictions: readonly number[], outcomes: readonly number[]): number {
  checkPredictions(predictions, outcomes);
  if (!isBinary(outcomes)) throw new DataError("Accuracy needs outcomes of 0 or 1, not partial credit.");
  return shareRight(predictions, outcomes);
}

/**
 * Returns every metric of predictions against outcomes. AUC and accuracy are `null` where they are undefined: AUC for
 * outcomes all equal, both for partial credit.
 *
 * @throws `DataError` when the predictions and outcomes are malformed.
 */
export function metrics(predictions: readonly number[], outcomes: readonly number[]): Metrics {
  checkPredictions(predictions, outcomes);
  const binary = isBinary(outcomes);
  return {
    count: predictions.length,
    auc: binary ? areaUnderCurve(predictions, outcomes) : null,
    rmse: rootMeanSquareError(predictions, outcomes),
    logLoss: meanLogLoss(predictions, outcomes),
    accuracy: binary ? shareRight(predictions, outcomes) : null,
  };
}

/** Returns the AUC of 0/1 outcomes from the ranks of the predictions, midranks for ties, or `null` for one class. */
function areaUnderCurve(predictions: readonly number[], outcomes: readonly number[]): number | null {
  const order = Array.from(predictions.keys()).sort((a, b) => (predictions[a] ?? 0) - (predictions[b] ?? 0));
  let correct = 0;
  let rankSum = 0;
  for (let start = 0; start < order.length; ) {
    const value = predictions[order[start] ?? 0];
    let end = start + 1;
    while (end < order.length && predictions[order[end] ?? 0] === value) end++;
    // Ranks start + 1 to end share their mean.
    const rank = (start + end + 1) / 2;
    for (let k = start; k < end; k++) {
      if (outcomes[order[k] ?? 0] === 1) {
        correct++;
        rankSum += rank;
      }
    }
    start = end;
  }
  const incorrect = order.length - correct;
  if (correct === 0 || incorrect === 0) return null;
  return (rankSum - (correct * (correct + 1)) / 2) / (correct * incorrect);
}

function rootMeanSquareError(predictions: readonly number[], outcomes: readonly number[]): number {
  return Math.sqrt(mean(predictions.map((prediction, index) => (prediction - (outcomes[index] ?? 0)) ** 2)));
}

function meanLogLoss(predictions: readonly number[], outcomes: readonly number[]): number {
  return mean(predictions.map((prediction, index) => -outcomeLogLikelihood(prediction, outcomes[index] ?? 0)));
}

function shareRight(predictions: readonly number[], outcomes: readonly number[]): number {
  return mean(predictions.map((prediction, index) => (prediction > 0.5 === (outcomes[index] === 1) ? 1 : 0)));
}

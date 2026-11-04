import { DataError, kappaOfPairs, mean } from "@measurekit/core";
import { outcomeLogLikelihood } from "../math/internal.js";
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
  /**
   * The share of outcomes predicted right, a prediction above 0.5 meaning correct, or `null` unless outcomes are 0 or
   * 1.
   */
  readonly accuracy: number | null;
  /**
   * The coefficient of determination 1 - Σ(y - p)² / Σ(y - ȳ)²: 1 for exact predictions, 0 for the mean outcome, below
   * 0 for predictions worse than it. `null` when all outcomes are equal.
   */
  readonly rSquared: number | null;
  /**
   * Cohen's kappa of the predictions above 0.5 and the outcomes: agreement beyond chance, from the margins. `null` for
   * partial credit, or when chance agreement is 1.
   */
  readonly kappa: number | null;
}

/**
 * Returns the area under the ROC curve: the probability that a correct outcome has a higher prediction than an
 * incorrect one, ties counting one half (Mann-Whitney). It depends only on the order of the predictions, so it is
 * invariant to increasing transformations of them.
 *
 * @param predictions - The predicted probabilities.
 * @param outcomes - The outcomes, one for each prediction.
 * @returns The area, from 0 to 1.
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
 * @param predictions - The predicted probabilities.
 * @param outcomes - The outcomes, one for each prediction.
 * @returns The root mean square error, from 0 to 1.
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
 * @param predictions - The predicted probabilities.
 * @param outcomes - The outcomes, one for each prediction.
 * @returns The mean negative log-likelihood, at least 0.
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
 * @param predictions - The predicted probabilities.
 * @param outcomes - The outcomes, one for each prediction.
 * @returns The share predicted right, from 0 to 1.
 * @throws `DataError` when the predictions and outcomes are malformed or an outcome is not 0 or 1.
 */
export function accuracy(predictions: readonly number[], outcomes: readonly number[]): number {
  checkPredictions(predictions, outcomes);
  if (!isBinary(outcomes)) throw new DataError("Accuracy needs outcomes of 0 or 1, not partial credit.");
  return shareRight(predictions, outcomes);
}

/**
 * Returns the coefficient of determination of predictions against outcomes, 1 - Σ(y - p)² / Σ(y - ȳ)², the share of
 * the variance of the outcomes that the predictions explain. It is below 0 for predictions worse than the mean outcome.
 *
 * @param predictions - The predicted probabilities.
 * @param outcomes - The outcomes, one for each prediction.
 * @returns The coefficient, at most 1.
 * @see {@link https://jedm.educationaldatamining.org/index.php/JEDM/article/view/JEDM087 | Pelánek (2015)}
 * @throws `DataError` when the predictions and outcomes are malformed or all outcomes are equal.
 */
export function rSquared(predictions: readonly number[], outcomes: readonly number[]): number {
  checkPredictions(predictions, outcomes);
  const value = coefficientOfDetermination(predictions, outcomes);
  if (value === null) {
    throw new DataError(`R² needs outcomes that are not all equal; all ${outcomes.length} are ${outcomes[0]}.`);
  }
  return value;
}

/**
 * Returns Cohen's kappa of the predictions and the outcomes: a prediction above 0.5 predicts a correct outcome, and
 * kappa is the agreement of these predicted outcomes with the observed ones beyond what their shares of correct
 * outcomes give by chance, (p_o - p_e) / (1 - p_e).
 *
 * @param predictions - The predicted probabilities.
 * @param outcomes - The outcomes, one for each prediction.
 * @returns Kappa, at most 1.
 * @see {@link https://jedm.educationaldatamining.org/index.php/JEDM/article/view/JEDM087 | Pelánek (2015)}
 * @throws `DataError` when the predictions and outcomes are malformed, an outcome is not 0 or 1, or chance agreement is
 * 1, as when every prediction and every outcome is correct.
 */
export function kappa(predictions: readonly number[], outcomes: readonly number[]): number {
  checkPredictions(predictions, outcomes);
  if (!isBinary(outcomes)) throw new DataError("Kappa needs outcomes of 0 or 1, not partial credit.");
  const value = agreementBeyondChance(predictions, outcomes);
  if (value === null) {
    throw new DataError(
      `Kappa is undefined when every prediction and every outcome is ${outcomes[0] === 1 ? "correct" : "incorrect"}.`,
    );
  }
  return value;
}

/**
 * Returns every metric of predictions against outcomes. AUC and accuracy are `null` where they are undefined: AUC for
 * outcomes all equal, both for partial credit; R² for outcomes all equal; kappa for partial credit and where chance
 * agreement is 1.
 *
 * @param predictions - The predicted probabilities.
 * @param outcomes - The outcomes, one for each prediction.
 * @returns Every metric, with `null` for those that are undefined.
 * @throws `DataError` when the predictions and outcomes are malformed.
 *
 * @example
 * ```ts
 * import { auc, metrics } from "@measurekit/knowledge-tracing/evaluate";
 *
 * auc([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]); // 0.75
 * metrics([0.1, 0.4, 0.35, 0.8], [0, 0, 1, 1]);
 * // { count: 4, auc: 0.75, rmse: 0.4, logLoss: 0.47, accuracy: 0.75, rSquared: 0.37, kappa: 0.5 }
 * ```
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
    rSquared: coefficientOfDetermination(predictions, outcomes),
    kappa: binary ? agreementBeyondChance(predictions, outcomes) : null,
  };
}

/** Returns the AUC of 0/1 outcomes from the ranks of the predictions, midranks for ties, or `null` for one class. */
function areaUnderCurve(predictions: readonly number[], outcomes: readonly number[]): number | null {
  let correct = 0;
  for (const outcome of outcomes) if (outcome === 1) correct++;
  const incorrect = outcomes.length - correct;
  if (correct === 0 || incorrect === 0) return null;
  // The predictions of correct and of incorrect outcomes sort apart by the engine's numeric sort of a typed array, and a
  // merge of the two walks the ranks in order: several times as fast as sorting the indices through a comparator.
  const right = new Float64Array(correct);
  const wrong = new Float64Array(incorrect);
  for (let k = 0, r = 0, w = 0; k < outcomes.length; k++) {
    if (outcomes[k] === 1) right[r++] = predictions[k] ?? 0;
    else wrong[w++] = predictions[k] ?? 0;
  }
  return (correctRankSum(right.sort(), wrong.sort()) - (correct * (correct + 1)) / 2) / (correct * incorrect);
}

/** Returns the sum of the ranks of the sorted predictions `right` among both lists, midranks for ties. */
function correctRankSum(right: Float64Array, wrong: Float64Array): number {
  let rankSum = 0;
  for (let start = 0, r = 0, w = 0; r < right.length || w < wrong.length; ) {
    // Predictions are probabilities, so Infinity marks a list that is used up.
    const value = Math.min(right[r] ?? Number.POSITIVE_INFINITY, wrong[w] ?? Number.POSITIVE_INFINITY);
    const first = r;
    while (r < right.length && right[r] === value) r++;
    while (w < wrong.length && wrong[w] === value) w++;
    // Ranks start + 1 to end share their mean.
    const end = r + w;
    const rank = (start + end + 1) / 2;
    for (let k = first; k < r; k++) rankSum += rank;
    start = end;
  }
  return rankSum;
}

// The means below sum in order, as `mean` does, without an array of the terms.

function rootMeanSquareError(predictions: readonly number[], outcomes: readonly number[]): number {
  let sum = 0;
  for (let k = 0; k < predictions.length; k++) sum += ((predictions[k] ?? 0) - (outcomes[k] ?? 0)) ** 2;
  return Math.sqrt(sum / predictions.length);
}

function meanLogLoss(predictions: readonly number[], outcomes: readonly number[]): number {
  let sum = 0;
  for (let k = 0; k < predictions.length; k++) sum += -outcomeLogLikelihood(predictions[k] ?? 0, outcomes[k] ?? 0);
  return sum / predictions.length;
}

function shareRight(predictions: readonly number[], outcomes: readonly number[]): number {
  let right = 0;
  for (let k = 0; k < predictions.length; k++) right += (predictions[k] ?? 0) > 0.5 === (outcomes[k] === 1) ? 1 : 0;
  return right / predictions.length;
}

function coefficientOfDetermination(predictions: readonly number[], outcomes: readonly number[]): number | null {
  // Compared as numbers, not through the mean, which rounds: equal outcomes have no variance to explain.
  if (outcomes.every((outcome) => outcome === outcomes[0])) return null;
  const center = mean(outcomes);
  let residual = 0;
  let total = 0;
  predictions.forEach((prediction, index) => {
    const outcome = outcomes[index] ?? 0;
    residual += (outcome - prediction) ** 2;
    total += (outcome - center) ** 2;
  });
  return 1 - residual / total;
}

function agreementBeyondChance(predictions: readonly number[], outcomes: readonly number[]): number | null {
  return kappaOfPairs(
    predictions.map((prediction) => (prediction > 0.5 ? 1 : 0)),
    outcomes,
  );
}

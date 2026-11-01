import { checkArray, DataError, describe, isUnitInterval } from "@measurekit/core";
import { checkLog, type EventLog } from "../data/event-log.js";

/** The predictions and outcomes of the events that have both, in log order. */
export interface ObservedPairs {
  /** The predictions of the events with an outcome and a prediction. */
  readonly predictions: readonly number[];
  /** Their outcomes. */
  readonly outcomes: readonly number[];
}

/**
 * Returns the events with both an outcome and a prediction, aligned, for the metrics.
 *
 * @remarks
 * A fit's predictions have one entry per event of its log, `null` for an event without an outcome, which the metrics
 * take no part of; this pairs them up.
 *
 * @param predictions - One entry per event of the log: a probability or `null`.
 * @throws `DataError` when the log is malformed, the predictions are not one per event, or one is neither a probability
 * nor `null`.
 */
export function observedPairs(log: EventLog, predictions: readonly (number | null)[]): ObservedPairs {
  checkLog(log);
  checkArray(predictions, "The predictions must be an array with a prediction or null for each event");
  if (predictions.length !== log.length) {
    throw new DataError(`Found ${predictions.length} predictions for ${log.length} events.`);
  }
  const pairedPredictions: number[] = [];
  const outcomes: number[] = [];
  log.forEach(({ outcome }, index) => {
    const prediction = predictions[index];
    if (prediction !== null && !isUnitInterval(prediction)) {
      throw new DataError(
        `Prediction ${index + 1} must be a probability from 0 to 1 or null, found ${describe(prediction)}.`,
      );
    }
    if (prediction === null || outcome === null) return;
    pairedPredictions.push(prediction);
    outcomes.push(outcome);
  });
  return { predictions: pairedPredictions, outcomes };
}

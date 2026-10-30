import { DataError, describe, isUnitInterval } from "@measurekit/core";

/**
 * Checks paired predictions and outcomes: equally many, at least one, predictions probabilities and outcomes from 0
 * to 1.
 */
export function checkPredictions(predictions: readonly number[], outcomes: readonly number[]): void {
  if (!(Array.isArray(predictions) && Array.isArray(outcomes))) {
    throw new DataError("Predictions and outcomes must be arrays of numbers.");
  }
  if (predictions.length !== outcomes.length) {
    throw new DataError(`Found ${predictions.length} predictions and ${outcomes.length} outcomes.`);
  }
  if (predictions.length === 0) throw new DataError("Metrics need at least one prediction.");
  predictions.forEach((prediction, index) => {
    if (!isUnitInterval(prediction)) {
      throw new DataError(`Prediction ${index + 1} must be a probability from 0 to 1, found ${describe(prediction)}.`);
    }
    const outcome = outcomes[index];
    if (!isUnitInterval(outcome)) {
      throw new DataError(`Outcome ${index + 1} must be from 0 to 1, found ${describe(outcome)}.`);
    }
  });
}

/** Returns whether every outcome is 0 or 1. */
export function isBinary(outcomes: readonly number[]): boolean {
  return outcomes.every((outcome) => outcome === 0 || outcome === 1);
}

/** Returns the mean of at least one value. */
export function mean(values: readonly number[]): number {
  let sum = 0;
  for (const value of values) sum += value;
  return sum / values.length;
}

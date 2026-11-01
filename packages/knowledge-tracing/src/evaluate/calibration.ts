import { checkOptions, checkPositiveInteger } from "@measurekit/core";
import { checkPredictions } from "./predictions.js";

/** One bin of a calibration table: the predictions within its bounds and the outcomes of their events. */
export interface CalibrationBin {
  /** The lower bound, included. */
  readonly lower: number;
  /** The upper bound, excluded except for the last bin, which includes 1. */
  readonly upper: number;
  /** The number of predictions in the bin. */
  readonly count: number;
  /** The mean prediction in the bin, or `null` when it is empty. */
  readonly meanPrediction: number | null;
  /** The mean outcome in the bin, the observed rate of correct outcomes, or `null` when it is empty. */
  readonly meanOutcome: number | null;
}

const DEFAULT_BIN_COUNT = 10;

/** Options of {@link calibration}. */
export interface CalibrationOptions {
  /**
   * The number of bins of equal width that divide [0, 1].
   *
   * @defaultValue `10`
   */
  readonly binCount?: number;
}

/**
 * Returns a calibration table: predictions grouped into bins of equal width, with the mean prediction and the mean
 * outcome of each bin. A calibrated model's mean outcome is close to its mean prediction in every bin. Bin k of
 * K covers [k / K, (k + 1) / K), the last one [(K - 1) / K, 1].
 *
 * @param predictions - The predicted probabilities, from 0 to 1.
 * @param outcomes - The outcomes, one for each prediction.
 * @param options - The number of bins.
 * @returns The bins in order, each with its range, count, mean prediction and mean outcome.
 * @throws `DataError` when the predictions and outcomes are malformed or `binCount` is not a positive integer.
 */
export function calibration(
  predictions: readonly number[],
  outcomes: readonly number[],
  options: CalibrationOptions = {},
): CalibrationBin[] {
  checkOptions(options);
  const { binCount = DEFAULT_BIN_COUNT } = options;
  checkPredictions(predictions, outcomes);
  checkPositiveInteger(binCount, "The number of bins");
  const counts = new Float64Array(binCount);
  const predictionSums = new Float64Array(binCount);
  const outcomeSums = new Float64Array(binCount);
  predictions.forEach((prediction, index) => {
    const bin = binOf(prediction, binCount);
    counts[bin] = (counts[bin] ?? 0) + 1;
    predictionSums[bin] = (predictionSums[bin] ?? 0) + prediction;
    outcomeSums[bin] = (outcomeSums[bin] ?? 0) + (outcomes[index] ?? 0);
  });
  return Array.from(counts, (count, bin) => ({
    lower: bin / binCount,
    upper: (bin + 1) / binCount,
    count,
    meanPrediction: count === 0 ? null : (predictionSums[bin] ?? 0) / count,
    meanOutcome: count === 0 ? null : (outcomeSums[bin] ?? 0) / count,
  }));
}

/**
 * Returns the bin k with k / K ≤ p < (k + 1) / K, comparing with the bounds as computed, not with p K, which rounds.
 */
function binOf(prediction: number, binCount: number): number {
  let bin = Math.min(Math.floor(prediction * binCount), binCount - 1);
  if (prediction < bin / binCount) bin--;
  else if (bin + 1 < binCount && prediction >= (bin + 1) / binCount) bin++;
  return bin;
}

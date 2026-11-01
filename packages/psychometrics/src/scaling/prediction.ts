import { checkOptions } from "@measurekit/core";
import { checkOpenUnitInterval } from "../checks.js";

/** Options of {@link predictFormScore}. */
export interface ScorePredictionOptions {
  /**
   * The probability of the central interval of the prediction, between 0 and 1, exclusive.
   *
   * @defaultValue `0.8`
   */
  readonly level?: number;
}

/** Options of {@link predictThetaScore}. */
export interface ThetaScorePredictionOptions extends ScorePredictionOptions {
  /**
   * The standard error of the official theta estimate, finite and at least 0: a computerized adaptive test's target
   * precision. 0 reports each theta as it is.
   *
   * @defaultValue `0`
   */
  readonly standardError?: number;
}

/** The probability of one reportable score. */
export interface ScoreProbability {
  /** The scale score. */
  readonly score: number;
  /** Its probability. */
  readonly probability: number;
}

/** A prediction of an official score. */
export interface ScorePrediction {
  /** The expected score. */
  readonly mean: number;
  /** The lowest score that has at least (1 - `level`) / 2 of the probability at or below it. */
  readonly lower: number;
  /** The lowest score that has at least (1 + `level`) / 2 of the probability at or below it. */
  readonly upper: number;
  /** The probability of each reportable score, in ascending order of score. */
  readonly distribution: readonly ScoreProbability[];
}

const DEFAULT_LEVEL = 0.8;

/**
 * Checks the options of a prediction and returns its level.
 *
 * @throws `DataError` when the options are not an object or the level is not a number between 0 and 1, exclusive.
 */
export function checkLevel(options: ScorePredictionOptions): number {
  checkOptions(options);
  const { level = DEFAULT_LEVEL } = options;
  checkOpenUnitInterval(level, "The option level");
  return level;
}

/**
 * Returns the prediction of a distribution of scores in ascending order: its probabilities scaled to sum to 1, their
 * mean, and the equal-tailed interval of the level.
 */
export function summarize(distribution: readonly ScoreProbability[], level: number): ScorePrediction {
  const total = distribution.reduce((sum, { probability }) => sum + probability, 0);
  const scaled = distribution.map(({ score, probability }) => ({ score, probability: probability / total }));
  let mean = 0;
  for (const { score, probability } of scaled) mean += score * probability;
  const quantile = (share: number): number => {
    let cumulative = 0;
    for (const { score, probability } of scaled) {
      cumulative += probability;
      if (cumulative >= share) return score;
    }
    return scaled[scaled.length - 1]?.score ?? 0;
  };
  return { mean, lower: quantile((1 - level) / 2), upper: quantile((1 + level) / 2), distribution: scaled };
}

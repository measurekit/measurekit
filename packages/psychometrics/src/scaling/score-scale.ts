import {
  checkFields,
  checkFinite,
  checkNonNegative,
  checkPositive,
  checkRecord,
  DataError,
  describe,
  type FieldNames,
} from "@measurekit/core";
import type { ScaleTransformation } from "../equating/internal.js";

/** The rounding and the bounds of reported scores. */
export interface ScoreRounding {
  /**
   * The lowest reported score: lower scores are reported as this one. A number, or `-Infinity` for none.
   *
   * @defaultValue `-Infinity`
   */
  readonly lowest?: number;
  /**
   * The highest reported score: higher scores are reported as this one. A number above `lowest`, or `Infinity` for
   * none.
   *
   * @defaultValue `Infinity`
   */
  readonly highest?: number;
  /**
   * The rounding increment: scores are reported as multiples of it, the nearest, halves up. A finite number of at least
   * 0; 0 leaves scores unrounded.
   *
   * @defaultValue `1`
   */
  readonly step?: number;
}

export const SCORE_ROUNDING_FIELDS: FieldNames<ScoreRounding> = { lowest: true, highest: true, step: true };

/** A linear scale of theta: `slope` θ + `intercept`, then rounded and bounded as `ScoreRounding` says. */
export interface ScoreScale extends ScaleTransformation, ScoreRounding {}

const SCORE_SCALE_FIELDS: FieldNames<ScoreScale> = {
  slope: true,
  intercept: true,
  lowest: true,
  highest: true,
  step: true,
};

/** The rounding and bounds of a scale, with their defaults. */
export interface CheckedRounding {
  readonly lowest: number;
  readonly highest: number;
  readonly step: number;
}

/** A scale checked, with the defaults of its rounding. */
export interface CheckedScale extends CheckedRounding, ScaleTransformation {}

const DEFAULT_STEP = 1;

/**
 * Checks the rounding and bounds of reported scores and returns them with their defaults.
 *
 * @param label - Names the rounding in error messages, such as `The score scale`.
 * @throws `DataError` unless the rounding is an object, `lowest` and `highest` are numbers with `lowest` below
 * `highest`, and `step` is a finite number of at least 0.
 */
export function checkRounding(rounding: ScoreRounding, label: string): CheckedRounding {
  checkRecord(rounding, `${label} must be an object`);
  const { lowest = Number.NEGATIVE_INFINITY, highest = Number.POSITIVE_INFINITY, step = DEFAULT_STEP } = rounding;
  for (const [bound, value] of [
    ["lowest", lowest],
    ["highest", highest],
  ] as const) {
    if (typeof value !== "number" || Number.isNaN(value)) {
      throw new DataError(`${label} needs a number as ${bound}, found ${describe(value)}.`);
    }
  }
  if (!(lowest < highest)) {
    throw new DataError(`${label} needs a lowest score below the highest, found ${lowest} and ${highest}.`);
  }
  checkNonNegative(step, `The step of ${label.toLowerCase()}`);
  return { lowest, highest, step };
}

/**
 * Checks a scale and returns it with the defaults of its rounding.
 *
 * @throws `DataError` unless the slope is positive and finite, the intercept finite, and the rounding valid.
 */
export function checkScale(scale: ScoreScale): CheckedScale {
  checkRecord(scale, "The score scale must be an object { slope, intercept }");
  checkFields(scale, SCORE_SCALE_FIELDS, "The score scale");
  checkPositive(scale.slope, "The slope of the score scale");
  checkFinite(scale.intercept, "The intercept of the score scale");
  return { slope: scale.slope, intercept: scale.intercept, ...checkRounding(scale, "The score scale") };
}

/** Returns a score rounded to the nearest multiple of the step, halves up, and bounded. */
export function roundScore(score: number, { lowest, highest, step }: CheckedRounding): number {
  const rounded = step > 0 ? multipleOf(Math.round(score / step), step) : score;
  // Adding 0 turns -0, which rounding a small negative score gives, into 0.
  return Math.min(Math.max(rounded, lowest), highest) + 0;
}

/**
 * Returns k times a step. A step such as 0.1 is no exact double, so k · 0.1 can miss the decimal, as 3 · 0.1 gives
 * 0.30000000000000004; where the step is the inverse of a whole number, k divided by it gives the double nearest the
 * decimal, 0.3.
 */
export function multipleOf(k: number, step: number): number {
  const inverse = 1 / step;
  return Number.isInteger(inverse) ? k / inverse : k * step;
}

/** Returns the reported score of theta on a checked scale. */
export function scoreOfTheta(theta: number, scale: CheckedScale): number {
  return roundScore(scale.slope * theta + scale.intercept, scale);
}

/**
 * Returns the reported score of theta on a scale: slope θ + intercept, rounded to the nearest multiple of `step`, halves
 * up, and bounded to `lowest` and `highest`.
 *
 * @remarks
 * Valid input: a finite theta, a finite positive slope, a finite intercept, and the rounding of {@link ScoreRounding}.
 * The result is monotone in theta.
 *
 * @param theta - The ability.
 * @param scale - The slope, intercept and rounding of the reported score.
 * @returns The reported score.
 *
 * @example
 * ```ts
 * import { scaleScore } from "@measurekit/psychometrics/scaling";
 *
 * const scale = { slope: 100, intercept: 500, lowest: 200, highest: 800, step: 10 };
 * scaleScore(0.234, scale); // => 520
 * scaleScore(-4.2, scale); // => 200
 * ```
 * @throws `DataError` when theta is not finite or the scale is invalid.
 */
export function scaleScore(theta: number, scale: ScoreScale): number {
  checkFinite(theta, "Theta");
  return scoreOfTheta(theta, checkScale(scale));
}

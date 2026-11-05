import {
  checkFields,
  checkFinite,
  checkOptions,
  checkPositiveInteger,
  checkRecord,
  checkUnitInterval,
  type FieldNames,
} from "@measurekit/core";
import { checkCurvePoints, type ErrorRatePoint } from "./curve-points.js";

/** The category of a skill's learning curve, in the order that {@link categorizeCurve} tests them. */
export type CurveCategory = "too little data" | "low and flat" | "still high" | "no learning" | "good";

/** DataShop's default error rate below which a curve is low and flat: Bloom's 20% who master without instruction. */
const DEFAULT_LOW_ERROR = 0.2;
/** DataShop's default error rate above which the last point of a curve is still high. */
const DEFAULT_HIGH_ERROR = 0.4;

/** DataShop's thresholds for the category of a curve, which depend on the curriculum, so every one is required. */
export interface CurveThresholds {
  /** Points of fewer learners are discarded: DataShop's student threshold. */
  readonly minLearners: number;
  /** Fewer points left mean too little data: DataShop's opportunity threshold. */
  readonly minPoints: number;
  /** The slope below which a curve shows no learning: DataShop's AFM slope threshold. */
  readonly minSlope: number;
}

const CURVE_THRESHOLD_FIELDS: FieldNames<CurveThresholds> = { minLearners: true, minPoints: true, minSlope: true };

/** Options of {@link categorizeCurve}. */
export interface CurveCategoryOptions {
  /**
   * The error rate that a point must stay at or above, or the curve is low and flat.
   *
   * @defaultValue `0.2`
   */
  readonly lowError?: number;
  /**
   * The error rate that the last point must stay at or below, or the curve is still high.
   *
   * @defaultValue `0.4`
   */
  readonly highError?: number;
}

const CURVE_CATEGORY_OPTIONS: FieldNames<CurveCategoryOptions> = { lowError: true, highError: true };

/**
 * Returns the category of a skill's learning curve by DataShop's rules. The points of fewer than `minLearners` learners
 * are discarded, and the first rule that holds decides: fewer than `minPoints` points left, `"too little data"`; a
 * point below `lowError`, `"low and flat"`; the last point above `highError`, `"still high"`; `slope` below
 * `minSlope`, `"no learning"`; otherwise `"good"`.
 *
 * @remarks
 * The thresholds depend on the curriculum, so `thresholds` is required. `slope` comes from a fit of AFM with the log's
 * Q-matrix: the skill's slope. Pass the points of one skill, such as `learningCurve(log).skills.fractions`.
 *
 * @param points - The error rate of each opportunity, as `learningCurve` gives them.
 * @param slope - The slope of the skill in AFM: its learning rate, the rise of the logit of P(correct) per opportunity.
 * @param thresholds - DataShop's thresholds for learners, points and slope.
 * @param options - DataShop's thresholds for error.
 * @returns The category: `"too little data"`, `"low and flat"`, `"still high"`, `"no learning"` or `"good"`.
 * @see {@link https://pslcdatashop.web.cmu.edu/help?page=learningCurve | DataShop, Learning Curve help}
 * @throws `DataError` when the points or an option are malformed.
 */
export function categorizeCurve(
  points: readonly ErrorRatePoint[],
  slope: number,
  thresholds: CurveThresholds,
  options: CurveCategoryOptions = {},
): CurveCategory {
  checkOptions(options, CURVE_CATEGORY_OPTIONS);
  checkRecord(thresholds, "The thresholds must be an object { minLearners, minPoints, minSlope }");
  checkFields(thresholds, CURVE_THRESHOLD_FIELDS, "The thresholds");
  const { minLearners, minPoints, minSlope } = thresholds;
  const { lowError = DEFAULT_LOW_ERROR, highError = DEFAULT_HIGH_ERROR } = options;
  checkPositiveInteger(minLearners, "The threshold minLearners");
  checkPositiveInteger(minPoints, "The threshold minPoints");
  checkFinite(slope, "The slope");
  checkFinite(minSlope, "The threshold minSlope");
  checkUnitInterval(lowError, "The option lowError");
  checkUnitInterval(highError, "The option highError");
  checkCurvePoints(points);
  const kept = points.filter((point) => point.count >= minLearners);
  const last = kept.reduce<ErrorRatePoint | null>(
    (latest, point) => (latest === null || point.opportunity > latest.opportunity ? point : latest),
    null,
  );
  if (last === null || kept.length < minPoints) return "too little data";
  if (kept.some((point) => point.errorRate < lowError)) return "low and flat";
  if (last.errorRate > highError) return "still high";
  return slope < minSlope ? "no learning" : "good";
}

import { checkFinite, checkOptions, checkPositiveInteger, checkUnitInterval } from "@measurekit/core";
import { checkCurvePoints, type ErrorRatePoint } from "./curve-points.js";

/** The category of a skill's learning curve, in the order that {@link categorizeCurve} tests them. */
export type CurveCategory = "too little data" | "low and flat" | "still high" | "no learning" | "good";

/** DataShop's default error rate below which a curve is low and flat: Bloom's 20% who master without instruction. */
const DEFAULT_LOW_ERROR = 0.2;
/** DataShop's default error rate above which the last point of a curve is still high. */
const DEFAULT_HIGH_ERROR = 0.4;

/** Options of {@link categorizeCurve}. */
export interface CurveCategoryOptions {
  /** Points of fewer learners are discarded: DataShop's student threshold. Required, as it depends on the data. */
  readonly minLearners: number;
  /** Fewer points left mean too little data: DataShop's opportunity threshold. Required. */
  readonly minPoints: number;
  /** The slope of the skill in AFM: its learning rate, the rise of the logit of P(correct) per opportunity. Required. */
  readonly slope: number;
  /** The slope below which a curve shows no learning: DataShop's AFM slope threshold. Required. */
  readonly minSlope: number;
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

/**
 * Returns the category of a skill's learning curve by DataShop's rules. The points of fewer than `minLearners` learners
 * are discarded, and the first rule that holds decides: fewer than `minPoints` points left, `"too little data"`; a
 * point below `lowError`, `"low and flat"`; the last point above `highError`, `"still high"`; `slope` below
 * `minSlope`, `"no learning"`; otherwise `"good"`.
 *
 * @remarks
 * The thresholds depend on the curriculum, so the first four options are required. `slope` comes from a fit of AFM
 * with the log's Q-matrix: the skill's slope. Pass the points of one skill, such as `learningCurve(log).skills.fractions`.
 *
 * @see {@link https://pslcdatashop.web.cmu.edu/help?page=learningCurve | DataShop, Learning Curve help}
 * @throws `DataError` when the points or an option are malformed.
 */
export function categorizeCurve(points: readonly ErrorRatePoint[], options: CurveCategoryOptions): CurveCategory {
  checkOptions(options);
  const {
    minLearners,
    minPoints,
    slope,
    minSlope,
    lowError = DEFAULT_LOW_ERROR,
    highError = DEFAULT_HIGH_ERROR,
  } = options;
  checkPositiveInteger(minLearners, "The option minLearners");
  checkPositiveInteger(minPoints, "The option minPoints");
  checkFinite(slope, "The option slope");
  checkFinite(minSlope, "The option minSlope");
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

/**
 * Score reporting and prediction: linear scales of theta with rounding and bounds, conversion tables from total scores
 * to scale scores by the EAP of the summed score or the test characteristic curve, conditional standard errors of
 * measurement of scale scores, scale scores of equated scores, growth between the groups of a vertical scale,
 * percentile ranks in a norm sample, and the prediction of an official score from a posterior of theta.
 *
 * @module @measurekit/psychometrics/scaling
 */

export { conditionalScaleError } from "./conditional-error.js";
export {
  type ConversionMethod,
  type ConversionRow,
  type ConversionTableOptions,
  conversionTable,
} from "./conversion-table.js";
export { type GroupScale, type ScaleGrowth, scaleGrowth } from "./growth.js";
export { percentileRanks } from "./percentile-ranks.js";
export { predictFormScore } from "./predict-form-score.js";
export { predictThetaScore } from "./predict-theta-score.js";
export type {
  ScorePrediction,
  ScorePredictionOptions,
  ScoreProbability,
  ThetaScorePredictionOptions,
} from "./prediction.js";
export { scaleConversion } from "./scale-conversion.js";
export { type ScoreRounding, type ScoreScale, scaleScore } from "./score-scale.js";

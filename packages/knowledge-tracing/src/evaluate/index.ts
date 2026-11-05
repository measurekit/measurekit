/**
 * Evaluation of learner models: AUC, RMSE, log-loss, accuracy, R², kappa and calibration of predictions; cross-validation
 * on folds by learner, item or time; model comparison, with a paired test over learners; metrics and learning curves
 * by opportunity, and their power law and categories.
 *
 * @module @measurekit/knowledge-tracing/evaluate
 */
export { type CalibrationBin, type CalibrationOptions, calibrationCurve } from "./calibration.js";
export {
  type CrossValidateOptions,
  type CrossValidation,
  crossValidate,
  type FoldEvaluation,
  type UnseenSkills,
} from "./cross-validation.js";
export {
  type CurveCategory,
  type CurveCategoryOptions,
  type CurveThresholds,
  categorizeCurve,
} from "./curve-category.js";
export { fitPowerLaw, type PowerLaw } from "./curve-fit.js";
export type { ErrorRatePoint } from "./curve-points.js";
export {
  type CompareByLearnerOptions,
  compareByLearner,
  type LearnerComparison,
  type LearnerLoss,
} from "./learner-comparison.js";
export {
  type LearningCurve,
  type LearningCurveOptions,
  type LearningCurvePoint,
  learningCurve,
} from "./learning-curve.js";
export { accuracy, auc, kappa, logLoss, type Metrics, metrics, rmse, rSquared } from "./metrics.js";
export { compareModels, type ModelEvaluation } from "./model-comparison.js";
export { alignPredictions, type ObservedPairs } from "./observed-pairs.js";
export { metricsByOpportunity, type OpportunityMetrics, type OpportunityPoint } from "./opportunity-metrics.js";

/**
 * Evaluation of learner models: AUC, RMSE, log-loss, accuracy and calibration of predictions; cross-validation on folds
 * by learner, item or time; model comparison; and learning curves.
 *
 * @module @measurekit/knowledge-tracing/evaluate
 */
export { type CalibrationBin, type CalibrationOptions, calibration } from "./calibration.js";
export { type CrossValidation, crossValidate, type FoldEvaluation } from "./cross-validation.js";
export { type LearningCurve, type LearningCurvePoint, learningCurve } from "./learning-curve.js";
export { accuracy, auc, logLoss, type Metrics, metrics, rmse } from "./metrics.js";
export { compareModels, type ModelEvaluation } from "./model-comparison.js";
export { type ObservedPairs, observedPairs } from "./observed-pairs.js";

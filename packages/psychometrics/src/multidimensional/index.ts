/**
 * Multidimensional item response theory: compensatory logistic and graded models, calibration by EM for up to three
 * dimensions, exploratory and confirmatory, standard errors, rotation, scoring and information.
 *
 * @module @measurekit/psychometrics/multidimensional
 */

export {
  type CalibratedMultidimensionalItem,
  calibrateMultidimensional,
  type MultidimensionalCalibration,
} from "./calibration.js";
export type { CalibrationMethod, MultidimensionalCalibrationOptions } from "./calibration-options.js";
export { type MultidimensionalGradedParameters, multidimensionalGradedResponse } from "./graded.js";
export { multidimensionalInformation } from "./information.js";
export {
  type MultidimensionalLogisticParameters,
  multidimensionalThreeParameterLogistic,
  multidimensionalTwoParameterLogistic,
} from "./logistic.js";
export type {
  LoadingPattern,
  MultidimensionalItem,
  MultidimensionalModel,
  MultidimensionalTestItem,
} from "./model.js";
export type { MultivariateNormalPrior } from "./prior.js";
export { type RotatedLoadings, type RotationMethod, type RotationOptions, rotateLoadings } from "./rotation.js";
export {
  type MultidimensionalEstimate,
  type MultidimensionalEstimator,
  type MultidimensionalPersonEstimate,
  type MultidimensionalPosteriorMeanOptions,
  type MultidimensionalPosteriorModeOptions,
  multidimensionalEstimates,
  multidimensionalPosteriorMeanEstimate,
  multidimensionalPosteriorModeEstimate,
} from "./scoring.js";

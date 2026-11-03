/**
 * Multidimensional item response theory: compensatory logistic and graded models, calibration by EM for up to three
 * dimensions and by MH-RM for any number, exploratory and confirmatory, in several groups and with fixed items,
 * bifactor and two-tier models by dimension reduction, standard errors, rotation, scoring and information, and the
 * calibration of a trait that changes over two or three occasions.
 *
 * @module @measurekit/psychometrics/multidimensional
 */

export type { RotationMethod, RotationOptions } from "../math/internal.js";
export { type BifactorOptions, calibrateBifactor } from "./bifactor.js";
export {
  type CalibratedMultidimensionalItem,
  calibrateMultidimensional,
  type MultidimensionalCalibration,
  type MultidimensionalGroup,
} from "./calibration.js";
export type { CalibrationMethod, MultidimensionalCalibrationOptions } from "./calibration-options.js";
export { multidimensionalDifficulty, multidimensionalDiscrimination } from "./direction.js";
export { type MultidimensionalGradedParameters, multidimensionalGradedResponse } from "./graded.js";
export { multidimensionalInformation } from "./information.js";
export {
  type MultidimensionalLogisticParameters,
  multidimensionalThreeParameterLogistic,
  multidimensionalTwoParameterLogistic,
} from "./logistic.js";
export {
  calibrateLongitudinal,
  type LongitudinalCalibration,
  type LongitudinalOptions,
  type OccasionChange,
  type OccasionMoments,
} from "./longitudinal.js";
export type {
  LoadingPattern,
  MultidimensionalItem,
  MultidimensionalModel,
  MultidimensionalTestItem,
} from "./model.js";
export type { MultivariateNormalPrior } from "./prior.js";
export { type RotatedLoadings, rotateLoadings } from "./rotation.js";
export {
  type MultidimensionalEstimate,
  type MultidimensionalEstimator,
  type MultidimensionalPersonEstimate,
  type MultidimensionalPosteriorMeanOptions,
  type MultidimensionalPosteriorModeOptions,
  multidimensionalEstimates,
  multidimensionalMaximumLikelihoodEstimate,
  multidimensionalPosteriorMeanEstimate,
  multidimensionalPosteriorModeEstimate,
  multidimensionalWeightedLikelihoodEstimate,
} from "./scoring.js";

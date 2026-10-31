/**
 * Cognitive diagnosis: the G-DINA, DINA, DINO, A-CDM, LLM and R-RUM models calibrated by marginal maximum likelihood,
 * persons' attribute mastery and profiles, classification accuracy and consistency, Q-matrix validation, and
 * nonparametric classification.
 *
 * @module @measurekit/psychometrics/cdm
 */

export type { QMatrix } from "@measurekit/core";
export { type CdmAccuracyOptions, type CdmClassificationAccuracy, cdmClassificationAccuracy } from "./accuracy.js";
export {
  type CdmCalibration,
  type CdmCalibrationOptions,
  type CdmItem,
  type CdmModel,
  calibrateCdm,
  type ProfileProportion,
} from "./calibration.js";
export {
  type NonparametricClassification,
  type NonparametricOptions,
  nonparametricClassification,
} from "./nonparametric.js";
export { type AttributeProfile, attributeProfiles, type ProfileEstimate } from "./persons.js";
export { type QMatrixValidationOptions, type QVectorSuggestion, validateQMatrix } from "./q-validation.js";

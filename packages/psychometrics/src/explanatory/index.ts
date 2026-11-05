/**
 * Explanatory item response theory: the linear logistic test model, which explains the difficulties of items by their
 * features, tested against the Rasch model; the linear logistic model of change, by conditional maximum likelihood; and the regression of item parameters on the features of the items,
 * estimated in two stages from a calibration by restricted maximum likelihood, with the prediction of the parameters
 * of new items from their features.
 *
 * @module @measurekit/psychometrics/explanatory
 */

export type { FeatureEffect, ItemFeatures } from "./features.js";
export {
  type ItemFeatureRegression,
  type ItemFeatureRegressionOptions,
  itemFeatureRegression,
} from "./item-feature-regression.js";
export {
  type LinearChange,
  type LinearChangeEffect,
  type LinearChangeOptions,
  linearChange,
} from "./linear-change.js";
export {
  type Lltm,
  type LltmModel,
  type LltmOptions,
  lltm,
  type RaschComparison,
} from "./lltm.js";
export { type ParameterPrediction, predictItemParameter } from "./predict-item-parameter.js";

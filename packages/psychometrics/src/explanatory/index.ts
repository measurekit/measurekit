/**
 * Explanatory item response theory: the linear logistic test model, which explains the difficulties of items by their
 * features, tested against the Rasch model; and the regression of item parameters on the features of the items,
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
export { type Lltm, type LltmOptions, type LltmStandardErrors, lltm, type RaschComparison } from "./lltm.js";
export { type ParameterPrediction, predictItemParameter } from "./predict-item-parameter.js";

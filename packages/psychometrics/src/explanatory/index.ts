/**
 * Explanatory item response theory: the regression of item parameters on the features of the items, estimated in two
 * stages from a calibration by restricted maximum likelihood, and the prediction of the parameters of new items from
 * their features.
 *
 * @module @measurekit/psychometrics/explanatory
 */

export type { FeatureEffect, ItemFeatures } from "./features.js";
export {
  type ItemFeatureRegression,
  type ItemFeatureRegressionOptions,
  itemFeatureRegression,
} from "./item-feature-regression.js";
export { type ParameterPrediction, predictItemParameter } from "./predict-item-parameter.js";

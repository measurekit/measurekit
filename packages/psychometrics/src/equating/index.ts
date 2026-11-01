/**
 * Linking and equating: scale transformations from common items by the mean-mean, mean-sigma, Haebara and
 * Stocking-Lord methods, composed over a chain of forms; linear and equipercentile equating of observed scores from
 * random groups, with log-linear presmoothing; equating through common items from nonequivalent groups by the Tucker,
 * Levine, chained and frequency estimation methods; bootstrap standard errors; the robust z screen of common items;
 * IRT true-score and observed-score equating; and delta-method standard errors of IRT equating.
 *
 * @module @measurekit/psychometrics/equating
 */

export type { BivariateDistribution } from "./bivariate.js";
export {
  type BivariateLogLinearPresmoothing,
  type BivariatePresmoothingOptions,
  bivariateLogLinearPresmoothing,
} from "./bivariate-presmoothing.js";
export { type BootstrapOptions, bootstrapEquating, type EquatingSample } from "./bootstrap.js";
export {
  type CharacteristicCurveLinking,
  haebaraLinking,
  type LinkingOptions,
  stockingLordLinking,
} from "./characteristic-curve.js";
export {
  type EquatingPopulations,
  type IrtObservedScoreOptions,
  irtObservedScoreEquating,
} from "./irt-observed-score.js";
export { type IrtEquatingErrorOptions, irtEquatingStandardErrors } from "./irt-standard-errors.js";
export {
  chainTransformations,
  meanMeanLinking,
  meanSigmaLinking,
  type ScaleTransformation,
  transformItem,
  transformTheta,
} from "./linking.js";
export {
  chainedLinearEquating,
  type LevineOptions,
  levineEquating,
  type SyntheticOptions,
  tuckerEquating,
} from "./neat.js";
export { chainedEquipercentileEquating, frequencyEstimationEquating } from "./neat-equipercentile.js";
export { equipercentileEquating, linearEquating } from "./observed-score.js";
export { type LogLinearPresmoothing, logLinearPresmoothing } from "./presmoothing.js";
export { type AnchorDrift, robustZ } from "./robust-z.js";
export { trueScoreEquating } from "./true-score.js";

/**
 * Linking and equating: scale transformations from common items by the mean-mean, mean-sigma, Haebara and
 * Stocking-Lord methods; linear and equipercentile equating of observed scores from random groups, with log-linear
 * presmoothing; and IRT true-score equating.
 *
 * @module @measurekit/psychometrics/equating
 */
export {
  type CharacteristicCurveLinking,
  haebaraLinking,
  type LinkingOptions,
  stockingLordLinking,
} from "./characteristic-curve.js";
export {
  meanMeanLinking,
  meanSigmaLinking,
  type ScaleTransformation,
  transformItem,
  transformTheta,
} from "./linking.js";
export { equipercentileEquating, linearEquating } from "./observed-score.js";
export { type LogLinearPresmoothing, logLinearPresmoothing } from "./presmoothing.js";
export { trueScoreEquating } from "./true-score.js";

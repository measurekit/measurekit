/**
 * Differential item functioning: the Mantel-Haenszel statistics with the ETS categories and logistic regression DIF,
 * on sparse data and with purification, and Lord's chi-square.
 *
 * @module @measurekit/psychometrics/dif
 */
export type { DifGroup } from "./groups.js";
export { type LikelihoodRatioTest, type LogisticRegressionDif, logisticRegressionDif } from "./logistic-regression.js";
export { type LordChiSquare, type LordDif, type LordOptions, lordChiSquare } from "./lord.js";
export {
  type EtsCategory,
  type MantelHaenszel,
  type MantelHaenszelOptions,
  mantelHaenszel,
} from "./mantel-haenszel.js";
export type { DifAnalysis, DifPurification, PurificationOptions } from "./purification.js";

/**
 * Differential item functioning: the Mantel-Haenszel statistics with the ETS categories, logistic regression DIF, the
 * Mantel test and ordinal logistic regression DIF for polytomous items, on sparse data and with purification, SIBTEST
 * of items and bundles, Lord's chi-square and Raju's areas, IRT likelihood-ratio tests for items of any model, and the
 * differential functioning of items and tests.
 *
 * @module @measurekit/psychometrics/dif
 */
export {
  type DifferentialFunctioningOptions,
  differentialFunctioning,
  type ItemFunctioning,
  type TestFunctioning,
} from "./differential-functioning.js";
export type { DifGroup } from "./groups.js";
export { type IrtDif, type IrtDifOptions, type IrtDifTest, irtLikelihoodRatioDif } from "./irt-likelihood-ratio.js";
export type { LikelihoodRatioTest } from "./likelihood-ratio.js";
export { type LogisticRegressionDif, logisticRegressionDif } from "./logistic-regression.js";
export { type LordChiSquare, type LordDif, type LordOptions, lordChiSquare } from "./lord.js";
export {
  type EtsCategory,
  type MantelHaenszel,
  type MantelHaenszelOptions,
  mantelHaenszel,
} from "./mantel-haenszel.js";
export {
  type GeneralizedMantelHaenszel,
  type MantelTest,
  mantelTest,
  type PolytomousEtsCategory,
} from "./mantel-test.js";
export {
  type MatchingOptions,
  type OrdinalLogisticDif,
  ordinalLogisticRegressionDif,
} from "./ordinal-logistic-regression.js";
export type { DifAnalysis, DifPurification, PurificationOptions } from "./purification.js";
export { type RajuArea, rajuAreaDif } from "./raju-area.js";
export { type Sibtest, type SibtestDif, type SibtestOptions, sibtest } from "./sibtest.js";

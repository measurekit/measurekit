// Helpers that the modules and tests of this package share. They are not exported from the package.

export { patternsOf } from "./bifactor-patterns.js";
export { structureOf } from "./bifactor-structure.js";
export { createWalk } from "./bifactor-walk.js";
export { setUp } from "./calibration-setup.js";
export { latentDistribution } from "./correlation.js";
export { designOf } from "./design.js";
export { type ExpectationItem, expectationStep } from "./expectation.js";
export { type MultidimensionalGradedParameters, multidimensionalGradedResponse } from "./graded.js";
export { productGrid } from "./grid.js";
export { gridLikelihood } from "./grid-likelihood.js";
export {
  checkMultidimensionalItem,
  checkMultidimensionalItems,
  checkMultidimensionalModel,
  checkScoringDimensions,
  indexMultidimensionalItems,
} from "./items.js";
export {
  type MultidimensionalLogisticParameters,
  multidimensionalThreeParameterLogistic,
  multidimensionalTwoParameterLogistic,
} from "./logistic.js";
export { createChain, responsesByItem, responsesByPerson } from "./mhrm-chain.js";
export {
  createCorrelations,
  latentObservedInformation,
  latentScore,
  personLatentScores,
  secondMoments,
} from "./mhrm-latent.js";
export { createScratch, itemMoments, observedItemInformation } from "./mhrm-moments.js";
export type { MultidimensionalItem, MultidimensionalModel, MultidimensionalTestItem } from "./model.js";
export { checkedPrior, type MultivariateNormalPrior, priorDimensions } from "./prior.js";
export {
  type MultidimensionalEstimate,
  type MultidimensionalEstimator,
  multidimensionalPosteriorModeEstimate,
  STANDARD_GRIDS,
} from "./scoring.js";
export { itemCorrelations } from "./starting-slopes.js";

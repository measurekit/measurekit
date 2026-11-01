/**
 * Dimensionality of a test: polychoric and tetrachoric correlations of item responses, parallel analysis of the
 * eigenvalues of their correlations against random data, exploratory factor analysis by the minimum residual method,
 * Velicer's minimum average partial test and the smoothing of correlation matrices.
 *
 * @module @measurekit/psychometrics/dimensionality
 */
export {
  type ExploratoryFactorOptions,
  type ExploratoryFactors,
  exploratoryFactorAnalysis,
  type FactorRotation,
} from "./exploratory-factors.js";
export type { ItemCorrelation } from "./item-correlations.js";
export {
  type MinimumAveragePartial,
  type MinimumAveragePartialOptions,
  minimumAveragePartial,
} from "./minimum-average-partial.js";
export {
  type EigenvalueComparison,
  type ParallelAnalysis,
  type ParallelAnalysisOptions,
  parallelAnalysis,
} from "./parallel-analysis.js";
export { type PolychoricCorrelations, type PolychoricOptions, polychoricCorrelations } from "./polychoric.js";
export { type SmoothingOptions, smoothCorrelations } from "./smoothing.js";

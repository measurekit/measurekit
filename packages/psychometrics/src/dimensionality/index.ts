/**
 * Dimensionality of a test: polychoric and tetrachoric correlations of item responses, and parallel analysis of the
 * eigenvalues of their correlations against random data.
 *
 * @module @measurekit/psychometrics/dimensionality
 */
export {
  type EigenvalueComparison,
  type ParallelAnalysis,
  type ParallelAnalysisOptions,
  parallelAnalysis,
} from "./parallel-analysis.js";
export { type PolychoricCorrelations, type PolychoricOptions, polychoricCorrelations } from "./polychoric.js";

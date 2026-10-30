/**
 * Classical test theory: item statistics, distractor analysis, reliability and classification.
 *
 * @module @measurekit/psychometrics/ctt
 */
export {
  classifyDifficulty,
  classifyDiscrimination,
  DIFFICULTY_THRESHOLDS,
  DISCRIMINATION_THRESHOLDS,
  type DifficultyClass,
  type DifficultyThresholds,
  type DiscriminationClass,
  type DiscriminationThresholds,
} from "./classification.js";
export { distractorAnalysis, type OptionAnalysis, type OptionStatistics } from "./distractors.js";
export type { GroupOptions } from "./groups.js";
export { type ItemStatistics, itemStatistics } from "./item-statistics.js";
export { type McDonaldOmega, mcdonaldOmega } from "./omega.js";
export {
  cronbachAlpha,
  kr20,
  kr21,
  type SplitHalf,
  type SplitHalfOptions,
  type SplitMethod,
  splitHalf,
  standardErrorOfMeasurement,
} from "./reliability.js";

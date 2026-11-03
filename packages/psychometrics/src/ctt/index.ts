/**
 * Classical test theory: item statistics, distractor analysis, reliability, the value of subscores, augmented subscores and
 * classification.
 *
 * @module @measurekit/psychometrics/ctt
 */
export { type AugmentedSection, type AugmentedSubscores, augmentedSubscores } from "./augmented-subscores.js";
export {
  classifyDifficulty,
  classifyDiscrimination,
  DIFFICULTY_THRESHOLDS,
  DISCRIMINATION_THRESHOLDS,
  type DifficultyClass,
  type DifficultyClassificationOptions,
  type DifficultyThresholds,
  type DiscriminationClass,
  type DiscriminationClassificationOptions,
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
  type StandardErrorOfMeasurementOptions,
  splitHalf,
  standardErrorOfMeasurement,
} from "./reliability.js";
export { type SubscoreValue, subscoreValue } from "./subscores.js";

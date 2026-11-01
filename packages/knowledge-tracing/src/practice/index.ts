/**
 * Practice: the next item by predicted success, mastery criteria replayed over a log (a probability of mastery, a streak,
 * a moving average, a window of outcomes and a predicted success), and the attempts expected until a streak of correct
 * answers.
 *
 * @module @measurekit/knowledge-tracing/practice
 */
export {
  type MasteryModel,
  type MasteryTable,
  type MovingAverageOptions,
  movingAverageMastery,
  type OpportunitiesToStreakOptions,
  opportunitiesToStreak,
  type PredictedOptions,
  predictedMastery,
  type StreakOptions,
  streakMastery,
  type ThresholdOptions,
  thresholdMastery,
  type WindowOptions,
  windowMastery,
} from "./mastery.js";
export { type Recommendation, type RecommendOptions, recommend } from "./recommend.js";

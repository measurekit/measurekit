/**
 * Practice: the next item by predicted success, mastery criteria replayed over a log, and the attempts expected until a
 * streak of correct answers.
 *
 * @module @measurekit/knowledge-tracing/practice
 */
export {
  type MasteryModel,
  type MasteryTable,
  type OpportunitiesToStreakOptions,
  opportunitiesToStreak,
  type StreakOptions,
  streakMastery,
  type ThresholdOptions,
  thresholdMastery,
} from "./mastery.js";
export { type Recommendation, type RecommendOptions, recommend } from "./recommend.js";

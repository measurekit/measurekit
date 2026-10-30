/**
 * Logistic learner models: the additive factors model, performance factors analysis and models of the features of
 * logistic knowledge tracing, with learner proficiency as fixed effects or a random intercept, fitted by Newton's
 * method with an optional ridge.
 *
 * @module @measurekit/knowledge-tracing/logistic
 */
export type { LogisticFeature } from "./features.js";
export {
  type AfmOptions,
  afm,
  type LogisticFit,
  type LogisticModel,
  type LogisticOptions,
  logistic,
  pfa,
} from "./logistic.js";
export type { LogisticParameters, LogisticSkill, Proficiency } from "./parameters.js";
export type { DecayedCounts, LogisticState, SkillHistory } from "./state.js";

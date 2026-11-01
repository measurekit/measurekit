/**
 * Logistic learner models: the additive factors model, performance factors analysis and models of the features of
 * logistic knowledge tracing, with intercepts for skills and items, coefficients for each skill or item or shared, terms
 * over the item or the learner, covariates and standard errors, learner proficiency as fixed effects or a random
 * intercept, fitted by Newton's method with an optional ridge.
 *
 * @module @measurekit/knowledge-tracing/logistic
 */
export type { LogisticFeature, PerformancePrediction, PracticeAge } from "./features.js";
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
export type { DecayedCounts, ExposureHistory, LeakyCounts, LogisticState, SkillHistory } from "./state.js";
export type { InterceptKind, LogisticTerm, TermCoefficient, TermScope } from "./terms.js";

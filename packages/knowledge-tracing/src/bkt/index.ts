/**
 * Bayesian knowledge tracing: the four-parameter model with pyBKT's extensions (forgetting, item guess and slip, item
 * learning, prior groups), fitted by expectation maximization and updated online by forward inference.
 *
 * @module @measurekit/knowledge-tracing/bkt
 */
export { type BktFit, type BktModel, type BktOptions, bkt, type OpportunitiesToMasteryOptions } from "./bkt.js";
export type { BktGuessSlip, BktLearning, BktParameters, BktSkill } from "./parameters.js";
export type { BktState } from "./state.js";

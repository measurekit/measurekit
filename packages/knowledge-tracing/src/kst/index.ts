/**
 * Knowledge space theory: knowledge structures over items or skills, quasi-ordinal spaces from surmise relations,
 * learning spaces and fringes; the basic local independence model fitted by maximum likelihood or minimum discrepancy;
 * inductive item tree analysis; and a learner model with adaptive assessment.
 *
 * @module @measurekit/knowledge-tracing/kst
 */

export { delineate } from "./delineate.js";
export { type IitaCandidate, type IitaOptions, type IitaResult, type IitaVersion, iita } from "./iita.js";
export { type KstFit, type KstModel, type KstOptions, kst, type LikelyState, type NextQuestionOptions } from "./kst.js";
export type { KstItem, KstParameters } from "./parameters.js";
export { type Fringes, fringes, isKnowledgeSpace, isLearningSpace } from "./space.js";
export type { KstState } from "./state.js";
export type { KnowledgeStructure } from "./structure.js";
export { type SurmiseStructureOptions, surmiseStructure } from "./surmise.js";

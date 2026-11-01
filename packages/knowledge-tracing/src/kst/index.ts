/**
 * Knowledge space theory: knowledge structures over items or skills, quasi-ordinal spaces from surmise relations and
 * spaces from clauses, closure under union, bases, atoms, traces and notions, learning spaces, fringes, learning paths
 * and neighbourhoods; the basic local independence model fitted by maximum likelihood or minimum discrepancy;
 * inductive item tree analysis; and a learner model with adaptive assessment.
 *
 * @module @measurekit/knowledge-tracing/kst
 */

export { notions, structureAtoms, structureBase, structureTrace } from "./base.js";
export type { StateLimitOptions } from "./build.js";
export { clauseStructure } from "./clauses.js";
export { unionClosure } from "./closure.js";
export { delineate } from "./delineate.js";
export { type IitaCandidate, type IitaOptions, type IitaResult, type IitaVersion, iita } from "./iita.js";
export { type KstFit, type KstModel, type KstOptions, kst, type LikelyState, type NextQuestionOptions } from "./kst.js";
export type { KstItem, KstParameters } from "./parameters.js";
export {
  isWellGraded,
  type LearningPathsOptions,
  learningPaths,
  type NeighbourhoodOptions,
  neighbourhood,
} from "./paths.js";
export { type Fringes, fringes, isKnowledgeSpace, isLearningSpace } from "./space.js";
export type { KstState } from "./state.js";
export type { KnowledgeStructure } from "./structure.js";
export { type SurmiseStructureOptions, surmiseStructure } from "./surmise.js";

/**
 * Knowledge space theory: knowledge structures over items or skills, quasi-ordinal spaces from surmise relations and
 * spaces from clauses, closure under union, bases, atoms, traces and notions, learning spaces, fringes, learning paths
 * and neighbourhoods; the basic local independence model fitted by maximum likelihood, minimum discrepancy or both, with
 * fixed and equal rates, its fit, comparison and local identifiability; the validity of a structure and the test of a
 * quasi-order against responses; inductive item tree analysis; and a learner model with adaptive assessment.
 *
 * @module @measurekit/knowledge-tracing/kst
 */

export { notions, structureAtoms, structureBase, structureTrace } from "./base.js";
export type { StateLimitOptions } from "./build.js";
export { clauseStructure } from "./clauses.js";
export { unionClosure } from "./closure.js";
export type { EqualRates, FixedRates } from "./constraints.js";
export { delineate } from "./delineate.js";
export { type DiffTest, type DiffTestOptions, diffTest } from "./diff-test.js";
export { compareFits, type FitComparison, type GoodnessOfFit } from "./goodness.js";
export { backwardGraded, forwardGraded } from "./graded.js";
export type { Identifiability } from "./identifiability.js";
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
export { type Prerequisites, type SurmiseStructureOptions, surmiseStructure } from "./surmise.js";
export { type StructureValidity, structureValidity } from "./validity.js";

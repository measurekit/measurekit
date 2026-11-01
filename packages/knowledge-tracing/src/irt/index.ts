/**
 * Online item response theory: the posterior of each learner's θ on a grid, updated by every outcome from calibrated
 * parameters of 1PL to 4PL, graded response and generalized partial credit items, with EAP and MAP estimates, and
 * carried across the gaps between events by an optional drift of θ.
 *
 * @module @measurekit/knowledge-tracing/irt
 */

export type { ThetaDistribution } from "@measurekit/core";
export type { Ability } from "./ability.js";
export {
  type AbilityOptions,
  type IrtDrift,
  type IrtFit,
  type IrtModel,
  type IrtOptions,
  type IrtPrior,
  irt,
  type PosteriorOptions,
} from "./irt.js";
export type { DichotomousItem, GradedItem, IrtItem, PartialCreditItem } from "./items.js";
export type { IrtParameters } from "./parameters.js";
export type { IrtState } from "./state.js";

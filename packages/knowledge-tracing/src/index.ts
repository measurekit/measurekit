/**
 * The learner model interface that every model implements, and what every module uses and core owns: the errors the
 * package throws, its seeded random number generator and the types of core that its functions take and return. Each
 * module is imported from its subpath, such as `@measurekit/knowledge-tracing/bkt`.
 *
 * @module @measurekit/knowledge-tracing
 */
export {
  createRandom,
  DataError,
  MeasurekitError,
  type NominalParameters,
  type QMatrix,
  type Random,
  type ThetaDistribution,
  type ThresholdItem,
} from "@measurekit/core";
export type { LearnerModel, ModelFit } from "./model.js";

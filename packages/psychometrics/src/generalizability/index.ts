/**
 * Generalizability theory for crossed and nested designs: variance components by the analysis of variance, balanced or
 * with empty cells by Henderson's Method 1, and decision studies with the generalizability and dependability
 * coefficients.
 *
 * @module @measurekit/psychometrics/generalizability
 */
export {
  type DecisionComponent,
  type DecisionStudy,
  type DecisionStudyOptions,
  decisionStudy,
} from "./decision-study.js";
export {
  type GeneralizabilityDesign,
  type GeneralizabilityStudy,
  generalizabilityStudy,
  type VarianceComponent,
} from "./study.js";

/**
 * Simulation: thetas, item parameters of every built-in model, response matrices of IRT, multidimensional IRT and
 * cognitive diagnosis models, drawn from a seeded `Random`.
 *
 * @module @measurekit/psychometrics/sim
 */
export { type CdmModelDescription, simulateCdmResponses } from "./cdm-responses.js";
export {
  type Distribution,
  logNormalDistribution,
  normalDistribution,
  uniformDistribution,
} from "./distributions.js";
export { type ItemSimulationOptions, simulateItems } from "./items.js";
export { simulateMultidimensionalResponses } from "./multidimensional-responses.js";
export { type ResponseSimulationOptions, simulateResponses } from "./responses.js";
export { simulateThetas } from "./thetas.js";

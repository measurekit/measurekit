/**
 * Conversion of item parameters to and from mirt's coefficients, for every built-in model, multidimensional ones included.
 *
 * @module @measurekit/psychometrics/parameterization
 */
export {
  type MirtCoefficients,
  readMirtIrt,
  readMirtMultidimensional,
  readMirtSlopeIntercept,
  writeMirtIrt,
  writeMirtMultidimensional,
  writeMirtSlopeIntercept,
} from "./mirt.js";

/**
 * Conversion of item parameters to and from mirt's coefficients, for every built-in model, multidimensional ones included.
 *
 * @module @measurekit/psychometrics/interop
 */
export {
  fromMirtIrt,
  fromMirtMultidimensional,
  fromMirtSlopeIntercept,
  type MirtCoefficients,
  toMirtIrt,
  toMirtMultidimensional,
  toMirtSlopeIntercept,
} from "./mirt.js";

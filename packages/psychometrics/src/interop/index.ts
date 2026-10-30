/**
 * Conversion of item parameters to and from mirt's coefficients, for every built-in model.
 *
 * @module @measurekit/psychometrics/interop
 */
export {
  fromMirtIrt,
  fromMirtSlopeIntercept,
  type MirtCoefficients,
  toMirtIrt,
  toMirtSlopeIntercept,
} from "./mirt.js";

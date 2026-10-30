import { logistic } from "@measurekit/core";

// Calibration estimates a bounded parameter x on an unbounded scale η: logistic between two finite bounds,
// exponential beyond one, identity without bounds.

/** Returns the parameter value at the unbounded coordinate `eta`. */
export function fromUnbounded(eta: number, lower: number, upper: number): number {
  if (Number.isFinite(lower) && Number.isFinite(upper)) return lower + (upper - lower) * logistic(eta);
  if (Number.isFinite(lower)) return lower + Math.exp(eta);
  if (Number.isFinite(upper)) return upper - Math.exp(-eta);
  return eta;
}

/** Returns the unbounded coordinate of a value strictly between the bounds. */
export function toUnbounded(x: number, lower: number, upper: number): number {
  if (Number.isFinite(lower) && Number.isFinite(upper)) return Math.log((x - lower) / (upper - x));
  if (Number.isFinite(lower)) return Math.log(x - lower);
  if (Number.isFinite(upper)) return -Math.log(upper - x);
  return x;
}

/** Returns dx/dη at the value `x`. */
export function unboundedDerivative(x: number, lower: number, upper: number): number {
  if (Number.isFinite(lower) && Number.isFinite(upper)) return ((x - lower) * (upper - x)) / (upper - lower);
  if (Number.isFinite(lower)) return x - lower;
  if (Number.isFinite(upper)) return upper - x;
  return 1;
}

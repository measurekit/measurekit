/** Returns the logistic function 1 / (1 + exp(-x)), the probability of a logit `x`. */
export function logistic(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Returns ln logistic(x) without underflow: -ln(1 + exp(-x)), computed as -(max(-x, 0) + ln(1 + exp(-|x|))), finite for
 * every finite x.
 */
export function logLogistic(x: number): number {
  return -(Math.max(-x, 0) + Math.log1p(Math.exp(-Math.abs(x))));
}

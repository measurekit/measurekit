/**
 * Returns the logistic function 1 / (1 + exp(-x)), the probability of a logit `x`.
 *
 * @param x - The logit.
 * @returns The probability, from 0 to 1.
 */
export function logistic(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

/**
 * Returns ln logistic(x) without underflow: -ln(1 + exp(-x)), computed as -(max(-x, 0) + ln(1 + exp(-|x|))), finite for
 * every finite x.
 *
 * @param x - The logit.
 * @returns The natural logarithm of the probability, at most 0.
 */
export function logLogistic(x: number): number {
  return -(Math.max(-x, 0) + Math.log1p(Math.exp(-Math.abs(x))));
}

/**
 * Writes logistic(x) to `out[0]` and logistic(-x) to `out[1]` from one exponential, each accurate in its own tail: with
 * t = exp(-|x|) and s = 1 / (1 + t), the larger is s and the smaller t s.
 *
 * @param x - The logit.
 * @param out - Receives logistic(x) at index 0 and logistic(-x) at index 1.
 */
export function logisticPair(x: number, out: Float64Array): void {
  const t = Math.exp(-Math.abs(x));
  const larger = 1 / (1 + t);
  const smaller = t * larger;
  out[0] = x >= 0 ? larger : smaller;
  out[1] = x >= 0 ? smaller : larger;
}

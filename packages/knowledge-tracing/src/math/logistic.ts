/**
 * Returns ln logistic(x) without underflow: -ln(1 + exp(-x)), computed as -(max(-x, 0) + ln(1 + exp(-|x|))), finite for
 * every finite x.
 */
export function logLogistic(x: number): number {
  return -(Math.max(-x, 0) + Math.log1p(Math.exp(-Math.abs(x))));
}

/**
 * Returns the log-likelihood of an outcome y in [0, 1] under the probability p of a correct response:
 * y ln p + (1 - y) ln(1 - p). A term of weight 0 is 0, so a prediction of 0 or 1 has log-likelihood 0 when the outcome
 * matches it and `-Infinity` when it is wrong.
 */
export function outcomeLogLikelihood(probability: number, outcome: number): number {
  let sum = 0;
  if (outcome > 0) sum += outcome * Math.log(probability);
  if (outcome < 1) sum += (1 - outcome) * Math.log1p(-probability);
  return sum;
}

/** Returns the logistic function 1 / (1 + exp(-x)), the probability of a logit `x`. */
export function logistic(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

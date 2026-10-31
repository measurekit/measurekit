/**
 * The smallest probability a likelihood uses. A probability that underflows to 0 at an extreme theta would make a
 * log-likelihood infinite and its derivatives NaN; clamped, it contributes about -744 and a slope of 0.
 */
export const MIN_PROBABILITY = Number.MIN_VALUE;

/** Returns the natural logarithm of a probability, with the probability at least {@link MIN_PROBABILITY}. */
export function logProbability(probability: number): number {
  return Math.log(Math.max(probability, MIN_PROBABILITY));
}

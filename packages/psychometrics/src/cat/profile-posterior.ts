import { logProbability } from "../irt/internal.js";
import type { DiagnosticTestItem } from "./diagnostic-items.js";

/**
 * Returns the posterior probability of each profile given responses to items: the prior times the probability of each
 * response in the item's latent group of the profile, normalized to sum to 1. A response's probability enters as at
 * least `MIN_PROBABILITY`, so that a success probability of 0 or 1 does not make the posterior 0 everywhere; a profile
 * with prior 0 has posterior 0.
 */
export function profilePosterior(
  prior: Float64Array,
  items: readonly DiagnosticTestItem[],
  responses: readonly number[],
): number[] {
  const logPosterior = prior.map((probability) => (probability > 0 ? Math.log(probability) : Number.NEGATIVE_INFINITY));
  items.forEach(({ probabilities, groups }, j) => {
    const correct = responses[j] === 1;
    const logProbabilities = Float64Array.from(probabilities, (p) => logProbability(correct ? p : 1 - p));
    for (let c = 0; c < logPosterior.length; c++) {
      logPosterior[c] = (logPosterior[c] ?? 0) + (logProbabilities[groups[c] ?? 0] ?? 0);
    }
  });
  const largest = logPosterior.reduce((most, value) => Math.max(most, value), Number.NEGATIVE_INFINITY);
  const posterior = logPosterior.map((value) => Math.exp(value - largest));
  const total = posterior.reduce((sum, value) => sum + value, 0);
  return Array.from(posterior, (value) => value / total);
}

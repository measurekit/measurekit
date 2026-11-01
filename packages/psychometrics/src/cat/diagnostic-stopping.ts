import { DataError, describe, isUnitInterval } from "@measurekit/core";
import type { ProfileEstimate } from "../cdm/internal.js";
import type { DiagnosticTestItem } from "./diagnostic-items.js";
import type { StoppingRule } from "./stopping.js";

/**
 * Returns a rule that stops once the largest posterior probability of a profile is at least `probability`: the test
 * has classified the person into one profile with that confidence.
 *
 * @param probability - From above 0 to 1.
 * @returns The stopping rule.
 * @throws `DataError` when `probability` is not above 0 and at most 1.
 * @see {@link https://doi.org/10.1111/1467-9876.00272 | Tatsuoka (2002)}
 */
export function posteriorThreshold(probability: number): StoppingRule<DiagnosticTestItem, ProfileEstimate> {
  if (!(isUnitInterval(probability) && probability > 0)) {
    throw new DataError(`A posterior threshold must be above 0 and at most 1, found ${describe(probability)}.`);
  }
  return { stops: ({ estimate }) => estimate.posterior.reduce((most, p) => Math.max(most, p), 0) >= probability };
}

/**
 * Returns a rule that stops once every attribute is classified with confidence: the posterior probability of mastering
 * it is at least `probability` or at most `1 - probability`.
 *
 * @param probability - Above 0.5 and at most 1.
 * @returns The stopping rule.
 * @throws `DataError` when `probability` is not above 0.5 and at most 1.
 * @see {@link https://doi.org/10.1177/0146621613488642 | Hsu, Wang and Chen (2013)}
 */
export function masteryThreshold(probability: number): StoppingRule<DiagnosticTestItem, ProfileEstimate> {
  if (!(isUnitInterval(probability) && probability > 0.5)) {
    throw new DataError(`A mastery threshold must be above 0.5 and at most 1, found ${describe(probability)}.`);
  }
  return {
    stops: ({ estimate }) => estimate.mastery.every((mastery) => mastery >= probability || mastery <= 1 - probability),
  };
}

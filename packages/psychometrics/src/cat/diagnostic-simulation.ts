import { checkArray, checkRandom, type Random } from "@measurekit/core";
import { simulateCdmResponses } from "../sim/cdm-responses.js";
import type { DiagnosticAdaptiveTest, DiagnosticAdaptiveTestState } from "./diagnostic-test.js";
import { checkTest, type ExposureRate, exposureRates } from "./simulation.js";

/** One simulated person of a diagnostic test: the true profile and the final state of the test. */
export interface SimulatedDiagnosticPerson {
  /** The person's true profile, in the test's attribute order. */
  readonly profile: readonly number[];
  /** The state at the end of the test. */
  readonly state: DiagnosticAdaptiveTestState;
}

/** The result of simulated administrations of a diagnostic test. */
export interface DiagnosticTestSimulation {
  /** Each person, in the order of the profiles. */
  readonly persons: readonly SimulatedDiagnosticPerson[];
  /** The share of persons whose MAP profile equals the true profile. */
  readonly patternAccuracy: number;
  /** For each attribute, in the test's order, the share of persons whose MAP profile agrees with the true one. */
  readonly attributeAccuracy: readonly number[];
  /** The mean number of items administered. */
  readonly meanLength: number;
  /** The exposure rate of every item of the pool, in its order. */
  readonly exposureRates: readonly ExposureRate[];
}

/**
 * Simulates a diagnostic adaptive test for persons with known profiles: each person's response to an item is correct
 * with the success probability of the item's latent group that the profile falls in, as {@link simulateCdmResponses}
 * draws it. The same `random` draws the responses, then makes the test's random choices, so a seeded generator
 * reproduces the simulation.
 *
 * @throws `DataError` when `test` is not a diagnostic adaptive test, `random` is not a random number generator, there
 * are no profiles or a profile is not a 0 or 1 for each attribute.
 */
export function simulateDiagnosticTest(
  test: DiagnosticAdaptiveTest,
  profiles: readonly (readonly number[])[],
  random: Random,
): DiagnosticTestSimulation {
  checkTest(test, "pool", "The test must be a diagnostic adaptive test from diagnosticAdaptiveTest");
  checkArray(test.attributes, "The test must be a diagnostic adaptive test from diagnosticAdaptiveTest");
  checkRandom(random);
  const { pool, attributes } = test;
  const matrix = simulateCdmResponses({ attributes, items: pool }, profiles, random);
  const columns = new Map(matrix.itemIds.map((id, column) => [id, column]));
  const persons = profiles.map((profile, person) => {
    let state = test.start(random);
    while (state.next !== null) {
      const response = matrix.responses[person]?.[columns.get(state.next) ?? -1] ?? 0;
      state = test.answer(state, response, random);
    }
    return { profile, state };
  });
  const share = (agrees: (person: SimulatedDiagnosticPerson) => boolean): number =>
    persons.filter(agrees).length / persons.length;
  return {
    persons,
    patternAccuracy: share(({ profile, state }) => profile.every((value, k) => state.estimate.map[k] === value)),
    attributeAccuracy: attributes.map((_, k) => share(({ profile, state }) => state.estimate.map[k] === profile[k])),
    meanLength: persons.reduce((sum, { state }) => sum + state.items.length, 0) / persons.length,
    exposureRates: exposureRates(
      pool,
      persons.map(({ state }) => state),
    ),
  };
}

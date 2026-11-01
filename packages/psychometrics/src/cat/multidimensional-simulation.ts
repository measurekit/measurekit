import { checkRandom, mean, type Random } from "@measurekit/core";
import { simulateMultidimensionalResponses } from "../sim/internal.js";
import type { MultidimensionalAdaptiveTest, MultidimensionalAdaptiveTestState } from "./multidimensional-test.js";
import { administer, checkTest, type ExposureRate, responseLookup, summarizeAdministrations } from "./simulation.js";

/** One simulated person of a multidimensional test: the true thetas and the final state of the test. */
export interface SimulatedMultidimensionalPerson {
  /** The person's true theta in each dimension. */
  readonly theta: readonly number[];
  /** The state at the end of the test. */
  readonly state: MultidimensionalAdaptiveTestState;
}

/** The result of simulated administrations of a multidimensional adaptive test. */
export interface MultidimensionalTestSimulation {
  /** Each person, in the order of the thetas. */
  readonly persons: readonly SimulatedMultidimensionalPerson[];
  /** For each dimension, the mean of the estimate minus the true theta. */
  readonly bias: readonly number[];
  /** For each dimension, the square root of the mean squared difference between the estimate and the true theta. */
  readonly rootMeanSquareError: readonly number[];
  /** The mean number of items administered. */
  readonly meanLength: number;
  /** The exposure rate of every item of the pool, in its order. */
  readonly exposureRates: readonly ExposureRate[];
}

/**
 * Simulates a multidimensional adaptive test for persons with known thetas: each person's response to every pool item is
 * drawn from the item's category probabilities at the person's thetas, as {@link simulateMultidimensionalResponses}
 * draws them. The same `random` then makes the test's random choices, so a seeded generator reproduces the simulation.
 *
 * @param test - The test to run.
 * @param thetas - The thetas of each person, one for each dimension.
 * @param random - The generator of the responses, the first item and ties.
 * @returns The simulated tests of the persons, the bias and root mean square error of the estimates, the mean length
 *   and the exposure rate of each item.
 * @throws `DataError` when `test` is not a multidimensional adaptive test, `random` is not a random number generator,
 * there are no persons, or a person's thetas are not as many finite numbers as the items have dimensions.
 */
export function simulateMultidimensionalAdaptiveTest(
  test: MultidimensionalAdaptiveTest,
  thetas: readonly (readonly number[])[],
  random: Random,
): MultidimensionalTestSimulation {
  checkTest(test, "pool", "The test must be a multidimensional adaptive test from multidimensionalAdaptiveTest");
  checkRandom(random);
  const { pool } = test;
  const matrix = simulateMultidimensionalResponses(pool, thetas, random);
  const response = responseLookup(
    matrix.responses,
    pool.map(({ id }) => id),
  );
  const persons = thetas.map((theta, person) => ({
    theta,
    state: administer(test, (id) => response(person, id), random),
  }));
  const dimensions = thetas[0]?.length ?? 0;
  const errors = (d: number): number[] =>
    persons.map(({ theta, state }) => (state.estimate.theta[d] ?? 0) - (theta[d] ?? 0));
  const perDimension = Array.from({ length: dimensions }, (_, d) => errors(d));
  return {
    persons,
    bias: perDimension.map((list) => mean(list)),
    rootMeanSquareError: perDimension.map((list) =>
      Math.sqrt(list.reduce((sum, error) => sum + error * error, 0) / list.length),
    ),
    ...summarizeAdministrations(
      pool,
      persons.map(({ state }) => state),
    ),
  };
}

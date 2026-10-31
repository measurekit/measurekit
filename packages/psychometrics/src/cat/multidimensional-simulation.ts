import { checkRandom, type Random } from "@measurekit/core";
import { simulateMultidimensionalResponses } from "../sim/multidimensional-responses.js";
import type { MultidimensionalAdaptiveTest, MultidimensionalAdaptiveTestState } from "./multidimensional-test.js";
import { checkTest, type ExposureRate, exposureRates } from "./simulation.js";

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
  const columns = new Map(pool.map((item, column) => [item.id, column]));
  const persons = thetas.map((theta, person) => {
    let state = test.start(random);
    while (state.next !== null) {
      const response = matrix.responses[person]?.[columns.get(state.next) ?? -1] ?? 0;
      state = test.answer(state, response, random);
    }
    return { theta, state };
  });
  const dimensions = thetas[0]?.length ?? 0;
  const errors = (d: number): number[] =>
    persons.map(({ theta, state }) => (state.estimate.theta[d] ?? 0) - (theta[d] ?? 0));
  const perDimension = Array.from({ length: dimensions }, (_, d) => errors(d));
  return {
    persons,
    bias: perDimension.map((list) => list.reduce((sum, error) => sum + error, 0) / list.length),
    rootMeanSquareError: perDimension.map((list) =>
      Math.sqrt(list.reduce((sum, error) => sum + error * error, 0) / list.length),
    ),
    meanLength: persons.reduce((sum, { state }) => sum + state.items.length, 0) / persons.length,
    exposureRates: exposureRates(
      pool,
      persons.map(({ state }) => state),
    ),
  };
}

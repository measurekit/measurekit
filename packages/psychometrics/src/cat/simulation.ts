import {
  checkOptions,
  checkRandom,
  DataError,
  describe,
  type FieldNames,
  isRecord,
  type Random,
} from "@measurekit/core";
import { checkFunction } from "../checks.js";
import { indexItems, largestCategoryCount, type TestItem } from "../irt/internal.js";
import { checkThetas, simulateResponse } from "../sim/internal.js";
import type { AdaptiveTest, AdaptiveTestState } from "./engine.js";
import { itemById } from "./items.js";
import { locateModules, type MultistageTest, type MultistageTestState } from "./multistage-test.js";
import type { TestState } from "./state.js";

/**
 * One simulated person: the true theta and the final state of the test.
 *
 * @typeParam TState - The state of the test.
 */
export interface SimulatedPerson<TState extends TestState> {
  /** The person's true theta. */
  readonly theta: number;
  /** The state at the end of the test. */
  readonly state: TState;
}

/** The exposure rate of an item: the share of persons who were administered it. */
export interface ExposureRate {
  /** The item's id. */
  readonly id: string;
  /** The share of persons who were administered the item. */
  readonly rate: number;
}

/**
 * The result of simulated administrations of a test.
 *
 * @typeParam TState - The state of the test.
 */
export interface TestSimulation<TState extends TestState> {
  /** Each person, in the order of the thetas. */
  readonly persons: readonly SimulatedPerson<TState>[];
  /** The mean of the estimate minus the true theta. */
  readonly bias: number;
  /** The square root of the mean squared difference between the estimate and the true theta. */
  readonly rootMeanSquareError: number;
  /** The mean number of items administered. */
  readonly meanLength: number;
  /** The exposure rate of every item of the pool or panel, in its order. */
  readonly exposureRates: readonly ExposureRate[];
}

/** Options of the simulations of adaptive tests, diagnostic and multidimensional ones included. */
export interface AdaptiveSimulationOptions {
  /**
   * Returns the generator of the test's random choices, such as the first item, ties and exposure control, at one step
   * of one person: step 0 starts the test and step n answers its nth item, and the persons are numbered from 0 in the
   * order given. A host that seeds each call of `start` and `answer` from a person's session passes its generators, so
   * that the simulation gives the items the host would; the responses still come from `random`.
   *
   * @defaultValue the `random` of the responses, which makes the choices between its draws
   */
  readonly testRandom?: (person: number, step: number) => Random;
}

const ADAPTIVE_SIMULATION_OPTIONS: FieldNames<AdaptiveSimulationOptions> = { testRandom: true };

/**
 * Simulates an adaptive test for persons with known thetas: each response is drawn from the item's category
 * probabilities at the person's theta. The same `random` draws the responses and makes the test's random choices, or the
 * option `testRandom` makes those choices, so seeded generators reproduce the simulation.
 *
 * @param test - The test to run.
 * @param thetas - The abilities of the persons.
 * @param random - The generator of the responses and, without `testRandom`, of the first item and ties.
 * @param options - The generator of the test's choices at each step of each person.
 * @returns The simulated tests of the persons, the bias and root mean square error of the estimates, the mean length
 *   and the exposure rate of each item.
 * @throws `DataError` when `test` is not an adaptive test, `random` or a generator of `testRandom` not a random number
 * generator, without thetas or when a theta is not finite.
 */
export function simulateAdaptiveTest(
  test: AdaptiveTest,
  thetas: readonly number[],
  random: Random,
  options: AdaptiveSimulationOptions = {},
): TestSimulation<AdaptiveTestState> {
  checkTest(test, "pool", "The test must be an adaptive test from createAdaptiveTest");
  checkRandom(random);
  const randomOf = stepRandoms(options, random);
  const { pool } = test;
  const indices = indexItems(pool);
  const probabilities = new Float64Array(largestCategoryCount(pool));
  checkThetas(thetas);
  const persons = thetas.map((theta, person) => ({
    theta,
    state: administer(
      test,
      (id) => simulateResponse(itemById(pool, indices, id), theta, random, probabilities),
      randomOf(person),
    ),
  }));
  return summarize(persons, pool);
}

/**
 * Simulates a multistage test for persons with known thetas: each response is drawn from the item's category
 * probabilities at the person's theta, so a seeded `random` reproduces the simulation.
 *
 * @param test - The test to run.
 * @param thetas - The abilities of the persons.
 * @param random - The generator of the responses and ties.
 * @returns The simulated tests of the persons, the bias and root mean square error of the estimates, the mean length
 *   and the exposure rate of each item.
 * @throws `DataError` when `test` is not a multistage test or `random` not a random number generator, without thetas or
 * when a theta is not finite.
 */
export function simulateMultistageTest(
  test: MultistageTest,
  thetas: readonly number[],
  random: Random,
): TestSimulation<MultistageTestState> {
  checkTest(test, "stages", "The test must be a multistage test from createMultistageTest");
  checkRandom(random);
  const modules = locateModules(test.stages);
  const items = [...modules.values()].flatMap(({ module }) => module.items);
  indexItems(items);
  const probabilities = new Float64Array(largestCategoryCount(items));
  checkThetas(thetas);
  const persons = thetas.map((theta) => {
    let state = test.start();
    while (state.next !== null) {
      const responses = (modules.get(state.next)?.module.items ?? []).map((item) =>
        simulateResponse(item, theta, random, probabilities),
      );
      state = test.answer(state, responses);
    }
    return { theta, state };
  });
  return summarize(persons, items);
}

/**
 * Checks that a test is of the kind a simulation takes: an object with the methods `start` and `answer` and the array
 * `field`, its pool or its stages.
 *
 * @throws `DataError` when it is not.
 */
export function checkTest(test: unknown, field: "pool" | "stages", message: string): void {
  const valid =
    isRecord(test) &&
    Array.isArray(test[field]) &&
    typeof test.start === "function" &&
    typeof test.answer === "function";
  if (!valid) throw new DataError(`${message}, found ${describe(test)}.`);
}

function summarize<TState extends TestState>(
  persons: readonly SimulatedPerson<TState>[],
  items: readonly TestItem[],
): TestSimulation<TState> {
  let error = 0;
  let squaredError = 0;
  for (const { theta, state } of persons) {
    const difference = state.estimate.theta - theta;
    error += difference;
    squaredError += difference * difference;
  }
  const n = persons.length;
  return {
    persons,
    bias: error / n,
    rootMeanSquareError: Math.sqrt(squaredError / n),
    ...summarizeAdministrations(
      items,
      persons.map(({ state }) => state),
    ),
  };
}

/**
 * Returns, for each person, the generator of each step of their test: those of the option `testRandom`, checked as
 * they come, or `random` at every step.
 *
 * @throws `DataError` when the options are not an object or `testRandom` is not a function.
 */
export function stepRandoms(
  options: AdaptiveSimulationOptions,
  random: Random,
): (person: number) => (step: number) => Random {
  checkOptions(options, ADAPTIVE_SIMULATION_OPTIONS);
  const { testRandom } = options;
  if (testRandom === undefined) return () => () => random;
  checkFunction(testRandom, "The option testRandom must be a function of a person and a step");
  return (person) => (step) => {
    const found = testRandom(person, step);
    checkRandom(found);
    return found;
  };
}

/**
 * Administers a test to one person and returns the state at its end.
 *
 * @param respond - Returns the person's response to the item with an id.
 * @param randomAt - Returns the generator of a step: 0 to start, n to answer the nth item.
 */
export function administer<TState extends { readonly items: readonly string[]; readonly next: string | null }>(
  test: {
    start(random: Random): TState;
    answer(state: TState, response: number, random: Random): TState;
  },
  respond: (id: string) => number,
  randomAt: (step: number) => Random,
): TState {
  let state = test.start(randomAt(0));
  while (state.next !== null) state = test.answer(state, respond(state.next), randomAt(state.items.length + 1));
  return state;
}

/** Returns the mean number of items administered, and the exposure rate of every item, in its order. */
export function summarizeAdministrations(
  items: readonly { readonly id: string }[],
  states: readonly TestState<unknown>[],
): { readonly meanLength: number; readonly exposureRates: ExposureRate[] } {
  let length = 0;
  for (const state of states) length += state.items.length;
  return { meanLength: length / states.length, exposureRates: exposureRates(items, states) };
}

/**
 * Returns the response of a person to the item with an id, from the simulated `responses` of persons, a row each, to
 * the items `itemIds`, a column each. A response that is missing is 0.
 */
export function responseLookup(
  responses: readonly (readonly (number | null)[])[],
  itemIds: readonly string[],
): (person: number, id: string) => number {
  const columns = new Map(itemIds.map((id, column) => [id, column]));
  return (person, id) => responses[person]?.[columns.get(id) ?? -1] ?? 0;
}

/** Returns the share of the states that administered each item, in the order of the items. */
function exposureRates(
  items: readonly { readonly id: string }[],
  states: readonly TestState<unknown>[],
): ExposureRate[] {
  const counts = new Map(items.map((item) => [item.id, 0]));
  for (const state of states) for (const id of state.items) counts.set(id, (counts.get(id) ?? 0) + 1);
  return items.map(({ id }) => ({ id, rate: (counts.get(id) ?? 0) / states.length }));
}

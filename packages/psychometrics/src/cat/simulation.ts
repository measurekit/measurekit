import { checkRandom, DataError, describe, isRecord, type Random } from "@measurekit/core";
import { indexItems, largestCategoryCount } from "../irt/items.js";
import type { TestItem } from "../irt/model.js";
import { checkThetas, simulateResponse } from "../sim/responses.js";
import type { AdaptiveTest, AdaptiveTestState } from "./adaptive-test.js";
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

/**
 * Simulates an adaptive test for persons with known thetas: each response is drawn from the item's category
 * probabilities at the person's theta. The same `random` draws the responses and makes the test's random choices, so
 * a seeded generator reproduces the simulation.
 *
 * @throws `DataError` when `test` is not an adaptive test or `random` not a random number generator, without thetas or
 * when a theta is not finite.
 */
export function simulateAdaptiveTest(
  test: AdaptiveTest,
  thetas: readonly number[],
  random: Random,
): TestSimulation<AdaptiveTestState> {
  checkTest(test, "pool", "The test must be an adaptive test from adaptiveTest");
  checkRandom(random);
  const { pool } = test;
  const indices = indexItems(pool);
  const probabilities = new Float64Array(largestCategoryCount(pool));
  checkThetas(thetas);
  const persons = thetas.map((theta) => ({
    theta,
    state: administer(
      test,
      (id) => simulateResponse(itemById(pool, indices, id), theta, random, probabilities),
      random,
    ),
  }));
  return summarize(persons, pool);
}

/**
 * Simulates a multistage test for persons with known thetas: each response is drawn from the item's category
 * probabilities at the person's theta, so a seeded `random` reproduces the simulation.
 *
 * @throws `DataError` when `test` is not a multistage test or `random` not a random number generator, without thetas or
 * when a theta is not finite.
 */
export function simulateMultistageTest(
  test: MultistageTest,
  thetas: readonly number[],
  random: Random,
): TestSimulation<MultistageTestState> {
  checkTest(test, "stages", "The test must be a multistage test from multistageTest");
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
 * Administers a test to one person and returns the state at its end.
 *
 * @param respond - Returns the person's response to the item with an id.
 */
export function administer<TState extends { readonly next: string | null }>(
  test: {
    start(random: Random): TState;
    answer(state: TState, response: number, random: Random): TState;
  },
  respond: (id: string) => number,
  random: Random,
): TState {
  let state = test.start(random);
  while (state.next !== null) state = test.answer(state, respond(state.next), random);
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

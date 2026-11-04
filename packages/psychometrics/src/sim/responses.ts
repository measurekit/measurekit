import {
  checkArray,
  checkFinite,
  checkOptions,
  checkRandom,
  checkUnitInterval,
  DataError,
  describe,
  type FieldNames,
  type Random,
} from "@measurekit/core";
import { buildResponseMatrix, type ResponseMatrix } from "../data/internal.js";
import { checkItemList, type IrtItem, largestCategoryCount, type TestItem } from "../irt/internal.js";

/** Options of {@link simulateResponses}. */
export interface ResponseSimulationOptions {
  /**
   * The probability that a response is missing, the same for every response and independent of the responses: missing
   * completely at random.
   *
   * @defaultValue `0`
   */
  readonly missingProbability?: number;
}

const RESPONSE_SIMULATION_OPTIONS: FieldNames<ResponseSimulationOptions> = { missingProbability: true };

/**
 * Simulates the responses of persons with known thetas to items of any models.
 *
 * @remarks
 * Each response is drawn from its item's category probabilities at the person's theta, by person, then item. Then, with
 * a `missingProbability`, each response is missing with that probability, so a seed gives the same responses with and
 * without missingness, less the missing ones. The matrix has a row per theta and a column per item, named by the items'
 * ids when they carry them, as a test's items do, and `item1` and on otherwise.
 *
 * @param items - The items, each with an `id` or none.
 * @param thetas - The ability of each person.
 * @param random - The generator of the responses.
 * @param options - The probability that a response is missing.
 * @returns The responses, one row for each person.
 * @throws `DataError` without items or thetas, when an item is invalid, some items carry an id and others none, the ids
 * are not distinct non-empty strings, a theta is not finite, `random` is not a random number generator, or
 * `missingProbability` is outside [0, 1].
 *
 * @example
 * ```ts
 * import { createRandom } from "@measurekit/psychometrics";
 * import { twoParameterLogistic } from "@measurekit/psychometrics/irt";
 * import { simulateItems, simulateResponses, simulateThetas } from "@measurekit/psychometrics/sim";
 *
 * const random = createRandom(1);
 * const items = simulateItems(twoParameterLogistic, 3, random);
 * const thetas = simulateThetas(4, random);
 * const data = simulateResponses(items, thetas, random);
 * data.itemIds; // ["item1", "item2", "item3"]
 * data.responses; // [[0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 1, 1]], a row for each person
 * ```
 */
export function simulateResponses(
  items: readonly (IrtItem | TestItem)[],
  thetas: readonly number[],
  random: Random,
  options: ResponseSimulationOptions = {},
): ResponseMatrix {
  const missingProbability = missingProbabilityOf(options);
  checkItemList(items, "A simulation");
  const itemIds = carriedItemIds(items);
  checkThetas(thetas);
  checkRandom(random);
  const probabilities = new Float64Array(largestCategoryCount(items));
  const rows: (number | null)[][] = thetas.map((theta) =>
    items.map((item) => simulateResponse(item, theta, random, probabilities)),
  );
  return matrixWithMissing(rows, missingProbability, random, itemIds);
}

/**
 * Returns the ids that items carry, as the items of a test or a calibration do, or `undefined` when none carries one.
 *
 * @throws `DataError` when some items carry an id and others none, or an id is not a string.
 */
export function carriedItemIds(items: readonly object[]): string[] | undefined {
  const first = items.findIndex((item) => Object.hasOwn(item, "id"));
  if (first === -1) return undefined;
  return items.map((item, index) => {
    const id: unknown = Reflect.get(item, "id");
    if (typeof id !== "string") {
      throw new DataError(`Item ${index + 1} needs a string id, as item ${first + 1} has one, found ${describe(id)}.`);
    }
    return id;
  });
}

/**
 * Checks the options of a simulation of responses and returns its missing probability.
 *
 * @throws `DataError` when `missingProbability` is outside [0, 1].
 */
export function missingProbabilityOf(options: ResponseSimulationOptions): number {
  checkOptions(options, RESPONSE_SIMULATION_OPTIONS);
  const { missingProbability = 0 } = options;
  checkUnitInterval(missingProbability, "The option missingProbability");
  return missingProbability;
}

/**
 * Makes each response missing with `missingProbability`, in row order, and returns the matrix of the rows, with the
 * item ids given or the default ones.
 */
export function matrixWithMissing(
  rows: (number | null)[][],
  missingProbability: number,
  random: Random,
  itemIds: readonly string[] | undefined,
): ResponseMatrix {
  if (missingProbability > 0) {
    for (const row of rows) {
      for (let j = 0; j < row.length; j++) if (random.uniform() < missingProbability) row[j] = null;
    }
  }
  return buildResponseMatrix(rows, itemIds === undefined ? {} : { itemIds });
}

/**
 * Draws the response of a person at `theta` to an item: category k with probability P(k | θ), by inverting the
 * distribution function of the categories at one uniform draw. `probabilities` is scratch space with room for the
 * item's categories, so that simulations do not allocate per response.
 */
export function simulateResponse(item: IrtItem, theta: number, random: Random, probabilities: Float64Array): number {
  item.model.probabilities(item.parameters, theta, probabilities);
  return drawCategory(probabilities, item.model.categoryCount(item.parameters), random);
}

/**
 * Draws a category from its `probabilities`, of which the first `categoryCount` are used, by inverting the
 * distribution function at one uniform draw.
 */
export function drawCategory(probabilities: Float64Array, categoryCount: number, random: Random): number {
  const last = categoryCount - 1;
  let remaining = random.uniform();
  let category = 0;
  while (category < last && remaining >= (probabilities[category] ?? 0)) {
    remaining -= probabilities[category] ?? 0;
    category++;
  }
  return category;
}

/**
 * Checks the true thetas of a simulation.
 *
 * @throws `DataError` when the thetas are not an array, without thetas, or when a theta is not finite.
 */
export function checkThetas(thetas: readonly number[]): void {
  checkArray(thetas, "A simulation needs an array of thetas");
  if (thetas.length === 0) throw new DataError("A simulation needs at least one theta.");
  thetas.forEach((theta, person) => {
    checkFinite(theta, `Theta ${person + 1}`);
  });
}

import { checkArray, checkFinite, checkRandom, DataError, type Random } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { largestCategoryCount } from "../irt/internal.js";
import {
  checkMultidimensionalItems,
  type MultidimensionalItem,
  type MultidimensionalTestItem,
} from "../multidimensional/internal.js";
import {
  carriedItemIds,
  drawCategory,
  matrixWithMissing,
  missingProbabilityOf,
  type ResponseSimulationOptions,
} from "./responses.js";

/**
 * Simulates the responses of persons with known vectors of thetas to multidimensional items of any models.
 *
 * @remarks
 * Each response is drawn from its item's category probabilities at the person's thetas, by person, then item, by
 * inverting the distribution function of the categories at one uniform draw. Then, with a `missingProbability`, each
 * response is missing with that probability, so a seed gives the same responses with and without missingness, less
 * the missing ones. The matrix has a row per person and a column per item, named by the items' ids when they carry
 * them, as a test's items do, and `item1` and on otherwise.
 *
 * @param items - The multidimensional items, each with an `id` or none.
 * @param thetas - One array of thetas per person, as many as the items have dimensions.
 * @param random - The generator of the responses.
 * @param options - The probability that a response is missing.
 * @returns The responses, one row for each person.
 * @throws `DataError` without items or thetas, when an item is invalid, the items differ in their number of dimensions,
 * some items carry an id and others none, the ids are not distinct strings, a person's thetas are not that
 * many finite numbers, `random` is not a random number generator, or `missingProbability` is outside [0, 1].
 */
export function simulateMultidimensionalResponses(
  items: readonly (MultidimensionalItem | MultidimensionalTestItem)[],
  thetas: readonly (readonly number[])[],
  random: Random,
  options: ResponseSimulationOptions = {},
): ResponseMatrix {
  const missingProbability = missingProbabilityOf(options);
  const dimensions = checkMultidimensionalItems(items, "A simulation");
  const itemIds = carriedItemIds(items);
  checkThetas(thetas, dimensions);
  checkRandom(random);
  const theta = new Float64Array(dimensions);
  const probabilities = new Float64Array(largestCategoryCount(items));
  const rows: (number | null)[][] = thetas.map((person) => {
    theta.set(person);
    return items.map((item) => simulateResponse(item, theta, random, probabilities));
  });
  return matrixWithMissing(rows, missingProbability, random, itemIds);
}

/**
 * Draws the response of a person at `theta` to an item: category k with probability P(k | θ), by inverting the
 * distribution function of the categories at one uniform draw.
 */
function simulateResponse(
  item: MultidimensionalItem,
  theta: Float64Array,
  random: Random,
  probabilities: Float64Array,
): number {
  item.model.probabilities(item.parameters, theta, probabilities);
  return drawCategory(probabilities, item.model.categoryCount(item.parameters), random);
}

/**
 * Checks the true thetas of a simulation.
 *
 * @throws `DataError` when the thetas are not an array, without thetas, or when a person's thetas are not an array of
 * `dimensions` finite numbers.
 */
function checkThetas(thetas: readonly (readonly number[])[], dimensions: number): void {
  checkArray(thetas, "A simulation needs an array of thetas by person");
  if (thetas.length === 0) throw new DataError("A simulation needs at least one person's thetas.");
  thetas.forEach((person, index) => {
    checkArray(person, `The thetas of person ${index + 1} must be an array`);
    if (person.length !== dimensions) {
      throw new DataError(
        `Person ${index + 1} needs ${dimensions} thetas for items on ${dimensions} dimensions, found ${person.length}.`,
      );
    }
    person.forEach((value, d) => {
      checkFinite(value, `Theta ${d + 1} of person ${index + 1}`);
    });
  });
}

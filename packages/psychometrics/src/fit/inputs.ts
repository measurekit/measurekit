import { checkArray, checkFinite, checkOptions, DataError, type ThetaDistribution } from "@measurekit/core";
import { checkResponseMatrix, type ResponseMatrix } from "../data/internal.js";
import {
  type CheckedThetaDistribution,
  checkItem,
  checkThetaDistribution,
  type IrtItem,
  quadratureRule,
} from "../irt/internal.js";

/** Options of {@link rootMeanSquareDeviation} and {@link localDependenceChiSquare}. */
export interface ResidualFitOptions {
  /**
   * The distribution of theta of the persons, such as a calibration's group at its nodes.
   *
   * @defaultValue N(0, 1) on 61 rectangular points from -6 to 6
   */
  readonly population?: ThetaDistribution;
}

/**
 * Checks the options of a residual fit statistic and returns its population's nodes and weights.
 *
 * @throws `DataError` when the options or the population are invalid.
 */
export function residualPopulation(options: ResidualFitOptions): CheckedThetaDistribution {
  checkOptions(options);
  const { population } = options;
  return population === undefined ? quadratureRule() : checkThetaDistribution(population, "population");
}

/**
 * Checks a response matrix and its items: one valid item per column, and every response a category of its item.
 *
 * @throws `DataError` naming the first problem found.
 */
export function checkItems(matrix: ResponseMatrix, items: readonly IrtItem[]): void {
  checkResponseMatrix(matrix);
  checkArray(items, "The items must be an array");
  const { itemIds, personIds, responses } = matrix;
  if (items.length !== itemIds.length) {
    throw new DataError(`Found ${items.length} items for ${itemIds.length} columns of responses.`);
  }
  items.forEach((item, column) => {
    const id = itemIds[column] ?? "";
    checkItem(item, `Item "${id}"`);
    const count = item.model.categoryCount(item.parameters);
    responses.forEach((row, person) => {
      const response = row[column] ?? null;
      if (response !== null && response >= count) {
        throw new DataError(
          `The response of person "${personIds[person]}" to item "${id}" is ${response}, but the item has categories 0 to ${count - 1}.`,
        );
      }
    });
  });
}

/**
 * Checks that there is one finite theta per person.
 *
 * @throws `DataError` when the thetas are not an array, their count differs or a theta is not finite.
 */
export function checkPersonThetas(matrix: ResponseMatrix, thetas: readonly number[]): void {
  checkArray(thetas, "The thetas must be an array");
  if (thetas.length !== matrix.personIds.length) {
    throw new DataError(`Found ${thetas.length} thetas for ${matrix.personIds.length} persons.`);
  }
  thetas.forEach((theta, person) => {
    checkFinite(theta, `The theta of person "${matrix.personIds[person]}"`);
  });
}

/**
 * Returns whether a person's answered responses are all in their items' lowest categories or all in their highest:
 * an extreme score, whose maximum likelihood theta is infinite. A person without responses is extreme too.
 */
export function isExtreme(items: readonly IrtItem[], row: readonly (number | null)[]): boolean {
  let lowest = true;
  let highest = true;
  row.forEach((response, column) => {
    const item = items[column];
    if (response === null || item === undefined) return;
    lowest &&= response === 0;
    highest &&= response === item.model.categoryCount(item.parameters) - 1;
  });
  return lowest || highest;
}

/**
 * Writes an item's category probabilities at theta into `probabilities`, and the mean, variance and fourth central
 * moment of its score into `moments[0]`, `moments[1]` and `moments[2]`.
 */
export function scoreMoments(item: IrtItem, theta: number, probabilities: Float64Array, moments: Float64Array): void {
  const count = item.model.categoryCount(item.parameters);
  item.model.probabilities(item.parameters, theta, probabilities);
  let mean = 0;
  for (let k = 1; k < count; k++) mean += k * (probabilities[k] ?? 0);
  let variance = 0;
  let fourth = 0;
  for (let k = 0; k < count; k++) {
    const square = (k - mean) ** 2;
    variance += square * (probabilities[k] ?? 0);
    fourth += square * square * (probabilities[k] ?? 0);
  }
  moments[0] = mean;
  moments[1] = variance;
  moments[2] = fourth;
}

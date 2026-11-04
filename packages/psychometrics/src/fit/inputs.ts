import {
  checkArray,
  checkOptions,
  DataError,
  describe,
  type FieldNames,
  type ThetaDistribution,
} from "@measurekit/core";
import {
  type CheckedThetaDistribution,
  checkHeldParameters,
  checkThetaDistribution,
  type FixedParameters,
  type IrtItem,
  quadratureRule,
  readFixedParameters,
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

const RESIDUAL_FIT_OPTIONS: FieldNames<ResidualFitOptions> = { population: true };

/**
 * Checks the options of a residual fit statistic and returns its population's nodes and weights.
 *
 * @throws `DataError` when the options or the population are invalid.
 */
export function residualPopulation(options: ResidualFitOptions): CheckedThetaDistribution {
  checkOptions(options, RESIDUAL_FIT_OPTIONS);
  const { population } = options;
  return population === undefined ? quadratureRule() : checkThetaDistribution(population, "population");
}

/**
 * Returns, for each item, whether its calibration estimated each entry of its parameter vector: none of an anchor's,
 * and not those `fixedParameters` holds.
 *
 * @param fixed - The ids of the calibration's anchors.
 * @param fixedParameters - The parameters the calibration held at their values; only their names are read.
 * @throws `DataError` when `fixed` is not an array of item ids, or `fixedParameters` is invalid for the items.
 */
export function estimatedEntries(
  items: readonly IrtItem[],
  itemIds: readonly string[],
  fixed: readonly string[],
  fixedParameters: FixedParameters,
): readonly (readonly boolean[])[] {
  checkArray(fixed, "The option fixed must be an array of item ids");
  const ids = new Set(itemIds);
  for (const id of fixed) {
    if (!(typeof id === "string" && ids.has(id))) {
      throw new DataError(`The option fixed names ${describe(id)}, which is not an item id.`);
    }
  }
  const anchors = new Set(fixed);
  const held = readFixedParameters(fixedParameters, itemIds, anchors);
  return items.map(({ model, parameters }, column) => {
    const id = itemIds[column] ?? "";
    const specifications = model.parameters(model.categoryCount(parameters));
    if (anchors.has(id)) return specifications.map(() => false);
    const names = held.get(id);
    if (names !== undefined) checkHeldParameters(id, model.name, specifications, names);
    return specifications.map(({ name }) => names?.has(name) !== true);
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

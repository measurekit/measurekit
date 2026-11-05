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
 * Returns, for each item, whether its calibration estimated each entry of its parameter vector: those its `estimated`
 * names, or every entry of an item without `estimated`.
 *
 * @throws `DataError` when an item's `estimated` is not an array of names of its model's parameters.
 */
export function estimatedEntries(
  items: readonly (IrtItem & { readonly estimated?: readonly string[] })[],
  itemIds: readonly string[],
): readonly (readonly boolean[])[] {
  return items.map(({ model, parameters, estimated }, column) => {
    const specifications = model.parameters(model.categoryCount(parameters));
    if (estimated === undefined) return specifications.map(() => true);
    const label = `The estimated parameters of item "${itemIds[column]}"`;
    checkArray(estimated, `${label} must be an array of the names of its parameters`);
    const known = new Set(specifications.map(({ name }) => name));
    for (const name of estimated) {
      if (!known.has(name)) {
        throw new DataError(
          `${label} name ${describe(name)}, which the ${model.name} model does not have; it has ${[...known].join(", ")}.`,
        );
      }
    }
    const named = new Set(estimated);
    return specifications.map(({ name }) => named.has(name));
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

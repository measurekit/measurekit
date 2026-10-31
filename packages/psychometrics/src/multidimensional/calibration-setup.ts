import { checkRecord, DataError } from "@measurekit/core";
import { checkPrior, type ParameterPrior } from "../irt/prior.js";
import { type CompressedResponses, categoryCounts, checkCategories } from "../irt/responses.js";
import type { MultidimensionalCalibrationOptions } from "./calibration-options.js";
import type { Design } from "./design.js";
import type { ExpectationItem } from "./expectation.js";
import { checkMultidimensionalModel } from "./items.js";
import type { MultidimensionalModel } from "./model.js";
import { startingSlopes } from "./starting-slopes.js";

/** The bounds and prior of one estimated entry of an item's parameter vector. */
export interface FreeEntry {
  readonly lower: number;
  readonly upper: number;
  readonly prior: ParameterPrior | undefined;
}

/** An estimated entry without bounds or prior. */
export const UNBOUNDED: FreeEntry = {
  lower: Number.NEGATIVE_INFINITY,
  upper: Number.POSITIVE_INFINITY,
  prior: undefined,
};

/** An item during calibration. */
export interface ItemState<TParameters> extends ExpectationItem {
  readonly id: string;
  readonly model: MultidimensionalModel<TParameters>;
  parameters: TParameters;
  /** The number of dimensions. */
  readonly dimensions: number;
  /** The names of the entries of the parameter vector. */
  readonly names: readonly string[];
  /** The parameter vector, with the entries that are not estimated at their fixed values. */
  readonly vector: Float64Array;
  /** The index in the vector of each estimated entry. */
  readonly free: Int32Array;
  /** The bounds and prior of each estimated entry. */
  readonly entries: readonly FreeEntry[];
}

/**
 * Returns the name that priors give to a parameter: the entry's name without its index, `"a"` for the slope `"a2"`.
 */
function priorName(entry: string): string {
  return entry.replace(/\d+$/, "");
}

/**
 * Checks the models and priors against the data and builds the items with their starting values.
 *
 * @throws `DataError` when the number of models is not the number of items, a model is invalid, a prior is not a prior
 * or names a parameter that no estimated item has, or an item cannot be estimated from its responses.
 */
export function setUp<TParameters>(
  responses: CompressedResponses,
  options: MultidimensionalCalibrationOptions<TParameters>,
  design: Design,
  nodeCount: number,
): ItemState<TParameters>[] {
  const { itemIds } = responses;
  const dimensions = design.dimensionNames.length;
  const counts = categoryCounts(responses);
  const models = itemModels(options.model, itemIds.length);
  const { priors = {} } = options;
  checkRecord(priors, "The option priors must be an object of priors by parameter name");
  for (const [name, prior] of Object.entries(priors)) checkPrior(prior, `The prior of "${name}"`);
  const slopes = startingSlopes(design, responses);
  const items = models.map((model, column): ItemState<TParameters> => {
    const id = itemIds[column] ?? "";
    const itemCounts = counts[column] ?? [];
    const categoryCount = checkCategories(id, model, itemCounts);
    const answered = itemCounts.reduce((sum, count) => sum + count, 0);
    const proportions = Float64Array.from(itemCounts, (count) => count / answered);
    const vector = model.toVector(model.startingValues(proportions, slopes[column] ?? new Float64Array(dimensions)));
    const specifications = model.parameters(categoryCount, dimensions);
    const loads = design.loads[column] ?? [];
    const free = Int32Array.from(specifications.keys()).filter((j) => j >= dimensions || loads[j] === true);
    return {
      id,
      model,
      categoryCount,
      dimensions,
      parameters: model.fromVector(vector, dimensions),
      counts: new Float64Array(nodeCount * categoryCount),
      names: specifications.map(({ name }) => name),
      vector,
      free,
      entries: Array.from(free, (j) => {
        const {
          name,
          lower = Number.NEGATIVE_INFINITY,
          upper = Number.POSITIVE_INFINITY,
        } = specifications[j] ?? {
          name: "",
        };
        return { lower, upper, prior: priors[priorName(name)] };
      }),
    };
  });
  const estimated = new Set(items.flatMap((item) => Array.from(item.free, (j) => priorName(item.names[j] ?? ""))));
  for (const name of Object.keys(priors)) {
    if (!estimated.has(name)) throw new DataError(`A prior is given for "${name}", which no estimated item has.`);
  }
  return items;
}

function itemModels<TParameters>(
  model: MultidimensionalModel<TParameters> | readonly MultidimensionalModel<TParameters>[],
  itemCount: number,
): readonly MultidimensionalModel<TParameters>[] {
  const models = isList(model) ? model : Array.from({ length: itemCount }, () => model);
  if (models.length !== itemCount) throw new DataError(`Found ${models.length} models for ${itemCount} items.`);
  for (const each of models) {
    checkMultidimensionalModel(
      each,
      "The option model must be a multidimensional model such as multidimensionalTwoParameterLogistic",
    );
  }
  return models;
}

// Array.isArray does not narrow a union with a readonly array type.
function isList<T>(value: T | readonly T[]): value is readonly T[] {
  return Array.isArray(value);
}

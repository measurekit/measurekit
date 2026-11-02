import { checkRecord, DataError } from "@measurekit/core";
import {
  type CompressedResponses,
  categoryTallies,
  checkCategories,
  checkFixedCategories,
  checkFixedIds,
  checkPrior,
  modelsOf,
  NO_RESPONSES,
  type ParameterPrior,
} from "../irt/internal.js";
import type { MultidimensionalCalibrationOptions } from "./calibration-options.js";
import type { Design } from "./design.js";
import type { ExpectationItem } from "./expectation.js";
import { checkMultidimensionalItem, checkMultidimensionalModel } from "./items.js";
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
  /** Whether the item is an anchor: its parameters are given, and no entry is estimated. */
  readonly fixed: boolean;
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
 * or names a parameter that no estimated item has, an item cannot be estimated from its responses, or a fixed item is
 * not an item of the data, has invalid parameters, other dimensions than the calibration or fewer categories than its
 * responses.
 */
export function setUp<TParameters>(
  responses: CompressedResponses,
  options: MultidimensionalCalibrationOptions<TParameters>,
  design: Design,
  nodeCount: number,
): ItemState<TParameters>[] {
  const { itemIds } = responses;
  const dimensions = design.dimensionNames.length;
  const tallies = categoryTallies(responses);
  const models = modelsOf(options.model, itemIds.length, (each) =>
    checkMultidimensionalModel(
      each,
      "The option model must be a multidimensional model such as multidimensionalTwoParameterLogistic",
    ),
  );
  const { priors = {}, fixed = {} } = options;
  checkRecord(priors, "The option priors must be an object of priors by parameter name");
  // A map, so that an id such as "constructor" finds only its own parameters.
  checkRecord(fixed, "The option fixed must be an object of item parameters by item id");
  const anchors = new Map(Object.entries(fixed));
  checkFixedIds(anchors.keys(), itemIds);
  for (const [name, prior] of Object.entries(priors)) checkPrior(prior, `The prior of "${name}"`);
  const slopes = startingSlopes(design, responses);
  const items = models.map((model, column): ItemState<TParameters> => {
    const id = itemIds[column] ?? "";
    const tally = tallies[column] ?? NO_RESPONSES;
    const anchor = anchors.get(id);
    if (anchor !== undefined) return fixedItem(id, model, anchor, tally.highest + 1, dimensions, nodeCount);
    const proportions = checkCategories(id, model, tally);
    const categoryCount = proportions.length;
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
      fixed: false,
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

/**
 * Returns an anchor: an item whose parameters are given and not estimated.
 *
 * @throws `DataError` when the parameters are invalid for the model, span other dimensions than the calibration or have
 * fewer categories than the item's responses.
 */
function fixedItem<TParameters>(
  id: string,
  model: MultidimensionalModel<TParameters>,
  parameters: TParameters,
  responseCategories: number,
  dimensions: number,
  nodeCount: number,
): ItemState<TParameters> {
  checkMultidimensionalItem({ model, parameters }, `Fixed item "${id}"`);
  const categoryCount = model.categoryCount(parameters);
  if (model.dimensionCount(parameters) !== dimensions) {
    throw new DataError(
      `Fixed item "${id}" has ${model.dimensionCount(parameters)} dimensions, but the calibration has ${dimensions}.`,
    );
  }
  checkFixedCategories(id, responseCategories, categoryCount);
  return {
    id,
    model,
    categoryCount,
    dimensions,
    parameters,
    counts: new Float64Array(nodeCount * categoryCount),
    names: model.parameters(categoryCount, dimensions).map(({ name }) => name),
    vector: model.toVector(parameters),
    fixed: true,
    free: new Int32Array(0),
    entries: [],
  };
}

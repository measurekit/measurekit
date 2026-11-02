import { checkRecord, DataError } from "@measurekit/core";
import type { Quadrature } from "../math/quadrature.js";
import type { CalibrationOptions } from "./calibration-options.js";
import type { ExpectationItem } from "./expectation.js";
import { checkFixedCategories, checkFixedIds, checkItem, checkModel, modelsOf } from "./items.js";
import { type Population, population } from "./latent.js";
import type { IrtModel, ModelParameter } from "./model.js";
import { checkPrior, type ParameterPrior } from "./prior.js";
import { type CompressedResponses, categoryTallies, checkCategories } from "./responses.js";
import { type TestletDesign, testletDesign, testletWeights } from "./testlets.js";

/** An item during calibration. */
export interface ItemState<TParameters> extends ExpectationItem {
  readonly id: string;
  readonly model: IrtModel<TParameters>;
  parameters: TParameters;
  /** Whether the item's parameters are estimated rather than fixed. */
  readonly estimated: boolean;
  /** The names of the vector entries of an estimated item. */
  readonly names: readonly string[];
  /** The parameter vector of an estimated item; empty for a fixed item. */
  readonly vector: Float64Array;
  /** The index of the free parameter behind each vector entry. */
  readonly slots: Int32Array;
}

/**
 * The items and the free parameters of a calibration: the items' parameters, then the means and standard deviations
 * of the normal distributions of theta.
 */
export interface Setup<TParameters> {
  readonly items: readonly ItemState<TParameters>[];
  /** The groups of persons with their distributions of theta. */
  readonly population: Population;
  /** The testlets, or `null` without them. */
  readonly design: TestletDesign | null;
  /** Bounds and priors of the free parameters. */
  readonly free: readonly FreeParameter[];
  /** The current value of each free parameter. */
  readonly values: Float64Array;
}

/** A parameter calibration estimates: per item, or once for all items of a model that share it. */
export interface FreeParameter {
  readonly lower: number;
  readonly upper: number;
  readonly prior: ParameterPrior | undefined;
}

/**
 * Checks the options against the data and builds the items, the groups and the free parameters with their starting
 * values.
 *
 * @param answered - Whether each person answered an item.
 */
export function setUp<TParameters>(
  responses: CompressedResponses,
  options: CalibrationOptions<TParameters>,
  answered: readonly boolean[],
  rule: Quadrature,
): Setup<TParameters> {
  const { itemIds } = responses;
  const tallies = categoryTallies(responses);
  const models = modelsOf(options.model, itemIds.length, (each) =>
    checkModel(each, "The option model must be an IRT model such as twoParameterLogistic"),
  );
  const { fixed = {}, priors = {}, itemPriors = {} } = options;
  checkRecord(fixed, "The option fixed must be an object of item parameters by item id");
  checkRecord(priors, "The option priors must be an object of priors by parameter name");
  for (const [name, prior] of Object.entries(priors)) checkPrior(prior, `The prior of "${name}"`);
  checkFixedIds(Object.keys(fixed), itemIds);
  const priorsOfItem = itemPriorMap(itemPriors, itemIds, fixed);
  const free: FreeParameter[] = [];
  const values: number[] = [];
  const addFree = (lower: number, start: number): number => {
    free.push({ lower, upper: Number.POSITIVE_INFINITY, prior: undefined });
    return values.push(start) - 1;
  };
  const rectangular = options.quadrature?.method !== "gauss-hermite";
  const design =
    options.testlets === undefined ? null : testletDesign(options.testlets, itemIds, rule, rectangular, addFree);
  const sharing = new Map<IrtModel<TParameters>, SharedParameters>();
  const items = models.map((model, column): ItemState<TParameters> => {
    const id = itemIds[column] ?? "";
    const tally = tallies[column] ?? { highest: -1, counts: new Map() };
    const parameters = fixed[id];
    const nodes = design !== null && (design.testletOf[column] ?? -1) >= 0 ? design.sums : rule.nodes;
    if (parameters !== undefined) return fixedItem(id, model, parameters, tally.highest + 1, nodes);
    const counts = checkCategories(id, model, tally);
    const categoryCount = counts.length;
    const answered = counts.reduce((sum, count) => sum + count, 0);
    const start = model.toVector(model.startingValues(Float64Array.from(counts, (count) => count / answered)));
    const specifications = model.parameters(categoryCount);
    const shared = sharedSlots(sharing, model, { id, categoryCount, specifications });
    const own = priorsOfItem.get(id);
    const slots = Int32Array.from(
      specifications,
      ({ name, lower = Number.NEGATIVE_INFINITY, upper = Number.POSITIVE_INFINITY, shared: common }, p) => {
        if (common === true && own?.has(name) === true) {
          throw new DataError(
            `A prior is given for "${name}" of item "${id}", whose model shares it among items; use the option priors.`,
          );
        }
        const existing = common === true ? shared.get(name) : undefined;
        if (existing !== undefined) return existing;
        free.push({ lower, upper, prior: own?.get(name) ?? priors[name] });
        values.push(start[p] ?? 0);
        if (common === true) shared.set(name, values.length - 1);
        return values.length - 1;
      },
    );
    const vector = Float64Array.from(slots, (slot) => values[slot] ?? 0);
    return {
      id,
      model,
      categoryCount,
      parameters: model.fromVector(vector),
      nodes,
      counts: new Float64Array(nodes.length * categoryCount),
      estimated: true,
      names: specifications.map(({ name }) => name),
      vector,
      slots,
    };
  });
  const estimatedNames = new Set(items.flatMap((item) => item.names));
  for (const name of Object.keys(priors)) {
    if (!estimatedNames.has(name)) throw new DataError(`A prior is given for "${name}", which no estimated item has.`);
  }
  checkItemPriorNames(priorsOfItem, items);
  const groups = population(options, answered, rule, rectangular, Object.keys(fixed).length > 0, addFree);
  if (design !== null) {
    checkTestletOptions(options);
    testletWeights(design, Float64Array.from(values), rule.nodes);
  }
  return { items, population: groups, design, free, values: Float64Array.from(values) };
}

/**
 * Checks the priors by item id and parameter name against the items and returns them as maps, so that an id such as
 * "constructor" finds only its own priors.
 *
 * @throws `DataError` when the priors are not objects of priors, an id is not an item id, or the item is fixed.
 */
function itemPriorMap(
  itemPriors: NonNullable<CalibrationOptions<unknown>["itemPriors"]>,
  itemIds: readonly string[],
  fixed: Readonly<Record<string, unknown>>,
): Map<string, Map<string, ParameterPrior>> {
  checkRecord(itemPriors, "The option itemPriors must be an object of priors by item id and parameter name");
  return new Map(
    Object.entries(itemPriors).map(([id, byName]) => {
      if (!itemIds.includes(id)) throw new DataError(`Priors are given for "${id}", which is not an item id.`);
      if (Object.keys(fixed).includes(id)) throw new DataError(`Priors are given for item "${id}", which is fixed.`);
      checkRecord(byName, `The priors of item "${id}" must be an object of priors by parameter name`);
      for (const [name, prior] of Object.entries(byName)) checkPrior(prior, `The prior of "${name}" of item "${id}"`);
      return [id, new Map(Object.entries(byName))];
    }),
  );
}

/**
 * Checks that the items have the parameters the priors name.
 *
 * @throws `DataError` when an item's model has no parameter of that name.
 */
function checkItemPriorNames<TParameters>(
  priorsOfItem: ReadonlyMap<string, ReadonlyMap<string, ParameterPrior>>,
  items: readonly ItemState<TParameters>[],
): void {
  const byId = new Map(items.map((item) => [item.id, item]));
  for (const [id, byName] of priorsOfItem) {
    const names = byId.get(id)?.names ?? [];
    for (const name of byName.keys()) {
      if (!names.includes(name)) {
        throw new DataError(`A prior is given for "${name}" of item "${id}", whose model has no such parameter.`);
      }
    }
  }
}

/** The parameters the estimated items of a model share, as its first estimated item has them. */
interface SharedParameters {
  readonly id: string;
  readonly categoryCount: number;
  readonly names: string;
  /** The free parameter of each shared name. */
  readonly slots: Map<string, number>;
}

/**
 * Returns the free parameters of the names an item shares with the other estimated items of its model.
 *
 * @throws `DataError` when the item's shared parameters differ from those of the model's first estimated item, as
 * the RSM's steps do for items with different numbers of categories.
 */
function sharedSlots<TParameters>(
  sharing: Map<IrtModel<TParameters>, SharedParameters>,
  model: IrtModel<TParameters>,
  item: { readonly id: string; readonly categoryCount: number; readonly specifications: readonly ModelParameter[] },
): Map<string, number> {
  const names = item.specifications
    .filter((parameter) => parameter.shared === true)
    .map((parameter) => parameter.name)
    .join();
  const first = sharing.get(model);
  if (first === undefined) {
    const slots = new Map<string, number>();
    sharing.set(model, { id: item.id, categoryCount: item.categoryCount, names, slots });
    return slots;
  }
  if (first.names !== names) {
    throw new DataError(
      `Items "${first.id}" and "${item.id}" of the ${model.name} model share parameters, so they need the same number of categories; found ${first.categoryCount} and ${item.categoryCount}.`,
    );
  }
  return first.slots;
}

/**
 * Checks that testlets come with standard errors the testlet model supports. The groups and their distributions of
 * theta need no check: the testlets' integrals do not depend on the group, and `population` checks the scale.
 *
 * @throws `DataError` when the standard errors are the cross-product.
 */
function checkTestletOptions<TParameters>(options: CalibrationOptions<TParameters>): void {
  if (options.standardErrors === "cross-product") {
    throw new DataError('Testlets need the standard errors "none" or "hessian".');
  }
}

function fixedItem<TParameters>(
  id: string,
  model: IrtModel<TParameters>,
  parameters: TParameters,
  responseCategories: number,
  nodes: Float64Array,
): ItemState<TParameters> {
  checkItem({ model, parameters }, `Fixed item "${id}"`);
  const categoryCount = model.categoryCount(parameters);
  checkFixedCategories(id, responseCategories, categoryCount);
  return {
    id,
    model,
    categoryCount,
    parameters,
    nodes,
    counts: new Float64Array(nodes.length * categoryCount),
    estimated: false,
    names: [],
    vector: new Float64Array(0),
    slots: new Int32Array(0),
  };
}

import { checkBoolean, checkPositiveInteger, checkRecord, DataError, describe } from "@measurekit/core";
import type { LoadingPattern } from "./model.js";

/** The dimensions of a calibration and which of them each item loads on. */
export interface Design {
  /** The names of the dimensions. */
  readonly dimensionNames: readonly string[];
  /** For each item, in item order, whether it loads on each dimension: a slope that is not estimated is 0. */
  readonly loads: readonly (readonly boolean[])[];
  /** Whether the dimensions are found by the calibration, without a pattern of loadings. */
  readonly exploratory: boolean;
  /** Whether the correlations between dimensions are estimated. */
  readonly correlated: boolean;
}

/** The options of a calibration that fix its dimensions. */
interface DesignOptions {
  readonly dimensions?: number;
  readonly loadings?: LoadingPattern;
  readonly correlated?: boolean;
}

/**
 * Returns the design of a calibration from its options and the item ids of its data.
 *
 * @remarks
 * An exploratory design has every item load on every dimension, except that item j loads only on the first j
 * dimensions for the first D - 1 items, which fixes the rotation; its dimensions are uncorrelated and named F1 to FD.
 * A confirmatory design names the dimensions of each item; the dimensions are named in order of first appearance among
 * the items in item order.
 *
 * @throws `DataError` when neither or both of `dimensions` and `loadings` are given, `dimensions` is not a positive
 * integer, there are fewer items than dimensions, `loadings` does not name an array of
 * dimensions for each item, or names an item that is not in the data.
 */
export function designOf(options: DesignOptions, itemIds: readonly string[]): Design {
  const { dimensions, loadings, correlated = true } = options;
  checkBoolean(correlated, "The option correlated");
  if (loadings !== undefined) {
    if (dimensions !== undefined) {
      throw new DataError("Calibration takes the option dimensions or the option loadings, not both.");
    }
    return confirmatory(loadings, itemIds, correlated);
  }
  if (dimensions === undefined) {
    throw new DataError(
      "Calibration needs the option dimensions, for an exploratory model, or loadings, for a confirmatory one.",
    );
  }
  return exploratory(dimensions, itemIds.length);
}

function exploratory(dimensions: number, itemCount: number): Design {
  checkPositiveInteger(dimensions, "The option dimensions");
  if (itemCount < dimensions) {
    throw new DataError(
      `An exploratory model of ${dimensions} dimensions needs at least ${dimensions} items, found ${itemCount}.`,
    );
  }
  return {
    dimensionNames: Array.from({ length: dimensions }, (_, d) => `F${d + 1}`),
    loads: Array.from({ length: itemCount }, (_, item) => Array.from({ length: dimensions }, (__, d) => d <= item)),
    exploratory: true,
    correlated: false,
  };
}

function confirmatory(loadings: LoadingPattern, itemIds: readonly string[], correlated: boolean): Design {
  checkRecord(loadings, "The option loadings must be an object of dimension names by item id");
  // A map, so that an id such as "constructor" finds only its own loadings.
  const byId = new Map(Object.entries(loadings));
  for (const id of byId.keys()) {
    if (!itemIds.includes(id)) throw new DataError(`Loadings are given for "${id}", which is not an item id.`);
  }
  const dimensionNames: string[] = [];
  const named = itemIds.map((id) => {
    const names = byId.get(id);
    if (!(Array.isArray(names) && names.length > 0 && names.every((name) => typeof name === "string" && name !== ""))) {
      throw new DataError(
        `The loadings of item "${id}" must be an array of at least one dimension name, found ${describe(names)}.`,
      );
    }
    if (new Set(names).size !== names.length) {
      throw new DataError(`Item "${id}" loads on a dimension twice: ${names.join(", ")}.`);
    }
    for (const name of names) if (!dimensionNames.includes(name)) dimensionNames.push(name);
    return names;
  });
  return {
    dimensionNames,
    loads: named.map((names) => dimensionNames.map((name) => names.includes(name))),
    exploratory: false,
    correlated: correlated && dimensionNames.length > 1,
  };
}

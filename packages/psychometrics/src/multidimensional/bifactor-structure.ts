import { checkRecordById, DataError, describe } from "@measurekit/core";
import type { Design } from "./design.js";
import { checkDimensions } from "./items.js";
import type { LoadingPattern } from "./model.js";

/** The most primary dimensions of a bifactor model: one is a bifactor model and two a two-tier model. */
const MAX_PRIMARY_DIMENSIONS = 2;

/** The name of the primary dimension when the items name none. */
const GENERAL = "general";

/** The items that share a specific dimension, or the items that have none. */
export interface ItemGroup {
  /** The index of the specific dimension among the dimensions of the calibration, or -1 for the items that have none. */
  readonly dimension: number;
  /** The indices of the items in the item order of the data, ascending. */
  readonly items: readonly number[];
}

/** The dimensions of a bifactor calibration and how its items divide among the specific dimensions. */
export interface BifactorStructure {
  /** The primary dimensions first, then the specific ones, each in order of first appearance among the items. */
  readonly design: Design;
  /** The number of primary dimensions, 1 or 2. */
  readonly primaryCount: number;
  /** The items of each specific dimension, in the order of the dimensions, then the items that have none, if any. */
  readonly groups: readonly ItemGroup[];
}

/** The options of a bifactor calibration that fix its dimensions. */
interface StructureOptions {
  readonly specific: Readonly<Record<string, string | null>>;
  readonly primary?: LoadingPattern;
}

/**
 * Returns the dimensions of a bifactor model and the groups of items that share a specific dimension.
 *
 * @throws `DataError` when `specific` or `primary` is not an object that names every item of the data and no other,
 * an item has more than one specific dimension or a primary dimension twice, a specific dimension has fewer than two
 * items, there are more than two primary dimensions, a name belongs to a primary and a specific dimension, or there are
 * more than 100 dimensions, primary and specific.
 */
export function structureOf(options: StructureOptions, itemIds: readonly string[]): BifactorStructure {
  const { specific, primary } = options;
  checkRecordById(specific, "The option specific must be an object of specific dimension names by item id");
  const specificByItem = specificDimensions(specific, itemIds);
  const primaryByItem = primary === undefined ? itemIds.map(() => [GENERAL]) : primaryDimensions(primary, itemIds);
  const primaryNames = namesInOrder(primaryByItem.flat());
  const specificNames = namesInOrder(specificByItem.filter((name) => name !== null));
  if (primaryNames.length > MAX_PRIMARY_DIMENSIONS) {
    throw new DataError(
      `A bifactor model has one primary dimension and a two-tier model two, found ${primaryNames.length}: ${primaryNames.join(", ")}. Use calibrateMultidimensional with the option loadings for more.`,
    );
  }
  checkDimensions(primaryNames.length + specificNames.length, "A bifactor or two-tier calibration");
  for (const name of specificNames) {
    if (primaryNames.includes(name)) throw new DataError(`"${name}" names a primary and a specific dimension.`);
    if (specificByItem.filter((each) => each === name).length < 2) {
      throw new DataError(`The specific dimension "${name}" has one item; a specific dimension needs at least two.`);
    }
  }
  const dimensionNames = [...primaryNames, ...specificNames];
  return {
    design: {
      dimensionNames,
      loads: itemIds.map((_, item) => {
        const loaded = [...(primaryByItem[item] ?? []), specificByItem[item] ?? null];
        return dimensionNames.map((name) => loaded.includes(name));
      }),
      exploratory: false,
      correlated: primaryNames.length === MAX_PRIMARY_DIMENSIONS,
    },
    primaryCount: primaryNames.length,
    groups: groupsOf(specificByItem, specificNames, primaryNames.length),
  };
}

/** Returns the names in order of first appearance, each once. */
function namesInOrder(names: readonly string[]): string[] {
  return [...new Set(names)];
}

/** Returns the items of each specific dimension, in the order of `specificNames`, then the items without one. */
function groupsOf(
  specificByItem: readonly (string | null)[],
  specificNames: readonly string[],
  primaryCount: number,
): ItemGroup[] {
  const itemsOf = (name: string | null): number[] =>
    specificByItem.flatMap((each, item) => (each === name ? [item] : []));
  const groups = specificNames.map((name, s) => ({ dimension: primaryCount + s, items: itemsOf(name) }));
  const rest = itemsOf(null);
  return rest.length > 0 ? [...groups, { dimension: -1, items: rest }] : groups;
}

/** Returns each item's specific dimension, `null` for an item with none. */
function specificDimensions(
  specific: Readonly<Record<string, unknown>>,
  itemIds: readonly string[],
): (string | null)[] {
  // A map, so that an id such as "constructor" finds only its own entry.
  const byId = new Map(Object.entries(specific));
  for (const id of byId.keys()) {
    if (!itemIds.includes(id))
      throw new DataError(`A specific dimension is given for "${id}", which is not an item id.`);
  }
  return itemIds.map((id) => {
    const name = byId.get(id);
    if (name !== null && !(typeof name === "string" && name !== "")) {
      throw new DataError(
        `The specific dimension of item "${id}" must be one dimension name or null, found ${describe(name)}; an item loads on at most one specific dimension.`,
      );
    }
    return name;
  });
}

/** Returns each item's primary dimensions. */
function primaryDimensions(primary: LoadingPattern, itemIds: readonly string[]): string[][] {
  checkRecordById(primary, "The option primary must be an object of dimension names by item id");
  const byId = new Map(Object.entries(primary));
  for (const id of byId.keys()) {
    if (!itemIds.includes(id))
      throw new DataError(`A primary dimension is given for "${id}", which is not an item id.`);
  }
  return itemIds.map((id) => {
    const names = byId.get(id);
    if (!(Array.isArray(names) && names.length > 0 && names.every((name) => typeof name === "string" && name !== ""))) {
      throw new DataError(
        `The primary dimensions of item "${id}" must be an array of at least one dimension name, found ${describe(names)}.`,
      );
    }
    if (new Set(names).size !== names.length) {
      throw new DataError(`Item "${id}" loads on a primary dimension twice: ${names.join(", ")}.`);
    }
    return [...names];
  });
}

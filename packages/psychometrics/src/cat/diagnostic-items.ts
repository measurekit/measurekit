import { checkArray, checkRecord, DataError, describe } from "@measurekit/core";
import {
  bitPattern,
  type CdmItem,
  checkDesignSize,
  checkItemAttributes,
  checkSuccessProbabilities,
  profileIndices,
} from "../cdm/internal.js";
import type { PoolControls } from "./items.js";

/**
 * An item of a diagnostic test's pool: a calibrated cognitive diagnosis item, such as an item of a `CdmCalibration`,
 * with the controls of every pool item.
 */
export interface DiagnosticPoolItem extends Pick<CdmItem, "id" | "attributes" | "probabilities">, PoolControls {}

/** A pool item bound to the profiles of its test, as the selection and stopping rules of the test see it. */
export interface DiagnosticTestItem extends DiagnosticPoolItem {
  /**
   * The index of the item's latent group in each profile of the test, in the order of the profiles: the index into
   * `probabilities` of the profile reduced to the item's attributes.
   */
  readonly groups: Int32Array;
}

/** A pool checked against the attributes of its test: each item with the positions of its attributes among them. */
export interface CheckedPool {
  readonly items: readonly { readonly item: DiagnosticPoolItem; readonly positions: readonly number[] }[];
  /** The index of each item in the pool, by id. */
  readonly indices: Map<string, number>;
}

/**
 * Checks the pool of a diagnostic test against its attributes, before the test holds anything of 2^K entries. Items
 * keep their order and their ids are unique.
 *
 * @throws `DataError` when the pool is not an array of objects, an id is not a string or repeats, an item has no
 * attributes, repeats one or names one that is not an attribute of the test, an item does not have one probability
 * from 0 to 1 per latent group, or the pool has more than 2^26 cells of items by profiles or more than 2^20 latent
 * groups over all items.
 */
export function checkPool(pool: readonly DiagnosticPoolItem[], attributes: readonly string[]): CheckedPool {
  checkArray(pool, "The option pool must be an array of items");
  const positions = new Map(attributes.map((attribute, k) => [attribute, k]));
  const indices = new Map<string, number>();
  const items = pool.map((item, index) => {
    checkRecord(item, `Item ${index + 1} must be an object { id, attributes, probabilities }`);
    if (typeof item.id !== "string")
      throw new DataError(`Item ${index + 1} needs a string id, found ${describe(item.id)}.`);
    if (indices.has(item.id)) throw new DataError(`Item ids must be unique; "${item.id}" appears twice.`);
    checkItemAttributes(item.id, item.attributes);
    const itemPositions = item.attributes.map((attribute) => {
      const position = positions.get(attribute);
      if (position === undefined) {
        throw new DataError(`Item "${item.id}" has attribute "${attribute}", which is not an attribute of the test.`);
      }
      return position;
    });
    checkSuccessProbabilities(item.id, item.probabilities, 2 ** itemPositions.length);
    indices.set(item.id, index);
    return { item, positions: itemPositions };
  });
  // The test holds an index of the 2^K profiles for each item, which a pool of short items names in a few bytes each.
  checkDesignSize(
    attributes.length,
    items.map((checked) => checked.positions.length),
  );
  return { items, indices };
}

/** Binds the items of a checked pool to the profiles of its test. */
export function bindPool(
  { items, indices }: CheckedPool,
  profiles: readonly (readonly number[])[],
): { readonly items: DiagnosticTestItem[]; readonly indices: Map<string, number> } {
  return {
    items: items.map(({ item, positions }) => {
      const latentGroups = profileIndices(positions.length);
      return {
        ...item,
        groups: Int32Array.from(profiles, (profile) => latentGroups[bitPattern(profile, positions)] ?? 0),
      };
    }),
    indices,
  };
}

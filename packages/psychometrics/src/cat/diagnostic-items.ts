import { checkArray, checkRecord, DataError, describe } from "@measurekit/core";
import type { CdmItem } from "../cdm/calibration.js";
import { checkSuccessProbabilities } from "../cdm/calibration.js";
import { bitPattern, checkItemAttributes, profileIndices } from "../cdm/q-matrix.js";
import type { PoolControls } from "./items.js";
import type { ProfileSpace } from "./profile-space.js";

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

/**
 * Checks the pool of a diagnostic test and binds its items to the profiles of the test. Items keep their order and
 * their ids are unique.
 *
 * @throws `DataError` when the pool is not an array of objects, an id is not a string or repeats, an item has no
 * attributes, repeats one or names one that is not an attribute of the test, or an item does not have one probability
 * from 0 to 1 per latent group.
 */
export function bindPool(
  pool: readonly DiagnosticPoolItem[],
  { attributes, profiles }: ProfileSpace,
): { readonly items: DiagnosticTestItem[]; readonly indices: Map<string, number> } {
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
    const latentGroups = profileIndices(itemPositions.length);
    return {
      ...item,
      groups: Int32Array.from(profiles, (profile) => latentGroups[bitPattern(profile, itemPositions)] ?? 0),
    };
  });
  return { items, indices };
}

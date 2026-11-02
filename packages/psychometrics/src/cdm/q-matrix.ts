import { checkRecord, DataError, describe, isNonEmptyString, type QMatrix } from "@measurekit/core";

/** A cognitive diagnosis design: the attributes, the profiles over them and each item's attributes. */
export interface CdmDesign {
  /** The attributes, in order of first appearance in the Q-matrix over the items. */
  readonly attributes: readonly string[];
  /** Each profile: 1 for each mastered attribute, 0 otherwise. */
  readonly profiles: readonly (readonly number[])[];
  /** Each item's attribute indices, in the order of `attributes`. */
  readonly itemAttributes: readonly (readonly number[])[];
  /** For each item, the latent group of each profile: its reduced profile over the item's attributes. */
  readonly groupOf: readonly Int32Array[];
}

/**
 * Returns the profiles over `count` attributes, 2^count of them: by the number of attributes mastered, then those
 * mastering earlier attributes first, as GDINA's `attributepattern`: 000, 100, 010, 001, 110, 101, 011, 111.
 */
export function profilesOf(count: number): number[][] {
  const profiles: number[][] = [];
  const combine = (size: number, from: number, chosen: number[]): void => {
    if (chosen.length === size) {
      profiles.push(Array.from({ length: count }, (_, k) => (chosen.includes(k) ? 1 : 0)));
      return;
    }
    for (let k = from; k < count; k++) combine(size, k + 1, [...chosen, k]);
  };
  for (let size = 0; size <= count; size++) combine(size, 0, []);
  return profiles;
}

/**
 * Returns, for each bit pattern over `count` attributes, attribute k mastered when bit k is set, the index of its
 * profile among {@link profilesOf}(count).
 */
export function profileIndices(count: number): Int32Array {
  const indices = new Int32Array(2 ** count);
  profilesOf(count).forEach((profile, index) => {
    indices[bitPattern(profile, profile.keys())] = index;
  });
  return indices;
}

/** Returns the bit pattern of a profile's attributes at `positions`: bit j is set when the j-th is mastered. */
export function bitPattern(profile: readonly number[], positions: Iterable<number>): number {
  let pattern = 0;
  let bit = 0;
  for (const k of positions) pattern |= (profile[k] ?? 0) << bit++;
  return pattern;
}

/** The most attributes of a design: 2^20 profiles. */
export const MAX_ATTRIBUTES = 20;
/**
 * The most cells of the items' tables from the profiles to their latent groups, items × 2^K: 256 MB of 32-bit integers,
 * 64 items at 20 attributes.
 */
const MAX_PROFILE_CELLS = 2 ** 26;
/** The most latent groups over all items, 2^k for an item of k attributes: the success probabilities of the G-DINA. */
const MAX_LATENT_GROUPS = 2 ** 20;

/**
 * Checks an item's attributes: at least one, each a non-empty string, none twice.
 *
 * @throws `DataError` when they are not.
 */
export function checkItemAttributes(
  id: string,
  attributes: readonly string[] | undefined,
): asserts attributes is readonly string[] {
  if (!(Array.isArray(attributes) && attributes.length > 0)) {
    throw new DataError(`Item "${id}" needs at least one attribute.`);
  }
  for (const attribute of attributes) {
    if (!isNonEmptyString(attribute)) {
      throw new DataError(`Item "${id}" has an attribute that is not a non-empty string: ${describe(attribute)}.`);
    }
  }
  if (new Set(attributes).size !== attributes.length) throw new DataError(`Item "${id}" lists an attribute twice.`);
}

/**
 * Checks a Q-matrix against item ids and returns its attributes, in order of first appearance, and each item's attribute
 * indices, sorted.
 *
 * @throws `DataError` when the Q-matrix is not an object, an item has no entry, an entry names an item that is not an
 * item id, or an item's attributes fail {@link checkItemAttributes}.
 */
export function designAttributes(
  itemIds: readonly string[],
  qMatrix: QMatrix,
): Pick<CdmDesign, "attributes" | "itemAttributes"> {
  // A Q-matrix shared with knowledge tracing or loaded from JSON may be another value.
  checkRecord(qMatrix, "A Q-matrix must be an object of attribute lists by item id");
  for (const id of Object.keys(qMatrix)) {
    if (!itemIds.includes(id)) throw new DataError(`The Q-matrix has item "${id}", which is not an item id.`);
  }
  const attributes: string[] = [];
  const positions = new Map<string, number>();
  const itemAttributes = itemIds.map((id) => {
    const listed = Object.hasOwn(qMatrix, id) ? qMatrix[id] : undefined;
    checkItemAttributes(id, listed);
    const indices = listed.map((attribute) => {
      const known = positions.get(attribute);
      if (known !== undefined) return known;
      positions.set(attribute, attributes.length);
      return attributes.push(attribute) - 1;
    });
    return indices.sort((a, b) => a - b);
  });
  return { attributes, itemAttributes };
}

/**
 * Checks a Q-matrix against item ids and returns its design.
 *
 * @throws `DataError` as {@link designAttributes} does, or when there are more than 20 attributes, more than 2^26 cells
 * of items by profiles or more than 2^20 latent groups over all items.
 */
export function cdmDesign(itemIds: readonly string[], qMatrix: QMatrix): CdmDesign {
  const { attributes, itemAttributes } = designAttributes(itemIds, qMatrix);
  if (attributes.length > MAX_ATTRIBUTES) {
    throw new DataError(`A design has at most ${MAX_ATTRIBUTES} attributes, found ${attributes.length}.`);
  }
  // A few hundred bytes of Q-matrix can name these, and a fit holds each of them: the limits keep its memory within
  // about 2 GB.
  const cells = itemAttributes.length * 2 ** attributes.length;
  if (cells > MAX_PROFILE_CELLS) {
    throw new DataError(
      `A design has at most ${MAX_PROFILE_CELLS} cells of items by profiles, items × 2^attributes, found ${cells} of ${itemAttributes.length} items and ${attributes.length} attributes.`,
    );
  }
  const groups = itemAttributes.reduce((sum, indices) => sum + 2 ** indices.length, 0);
  if (groups > MAX_LATENT_GROUPS) {
    throw new DataError(
      `A design has at most ${MAX_LATENT_GROUPS} latent groups over all items, 2^k for an item of k attributes, found ${groups}.`,
    );
  }
  const profiles = profilesOf(attributes.length);
  const groupOf = itemAttributes.map((indices) => {
    const groups = profileIndices(indices.length);
    return Int32Array.from(profiles, (profile) => groups[bitPattern(profile, indices)] ?? 0);
  });
  return { attributes, profiles, itemAttributes, groupOf };
}

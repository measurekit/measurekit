import { checkRecordById, DataError, type QMatrix } from "@measurekit/core";
import { designAttributes } from "./q-matrix.js";

/** The conditions under which a Q-matrix identifies the DINA and DINO models. */
// biome-ignore lint/style/useNamingConvention: Q-matrix is the name of this matrix in the field.
export interface QMatrixIdentifiability {
  /** Whether each attribute has an item that measures it alone, by attribute. */
  readonly singleItems: Readonly<Record<string, boolean>>;
  /** Whether each attribute is in at least three items, by attribute. */
  readonly threeItems: Readonly<Record<string, boolean>>;
  /** With one single-attribute item of each attribute set aside, whether the columns of the rest differ pairwise. */
  readonly distinctColumns: boolean;
  /** All three: Xu and Zhang's conditions for the DINA, as `CDM::din_identifiability`. */
  readonly dinaIdentified: boolean;
}

/** The items an attribute needs, by Xu and Zhang's condition. */
const MIN_ITEMS = 3;

/**
 * Returns whether a Q-matrix meets Xu and Zhang's conditions for the identifiability of the DINA model, which by
 * duality hold for the DINO, as `CDM::din_identifiability` checks them.
 *
 * @remarks
 * The conditions: every attribute has an item that measures it alone; every attribute is in at least three items; and
 * once the first such single-attribute item of each attribute is set aside, the columns of the remaining Q-matrix differ
 * pairwise. Where no item is set aside the whole Q-matrix is compared; an empty remainder has equal columns. Each column
 * is compared by a key of its items, so that the cost grows with the entries of the Q-matrix. The conditions cover the DINA and
 * DINO only; those of general models are under Not included in the guide.
 *
 * @param qMatrix - The attributes of each item, by item id.
 * @returns Each condition by attribute, and whether all three hold.
 * @throws `DataError` when the Q-matrix is not an object, has no item, or an item's attributes are not a non-empty list
 * of distinct non-empty strings.
 * @see {@link https://doi.org/10.1007/s11336-015-9471-z | Xu and Zhang (2016)}
 */
export function qMatrixIdentifiability(qMatrix: QMatrix): QMatrixIdentifiability {
  checkRecordById(qMatrix, "A Q-matrix must be an object of attribute lists by item id");
  const ids = Object.keys(qMatrix);
  const { attributes, itemAttributes } = designAttributes(ids, qMatrix);
  if (ids.length === 0) throw new DataError("A Q-matrix needs at least one item.");
  const counts = new Array<number>(attributes.length).fill(0);
  const firstSingle = new Array<number>(attributes.length).fill(-1);
  itemAttributes.forEach((indices, item) => {
    for (const k of indices) counts[k] = (counts[k] ?? 0) + 1;
    const [only] = indices;
    if (indices.length === 1 && only !== undefined && firstSingle[only] === -1) firstSingle[only] = item;
  });
  // A column's key lists the remaining items that measure its attribute; equal columns have equal keys.
  const setAside = new Set(firstSingle.filter((item) => item >= 0));
  const columns = attributes.map((): number[] => []);
  itemAttributes.forEach((indices, item) => {
    if (!setAside.has(item)) for (const k of indices) columns[k]?.push(item);
  });
  const distinctColumns = new Set(columns.map((items) => items.join())).size === attributes.length;
  const singleItems = Object.fromEntries(attributes.map((attribute, k) => [attribute, (firstSingle[k] ?? -1) >= 0]));
  const threeItems = Object.fromEntries(attributes.map((attribute, k) => [attribute, (counts[k] ?? 0) >= MIN_ITEMS]));
  return {
    singleItems,
    threeItems,
    distinctColumns,
    dinaIdentified:
      distinctColumns && Object.values(singleItems).every(Boolean) && Object.values(threeItems).every(Boolean),
  };
}

import { checkArray, DataError } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { type CompressedResponses, compressResponses } from "../irt/internal.js";
import { linkedGroups } from "./groups.js";

/** The most occasions a longitudinal calibration takes: its grid has a dimension for each. */
const MAX_OCCASIONS = 3;

/** The responses of several occasions on one set of persons and items. */
export interface Occasions {
  /** The ids of the persons who took part in some occasion, in order of first appearance. */
  readonly personIds: readonly string[];
  /** The ids of the items of any occasion, in order of first appearance. */
  readonly itemIds: readonly string[];
  /**
   * The responses of each occasion over all persons and items: a person who did not take part has no responses, and
   * an item that the occasion does not have is never answered.
   */
  readonly occasions: readonly CompressedResponses[];
  /** The responses of all occasions as those of one set of persons: a row for each person and occasion, by occasion. */
  readonly pooled: CompressedResponses;
}

/**
 * Checks the response matrices of the occasions and lines their persons and items up by id.
 *
 * @throws `DataError` when `occasions` is not an array of 2 or 3 response matrices, a matrix is malformed, no one
 * responded, or an occasion shares no item with the first, directly or through the others.
 */
export function arrangeOccasions(occasions: readonly ResponseMatrix[]): Occasions {
  checkArray(occasions, "The occasions must be an array of response matrices");
  if (occasions.length < 2 || occasions.length > MAX_OCCASIONS) {
    throw new DataError(
      `Longitudinal calibration takes 2 or ${MAX_OCCASIONS} occasions, found ${occasions.length}: a grid with a dimension for each coarser than that cannot hold the correlations between occasions.`,
    );
  }
  const compressed = occasions.map((occasion) => compressResponses(occasion));
  const personIds = [...new Set(compressed.flatMap((each) => each.personIds))];
  const itemIds = [...new Set(compressed.flatMap((each) => each.itemIds))];
  const aligned = compressed.map((each) => align(each, personIds, itemIds));
  if (!aligned.some((each) => each.codes.length > 0)) throw new DataError("Calibration needs at least one response.");
  checkLinked(aligned, itemIds);
  return { personIds, itemIds, occasions: aligned, pooled: pool(aligned, personIds, itemIds) };
}

/** Re-indexes an occasion's responses by the union of the persons and the items of all occasions. */
function align(
  responses: CompressedResponses,
  personIds: readonly string[],
  itemIds: readonly string[],
): CompressedResponses {
  const rows = new Map(personIds.map((id, person) => [id, person]));
  const columns = new Map(itemIds.map((id, item) => [id, item]));
  const byPerson = personIds.map((): number[] => []);
  responses.personIds.forEach((id, person) => {
    const target = byPerson[rows.get(id) ?? 0] ?? [];
    for (let r = responses.offsets[person] ?? 0; r < (responses.offsets[person + 1] ?? 0); r++) target.push(r);
  });
  const offsets = new Int32Array(personIds.length + 1);
  const items: number[] = [];
  const codes: number[] = [];
  byPerson.forEach((positions, person) => {
    for (const r of positions) {
      items.push(columns.get(responses.itemIds[responses.items[r] ?? 0] ?? "") ?? 0);
      codes.push(responses.codes[r] ?? 0);
    }
    offsets[person + 1] = items.length;
  });
  return { personIds, itemIds, offsets, items: Int32Array.from(items), codes: Int32Array.from(codes), times: null };
}

/**
 * Checks that every occasion shares an item with the first, directly or through the others: otherwise the means of the
 * occasions are not on one scale.
 */
function checkLinked(occasions: readonly CompressedResponses[], itemIds: readonly string[]): void {
  const answered = occasions.map((occasion) => {
    const row = new Array<boolean>(itemIds.length).fill(false);
    for (const item of occasion.items) row[item] = true;
    return row;
  });
  const apart = linkedGroups(answered).findIndex((component) => component !== 0);
  if (apart >= 0) {
    throw new DataError(
      `Occasion ${apart + 1} shares no item with occasion 1, directly or through the other occasions, so their means cannot be put on one scale; give them common items.`,
    );
  }
}

/** Stacks the occasions: the persons of the first, then those of the second, and so on. */
function pool(
  occasions: readonly CompressedResponses[],
  personIds: readonly string[],
  itemIds: readonly string[],
): CompressedResponses {
  const persons = personIds.length;
  const offsets = new Int32Array(occasions.length * persons + 1);
  let start = 0;
  occasions.forEach((occasion, t) => {
    for (let person = 1; person <= persons; person++)
      offsets[t * persons + person] = start + (occasion.offsets[person] ?? 0);
    start += occasion.codes.length;
  });
  return {
    personIds: occasions.flatMap(() => personIds),
    itemIds,
    offsets,
    items: Int32Array.from(occasions.flatMap((occasion) => Array.from(occasion.items))),
    codes: Int32Array.from(occasions.flatMap((occasion) => Array.from(occasion.codes))),
    times: null,
  };
}

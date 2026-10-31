import { DataError } from "@measurekit/core";
import { checkResponseMatrix, type ResponseMatrix } from "../data/response-matrix.js";
import { checkResponseRecords, type ResponseRecords } from "../data/response-records.js";
import type { IrtModel } from "./model.js";

/**
 * Responses stored by person, the answered ones only, in item order within each person: the form calibration works
 * on, which grows with the responses rather than with persons times items.
 */
export interface CompressedResponses {
  readonly personIds: readonly string[];
  readonly itemIds: readonly string[];
  /** The start of each person's responses in `items` and `codes`, and their end as the last entry. */
  readonly offsets: Int32Array;
  /** The item index of each response. */
  readonly items: Int32Array;
  /** Each response. */
  readonly codes: Int32Array;
  /** Each response's time in seconds, 0 without one; `null` when the data have no times. */
  readonly times: Float64Array | null;
}

/**
 * Checks a response matrix or response records and returns their answered responses by person.
 *
 * @throws `DataError` when the data are malformed, naming the first problem.
 */
export function compressResponses(data: ResponseMatrix | ResponseRecords): CompressedResponses {
  if (data !== null && typeof data === "object" && "records" in data) {
    const byPerson = checkResponseRecords(data);
    const timed = data.records.some((record) => record.time !== undefined);
    return compressed(data, byPerson.length, timed, (person, add) => {
      for (const { item, response, time } of byPerson[person] ?? []) add(item, response, time);
    });
  }
  checkResponseMatrix(data);
  const { times } = data;
  return compressed(data, data.responses.length, times !== undefined, (person, add) => {
    data.responses[person]?.forEach((response, item) => {
      if (response !== null) add(item, response, times?.[person]?.[item] ?? 0);
    });
  });
}

function compressed(
  ids: { readonly personIds: readonly string[]; readonly itemIds: readonly string[] },
  personCount: number,
  timed: boolean,
  forEachResponse: (person: number, add: (item: number, response: number, time: number) => void) => void,
): CompressedResponses {
  const offsets = new Int32Array(personCount + 1);
  const items: number[] = [];
  const codes: number[] = [];
  const times: number[] = [];
  for (let person = 0; person < personCount; person++) {
    forEachResponse(person, (item, response, time) => {
      items.push(item);
      codes.push(response);
      times.push(time);
    });
    offsets[person + 1] = items.length;
  }
  return {
    personIds: ids.personIds,
    itemIds: ids.itemIds,
    offsets,
    items: Int32Array.from(items),
    codes: Int32Array.from(codes),
    times: timed ? Float64Array.from(times) : null,
  };
}

/** Returns the responses by person, then item, of a checked matrix, with -1 for missing ones. */
export function responseCodes(matrix: ResponseMatrix): Int32Array {
  const itemCount = matrix.itemIds.length;
  const codes = new Int32Array(matrix.responses.length * itemCount);
  matrix.responses.forEach((row, person) => {
    row.forEach((response, item) => {
      codes[person * itemCount + item] = response ?? -1;
    });
  });
  return codes;
}

/** Returns the number of persons in each category of each item, up to the item's highest response. */
export function categoryCounts({ itemIds, items, codes }: CompressedResponses): number[][] {
  const counts = itemIds.map((): number[] => []);
  items.forEach((item, r) => {
    const response = codes[r] ?? 0;
    const itemCounts = counts[item] ?? [];
    while (itemCounts.length <= response) itemCounts.push(0);
    itemCounts[response] = (itemCounts[response] ?? 0) + 1;
  });
  return counts;
}

/**
 * Checks that an item's responses can be estimated with its model, of either kind, and returns its number of
 * categories.
 *
 * @throws `DataError` when the item has a response above the model's categories, responses in fewer than two
 * categories, or no response in a category below its highest.
 */
export function checkCategories(
  id: string,
  model: Pick<IrtModel, "name" | "maxCategories">,
  counts: readonly number[],
): number {
  if (counts.length > model.maxCategories) {
    throw new DataError(
      `Item "${id}" has a response of ${counts.length - 1}, but the ${model.name} model allows responses from 0 to ${model.maxCategories - 1}.`,
    );
  }
  if (counts.length < 2) {
    throw new DataError(`Item "${id}" has responses in fewer than two categories; calibration needs at least two.`);
  }
  const empty = counts.indexOf(0);
  if (empty >= 0) {
    throw new DataError(
      `Item "${id}" has no response in category ${empty}; calibration needs responses in every category up to the highest.`,
    );
  }
  return counts.length;
}

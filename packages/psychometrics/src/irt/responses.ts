import { DataError, describe } from "@measurekit/core";
import {
  checkResponseMatrix,
  checkResponseRecords,
  type ResponseMatrix,
  type ResponseRecords,
} from "../data/internal.js";
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

/** An item's responses by category. */
export interface CategoryTally {
  /** The highest response to the item; -1 when it has none. */
  readonly highest: number;
  /** The number of responses in each category that has one. */
  readonly counts: ReadonlyMap<number, number>;
}

/** The tally of an item without responses. */
export const NO_RESPONSES: CategoryTally = { highest: -1, counts: new Map() };

/** Returns the responses to each item by category, in space that grows with the categories used, not with the highest. */
export function categoryTallies({ itemIds, items, codes }: CompressedResponses): CategoryTally[] {
  const counts = itemIds.map(() => new Map<number, number>());
  const highest = new Int32Array(itemIds.length).fill(-1);
  items.forEach((item, r) => {
    const response = codes[r] ?? 0;
    const itemCounts = counts[item];
    itemCounts?.set(response, (itemCounts.get(response) ?? 0) + 1);
    highest[item] = Math.max(highest[item] ?? -1, response);
  });
  return counts.map((itemCounts, item) => ({ highest: highest[item] ?? -1, counts: itemCounts }));
}

/**
 * Checks that an item's responses can be estimated with its model, of either kind, and returns the share of its
 * responses in each category from 0 to the highest.
 *
 * @throws `DataError` when the item has a response above the model's categories, responses in fewer than two
 * categories, or no response in a category below its highest.
 */
export function checkCategories(
  id: string,
  model: Pick<IrtModel, "name" | "maxCategories">,
  { highest, counts }: CategoryTally,
): Float64Array {
  if (highest >= model.maxCategories) {
    throw new DataError(
      `Item "${id}" has a response of ${highest}, but the ${model.name} model allows responses from 0 to ${model.maxCategories - 1}.`,
    );
  }
  if (highest < 1) {
    throw new DataError(`Item "${id}" has responses in fewer than two categories; calibration needs at least two.`);
  }
  if (counts.size <= highest) {
    let empty = 0;
    while (counts.has(empty)) empty++;
    throw new DataError(
      `Item "${id}" has no response in category ${empty}; calibration needs responses in every category up to the highest.`,
    );
  }
  let answered = 0;
  for (const count of counts.values()) answered += count;
  return Float64Array.from({ length: highest + 1 }, (_, category) => (counts.get(category) ?? 0) / answered);
}

/** The groups of persons in a calibration. */
export interface PersonGroups {
  /** The distinct labels in order of first appearance; `null` without the option `groups`. */
  readonly labels: readonly string[] | null;
  /** The index in `labels` of each person's group; 0 for every person without groups. */
  readonly groupOf: Int32Array;
}

/**
 * Checks the labels that the option `groups` gives to `personCount` persons and returns their groups.
 *
 * @throws `DataError` when `groups` is given and does not hold one string for each person.
 */
export function personGroups(groups: readonly string[] | undefined, personCount: number): PersonGroups {
  if (groups === undefined) return { labels: null, groupOf: new Int32Array(personCount) };
  if (!Array.isArray(groups) || groups.length !== personCount) {
    throw new DataError(`Found ${Array.isArray(groups) ? groups.length : "no"} groups for ${personCount} persons.`);
  }
  groups.forEach((label, person) => {
    if (typeof label !== "string") {
      throw new DataError(`The group of person ${person + 1} must be a string, found ${describe(label)}.`);
    }
  });
  const labels = [...new Set(groups)];
  return { labels, groupOf: Int32Array.from(groups, (label) => labels.indexOf(label)) };
}

/**
 * Returns whether each person has a response, and how many persons do.
 *
 * @throws `DataError` when no person has a response.
 */
export function respondentsOf({ offsets }: Pick<CompressedResponses, "offsets">): {
  readonly answered: boolean[];
  readonly respondents: number;
} {
  const answered = Array.from(
    { length: offsets.length - 1 },
    (_, person) => (offsets[person + 1] ?? 0) > (offsets[person] ?? 0),
  );
  const respondents = answered.filter(Boolean).length;
  if (respondents === 0) throw new DataError("Calibration needs at least one response.");
  return { answered, respondents };
}

/**
 * Calls `visit` with each person who has a response, the bounds of the person's responses and the person's group.
 *
 * @param groupOf - The index of each person's group; without it every person is in group 0.
 */
export function forEachRespondent(
  { offsets }: Pick<CompressedResponses, "offsets">,
  groupOf: ArrayLike<number> | undefined,
  visit: (person: number, start: number, end: number, group: number) => void,
): void {
  for (let person = 0; person < offsets.length - 1; person++) {
    const start = offsets[person] ?? 0;
    const end = offsets[person + 1] ?? 0;
    if (end > start) visit(person, start, end, groupOf?.[person] ?? 0);
  }
}

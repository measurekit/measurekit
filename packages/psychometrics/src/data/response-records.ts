import { checkArray, checkOptions, checkRecord, DataError, describe, isPositive, isRecord } from "@measurekit/core";
import { indexIds } from "./ids.js";

/** One response in long format: a person's score on an item. */
export interface ResponseRecord {
  /** The person's id. */
  readonly person: string;
  /** The item's id. */
  readonly item: string;
  /** The score or category, an integer from 0. */
  readonly response: number;
  /** The response time in seconds, positive. */
  readonly time?: number;
}

/**
 * Responses in long format, one record per answered item, with the ids of the persons and items: the input of
 * calibration and ability estimation for sparse data, such as adaptive tests on a large pool, whose size grows with the
 * responses, not with persons times items.
 */
export interface ResponseRecords {
  /** Person ids, in the order of results by person. */
  readonly personIds: readonly string[];
  /** Item ids, in the order of results by item. */
  readonly itemIds: readonly string[];
  /** The responses, in any order. */
  readonly records: readonly ResponseRecord[];
}

/** Ids of the persons and items of response records. */
export interface ResponseRecordsOptions {
  /**
   * Person ids, which may include persons without records.
   *
   * @defaultValue the persons of the records, in order
   */
  readonly personIds?: readonly string[];
  /**
   * Item ids, which may include items without records.
   *
   * @defaultValue the items of the records, in order
   */
  readonly itemIds?: readonly string[];
}

/**
 * Builds response records, by default with the persons and items in the order they first appear. The records are
 * copied.
 *
 * @throws `DataError` when the records are not an array, a record's person or item is not a string among the ids, a
 * response is not a non-negative integer, a time is not a positive finite number, a person has two records of one
 * item, or the ids repeat.
 */
export function responseRecords(
  records: readonly ResponseRecord[],
  options: ResponseRecordsOptions = {},
): ResponseRecords {
  checkArray(records, "Response records must be an array");
  checkOptions(options);
  const objects = records.filter(isRecord);
  const {
    personIds = [...new Set(objects.map((record) => record.person))],
    itemIds = [...new Set(objects.map((record) => record.item))],
  } = options;
  const data: ResponseRecords = { personIds, itemIds, records };
  checkResponseRecords(data);
  return {
    ...data,
    records: records.map(({ person, item, response, time }) => ({
      person,
      item,
      response,
      ...(time !== undefined && { time }),
    })),
  };
}

/** A checked record by indices: the item, the response and the time, 0 without one. */
export interface IndexedRecord {
  readonly item: number;
  readonly response: number;
  readonly time: number;
}

/**
 * Checks response records and returns, for each person, the item indices, responses and times of the person's
 * records, ordered by item.
 *
 * @throws `DataError` naming the first problem found, in the cases of {@link responseRecords}.
 */
export function checkResponseRecords(data: ResponseRecords): IndexedRecord[][] {
  const { personIds, itemIds, records } = data;
  // Records built by hand or loaded from JSON may lack a field or hold another type.
  if (!(Array.isArray(personIds) && Array.isArray(itemIds) && Array.isArray(records))) {
    throw new DataError("Response records need the arrays personIds, itemIds and records.");
  }
  const persons = indexIds("person", personIds);
  const items = indexIds("item", itemIds);
  const byPerson = personIds.map((): IndexedRecord[] => []);
  records.forEach((record, index) => {
    checkRecord(record, `Record ${index + 1} must be an object { person, item, response }`);
    const person = persons.get(record.person);
    const item = items.get(record.item);
    if (person === undefined || item === undefined) {
      throw new DataError(
        `Record ${index + 1} needs a person and an item among the ids, found person ${describe(record.person)} and item ${describe(record.item)}.`,
      );
    }
    const { response } = record;
    if (!(Number.isSafeInteger(response) && response >= 0)) {
      throw new DataError(
        `The response of person "${record.person}" to item "${record.item}" must be a non-negative integer, found ${describe(response)}.`,
      );
    }
    const { time } = record;
    if (time !== undefined && !isPositive(time)) {
      throw new DataError(
        `The time of person "${record.person}" on item "${record.item}" must be a positive number, found ${describe(time)}.`,
      );
    }
    byPerson[person]?.push({ item, response, time: time ?? 0 });
  });
  byPerson.forEach((list, person) => {
    list.sort((x, y) => x.item - y.item);
    for (let r = 1; r < list.length; r++) {
      if (list[r]?.item === list[r - 1]?.item) {
        throw new DataError(
          `Person "${personIds[person]}" has two records of item "${itemIds[list[r]?.item ?? 0]}"; a person answers an item once.`,
        );
      }
    }
  });
  return byPerson;
}

import { checkArray, checkFinite, checkPositive, checkRecord, DataError } from "@measurekit/core";
import { indexIds } from "../data/ids.js";
import type { ResponseMatrix } from "../data/response-matrix.js";
import type { ResponseRecords } from "../data/response-records.js";
import { type CompressedResponses, compressResponses } from "../irt/responses.js";

/** Answered responses by person with their log times, `NaN` where a response has no time. */
export interface TimedResponses extends CompressedResponses {
  /** The natural log of each response's time; `NaN` without a time. */
  readonly logTimes: Float64Array;
}

/**
 * Checks data with response times and returns their responses with the log times.
 *
 * @throws `DataError` when the data are malformed or have no times.
 */
export function timedResponses(data: ResponseMatrix | ResponseRecords, statistic: string): TimedResponses {
  const responses = compressResponses(data);
  const { times } = responses;
  if (times === null) throw new DataError(`${statistic} needs response times.`);
  return { ...responses, logTimes: times.map((time) => (time > 0 ? Math.log(time) : Number.NaN)) };
}

/** Calls `visit` with the person, the item and the log time of each response with a time. */
export function forEachTime(
  timed: TimedResponses,
  visit: (person: number, item: number, logTime: number) => void,
): void {
  const { offsets, items, logTimes } = timed;
  for (let person = 0; person + 1 < offsets.length; person++) {
    for (let r = offsets[person] ?? 0; r < (offsets[person + 1] ?? 0); r++) {
      const logTime = logTimes[r] ?? Number.NaN;
      if (!Number.isNaN(logTime)) visit(person, items[r] ?? 0, logTime);
    }
  }
}

/** An item's parameters in the lognormal model of response times. */
export interface TimeItem {
  /** The item's id. */
  readonly id: string;
  /** β, the time intensity: the mean log time of a person of speed 0. */
  readonly intensity: number;
  /** α, the time discrimination: 1 / α is the SD of the log times at a given speed. */
  readonly discrimination: number;
}

/**
 * Returns the time items in the order of the data's item ids.
 *
 * @throws `DataError` when the time items are not an array of objects, their ids repeat, an item of the data has no
 * time item with its id, or a time item's parameters are not finite with a positive discrimination.
 */
export function itemsOfData(itemIds: readonly string[], items: readonly TimeItem[]): TimeItem[] {
  checkArray(items, "The time items must be an array");
  items.forEach((item, index) => {
    checkRecord(item, `Time item ${index + 1} must be an object { id, intensity, discrimination }`);
  });
  const indices = indexIds(
    "time item",
    items.map((item) => item.id),
  );
  return itemIds.map((id) => {
    const item = items[indices.get(id) ?? -1];
    if (item === undefined) throw new DataError(`The data have item "${id}", which is not among the time items.`);
    checkFinite(item.intensity, `The intensity of time item "${id}"`);
    checkPositive(item.discrimination, `The discrimination of time item "${id}"`);
    return item;
  });
}

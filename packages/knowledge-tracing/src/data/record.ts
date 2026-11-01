import { DataError } from "@measurekit/core";

/**
 * Returns a record's own entry for a key, or `undefined`. Ids are user data, so a lookup must not reach inherited
 * properties such as `constructor`.
 */
export function entry<T>(record: Readonly<Record<string, T>> | undefined, key: string): T | undefined {
  return record !== undefined && Object.hasOwn(record, key) ? record[key] : undefined;
}

/** Returns a record with `map` applied to each value, given the value and its key. */
export function mapRecord<TValue, TResult>(
  record: Readonly<Record<string, TValue>>,
  map: (value: TValue, key: string) => TResult,
): Record<string, TResult> {
  return Object.fromEntries(Object.entries(record).map(([key, value]) => [key, map(value, key)]));
}

/**
 * Returns a model's parameters of a skill or an item.
 *
 * @throws `DataError` when the record has none.
 */
export function parametersOf<T>(
  record: Readonly<Record<string, T>> | undefined,
  kind: "skill" | "item",
  id: string,
): T {
  const found = entry(record, id);
  if (found === undefined) throw new DataError(`The model has no parameters for ${kind} "${id}".`);
  return found;
}

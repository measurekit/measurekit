import { DataError, describe } from "@measurekit/core";

/**
 * Checks that ids are unique strings and returns the index of each.
 *
 * @param kind - Names the ids in error messages, such as `person`.
 * @throws `DataError` when an id is not a string or appears twice.
 */
export function indexIds(kind: string, ids: readonly string[]): Map<string, number> {
  const indices = new Map<string, number>();
  ids.forEach((id, index) => {
    if (typeof id !== "string") throw new DataError(`Every ${kind} id must be a string, found ${describe(id)}.`);
    if (indices.has(id)) throw new DataError(`The ${kind} id "${id}" appears more than once.`);
    indices.set(id, index);
  });
  return indices;
}

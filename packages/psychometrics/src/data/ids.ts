import { DataError, describe } from "@measurekit/core";

/**
 * Checks that ids are unique strings and returns the index of each.
 *
 * @param kind - Names the ids in error messages, such as `person`.
 * @throws `DataError` when an id is not a string or appears twice.
 */
export function indexIds(kind: string, ids: readonly string[]): Map<string, number> {
  const indices = new Map<string, number>();
  // By index, so that a hole in the ids is an id that is not a string, which forEach would skip.
  for (let index = 0; index < ids.length; index++) {
    const id: unknown = ids[index];
    if (typeof id !== "string") throw new DataError(`Every ${kind} id must be a string, found ${describe(id)}.`);
    const known = indices.size;
    indices.set(id, index);
    if (indices.size === known) throw new DataError(`The ${kind} id "${id}" appears more than once.`);
  }
  return indices;
}

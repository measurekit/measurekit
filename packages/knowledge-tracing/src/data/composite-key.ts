/**
 * Returns one key for several ids, such as a learner and a skill, that no other list of ids shares. Ids are user data and
 * may hold any character, so joining them with a separator would let two lists give the same key.
 */
export function compositeKey(ids: readonly string[]): string {
  return JSON.stringify(ids);
}

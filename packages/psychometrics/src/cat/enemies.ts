import { checkArray, DataError, describe } from "@measurekit/core";
import type { PoolItem } from "./items.js";

/**
 * Returns the enemies of each pool item by index: the other items of every enemy set it is in.
 *
 * @throws `DataError` when the sets are not an array, or a set is not an array of at least two distinct ids of pool
 * items.
 */
export function enemyIndices(
  pool: readonly PoolItem[],
  indices: ReadonlyMap<string, number>,
  sets: readonly (readonly string[])[],
): readonly (readonly number[])[] {
  checkArray(sets, "The option enemies must be an array of enemy sets");
  const enemies = pool.map(() => new Set<number>());
  sets.forEach((set, s) => {
    if (!Array.isArray(set) || set.length < 2) {
      throw new DataError(`Enemy set ${s + 1} needs at least two item ids.`);
    }
    const members = set.map((id) => {
      const index = indices.get(id);
      if (index === undefined) {
        throw new DataError(`Enemy set ${s + 1} names ${describe(id)}, which is not a pool item.`);
      }
      return index;
    });
    if (new Set(members).size !== members.length) throw new DataError(`Enemy set ${s + 1} names an item twice.`);
    for (const member of members) {
      for (const other of members) if (other !== member) enemies[member]?.add(other);
    }
  });
  return enemies.map((set) => [...set]);
}

/** Clears the eligibility of every enemy of the administered items. */
export function excludeEnemies(
  enemies: readonly (readonly number[])[],
  administered: readonly number[],
  eligible: Uint8Array,
): void {
  for (const index of administered) for (const enemy of enemies[index] ?? []) eligible[enemy] = 0;
}

/**
 * Checks that no two administered items are enemies.
 *
 * @throws `DataError` naming the first two that are.
 */
export function checkEnemies(
  enemies: readonly (readonly number[])[],
  administered: readonly number[],
  pool: readonly PoolItem[],
): void {
  const taken = new Set<number>();
  for (const index of administered) {
    const enemy = (enemies[index] ?? []).find((other) => taken.has(other));
    if (enemy !== undefined) {
      throw new DataError(
        `A test state administers items "${pool[enemy]?.id}" and "${pool[index]?.id}", which are enemies.`,
      );
    }
    taken.add(index);
  }
}

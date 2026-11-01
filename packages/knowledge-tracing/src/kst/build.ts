import { checkIntegerFrom, DataError } from "@measurekit/core";
import { compareStates, type KnowledgeStructure, seal } from "./structure.js";

/** The default limit on the number of knowledge states a structure is built with. */
const DEFAULT_MAX_STATES = 10_000;

/** Options of the functions that build a structure by closure or by prerequisites. */
export interface StateLimitOptions {
  /**
   * The most knowledge states to build; more throw a `DataError`, as a learner's state holds a probability for each.
   *
   * @defaultValue `10000`
   */
  readonly maxStates?: number;
}

/**
 * Checks the limit on the number of states and returns it.
 *
 * @throws `DataError` when it is not an integer from 2.
 */
export function checkMaxStates(maxStates: number = DEFAULT_MAX_STATES): number {
  checkIntegerFrom(maxStates, "The option maxStates", 2);
  return maxStates;
}

/** Returns the error of a structure that has more than `maxStates` knowledge states. */
export function tooManyStates(maxStates: number): DataError {
  return new DataError(
    `The structure has more than ${maxStates} knowledge states. Split the domain into units, such as chapters, and build a structure for each, or raise maxStates.`,
  );
}

/**
 * Returns the frozen structure whose states are given as ascending lists of positions in `domain`: each state once,
 * ordered by size, then by the domain's order.
 */
export function buildStructure(domain: readonly string[], states: readonly (readonly number[])[]): KnowledgeStructure {
  const unique = new Map(states.map((state) => [state.join(","), state]));
  return seal({
    domain: [...domain],
    states: [...unique.values()].sort(compareStates).map((state) => state.map((e) => domain[e] ?? "")),
  });
}

/** Returns whether element e is in a bitset. */
export function hasElement(bits: Uint32Array, e: number): boolean {
  return (((bits[e >>> 5] ?? 0) >>> (e & 31)) & 1) === 1;
}

/** Returns the positions of the elements in a bitset of `size` elements, ascending. */
export function positionsOf(bits: Uint32Array, size: number): number[] {
  const positions: number[] = [];
  for (let e = 0; e < size; e++) if (hasElement(bits, e)) positions.push(e);
  return positions;
}

/** Returns the number of bits set in a 32-bit word. */
export function popcount(word: number): number {
  let value = word - ((word >>> 1) & 0x55555555);
  value = (value & 0x33333333) + ((value >>> 2) & 0x33333333);
  return Math.imul((value + (value >>> 4)) & 0x0f0f0f0f, 0x01010101) >>> 24;
}

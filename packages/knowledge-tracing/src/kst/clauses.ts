import { checkOptions, DataError, describe, isRecord } from "@measurekit/core";
import {
  buildStructure,
  checkMaxStates,
  hasElement,
  positionsOf,
  STATE_LIMIT_OPTIONS,
  type StateLimitOptions,
  tooManyStates,
} from "./build.js";
import { addElement, compareStates, type KnowledgeStructure, keyOf } from "./structure.js";

/**
 * Returns the knowledge space of elements that each need one of several sets of other elements: an element joins a
 * state when the state holds all elements of one of the element's clauses. States are ordered by size, then by the
 * domain's order, and the structure is frozen.
 *
 * @remarks
 * `{ c: [["a"], ["b", "d"]] }` lets a learner who has a, or has both b and d, learn c. An element with no clauses, or
 * with an empty clause, needs nothing. The states grow one element at a time from the empty state, so that every state
 * can be learned element by element, and the result is a learning space. A surmise relation, with one clause for each
 * element, gives the quasi-ordinal space of {@link surmiseStructure}, except that elements that require each other are
 * learned together there and cannot be learned here.
 *
 * @param clauses - Each element's clauses, by element: every element of the domain is a key, with `[]` when it has
 *   none, and the domain is the keys in order. A clause is a list of keys.
 * @param options - The most states to build.
 * @returns The knowledge space, with its states in the domain's order.
 * @throws `DataError` when `clauses` is not an object of lists of lists of keys, an element has a clause with
 * something that is not a key, an element cannot be learned from the empty state, such as one in a cycle of clauses, or
 * the space has more states than `maxStates`.
 * @see {@link https://doi.org/10.1007/978-3-642-01039-2 | Falmagne and Doignon (2011), chapter 4}
 */
export function clauseStructure(
  clauses: Readonly<Record<string, readonly (readonly string[])[]>>,
  options: StateLimitOptions = {},
): KnowledgeStructure {
  checkOptions(options, STATE_LIMIT_OPTIONS);
  const maxStates = checkMaxStates(options.maxStates);
  if (!isRecord(clauses)) {
    throw new DataError(`Clauses must be an object of clause lists by element, found ${describe(clauses)}.`);
  }
  const domain = Object.keys(clauses);
  if (domain.length === 0) throw new DataError("Clauses need at least one element.");
  const words = Math.ceil(domain.length / 32);
  const index = new Map(domain.map((element, e) => [element, e]));
  const needs = domain.map((element) => clausesOf(element, clauses[element], index, words));
  const { states, reached } = grow(needs, words, maxStates);
  const unreachable = domain.filter((_, e) => !hasElement(reached, e));
  if (unreachable.length > 0) {
    throw new DataError(
      `${unreachable.map((element) => `"${element}"`).join(", ")} cannot be learned: no clause can be completed from the empty state. Look for a cycle of clauses, or add an element with no clauses.`,
    );
  }
  return buildStructure(domain, states.map((bits) => positionsOf(bits, domain.length)).sort(compareStates));
}

/** Returns the clauses of an element as bitsets, throwing a `DataError` when they are not lists of keys. */
function clausesOf(element: string, listed: unknown, index: ReadonlyMap<string, number>, words: number): Uint32Array[] {
  if (!Array.isArray(listed)) throw new DataError(`The clauses of "${element}" must be a list of lists of elements.`);
  return listed.map((clause: unknown) => {
    if (!Array.isArray(clause)) throw new DataError(`Each clause of "${element}" must be a list of elements.`);
    const bits = new Uint32Array(words);
    for (const required of clause) {
      const e = index.get(required);
      if (e === undefined) {
        throw new DataError(`A clause of "${element}" has ${describe(required)}, which is not a key of the clauses.`);
      }
      addElement(bits, 0, e);
    }
    return bits;
  });
}

/**
 * Returns the sets reached from the empty set by adding, one at a time, an element with no clauses or with a clause
 * inside the set, and the elements that some set holds.
 */
function grow(
  needs: readonly (readonly Uint32Array[])[],
  words: number,
  maxStates: number,
): { states: Uint32Array[]; reached: Uint32Array } {
  const empty = new Uint32Array(words);
  const states = [empty];
  const keys = new Set([keyOf(empty, 0, words)]);
  const reached = new Uint32Array(words);
  for (let s = 0; s < states.length; s++) {
    const state = states[s] ?? empty;
    for (const e of learnableFrom(needs, state)) {
      const grown = state.slice();
      addElement(grown, 0, e);
      const key = keyOf(grown, 0, words);
      if (keys.has(key)) continue;
      if (states.length === maxStates) throw tooManyStates(maxStates);
      keys.add(key);
      states.push(grown);
      addElement(reached, 0, e);
    }
  }
  return { states, reached };
}

/** Returns the elements outside a state that have no clauses, or a clause inside the state. */
function learnableFrom(needs: readonly (readonly Uint32Array[])[], state: Uint32Array): number[] {
  return needs.flatMap((clauses, e) => {
    const ready =
      clauses.length === 0 || clauses.some((clause) => clause.every((word, w) => (word & ~(state[w] ?? 0)) === 0));
    return !hasElement(state, e) && ready ? [e] : [];
  });
}

import { checkArray, checkRecord, DataError, isNonEmptyString } from "@measurekit/core";

/**
 * A knowledge structure: the knowledge states, each a set of elements of the domain, items or skills. It contains the
 * empty state and the whole domain. Plain data, which JSON serializes.
 */
export interface KnowledgeStructure {
  /** The elements, items or skills, in order. */
  readonly domain: readonly string[];
  /** The knowledge states, each a list of elements of the domain. */
  readonly states: readonly (readonly string[])[];
}

/** A structure in bitsets: state s holds element e when bit e of its `words` words starting at `s × words` is set. */
export interface CompiledStructure {
  readonly domain: readonly string[];
  /** The position of each element in the domain. */
  readonly index: ReadonlyMap<string, number>;
  /** The position of each state, by its {@link keyOf} key. */
  readonly stateOf: ReadonlyMap<string, number>;
  readonly words: number;
  readonly count: number;
  readonly bits: Uint32Array;
}

/** Element e of a bitset is bit `e & WORD_MASK` of word `e >>> WORD_SHIFT`: words of 32 bits. */
const WORD_SHIFT = 5;
const WORD_MASK = 31;

/** Returns the number of 32-bit words that hold `size` elements. */
function wordCount(size: number): number {
  return Math.ceil(size / (WORD_MASK + 1));
}

/** Returns whether state s of a compiled structure holds element e. */
export function holds(structure: CompiledStructure, s: number, e: number): boolean {
  return (((structure.bits[s * structure.words + (e >>> WORD_SHIFT)] ?? 0) >>> (e & WORD_MASK)) & 1) === 1;
}

/** Adds element e to the bitset of the words starting at `offset`; returns whether it was already there. */
export function addElement(bits: Uint32Array, offset: number, e: number): boolean {
  const word = offset + (e >>> WORD_SHIFT);
  const bit = 1 << (e & WORD_MASK);
  const present = ((bits[word] ?? 0) & bit) !== 0;
  bits[word] = (bits[word] ?? 0) | bit;
  return present;
}

/** Adds element e to a one-state bitset if it is absent, removes it otherwise. */
export function flipElement(words: Uint32Array, e: number): void {
  words[e >>> WORD_SHIFT] = (words[e >>> WORD_SHIFT] ?? 0) ^ (1 << (e & WORD_MASK));
}

/** Returns a state's key: its words, for lookups of states by content. */
export function keyOf(bits: Uint32Array, start: number, words: number): string {
  return bits.subarray(start, start + words).join(",");
}

/** The bitsets of the structures {@link seal} has validated and frozen, which later calls trust. */
const SEALED = new WeakMap<KnowledgeStructure, CompiledStructure>();

/**
 * Returns a structure in bitsets: a sealed structure from its record, any other after {@link validate}, so that a
 * structure from storage or from a caller is checked in full at each call.
 *
 * @throws `DataError` as {@link validate} does.
 */
export function compile(structure: KnowledgeStructure): CompiledStructure {
  return SEALED.get(structure) ?? validate(structure);
}

/**
 * Validates a structure that the library has built, freezes it, its domain, states and each state, and records its
 * bitsets, so that {@link compile} trusts it instead of checking it again; frozen, it cannot become invalid.
 *
 * @throws `DataError` as {@link validate} does.
 */
export function seal(structure: KnowledgeStructure): KnowledgeStructure {
  const compiled = validate(structure);
  for (const state of structure.states) Object.freeze(state);
  Object.freeze(structure.states);
  Object.freeze(structure.domain);
  SEALED.set(structure, compiled);
  return Object.freeze(structure);
}

/**
 * Checks a structure and returns it in bitsets.
 *
 * @throws `DataError` when the domain is not a non-empty list of distinct ids, a state names an element outside it or
 * twice, two states are equal, or the empty state or the whole domain is missing.
 */
function validate(structure: KnowledgeStructure): CompiledStructure {
  checkRecord(structure, "A knowledge structure must be an object of domain and states");
  const { domain, states } = structure;
  if (!(Array.isArray(domain) && domain.length > 0 && Array.from(domain).every(isNonEmptyString))) {
    throw new DataError("A knowledge structure needs a domain, a non-empty list of element ids.");
  }
  const index = new Map(domain.map((element, e) => [element, e]));
  if (index.size !== domain.length) throw new DataError("The domain of a knowledge structure lists an element twice.");
  checkArray(states, "A knowledge structure needs states, a list of lists of elements");
  const words = wordCount(domain.length);
  const bits = new Uint32Array(states.length * words);
  const stateOf = new Map<string, number>();
  states.forEach((state, s) => {
    checkArray(state, `State ${s + 1} must be a list of elements`);
    for (const element of state) {
      const e = index.get(element);
      if (e === undefined) throw new DataError(`State ${s + 1} has "${element}", which is not in the domain.`);
      if (addElement(bits, s * words, e)) throw new DataError(`State ${s + 1} lists "${element}" twice.`);
    }
    const key = keyOf(bits, s * words, words);
    if (stateOf.has(key)) throw new DataError(`State ${s + 1} repeats an earlier state.`);
    stateOf.set(key, s);
  });
  const full = new Uint32Array(words);
  for (let e = 0; e < domain.length; e++) addElement(full, 0, e);
  if (!stateOf.has(keyOf(new Uint32Array(words), 0, words))) {
    throw new DataError("A knowledge structure needs the empty state.");
  }
  if (!stateOf.has(keyOf(full, 0, words))) {
    throw new DataError("A knowledge structure needs the state of the whole domain.");
  }
  return { domain: [...domain], index, stateOf, words, count: states.length, bits };
}

/** Returns the elements of state s of a compiled structure, in the domain's order. */
export function elementsOf(compiled: CompiledStructure, s: number): string[] {
  return compiled.domain.filter((_, e) => holds(compiled, s, e));
}

/** Orders states, as sorted lists of element indices, by size, then lexicographically. */
export function compareStates(x: readonly number[], y: readonly number[]): number {
  if (x.length !== y.length) return x.length - y.length;
  for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return (x[i] ?? 0) - (y[i] ?? 0);
  return 0;
}

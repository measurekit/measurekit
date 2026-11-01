import { checkArray, DataError } from "@measurekit/core";
import {
  addElement,
  type CompiledStructure,
  compile,
  flipElement,
  holds,
  type KnowledgeStructure,
  keyOf,
} from "./structure.js";

/**
 * Returns whether a structure is a knowledge space: closed under union. It compares every pair of states, so takes time
 * quadratic in their number.
 *
 * @throws `DataError` when the structure is malformed.
 */
export function isKnowledgeSpace(structure: KnowledgeStructure): boolean {
  return closedUnderUnion(compile(structure));
}

/**
 * Returns whether a structure is a learning space: a knowledge space in which every non-empty state has an element
 * whose removal leaves a state, so that every state can be learned one element at a time.
 *
 * @throws `DataError` when the structure is malformed.
 * @see {@link https://doi.org/10.1016/j.jmp.2008.09.005 | Cosyn and Uzun (2009)}
 */
export function isLearningSpace(structure: KnowledgeStructure): boolean {
  const compiled = compile(structure);
  return (
    closedUnderUnion(compiled) &&
    Array.from({ length: compiled.count }, (_, s) => s).every(
      (s) => isEmpty(compiled, s) || innerFringeOf(compiled, s).length > 0,
    )
  );
}

function closedUnderUnion({ count, words, bits, stateOf }: CompiledStructure): boolean {
  const union = new Uint32Array(words);
  for (let s = 0; s < count; s++) {
    for (let t = s + 1; t < count; t++) {
      for (let w = 0; w < words; w++) union[w] = (bits[s * words + w] ?? 0) | (bits[t * words + w] ?? 0);
      if (!stateOf.has(keyOf(union, 0, words))) return false;
    }
  }
  return true;
}

/** The fringes of a knowledge state. */
export interface Fringes {
  /** The elements of the state whose removal leaves a state: the last ones learned. */
  readonly inner: readonly string[];
  /** The elements outside the state whose addition makes a state: the ones ready to learn. */
  readonly outer: readonly string[];
}

/**
 * Returns the inner and outer fringes of a state of the structure, in the domain's order.
 *
 * @throws `DataError` when the structure is malformed or the state is not one of its states.
 */
export function fringes(structure: KnowledgeStructure, state: readonly string[]): Fringes {
  const compiled = compile(structure);
  const s = stateIndex(compiled, state);
  const outer: string[] = [];
  const probe = compiled.bits.slice(s * compiled.words, (s + 1) * compiled.words);
  compiled.domain.forEach((element, e) => {
    if (holds(compiled, s, e)) return;
    flipElement(probe, e);
    if (compiled.stateOf.has(keyOf(probe, 0, probe.length))) outer.push(element);
    flipElement(probe, e);
  });
  return { inner: innerFringeOf(compiled, s), outer };
}

/** Returns the index of a state given as elements, throwing a `DataError` when it is not a state of the structure. */
export function stateIndex(compiled: CompiledStructure, state: readonly string[]): number {
  checkArray(state, "A knowledge state must be a list of elements");
  const probe = new Uint32Array(compiled.words);
  for (const element of state) {
    const e = compiled.index.get(element);
    if (e === undefined) throw new DataError(`The state has "${element}", which is not in the domain.`);
    addElement(probe, 0, e);
  }
  const s = compiled.stateOf.get(keyOf(probe, 0, probe.length));
  if (s === undefined) throw new DataError(`The state [${state.join(", ")}] is not a state of the structure.`);
  return s;
}

function isEmpty(compiled: CompiledStructure, s: number): boolean {
  return compiled.bits.subarray(s * compiled.words, (s + 1) * compiled.words).every((word) => word === 0);
}

function innerFringeOf(compiled: CompiledStructure, s: number): string[] {
  const probe = compiled.bits.slice(s * compiled.words, (s + 1) * compiled.words);
  return compiled.domain.filter((_, e) => {
    if (!holds(compiled, s, e)) return false;
    flipElement(probe, e);
    const found = compiled.stateOf.has(keyOf(probe, 0, probe.length));
    flipElement(probe, e);
    return found;
  });
}

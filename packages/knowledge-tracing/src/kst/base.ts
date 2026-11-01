import { checkArray, DataError, describe } from "@measurekit/core";
import { buildStructure, popcount, positionsOf } from "./build.js";
import { closeUnderUnion } from "./closure.js";
import {
  type CompiledStructure,
  compareStates,
  compile,
  elementsOf,
  holds,
  type KnowledgeStructure,
  keyOf,
} from "./structure.js";

/**
 * Returns the base of a knowledge space: its states that are not the union of other states, which is the smallest
 * family whose unions give the space. States are ordered by size, then by the domain's order.
 *
 * @remarks
 * The base is the union of the atoms of every element. It is `kbase` of the kst package, which also requires a space.
 *
 * @throws `DataError` when the structure is malformed or is not a knowledge space.
 * @see {@link https://doi.org/10.1007/978-3-642-01039-2 | Falmagne and Doignon (2011), chapter 3}
 */
export function structureBase(structure: KnowledgeStructure): readonly (readonly string[])[] {
  const compiled = compile(structure);
  const { count, words, bits, stateOf } = compiled;
  const atoms = inOrder(compiled, unionIrreducible(compiled));
  // Every state of a knowledge space is a union of atoms, and a union of atoms of a structure that is not closed under
  // union is outside it or leaves a state out, so the space is the closure of its base exactly.
  const closed = closeUnderUnion(
    atoms.map((s) => bits.subarray(s * words, (s + 1) * words)),
    words,
    count,
  );
  if (closed === null || closed.length !== count || !closed.every((state) => stateOf.has(keyOf(state, 0, words)))) {
    throw new DataError("The base is defined for a knowledge space, and this structure is not closed under union.");
  }
  return atoms.map((s) => elementsOf(compiled, s));
}

/**
 * Returns the atoms of an element: the states that hold it and no smaller state does, the smallest states that a
 * learner who has it can be in. States are ordered by size, then by the domain's order.
 *
 * @remarks
 * It is `katoms` of the kst package, which matches an element by its name as a pattern, so that its result is right
 * only for ids that no other id contains.
 *
 * @throws `DataError` when the structure is malformed or the element is not in its domain.
 * @see {@link https://doi.org/10.1007/978-3-642-01039-2 | Falmagne and Doignon (2011), chapter 3}
 */
export function structureAtoms(structure: KnowledgeStructure, element: string): readonly (readonly string[])[] {
  const compiled = compile(structure);
  const e = compiled.index.get(element);
  if (e === undefined) throw new DataError(`The element ${describe(element)} is not in the domain.`);
  const found: number[] = [];
  for (const s of bySize(compiled)) {
    if (!holds(compiled, s, e)) continue;
    if (!found.some((atom) => isSubset(compiled, atom, s))) found.push(s);
  }
  return inOrder(compiled, found).map((s) => elementsOf(compiled, s));
}

/**
 * Returns the trace of a structure on some of its elements: the structure that its states leave on them, each state
 * cut to these elements, once. Elements keep the domain's order. States are ordered by size, then by the domain's
 * order, and the structure is frozen.
 *
 * @remarks
 * It is `ktrace` of the kst package. The trace of a knowledge space is a knowledge space.
 *
 * @param elements - A non-empty list of distinct elements of the domain.
 * @throws `DataError` when the structure is malformed, or `elements` is not a non-empty list of distinct elements of
 * its domain.
 * @see {@link https://doi.org/10.1007/978-3-642-01039-2 | Falmagne and Doignon (2011), chapter 2}
 */
export function structureTrace(structure: KnowledgeStructure, elements: readonly string[]): KnowledgeStructure {
  const compiled = compile(structure);
  checkArray(elements, "The elements of a trace must be a list of elements");
  const kept = new Set<number>();
  for (const element of elements) {
    const e = compiled.index.get(element);
    if (e === undefined) throw new DataError(`The element ${describe(element)} is not in the domain.`);
    if (kept.has(e)) throw new DataError(`The elements of a trace list "${element}" twice.`);
    kept.add(e);
  }
  if (kept.size === 0) throw new DataError("A trace needs at least one element.");
  const domain = compiled.domain.filter((_, e) => kept.has(e));
  const place = new Map([...kept].sort((x, y) => x - y).map((e, position) => [e, position]));
  const states = Array.from({ length: compiled.count }, (_, s) =>
    positionsOf(compiled.bits.subarray(s * compiled.words, (s + 1) * compiled.words), compiled.domain.length)
      .filter((e) => kept.has(e))
      .map((e) => place.get(e) ?? 0),
  );
  return buildStructure(domain, states);
}

/**
 * Returns the notions of a structure: the classes of elements that are in the same states, as lists in the domain's
 * order, the classes ordered by their first element.
 *
 * @remarks
 * Elements of one notion cannot be told apart by any state, so a learner has all of them or none. It is `knotions` of
 * the kst package.
 *
 * @throws `DataError` when the structure is malformed.
 * @see {@link https://doi.org/10.1007/978-3-642-01039-2 | Falmagne and Doignon (2011), chapter 2}
 */
export function notions(structure: KnowledgeStructure): readonly (readonly string[])[] {
  const compiled = compile(structure);
  const classes = new Map<string, string[]>();
  compiled.domain.forEach((element, e) => {
    const profile = Array.from({ length: compiled.count }, (_, s) => (holds(compiled, s, e) ? "1" : "0")).join("");
    classes.set(profile, [...(classes.get(profile) ?? []), element]);
  });
  return [...classes.values()];
}

/** Returns the states `list` of a compiled structure ordered by size, then by the domain's order. */
function inOrder(compiled: CompiledStructure, list: readonly number[]): number[] {
  const positions = new Map(
    list.map((s) => [
      s,
      positionsOf(compiled.bits.subarray(s * compiled.words, (s + 1) * compiled.words), compiled.domain.length),
    ]),
  );
  return [...list].sort((x, y) => compareStates(positions.get(x) ?? [], positions.get(y) ?? []));
}

/** Returns the states of a compiled structure ordered by size, then by their position. */
function bySize(compiled: CompiledStructure): number[] {
  const sizes = Array.from({ length: compiled.count }, (_, s) => size(compiled, s));
  return sizes.map((_, s) => s).sort((x, y) => (sizes[x] ?? 0) - (sizes[y] ?? 0) || x - y);
}

function size(compiled: CompiledStructure, s: number): number {
  let total = 0;
  for (let w = 0; w < compiled.words; w++) total += popcount(compiled.bits[s * compiled.words + w] ?? 0);
  return total;
}

/** Returns whether state `a` is a subset of state `b` of a compiled structure. */
function isSubset(compiled: CompiledStructure, a: number, b: number): boolean {
  for (let w = 0; w < compiled.words; w++) {
    const bitsA = compiled.bits[a * compiled.words + w] ?? 0;
    if ((bitsA & ~(compiled.bits[b * compiled.words + w] ?? 0)) !== 0) return false;
  }
  return true;
}

/**
 * Returns the non-empty states that are not the union of the states inside them: a state is an atom of an element when
 * no state inside it holds the element, so it is the union of the states inside it exactly when it is an atom of none.
 */
function unionIrreducible(compiled: CompiledStructure): number[] {
  const irreducible: number[] = [];
  const covered = new Uint32Array(compiled.words);
  for (const s of bySize(compiled)) {
    unionInside(compiled, irreducible, s, covered);
    if (!equalsState(compiled, s, covered)) irreducible.push(s);
  }
  return irreducible;
}

/** Writes into `out` the union of those of the states `inside` that are subsets of state `s`. */
function unionInside(compiled: CompiledStructure, inside: readonly number[], s: number, out: Uint32Array): void {
  out.fill(0);
  for (const t of inside) {
    if (!isSubset(compiled, t, s)) continue;
    for (let w = 0; w < compiled.words; w++) out[w] = (out[w] ?? 0) | (compiled.bits[t * compiled.words + w] ?? 0);
  }
}

/** Returns whether the bitset `words` is state `s` of a compiled structure. */
function equalsState(compiled: CompiledStructure, s: number, words: Uint32Array): boolean {
  return words.every((word, w) => word === (compiled.bits[s * compiled.words + w] ?? 0));
}

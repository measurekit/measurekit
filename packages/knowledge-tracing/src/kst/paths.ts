import { checkIntegerFrom, checkOptions, DataError, type FieldNames } from "@measurekit/core";
import { popcount } from "./build.js";
import { stateIndex } from "./space.js";
import {
  addElement,
  type CompiledStructure,
  compile,
  elementsOf,
  flipElement,
  holds,
  type KnowledgeStructure,
  keyOf,
} from "./structure.js";

/** The default limit on the number of learning paths: they grow as a factorial. */
const DEFAULT_MAX_PATHS = 10_000;

/** Options of {@link neighbourhood}. */
export interface NeighbourhoodOptions {
  /**
   * The largest number of elements by which a state may differ from the given one.
   *
   * @defaultValue `1`
   */
  readonly distance?: number;
}

const NEIGHBOURHOOD_OPTIONS: FieldNames<NeighbourhoodOptions> = { distance: true };

/** Options of {@link learningPaths}. */
export interface LearningPathsOptions {
  /**
   * The most learning paths to list; more throw a `DataError`.
   *
   * @defaultValue `10000`
   */
  readonly maxPaths?: number;
}

const LEARNING_PATHS_OPTIONS: FieldNames<LearningPathsOptions> = { maxPaths: true };

/**
 * Returns the states of the structure that differ from `state` by at least one and at most `distance` elements, in the
 * structure's order.
 *
 * @remarks
 * With the default distance of 1 these are the states that add or lose one element, `kneighbourhood` of the kst
 * package; a larger distance is `knneighbourhood`.
 *
 * @param structure - The knowledge structure.
 * @param state - The state, in any order.
 * @param options - The distance.
 * @returns The states of the structure at a distance of 1 to `distance` from the state, in the domain's order.
 * @throws `DataError` when the structure is malformed, `state` is not one of its states, or `distance` is not a
 * positive integer.
 * @see {@link https://doi.org/10.1007/978-3-642-01039-2 | Falmagne and Doignon (2011), chapter 2}
 */
export function neighbourhood(
  structure: KnowledgeStructure,
  state: readonly string[],
  options: NeighbourhoodOptions = {},
): readonly (readonly string[])[] {
  checkOptions(options, NEIGHBOURHOOD_OPTIONS);
  const { distance = 1 } = options;
  checkIntegerFrom(distance, "The option distance", 1);
  const compiled = compile(structure);
  const s = stateIndex(compiled, state);
  const { words, bits } = compiled;
  const found: (readonly string[])[] = [];
  for (let t = 0; t < compiled.count; t++) {
    let apart = 0;
    for (let w = 0; w < words; w++) apart += popcount((bits[s * words + w] ?? 0) ^ (bits[t * words + w] ?? 0));
    if (apart >= 1 && apart <= distance) found.push(elementsOf(compiled, t));
  }
  return found;
}

/**
 * Returns the learning paths of a structure: the orders in which one learner can acquire the whole domain, starting
 * from the empty state, each step a state that adds one element. Paths are in the domain's order of their first
 * difference.
 *
 * @remarks
 * A path is a maximal chain of states in which each state adds one element to the one before, a gradation, which is
 * `lpath` of the kst package for a well-graded structure. A step of more than one element is not a step of a learning
 * path, so a structure that is not well-graded may have none. The number of paths grows as a factorial, so the
 * function stops at `maxPaths`.
 *
 * @param structure - The knowledge structure.
 * @param options - The most paths to list.
 * @returns The paths, each an order of the whole domain.
 * @throws `DataError` when the structure is malformed, `maxPaths` is not a positive integer, or the structure has more
 * learning paths than `maxPaths`.
 * @see {@link https://doi.org/10.1007/978-3-642-01039-2 | Falmagne and Doignon (2011), chapter 2}
 */
export function learningPaths(
  structure: KnowledgeStructure,
  options: LearningPathsOptions = {},
): readonly (readonly string[])[] {
  checkOptions(options, LEARNING_PATHS_OPTIONS);
  const { maxPaths = DEFAULT_MAX_PATHS } = options;
  checkIntegerFrom(maxPaths, "The option maxPaths", 1);
  const compiled = compile(structure);
  const next = steps(compiled);
  const full = compiled.stateOf.get(fullKey(compiled)) ?? 0;
  const empty = compiled.stateOf.get(keyOf(new Uint32Array(compiled.words), 0, compiled.words)) ?? 0;
  // A step that leads where the whole domain cannot be reached by single steps is left out first, so that every branch
  // of the search ends in a path and the search takes time in proportion to the paths it lists.
  const reaches = reachesFull(next, full, compiled.count);
  const paths: string[][] = [];
  const order: string[] = [];
  const walk = (s: number): void => {
    if (s === full) {
      if (paths.length === maxPaths) {
        throw new DataError(
          `The structure has more than ${maxPaths} learning paths, which grow as a factorial. Use structureBase for a summary of the structure, or raise maxPaths.`,
        );
      }
      paths.push([...order]);
      return;
    }
    for (const [element, t] of next[s] ?? []) {
      if (!reaches[t]) continue;
      order.push(element);
      walk(t);
      order.pop();
    }
  };
  if (reaches[empty]) walk(empty);
  return paths;
}

/**
 * Returns whether a structure is well-graded: from any state to any other there is a path of states that each differ
 * from the one before by one element, as many steps as the two states differ by. It compares every pair of states, so
 * takes time quadratic in their number.
 *
 * @remarks
 * It is `kstructure_is_wellgraded` of the kst package. A well-graded structure is the case in which every state has a
 * state one element away, toward any other.
 *
 * @param structure - The knowledge structure.
 * @returns Whether the structure is well-graded.
 * @throws `DataError` when the structure is malformed.
 * @see {@link https://doi.org/10.1007/978-3-642-01039-2 | Falmagne and Doignon (2011), chapter 2}
 */
export function isWellGraded(structure: KnowledgeStructure): boolean {
  const compiled = compile(structure);
  const adjacent = new Uint32Array(compiled.words);
  for (let s = 0; s < compiled.count; s++) {
    adjacentElements(compiled, s, adjacent);
    if (
      !Array.from({ length: compiled.count }, (_, t) => t).every(
        (t) => t === s || stepsToward(compiled, s, t, adjacent),
      )
    ) {
      return false;
    }
  }
  return true;
}

/**
 * Returns whether a tight path from state `s` to state `t` can start: the first step flips an element in which they
 * differ and that gives a state, one of the `adjacent` elements of `s`.
 */
function stepsToward(compiled: CompiledStructure, s: number, t: number, adjacent: Uint32Array): boolean {
  const { words, bits } = compiled;
  for (let w = 0; w < words; w++) {
    if ((((bits[s * words + w] ?? 0) ^ (bits[t * words + w] ?? 0)) & (adjacent[w] ?? 0)) !== 0) return true;
  }
  return false;
}

/** Writes into `out` the elements whose flip turns state `s` into a state of the structure. */
function adjacentElements(compiled: CompiledStructure, s: number, out: Uint32Array): void {
  const { words, bits, stateOf } = compiled;
  out.fill(0);
  const probe = bits.slice(s * words, (s + 1) * words);
  for (let e = 0; e < compiled.domain.length; e++) {
    flipElement(probe, e);
    if (stateOf.has(keyOf(probe, 0, words))) flipElement(out, e);
    flipElement(probe, e);
  }
}

/** Returns, for each state, the states that add one element to it, as pairs of the element and the state. */
function steps(compiled: CompiledStructure): (readonly [string, number])[][] {
  const { words, bits, stateOf } = compiled;
  return Array.from({ length: compiled.count }, (_, s) => {
    const found: [string, number][] = [];
    const probe = bits.slice(s * words, (s + 1) * words);
    compiled.domain.forEach((element, e) => {
      if (holds(compiled, s, e)) return;
      flipElement(probe, e);
      const t = stateOf.get(keyOf(probe, 0, words));
      if (t !== undefined) found.push([element, t]);
      flipElement(probe, e);
    });
    return found;
  });
}

/** Returns whether the whole domain can be reached from each state by single steps. */
function reachesFull(
  next: readonly (readonly (readonly [string, number])[])[],
  full: number,
  count: number,
): boolean[] {
  // A step adds an element, so the search is as deep as the domain is large and each state is decided once.
  const known = new Map<number, boolean>([[full, true]]);
  const decide = (s: number): boolean => {
    const decided = known.get(s);
    if (decided !== undefined) return decided;
    const reaches = (next[s] ?? []).some(([, t]) => decide(t));
    known.set(s, reaches);
    return reaches;
  };
  return Array.from({ length: count }, (_, s) => decide(s));
}

function fullKey(compiled: CompiledStructure): string {
  const whole = new Uint32Array(compiled.words);
  for (let e = 0; e < compiled.domain.length; e++) addElement(whole, 0, e);
  return keyOf(whole, 0, compiled.words);
}

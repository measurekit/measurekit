import { checkOptions, DataError, describe, type FieldNames, isRecord } from "@measurekit/core";
import { checkMaxStates, type StateLimitOptions, tooManyStates } from "./build.js";
import { checkElementCount } from "./element-limit.js";
import { compareStates, type KnowledgeStructure, seal } from "./structure.js";

/** Options of {@link surmiseStructure}. */
export type SurmiseStructureOptions = StateLimitOptions;

const SURMISE_STRUCTURE_OPTIONS: FieldNames<SurmiseStructureOptions> = { maxStates: true };

/**
 * A quasi-order of elements as `iita` returns it: each element's prerequisites, the elements that every learner who
 * masters it is expected to master.
 */
export type Prerequisites = Readonly<Record<string, readonly string[]>>;

/**
 * Returns the quasi-ordinal knowledge space of a surmise relation: every set of elements that holds the prerequisites
 * of each of its elements. States are ordered by size, then by the domain's order. The structure is frozen, and the
 * functions of this module trust it instead of checking it again.
 *
 * @param prerequisites - Each element's prerequisites, by element: every element of the domain is a key, with `[]` when
 *   it has none. Prerequisites of prerequisites are implied; elements that require each other are learned together.
 * @param options - The most states to build.
 * @returns The quasi-ordinal knowledge space.
 * @throws `DataError` when an element is the empty string, there are more than 1,000 elements, a prerequisite is not a
 * key, or the space has more states than `maxStates`.
 *
 * @example
 * ```ts
 * import { fringes, isLearningSpace, surmiseStructure } from "@measurekit/knowledge-tracing/kst";
 *
 * const prerequisites = { add: [], subtract: ["add"], multiply: ["add"], divide: ["subtract", "multiply"] };
 * const structure = surmiseStructure(prerequisites);
 * structure.states;
 * // [[], ["add"], ["add", "subtract"], ["add", "multiply"], ["add", "subtract", "multiply"],
 * //  ["add", "subtract", "multiply", "divide"]]
 * isLearningSpace(structure); // true
 * fringes(structure, ["add", "subtract"]); // { inner: ["subtract"], outer: ["multiply"] }
 * ```
 */
export function surmiseStructure(
  prerequisites: Prerequisites,
  options: SurmiseStructureOptions = {},
): KnowledgeStructure {
  checkOptions(options, SURMISE_STRUCTURE_OPTIONS);
  const maxStates = checkMaxStates(options.maxStates);
  if (!isRecord(prerequisites)) {
    throw new DataError(
      `Prerequisites must be an object of element lists by element, found ${describe(prerequisites)}.`,
    );
  }
  const domain = Object.keys(prerequisites);
  if (domain.length === 0) throw new DataError("Prerequisites need at least one element.");
  checkElementCount(domain.length, "A surmise relation", "elements");
  const below = transitiveClosure(prerequisites, domain);
  const m = domain.length;
  const states = downsets(below, m, maxStates).map((state) => state.map((e) => domain[e] ?? ""));
  return seal({ domain, states });
}

/**
 * Returns `below`, an m × m matrix with 1 at j × m + i when i is a prerequisite of j, directly or through others; every
 * element is its own.
 *
 * @remarks
 * Each element's row is the set its prerequisites reach, found by a search over the direct ones, so that the time is m
 * times the elements and prerequisites, not the cube of m that closing the matrix in place takes for a long cycle.
 *
 * @throws `DataError` when an element's prerequisites are not a list, or one is not in `domain`.
 */
export function transitiveClosure(prerequisites: Prerequisites, domain: readonly string[]): Uint8Array {
  const m = domain.length;
  const index = new Map(domain.map((element, e) => [element, e]));
  const direct = domain.map((element) => {
    const listed = prerequisites[element];
    if (!Array.isArray(listed)) throw new DataError(`The prerequisites of "${element}" must be a list of elements.`);
    return listed.map((required) => {
      const i = index.get(required);
      if (i === undefined) {
        throw new DataError(`"${element}" requires "${required}", which is not a key of the prerequisites.`);
      }
      return i;
    });
  });
  const below = new Uint8Array(m * m);
  const pending: number[] = [];
  for (let j = 0; j < m; j++) {
    below[j * m + j] = 1;
    pending.push(j);
    for (let k = pending.pop(); k !== undefined; k = pending.pop()) {
      for (const i of direct[k] ?? []) {
        if (below[j * m + i] === 1) continue;
        below[j * m + i] = 1;
        pending.push(i);
      }
    }
  }
  return below;
}

/**
 * Returns the sets closed under `below`, ordered by size, then by the domain's order. Elements that require each other
 * form a class, decided together; classes are decided with prerequisites first, and a class can join only when the
 * classes below it have, so every branch ends in a state, each once.
 */
function downsets(below: Uint8Array, m: number, maxStates: number): number[][] {
  const classes: number[][] = [];
  const classOf = new Int32Array(m).fill(-1);
  for (let e = 0; e < m; e++) {
    if (classOf[e] !== -1) continue;
    const members = [];
    for (let i = e; i < m; i++) if (below[e * m + i] === 1 && below[i * m + e] === 1) members.push(i);
    for (const i of members) classOf[i] = classes.length;
    classes.push(members);
  }
  const size = (c: number): number => countBelow(below, m, classes[c]?.[0] ?? 0);
  const order = classes.map((_, c) => c).sort((x, y) => size(x) - size(y) || x - y);
  const requires = classes.map((members, c) => {
    const representative = members[0] ?? 0;
    const required = new Set<number>();
    for (let i = 0; i < m; i++) if (below[representative * m + i] === 1) required.add(classOf[i] ?? 0);
    required.delete(c);
    return [...required];
  });
  const chosen = new Uint8Array(classes.length);
  const states: number[][] = [];
  const visit = (position: number): void => {
    if (position === order.length) {
      if (states.length === maxStates) {
        throw tooManyStates(maxStates);
      }
      const state: number[] = [];
      chosen.forEach((flag, c) => {
        if (flag === 1) for (const member of classes[c] ?? []) state.push(member);
      });
      states.push(state.sort((x, y) => x - y));
      return;
    }
    const c = order[position] ?? 0;
    visit(position + 1);
    if ((requires[c] ?? []).every((r) => chosen[r] === 1)) {
      chosen[c] = 1;
      visit(position + 1);
      chosen[c] = 0;
    }
  };
  visit(0);
  return states.sort(compareStates);
}

function countBelow(below: Uint8Array, m: number, e: number): number {
  let count = 0;
  for (let i = 0; i < m; i++) count += below[e * m + i] ?? 0;
  return count;
}

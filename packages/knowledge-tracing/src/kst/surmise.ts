import { checkOptions, DataError, describe, isRecord } from "@measurekit/core";
import { checkMaxStates, type StateLimitOptions, tooManyStates } from "./build.js";
import { compareStates, type KnowledgeStructure, seal } from "./structure.js";

/** Options of {@link surmiseStructure}. */
export type SurmiseStructureOptions = StateLimitOptions;

/**
 * Returns the quasi-ordinal knowledge space of a surmise relation: every set of elements that holds the prerequisites
 * of each of its elements. States are ordered by size, then by the domain's order. The structure is frozen, and the
 * functions of this module trust it instead of checking it again.
 *
 * @param prerequisites - Each element's prerequisites, by element: every element of the domain is a key, with `[]`
 * when it has none. Prerequisites of prerequisites are implied; elements that require each other are learned together.
 * @throws `DataError` when an element is the empty string, a prerequisite is not a key, or the space has more states
 * than `maxStates`.
 */
export function surmiseStructure(
  prerequisites: Readonly<Record<string, readonly string[]>>,
  options: SurmiseStructureOptions = {},
): KnowledgeStructure {
  checkOptions(options);
  const maxStates = checkMaxStates(options.maxStates);
  if (!isRecord(prerequisites)) {
    throw new DataError(
      `Prerequisites must be an object of element lists by element, found ${describe(prerequisites)}.`,
    );
  }
  const domain = Object.keys(prerequisites);
  if (domain.length === 0) throw new DataError("Prerequisites need at least one element.");
  const below = transitiveClosure(prerequisites, domain);
  const m = domain.length;
  const states = downsets(below, m, maxStates).map((state) => state.map((e) => domain[e] ?? ""));
  return seal({ domain, states });
}

/**
 * Returns `below`, an m × m matrix with 1 at j × m + i when i is a prerequisite of j, directly or through others; every
 * element is its own.
 */
function transitiveClosure(
  prerequisites: Readonly<Record<string, readonly string[]>>,
  domain: readonly string[],
): Uint8Array {
  const m = domain.length;
  const index = new Map(domain.map((element, e) => [element, e]));
  const below = new Uint8Array(m * m);
  domain.forEach((element, j) => {
    below[j * m + j] = 1;
    const listed = prerequisites[element];
    if (!Array.isArray(listed)) throw new DataError(`The prerequisites of "${element}" must be a list of elements.`);
    for (const required of listed) {
      const i = index.get(required);
      if (i === undefined) {
        throw new DataError(`"${element}" requires "${required}", which is not a key of the prerequisites.`);
      }
      below[j * m + i] = 1;
    }
  });
  for (let k = 0; k < m; k++) {
    for (let j = 0; j < m; j++) {
      if (below[j * m + k] !== 1) continue;
      for (let i = 0; i < m; i++) if (below[k * m + i] === 1) below[j * m + i] = 1;
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
        if (flag === 1) state.push(...(classes[c] ?? []));
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

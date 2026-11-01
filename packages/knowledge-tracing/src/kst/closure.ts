import { checkArray, checkOptions, DataError, describe } from "@measurekit/core";
import { buildStructure, checkMaxStates, positionsOf, type StateLimitOptions, tooManyStates } from "./build.js";
import { addElement, compile, type KnowledgeStructure, keyOf } from "./structure.js";

/**
 * Returns the smallest knowledge space that holds the states given: their unions, the empty state and the whole
 * domain. States are ordered by size, then by the domain's order, and the structure is frozen.
 *
 * @remarks
 * The closure grows one generator at a time, each union once, so that the time is that of the states built times the
 * states given. It is `kspace` of the kst package.
 *
 * @param domain - The elements, items or skills, as a non-empty list of distinct ids.
 * @param states - The states to start from, each a list of elements of the domain; repeats are taken once.
 * @throws `DataError` when the domain is not a non-empty list of distinct ids, a state is not a list of distinct
 * elements of the domain, or the space has more states than `maxStates`.
 * @see {@link https://doi.org/10.1007/978-3-642-01039-2 | Falmagne and Doignon (2011), chapter 2}
 */
export function unionClosure(
  domain: readonly string[],
  states: readonly (readonly string[])[],
  options: StateLimitOptions = {},
): KnowledgeStructure {
  checkOptions(options);
  const maxStates = checkMaxStates(options.maxStates);
  // The structure of the empty state and the whole domain checks the domain, and gives the positions and the words.
  const { index, words, bits } = compile({ domain, states: [[], domain] });
  checkArray(states, "The states must be a list of lists of elements");
  const generators = states.map((state, s) => {
    checkArray(state, `State ${s + 1} must be a list of elements`);
    const generator = new Uint32Array(words);
    for (const element of state) {
      const e = index.get(element);
      if (e === undefined) throw new DataError(`State ${s + 1} has ${describe(element)}, which is not in the domain.`);
      if (addElement(generator, 0, e)) throw new DataError(`State ${s + 1} lists "${element}" twice.`);
    }
    return generator;
  });
  const whole = bits.slice(words, 2 * words);
  const closed = closeUnderUnion([...generators, whole], words, maxStates);
  if (closed === null) throw tooManyStates(maxStates);
  return buildStructure(
    domain,
    closed.map((state) => positionsOf(state, domain.length)),
  );
}

/**
 * Returns the unions of every subset of `generators`, the empty state included, or `null` when there are more than
 * `limit` of them. Each new union is found by joining a generator to the unions found before it.
 */
export function closeUnderUnion(
  generators: readonly Uint32Array[],
  words: number,
  limit: number,
): Uint32Array[] | null {
  const empty = new Uint32Array(words);
  const closed = [empty];
  const keys = new Set([keyOf(empty, 0, words)]);
  for (const generator of generators) {
    const known = closed.length;
    for (let i = 0; i < known; i++) {
      const union = new Uint32Array(words);
      const base = closed[i];
      for (let w = 0; w < words; w++) union[w] = (base?.[w] ?? 0) | (generator[w] ?? 0);
      const key = keyOf(union, 0, words);
      if (keys.has(key)) continue;
      if (closed.length === limit) return null;
      keys.add(key);
      closed.push(union);
    }
  }
  return closed;
}

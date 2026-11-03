import { DataError } from "@measurekit/core";
import type { EventLog } from "../data/internal.js";
import { popcount } from "./build.js";
import { counterexamples, responsePatterns } from "./patterns.js";
import { type CompiledStructure, compile, holds, type KnowledgeStructure } from "./structure.js";

/**
 * The most elements for which the distance agreement is computed: it takes a table of every pattern of responses, which
 * has 2 ^ elements entries.
 */
const MAX_AGREEMENT_ELEMENTS = 20;

/** The validity of a knowledge structure against learners' responses to every item. */
export interface StructureValidity {
  /**
   * Goodman and Kruskal's gamma over the pairs the structure orders: (concordant - discordant) / (concordant +
   * discordant); `null` without any.
   */
  readonly gamma: number | null;
  /** Learner and pair counts in which the pair's prerequisite is solved and the other item is not. */
  readonly concordant: number;
  /** Learner and pair counts in which the other item is solved and the pair's prerequisite is not. */
  readonly discordant: number;
  /** The share of learners and pairs that are discordant: discordant / (learners × pairs); `null` without pairs. */
  readonly violationalCoefficient: number | null;
  /** The mean distance from a learner's responses to the nearest state: the number of items to change. */
  readonly discrepancyIndex: number;
  /** The number of learners at each distance to their nearest state, by distance. */
  readonly distances: Readonly<Record<string, number>>;
  /**
   * The discrepancy index over the mean distance from every pattern of responses to the nearest state; `null` above 20
   * elements, or when every pattern is a state.
   */
  readonly distanceAgreement: number | null;
  /** The share of learners who solved each item, by item. */
  readonly solutionRates: Readonly<Record<string, number>>;
}

/**
 * Measures how well a knowledge structure agrees with learners' responses to its items.
 *
 * @remarks
 * These are the validity measures of `kvalidate` in the kst package, `percent` as a share. The structure's surmise
 * relation orders the pair i, j when every state that holds j holds i: a learner who solves i and not j agrees with
 * it, and one who solves j and not i violates it. The discrepancy index is the mean number of items by which a
 * learner's responses differ from the nearest state, and the distance agreement compares it with that of responses
 * chosen at random. Each learner's first outcome at each item is used; every learner must have answered every item of
 * the structure's domain, with outcomes 0 or 1. The elements of the structure are the items of the log.
 *
 * @param structure - The knowledge structure over items.
 * @param log - The log whose first attempts give each learner's response pattern.
 * @returns The indices of agreement, such as γ and the discrepancy index, and the distances of the patterns to the
 *   structure.
 * @throws `DataError` when the structure or the log is malformed, a learner has not answered every item, an outcome is
 * not 0 or 1, the log has more than 1,000 items, or its items are not the structure's domain.
 * @see {@link https://doi.org/10.1007/978-3-642-01039-2 | Falmagne and Doignon (2011), chapter 2}
 * @see {@link https://doi.org/10.18637/jss.v037.i02 | Ünlü and Sargin (2010)}
 */
export function structureValidity(structure: KnowledgeStructure, log: EventLog): StructureValidity {
  const compiled = compile(structure);
  const { domain } = compiled;
  const patterns = domainPatterns(compiled, log);
  const m = domain.length;
  const learners = patterns.length;
  const b = counterexamples(patterns, m);
  let pairs = 0;
  let concordant = 0;
  let discordant = 0;
  forEachPrerequisitePair(compiled, (i, j) => {
    pairs++;
    concordant += b[j * m + i] ?? 0;
    discordant += b[i * m + j] ?? 0;
  });
  const distances = distancesToNearest(compiled, patterns);
  const discrepancyIndex = distances.total / learners;
  const potential = m <= MAX_AGREEMENT_ELEMENTS ? meanDistanceOfAllPatterns(compiled) : null;
  return {
    gamma: concordant + discordant === 0 ? null : (concordant - discordant) / (concordant + discordant),
    concordant,
    discordant,
    violationalCoefficient: pairs === 0 ? null : discordant / (learners * pairs),
    discrepancyIndex,
    distances: distances.counts,
    distanceAgreement: potential === null || potential === 0 ? null : discrepancyIndex / potential,
    solutionRates: Object.fromEntries(
      domain.map((element, e) => [element, patterns.reduce((sum, pattern) => sum + (pattern[e] ?? 0), 0) / learners]),
    ),
  };
}

/** Returns each learner's responses in the order of the structure's domain, checking that the items are its elements. */
function domainPatterns(compiled: CompiledStructure, log: EventLog): number[][] {
  const { items, patterns } = responsePatterns(log, "The validity of a structure");
  for (const item of items) {
    if (!compiled.index.has(item)) {
      throw new DataError(`The log has item "${item}", which is not in the structure's domain.`);
    }
  }
  const at = compiled.domain.map((element) => items.indexOf(element));
  const absent = compiled.domain[at.indexOf(-1)];
  if (absent !== undefined) {
    throw new DataError(
      `The structure's element "${absent}" is not an item of the log: no learner has an outcome there.`,
    );
  }
  return patterns.map((pattern) => at.map((position) => pattern[position] ?? 0));
}

/**
 * Passes `use` each pair i, j, i ≠ j, of the surmise relation, in order: every state that holds j holds i. The pairs are
 * counted as they are found, as a list of them would hold up to m² of them.
 */
function forEachPrerequisitePair(compiled: CompiledStructure, use: (i: number, j: number) => void): void {
  const m = compiled.domain.length;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      if (i === j) continue;
      let implied = true;
      for (let s = 0; implied && s < compiled.count; s++) implied = !holds(compiled, s, j) || holds(compiled, s, i);
      if (implied) use(i, j);
    }
  }
}

/** Returns the number of learners at each distance to their nearest state, and the sum of the distances. */
function distancesToNearest(
  compiled: CompiledStructure,
  patterns: readonly (readonly number[])[],
): { counts: Record<string, number>; total: number } {
  const { words, bits } = compiled;
  const nearest = new Map<string, number>();
  const counts = new Map<number, number>();
  let total = 0;
  const response = new Uint32Array(words);
  for (const pattern of patterns) {
    const key = pattern.join("");
    let distance = nearest.get(key);
    if (distance === undefined) {
      response.fill(0);
      pattern.forEach((outcome, e) => {
        if (outcome === 1) response[e >>> 5] = (response[e >>> 5] ?? 0) | (1 << (e & 31));
      });
      distance = nearestDistance(bits, response, words);
      nearest.set(key, distance);
    }
    counts.set(distance, (counts.get(distance) ?? 0) + 1);
    total += distance;
  }
  return { counts: Object.fromEntries([...counts].sort(([x], [y]) => x - y)), total };
}

/** Returns the fewest elements in which a response differs from a state. */
function nearestDistance(bits: Uint32Array, response: Uint32Array, words: number): number {
  let least = Number.POSITIVE_INFINITY;
  for (let at = 0; at < bits.length; at += words) {
    let apart = 0;
    for (let w = 0; w < words; w++) apart += popcount((bits[at + w] ?? 0) ^ (response[w] ?? 0));
    least = Math.min(least, apart);
  }
  return least;
}

/**
 * Returns the mean, over the 2 ^ m patterns of responses, of the distance to the nearest state. The table of distances
 * takes one pass per element: after the pass for an element, each pattern knows its distance using only changes in the
 * elements so far.
 */
function meanDistanceOfAllPatterns(compiled: CompiledStructure): number {
  const m = compiled.domain.length;
  const size = 2 ** m;
  const unreached = 255;
  const distance = new Uint8Array(size).fill(unreached);
  // Elements are at most 20, so a state is the one word of its bits.
  for (let s = 0; s < compiled.count; s++) distance[compiled.bits[s * compiled.words] ?? 0] = 0;
  for (let e = 0; e < m; e++) {
    const bit = 1 << e;
    for (let x = 0; x < size; x++) {
      const via = (distance[x ^ bit] ?? unreached) + 1;
      if (via < (distance[x] ?? unreached)) distance[x] = via;
    }
  }
  let total = 0;
  for (let x = 0; x < size; x++) total += distance[x] ?? 0;
  return total / size;
}

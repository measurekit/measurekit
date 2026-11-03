import { checkChoice, checkOptions, DataError } from "@measurekit/core";
import type { EventLog } from "../data/internal.js";
import { counterexamples, responsePatterns } from "./patterns.js";
import type { Prerequisites } from "./surmise.js";

/**
 * The fit measure of inductive item tree analysis.
 *
 * @see {@link https://doi.org/10.1016/j.mathsocsci.2009.06.001 | Sargin and Ünlü (2009)}
 * @see {@link https://doi.org/10.18637/jss.v037.i02 | Ünlü and Sargin (2010)}
 */
export type IitaVersion = "minimized" | "corrected" | "original";

const DEFAULT_VERSION = "minimized";

/**
 * The most cells that the candidates of IITA hold: a cell for each prerequisite of each candidate, 8 bytes in its list,
 * and {@link CELLS_PER_ITEM} for each item of each candidate, whose list and key take about 64. Each candidate lists the
 * prerequisites of every item, and 24 million cells take about 260 MB.
 */
const MAX_CANDIDATE_CELLS = 24_000_000;
/** The cells that an item's list of prerequisites takes in a candidate besides its prerequisites: its array and key. */
const CELLS_PER_ITEM = 8;

/** Options of {@link iita}. */
export interface IitaOptions {
  /**
   * `"minimized"`, the minimized corrected algorithm, with the error rate that minimizes the fit measure;
   * `"corrected"`, with Sargin and Ünlü's corrected expected counterexamples; or `"original"`, Schrepp's.
   *
   * @defaultValue `"minimized"`
   * @see {@link https://doi.org/10.23668/psycharchives.12781 | Schrepp (2003)}
   */
  readonly version?: IitaVersion;
}

/** A quasi-order of the selection set, with its fit. */
export interface IitaCandidate {
  /** Each item's prerequisites, by item: the items every learner who masters it is expected to master. */
  readonly prerequisites: Prerequisites;
  /** The fit measure: the mean squared difference of observed and expected counterexamples. */
  readonly diff: number;
  /** The error rate γ the expected counterexamples use. */
  readonly errorRate: number;
}

/** The result of inductive item tree analysis: the candidate of least `diff`, and every candidate. */
export interface IitaResult extends IitaCandidate {
  /** The quasi-orders of the selection set, in the order they were generated, with their fit. */
  readonly candidates: readonly IitaCandidate[];
}

/**
 * Derives a surmise relation between items from learners' responses by inductive item tree analysis.
 *
 * @remarks
 * It generates quasi-orders that admit more and more counterexamples, b_ij learners who fail i but solve j, and returns
 * the one whose counterexamples best match those expected with one error rate. Each learner's first outcome at each
 * item is used; every learner must have answered every item of the log, with outcomes 0 or 1. Items are in the order of
 * their first event with an outcome, and events without one are left out. Pass the prerequisites to `surmiseStructure` for the knowledge space.
 *
 * @param log - The log whose first attempts give each learner's response pattern.
 * @param options - The version of the algorithm.
 * @returns The candidate relations with their fit, the prerequisites of the best, its diff and its error rate.
 * @throws `DataError` when the log is malformed, a learner has not answered every item, an outcome is not 0 or 1, there
 * are fewer than 2 or more than 1,000 items with an outcome, an item is solved by no learner, or the candidates would
 * hold more than 24 million cells, a cell for each prerequisite and 8 for each item.
 */
export function iita(log: EventLog, options: IitaOptions = {}): IitaResult {
  checkOptions(options);
  const { version = DEFAULT_VERSION } = options;
  checkChoice(version, "The IITA version", ["minimized", "corrected", "original"]);
  const { items, patterns } = responsePatterns(log, "IITA");
  if (items.length < 2) throw new DataError("IITA needs at least 2 items.");
  const data = counterexampleData(items, patterns);
  const candidates: IitaCandidate[] = [];
  let cells = 0;
  forEachCandidate(data.b, data.m, (relation, implications) => {
    cells += CELLS_PER_ITEM * data.m + implications;
    if (cells > MAX_CANDIDATE_CELLS) {
      throw new DataError(
        `The candidates of IITA would hold more than ${MAX_CANDIDATE_CELLS} cells, a cell for each prerequisite and ${CELLS_PER_ITEM} for each item of each candidate. Analyze fewer items, such as those of a unit at a time.`,
      );
    }
    candidates.push({ prerequisites: prerequisitesOf(relation, items), ...fitRelation(relation, data, version) });
  });
  const best = candidates.reduce((x, y) => (y.diff < x.diff ? y : x));
  return { ...best, candidates };
}

/** The counterexamples b and solution rates p of m items answered by n learners. */
export interface Counterexamples {
  readonly b: Int32Array;
  readonly p: readonly number[];
  readonly m: number;
  readonly n: number;
}

/**
 * Returns the counterexamples and solution rates of the response patterns of `items`.
 *
 * @throws `DataError` when an item is solved by no learner.
 */
export function counterexampleData(
  items: readonly string[],
  patterns: readonly (readonly number[])[],
): Counterexamples {
  const n = patterns.length;
  const p = items.map((item, i) => {
    const share = patterns.reduce((sum, pattern) => sum + (pattern[i] ?? 0), 0) / n;
    if (share === 0) throw new DataError(`Item "${item}" is solved by no learner, so IITA cannot weigh it.`);
    return share;
  });
  return { b: counterexamples(patterns, items.length), p, m: items.length, n };
}

/**
 * Passes `use` each relation of the selection set, in order, with its number of implications: ⊑_0 holds i ⊑ j when
 * b_ij = 0; at each further count of counterexamples, the pairs with at most that many join, except those that take
 * part in an intransitivity. Pairs are checked in order, i then j, each against the pairs not yet excluded, and passes
 * repeat until none is; this order resolves conflicts as DAKS does. Relations without implications, or equal to the
 * previous one, are left out. A relation is an m × m matrix: 1 at i × m + j when i ⊑ j, i a prerequisite of j; the
 * diagonal is 1. Only the current relation is kept, so that the set costs the candidates made of it, not m² for each.
 *
 * @see {@link https://doi.org/10.23668/psycharchives.12781 | Schrepp (2003)}
 */
function forEachCandidate(b: Int32Array, m: number, use: (relation: Uint8Array, implications: number) => void): void {
  let relation: Uint8Array = new Uint8Array(m * m);
  for (let k = 0; k < relation.length; k++) relation[k] = b[k] === 0 ? 1 : 0;
  const take = (candidate: Uint8Array): void => {
    const implications = implicationCount(candidate, m);
    if (implications > 0) use(candidate, implications);
  };
  take(relation);
  for (const level of [...new Set(b)].filter((value) => value > 0).sort((x, y) => x - y)) {
    const next = extend(relation, b, m, level);
    if (next === null) continue;
    relation = next;
    take(relation);
  }
}

/** Returns the number of pairs i ⊑ j with i ≠ j that a relation holds. */
function implicationCount(relation: Uint8Array, m: number): number {
  let count = 0;
  for (let k = 0; k < relation.length; k++) if (relation[k] === 1 && k % (m + 1) !== 0) count++;
  return count;
}

/**
 * Returns the relation with the pairs of at most `level` counterexamples that keep it transitive, or `null` if none.
 */
function extend(relation: Uint8Array, b: Int32Array, m: number, level: number): Uint8Array | null {
  const added = pairsUpTo(relation, b, level);
  const union = relation.slice();
  for (const k of added) union[k] = 1;
  for (let changed = true; changed; ) {
    changed = false;
    for (const k of added) {
      if (union[k] === 1 && intransitive(union, m, Math.floor(k / m), k % m)) {
        union[k] = 0;
        changed = true;
      }
    }
  }
  return added.some((k) => union[k] === 1) ? union : null;
}

/** Returns the cells of the pairs outside a relation with at most `level` counterexamples, in order. */
function pairsUpTo(relation: Uint8Array, b: Int32Array, level: number): Int32Array {
  let count = 0;
  for (let k = 0; k < b.length; k++) if (relation[k] === 0 && (b[k] ?? 0) <= level) count++;
  const pairs = new Int32Array(count);
  for (let k = 0, at = 0; k < b.length; k++) if (relation[k] === 0 && (b[k] ?? 0) <= level) pairs[at++] = k;
  return pairs;
}

/**
 * Returns whether the pair i ⊑ j takes part in an intransitivity of the relation: k with j ⊑ k but not i ⊑ k, or
 * k ⊑ i but not k ⊑ j.
 */
function intransitive(relation: Uint8Array, m: number, i: number, j: number): boolean {
  for (let k = 0; k < m; k++) {
    if (relation[j * m + k] === 1 && relation[i * m + k] !== 1) return true;
    if (relation[k * m + i] === 1 && relation[k * m + j] !== 1) return true;
  }
  return false;
}

/** Returns diff and γ of a relation for a version of IITA. */
export function fitRelation(
  relation: Uint8Array,
  data: Counterexamples,
  version: IitaVersion,
): { diff: number; errorRate: number } {
  const { b, m } = data;
  const gamma = version === "minimized" ? minimizingRate(relation, data) : meanRate(relation, data);
  let sum = 0;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      if (i !== j) sum += ((b[i * m + j] ?? 0) - expected(relation, data, version, gamma, i, j)) ** 2;
    }
  }
  return { diff: sum / (m * (m - 1)), errorRate: gamma };
}

/** Returns γ of the original and corrected versions: the mean of b_ij / (p_j n) over the implications i ⊑ j, i ≠ j. */
function meanRate(relation: Uint8Array, { b, p, m, n }: Counterexamples): number {
  let pairs = 0;
  let sum = 0;
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < m; j++) {
      if (i === j || relation[i * m + j] !== 1) continue;
      pairs += 1;
      sum += (b[i * m + j] ?? 0) / ((p[j] ?? 0) * n);
    }
  }
  return sum / pairs;
}

/**
 * Returns b*_ij, the expected counterexamples: γ p_j n for an implication i ⊑ j; otherwise (1 - p_i) p_j n (1 - γ) in
 * the original version, and in the corrected ones (p_j - p_i + γ p_i) n when j ⊑ i and (1 - p_i) p_j n when not.
 */
function expected(
  relation: Uint8Array,
  { p, m, n }: Counterexamples,
  version: IitaVersion,
  gamma: number,
  i: number,
  j: number,
): number {
  const pi = p[i] ?? 0;
  const pj = p[j] ?? 0;
  if (relation[i * m + j] === 1) return gamma * pj * n;
  if (version === "original") return (1 - pi) * pj * n * (1 - gamma);
  return relation[j * m + i] === 1 ? (pj - pi + gamma * pi) * n : (1 - pi) * pj * n;
}

/**
 * Returns the γ that minimizes the corrected diff: -(x1 + x2) / (x3 + x4).
 *
 * @see {@link https://doi.org/10.1016/j.mathsocsci.2009.06.001 | Sargin and Ünlü (2009)}
 */
function minimizingRate(relation: Uint8Array, data: Counterexamples): number {
  const x = new Float64Array(4);
  for (let i = 0; i < data.m; i++) {
    for (let j = 0; j < data.m; j++) if (i !== j) addMinimizingTerms(relation, data, i, j, x);
  }
  const [x1 = 0, x2 = 0, x3 = 0, x4 = 0] = x;
  return x3 + x4 === 0 ? 0 : -(x1 + x2) / (x3 + x4);
}

/** Adds the pair's terms to x1 to x4: x2 and x4 for an implication i ⊑ j, x1 and x3 for j ⊑ i alone. */
function addMinimizingTerms(
  relation: Uint8Array,
  { b, p, m, n }: Counterexamples,
  i: number,
  j: number,
  x: Float64Array,
): void {
  const pi = p[i] ?? 0;
  const pj = p[j] ?? 0;
  const bij = b[i * m + j] ?? 0;
  if (relation[i * m + j] === 1) {
    x[1] = (x[1] ?? 0) - 2 * bij * pj * n;
    x[3] = (x[3] ?? 0) + 2 * pj * pj * n * n;
  } else if (relation[j * m + i] === 1) {
    x[0] = (x[0] ?? 0) - 2 * bij * pi * n + 2 * pi * pj * n * n - 2 * pi * pi * n * n;
    x[2] = (x[2] ?? 0) + 2 * pi * pi * n * n;
  }
}

/** Returns each item's prerequisites in a relation, in lists of their exact length, which take 8 bytes an entry. */
function prerequisitesOf(relation: Uint8Array, items: readonly string[]): Record<string, string[]> {
  const m = items.length;
  return Object.fromEntries(
    items.map((item, j) => {
      let count = 0;
      for (let i = 0; i < m; i++) if (i !== j && relation[i * m + j] === 1) count++;
      const prerequisites = new Array<string>(count);
      for (let i = 0, at = 0; i < m; i++)
        if (i !== j && relation[i * m + j] === 1) prerequisites[at++] = items[i] ?? "";
      return [item, prerequisites];
    }),
  );
}

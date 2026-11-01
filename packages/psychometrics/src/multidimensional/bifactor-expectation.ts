import type { GroupPatterns, PatternTable } from "./bifactor-patterns.js";
import { createWalk, type Walk } from "./bifactor-walk.js";
import type { ItemState } from "./calibration-setup.js";
import type { Grid } from "./grid.js";

/** The nodes of one group of items and its items, as the E-step and M-step see them. */
export interface GroupNodes<TParameters> {
  /** The distinct responses to the items of the group. */
  readonly patterns: GroupPatterns;
  /** The nodes of the group, the primary nodes times the specific ones, the specific varying fastest. */
  readonly grid: Grid;
  /** The items of the group, whose `counts` are over its nodes. */
  readonly items: readonly ItemState<TParameters>[];
  /** The weights of the specific nodes: one 1 for the items without a specific dimension. */
  readonly weights: Float64Array;
}

/** The E-step of a bifactor model: dimension reduction over the groups of items. */
export interface BifactorExpectation {
  /**
   * Evaluates every pattern's posterior over the primary nodes under the weights and the items' current parameters;
   * writes the items' expected counts at the nodes of their groups and the expected number of persons at each primary
   * node; and returns the marginal log-likelihood.
   */
  run(): number;
  /** The expected number of persons at each primary node, as of the last run. */
  readonly nodeCounts: Float64Array;
  /**
   * Calls `visit` with each pattern's number of persons, its posterior over the primary nodes (which sums to 1) and its
   * sub-pattern in each group, at the parameters of the last run. The arrays are reused between calls.
   */
  forEachPattern(visit: (frequency: number, posterior: Float64Array, subPatterns: Int32Array) => void): void;
  /**
   * Calls `visit` with each sub-pattern of each group that answers an item: the group's index, the sub-pattern, the
   * distribution of the group's nodes given the primary node (which sums to 1 over the specific nodes of each primary
   * node) and the expected number of persons at each primary node. Both arrays are reused between calls.
   */
  forEachSubPattern(
    visit: (group: number, subPattern: number, conditional: Float64Array, mass: Float64Array) => void,
  ): void;
}

/** A group during the E-step: its probability tables, the likelihood of each sub-pattern and its expected counts. */
interface GroupState<TParameters> {
  readonly nodes: GroupNodes<TParameters>;
  readonly patterns: GroupPatterns;
  /** The probabilities of each item by category, then node. */
  readonly tables: readonly Float64Array[];
  readonly walk: Walk;
  /** For each sub-pattern and primary node, the likelihood summed over the specific nodes, as the walk scales it. */
  readonly scaled: Float64Array;
  /** The log of the likelihood of each sub-pattern at each primary node, scale included. */
  readonly logLikelihood: Float64Array;
  /** The expected number of persons at each primary node whose pattern has each sub-pattern. */
  readonly mass: Float64Array;
  /** The most frequent category of each item, when every sub-pattern answers all of the items or none of them. */
  readonly references: Int32Array | null;
  /** The expected number of persons at each node of the group who answered its items, with references. */
  readonly total: Float64Array;
  /** The expected number of persons at each node for one sub-pattern. */
  readonly expected: Float64Array;
}

/**
 * Returns the E-step of a bifactor model over distinct patterns.
 *
 * @remarks
 * Given the primary dimensions, the specific dimensions are independent, so a person's likelihood is a sum over the
 * primary nodes of the weight times, for each group, a sum over its specific nodes. The latter depends on the responses
 * to the group's items only, and is computed once for each distinct sub-pattern. The posterior of a person's primary
 * node then gives each sub-pattern's expected number of persons at each primary node, and the expected counts of its
 * items at the group's nodes follow from the products again, once for each sub-pattern.
 *
 * @param logWeights - The log weights of the primary nodes, read at each run, so that a caller whose weights depend on
 * its own parameters rewrites them between runs.
 */
export function bifactorExpectation<TParameters>(
  table: PatternTable,
  groups: readonly GroupNodes<TParameters>[],
  logWeights: Float64Array,
): BifactorExpectation {
  const primaryNodes = logWeights.length;
  const states = groups.map((nodes, s) => groupState(nodes, table, s, primaryNodes));
  const nodeCounts = new Float64Array(primaryNodes);
  const joint = new Float64Array(primaryNodes);
  const groupCount = states.length;

  /** Writes a pattern's log joint densities over the primary nodes into `joint`. */
  const patternJoint = (p: number): void => {
    joint.set(logWeights);
    states.forEach((state, s) => {
      addRun(joint, 0, state.logLikelihood, (table.subPattern[p * groupCount + s] ?? 0) * primaryNodes, primaryNodes);
    });
  };

  /** Sums the persons' posteriors over the primary nodes into the node counts and the mass of the sub-patterns. */
  const summarize = (): number => {
    nodeCounts.fill(0);
    for (const state of states) state.mass.fill(0);
    let logLikelihood = 0;
    for (let p = 0; p < table.count; p++) {
      patternJoint(p);
      logLikelihood += weighPosterior(joint, table.frequency[p] ?? 0);
      addRun(nodeCounts, 0, joint, 0, primaryNodes);
      states.forEach((state, s) => {
        addRun(state.mass, (table.subPattern[p * groupCount + s] ?? 0) * primaryNodes, joint, 0, primaryNodes);
      });
    }
    return logLikelihood;
  };

  return {
    nodeCounts,
    run() {
      for (const state of states) {
        tabulate(state);
        likelihoods(state, primaryNodes);
      }
      const logLikelihood = summarize();
      for (const state of states) expectedCounts(state, primaryNodes);
      return logLikelihood;
    },
    forEachPattern(visit) {
      for (let p = 0; p < table.count; p++) {
        patternJoint(p);
        weighPosterior(joint, 1);
        visit(table.frequency[p] ?? 0, joint, table.subPattern.subarray(p * groupCount, (p + 1) * groupCount));
      }
    },
    forEachSubPattern(visit) {
      states.forEach((state, s) => {
        const conditional = new Float64Array(state.nodes.grid.count);
        state.walk((u, row) => {
          conditionalDistribution(state, u, row, conditional);
          visit(s, u, conditional, state.mass.subarray(u * primaryNodes, (u + 1) * primaryNodes));
        });
      });
    },
  };
}

/** Writes into `out` the share of each node among the specific nodes of its primary node, for a sub-pattern's row. */
function conditionalDistribution<TParameters>(
  state: GroupState<TParameters>,
  u: number,
  row: Float64Array,
  out: Float64Array,
): void {
  const block = state.nodes.weights.length;
  const primaryNodes = out.length / block;
  for (let q = 0; q < primaryNodes; q++) {
    const scaled = state.scaled[u * primaryNodes + q] ?? 0;
    for (let m = q * block; m < (q + 1) * block; m++) out[m] = scaled > 0 ? (row[m] ?? 0) / scaled : 0;
  }
}

/**
 * Turns the log joint densities of a pattern over the primary nodes into the expected number of persons at each node,
 * `frequency` times the posterior, and returns `frequency` times the pattern's log-likelihood.
 */
function weighPosterior(joint: Float64Array, frequency: number): number {
  const largest = joint.reduce((most, value) => Math.max(most, value), Number.NEGATIVE_INFINITY);
  let sum = 0;
  for (let q = 0; q < joint.length; q++) {
    joint[q] = Math.exp((joint[q] ?? 0) - largest);
    sum += joint[q] ?? 0;
  }
  for (let q = 0; q < joint.length; q++) joint[q] = ((joint[q] ?? 0) * frequency) / sum;
  return frequency * (largest + Math.log(sum));
}

/** Builds a group's state, with the tables and the sums it keeps between runs. */
function groupState<TParameters>(
  nodes: GroupNodes<TParameters>,
  table: PatternTable,
  index: number,
  primaryNodes: number,
): GroupState<TParameters> {
  const { patterns } = nodes;
  const tables = nodes.items.map((item) => new Float64Array(nodes.grid.count * item.categoryCount));
  const size = patterns.count * primaryNodes;
  return {
    nodes,
    patterns,
    tables,
    walk: createWalk({ patterns, tables, primaryNodes, weights: nodes.weights }),
    scaled: new Float64Array(size),
    logLikelihood: new Float64Array(size),
    mass: new Float64Array(size),
    references: referenceCategories(nodes, table, index),
    total: new Float64Array(nodes.grid.count),
    expected: new Float64Array(nodes.grid.count),
  };
}

/**
 * Returns the most frequent category of each item among the persons, or `null` when some sub-pattern answers some items
 * and leaves others unanswered: then the persons at a node who answered one item are not those who answered another.
 */
function referenceCategories<TParameters>(
  nodes: GroupNodes<TParameters>,
  table: PatternTable,
  index: number,
): Int32Array | null {
  const { width, count, codes } = nodes.patterns;
  for (let u = 0; u < count; u++) {
    const answered = codes.subarray(u * width, (u + 1) * width).filter((code) => code >= 0).length;
    if (answered > 0 && answered < width) return null;
  }
  const persons = new Float64Array(count);
  for (let p = 0; p < table.count; p++) {
    const u = table.subPattern[p * table.groups.length + index] ?? 0;
    persons[u] = (persons[u] ?? 0) + (table.frequency[p] ?? 0);
  }
  return Int32Array.from(nodes.items, (item, j) => {
    const tally = new Float64Array(item.categoryCount);
    for (let u = 0; u < count; u++) {
      const code = codes[u * width + j] ?? -1;
      if (code >= 0) tally[code] = (tally[code] ?? 0) + (persons[u] ?? 0);
    }
    return tally.reduce((best, weight, k) => (weight > (tally[best] ?? 0) ? k : best), 0);
  });
}

/** Writes each item's probabilities at the nodes of its group into its table. */
function tabulate<TParameters>({ nodes, tables }: GroupState<TParameters>): void {
  const { dimensions, count, nodes: coordinates } = nodes.grid;
  const theta = new Float64Array(dimensions);
  const probabilities = new Float64Array(nodes.items.reduce((most, item) => Math.max(most, item.categoryCount), 0));
  for (let m = 0; m < count; m++) {
    for (let d = 0; d < dimensions; d++) theta[d] = coordinates[m * dimensions + d] ?? 0;
    nodes.items.forEach((item, j) => {
      item.model.probabilities(item.parameters, theta, probabilities);
      const table = tables[j];
      if (table === undefined) return;
      for (let k = 0; k < item.categoryCount; k++) table[k * count + m] = probabilities[k] ?? 0;
    });
  }
}

/** Writes the likelihood of every sub-pattern of a group at each primary node, summed over the specific nodes. */
function likelihoods<TParameters>(state: GroupState<TParameters>, primaryNodes: number): void {
  const block = state.nodes.weights.length;
  state.walk((u, row, shift) => {
    for (let q = 0; q < primaryNodes; q++) {
      let sum = 0;
      for (let r = 0; r < block; r++) sum += row[q * block + r] ?? 0;
      state.scaled[u * primaryNodes + q] = sum;
      // A sum that underflows to 0 is the smallest probability, as in the other calibrations, so that no density is 0.
      state.logLikelihood[u * primaryNodes + q] = Math.log(Math.max(sum, Number.MIN_VALUE)) + (shift[q] ?? 0);
    }
  });
}

/** Writes the expected counts of a group's items at its nodes. */
function expectedCounts<TParameters>(state: GroupState<TParameters>, primaryNodes: number): void {
  const { items } = state.nodes;
  const { width, codes } = state.patterns;
  const { count: nodes } = state.nodes.grid;
  const block = state.nodes.weights.length;
  const { references, total, expected } = state;
  for (const item of items) item.counts.fill(0);
  total.fill(0);
  state.walk((u, row) => {
    for (let q = 0; q < primaryNodes; q++) {
      const scaled = state.scaled[u * primaryNodes + q] ?? 0;
      // Persons at a node whose likelihood underflowed are expected nowhere in the group.
      const factor = scaled > 0 ? (state.mass[u * primaryNodes + q] ?? 0) / scaled : 0;
      for (let r = q * block; r < (q + 1) * block; r++) expected[r] = factor * (row[r] ?? 0);
    }
    if (references !== null) addRun(total, 0, expected, 0, nodes);
    items.forEach((item, j) => {
      const code = codes[u * width + j] ?? -1;
      if (code >= 0 && code !== (references?.[j] ?? -1)) addRun(item.counts, code * nodes, expected, 0, nodes);
    });
  });
  if (references !== null) referenceCounts(items, references, total, nodes);
}

/** Writes the expected count of each item's reference category: the persons at the node less those in the others. */
function referenceCounts<TParameters>(
  items: readonly ItemState<TParameters>[],
  references: Int32Array,
  total: Float64Array,
  nodes: number,
): void {
  items.forEach((item, j) => {
    const reference = (references[j] ?? 0) * nodes;
    for (let m = 0; m < nodes; m++) {
      let others = 0;
      for (let k = 0; k < item.categoryCount; k++)
        if (k * nodes !== reference) others += item.counts[k * nodes + m] ?? 0;
      item.counts[reference + m] = Math.max((total[m] ?? 0) - others, 0);
    }
  });
}

/** Adds `source[from + q]` to `out[start + q]` for each q below `length`. */
function addRun(out: Float64Array, start: number, source: Float64Array, from: number, length: number): void {
  for (let q = 0; q < length; q++) out[start + q] = (out[start + q] ?? 0) + (source[from + q] ?? 0);
}

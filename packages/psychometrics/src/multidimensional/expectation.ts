import { normalizeLogs } from "@measurekit/core";
import { logProbability } from "../irt/probability.js";
import { type CompressedResponses, forEachRespondent } from "../irt/responses.js";
import type { Grid } from "./grid.js";
import type { MultidimensionalModel } from "./model.js";

/** An item as the E-step sees it: its model, its current parameters and the expected counts the E-step writes. */
export interface ExpectationItem {
  /** The item's model. */
  readonly model: MultidimensionalModel;
  /** The number of response categories. */
  readonly categoryCount: number;
  /** The current parameters. */
  parameters: unknown;
  /** The expected number of persons in each category at each node, by category, then node. */
  readonly counts: Float64Array;
}

/** The E-step of Bock-Aitkin EM over a grid of nodes. */
export interface ExpectationStep {
  /**
   * Evaluates every person's posterior over the nodes, under the weights of the person's group and the items' current
   * parameters; writes the items' expected counts and the expected number of persons at each node; and returns the
   * marginal log-likelihood: the sum over persons of log Σ_q w_q Π_i P_i(x_i | θ_q).
   */
  run(): number;
  /** The expected number of persons at each node, all groups together, as of the last run. */
  readonly nodeCounts: Float64Array;
  /** The expected number of persons at each node in each group, as of the last run. */
  readonly groupCounts: readonly Float64Array[];
  /**
   * Calls `visit` with each person who answered an item and the person's posterior over the nodes, at the parameters
   * of the last run. `posterior` is reused between calls.
   */
  forEachPosterior(visit: (person: number, posterior: Float64Array) => void): void;
}

/**
 * Returns the E-step of Bock-Aitkin EM over answered responses.
 *
 * @remarks
 * Persons without responses change nothing. Each item's log-probabilities at the nodes sit in one array, by category,
 * then node, so that a response adds the nodes of its category in one contiguous run.
 *
 * When every person who answered an item answered all of them, each item has a reference category, its most frequent:
 * a person's log joint density starts from the density of the reference categories of all items and adds the rise
 * to the category of each response that is not a reference, and the expected count of a reference category is the
 * persons expected at the node less those in the item's other categories. The additions go with the responses that are
 * not the most frequent ones.
 *
 * @param logWeights - The log weights of the nodes, or one array of them for each group, read at each run, so that a
 * caller whose weights depend on its own parameters rewrites them between runs.
 * @param groupOf - The index in `logWeights` of the group of each person, in the order of the persons of `data`. Without
 * it, every person is in the first group.
 */
export function expectationStep(
  data: CompressedResponses,
  items: readonly ExpectationItem[],
  grid: Grid,
  logWeights: Float64Array | readonly Float64Array[],
  groupOf?: ArrayLike<number>,
): ExpectationStep {
  const groupWeights = logWeights instanceof Float64Array ? [logWeights] : logWeights;
  const nodeCount = grid.count;
  const { codes } = data;
  const responseItems = data.items;
  const tableStarts = new Int32Array(items.length + 1);
  items.forEach((item, i) => {
    tableStarts[i + 1] = (tableStarts[i] ?? 0) + nodeCount * item.categoryCount;
  });
  const logTable = new Float64Array(tableStarts[items.length] ?? 0);
  const posterior = new Float64Array(nodeCount);
  const nodeCounts = new Float64Array(nodeCount);
  const groupCounts = groupWeights.length === 1 ? [nodeCounts] : groupWeights.map(() => new Float64Array(nodeCount));
  const references = referenceCategories(data, items);
  const baselines = references === null ? groupWeights : groupWeights.map(() => new Float64Array(nodeCount));
  const referenceTotal = new Float64Array(nodeCount);

  /** Writes the log joint densities over the nodes of a person of a group into `posterior`. */
  const joint = (start: number, end: number, group: number): void => {
    posterior.set(baselines[group] ?? nodeCounts);
    for (let r = start; r < end; r++) {
      const item = responseItems[r] ?? 0;
      const code = codes[r] ?? 0;
      if (code === (references?.[item] ?? -1)) continue;
      addRun(posterior, 0, logTable, (tableStarts[item] ?? 0) + code * nodeCount, nodeCount);
    }
  };

  /** Adds the posterior to the counts of the categories a person answered in and to the counts of the group. */
  const count = (start: number, end: number, group: number): void => {
    for (let r = start; r < end; r++) {
      const item = items[responseItems[r] ?? 0];
      const code = codes[r] ?? 0;
      if (item !== undefined && code !== (references?.[responseItems[r] ?? 0] ?? -1)) {
        addRun(item.counts, code * nodeCount, posterior, 0, nodeCount);
      }
    }
    addRun(groupCounts[group] ?? nodeCounts, 0, posterior, 0, nodeCount);
  };

  return {
    nodeCounts,
    groupCounts,
    run() {
      tabulate(items, grid, tableStarts, logTable);
      if (references !== null) {
        referenceDensities(items, references, tableStarts, logTable, referenceTotal);
        baselines.forEach((baseline, g) => {
          baseline.set(groupWeights[g] ?? referenceTotal);
          addRun(baseline, 0, referenceTotal, 0, nodeCount);
        });
      }
      for (const item of items) item.counts.fill(0);
      for (const counts of groupCounts) counts.fill(0);
      let logLikelihood = 0;
      forEachRespondent(data, groupOf, (_, start, end, group) => {
        joint(start, end, group);
        logLikelihood += normalizeLogs(posterior);
        count(start, end, group);
      });
      if (groupCounts.length > 1) {
        nodeCounts.fill(0);
        for (const counts of groupCounts) addRun(nodeCounts, 0, counts, 0, nodeCount);
      }
      if (references !== null) referenceCounts(items, references, nodeCounts);
      return logLikelihood;
    },
    forEachPosterior(visit) {
      forEachRespondent(data, groupOf, (person, start, end, group) => {
        joint(start, end, group);
        normalizeLogs(posterior);
        visit(person, posterior);
      });
    },
  };
}

/**
 * Returns the most frequent category of each item, or `null` when some person who answered an item left another
 * unanswered: then the density of the unanswered items would have to be taken off the baseline, for all of them.
 */
function referenceCategories(data: CompressedResponses, items: readonly ExpectationItem[]): Int32Array | null {
  const { offsets, codes } = data;
  const answered = items.length;
  for (let person = 0; person < offsets.length - 1; person++) {
    const responses = (offsets[person + 1] ?? 0) - (offsets[person] ?? 0);
    if (responses > 0 && responses < answered) return null;
  }
  const tallies = items.map((item) => new Int32Array(item.categoryCount));
  data.items.forEach((item, r) => {
    const tally = tallies[item];
    const code = codes[r] ?? 0;
    if (tally !== undefined) tally[code] = (tally[code] ?? 0) + 1;
  });
  return Int32Array.from(tallies, (tally) =>
    tally.reduce((best, count, k) => (count > (tally[best] ?? 0) ? k : best), 0),
  );
}

/**
 * Turns the table of log-probabilities into rises over the reference category of each item, which every category
 * but the reference keeps, and writes into `total` the reference categories' log-probabilities, which every group's
 * baseline adds to its log weights.
 */
function referenceDensities(
  items: readonly ExpectationItem[],
  references: Int32Array,
  tableStarts: Int32Array,
  logTable: Float64Array,
  total: Float64Array,
): void {
  const nodeCount = total.length;
  total.fill(0);
  items.forEach((item, i) => {
    const start = tableStarts[i] ?? 0;
    const reference = start + (references[i] ?? 0) * nodeCount;
    addRun(total, 0, logTable, reference, nodeCount);
    for (let k = 0; k < item.categoryCount; k++) {
      const from = start + k * nodeCount;
      if (from !== reference) subtractRun(logTable, from, reference, nodeCount);
    }
  });
}

/** Writes the expected count of each item's reference category: the persons at the node less those in the others. */
function referenceCounts(items: readonly ExpectationItem[], references: Int32Array, nodeCounts: Float64Array): void {
  const nodeCount = nodeCounts.length;
  items.forEach((item, i) => {
    const reference = (references[i] ?? 0) * nodeCount;
    for (let q = 0; q < nodeCount; q++) {
      let others = 0;
      for (let k = 0; k < item.categoryCount; k++) {
        if (k * nodeCount !== reference) others += item.counts[k * nodeCount + q] ?? 0;
      }
      item.counts[reference + q] = Math.max((nodeCounts[q] ?? 0) - others, 0);
    }
  });
}

/** Writes each item's log-probabilities at the nodes into the table, which holds them from `tableStarts`. */
function tabulate(
  items: readonly ExpectationItem[],
  { dimensions, count, nodes }: Grid,
  tableStarts: Int32Array,
  logTable: Float64Array,
): void {
  const theta = new Float64Array(dimensions);
  const probabilities = new Float64Array(items.reduce((most, item) => Math.max(most, item.categoryCount), 0));
  for (let q = 0; q < count; q++) {
    for (let d = 0; d < dimensions; d++) theta[d] = nodes[q * dimensions + d] ?? 0;
    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item === undefined) continue;
      item.model.probabilities(item.parameters, theta, probabilities);
      const start = tableStarts[i] ?? 0;
      for (let k = 0; k < item.categoryCount; k++) {
        logTable[start + k * count + q] = logProbability(probabilities[k] ?? 0);
      }
    }
  }
}

/** Adds `source[from + q]` to `out[start + q]` for each q below `length`. */
export function addRun(out: Float64Array, start: number, source: Float64Array, from: number, length: number): void {
  for (let q = 0; q < length; q++) out[start + q] = (out[start + q] ?? 0) + (source[from + q] ?? 0);
}

/** Takes `table[from + q]` off `table[start + q]` for each q below `length`, for the rise over a reference. */
function subtractRun(table: Float64Array, start: number, from: number, length: number): void {
  for (let q = 0; q < length; q++) table[start + q] = (table[start + q] ?? 0) - (table[from + q] ?? 0);
}

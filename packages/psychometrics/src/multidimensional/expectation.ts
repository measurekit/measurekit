import { normalizeLogs } from "@measurekit/core";
import { MIN_PROBABILITY } from "../irt/probability.js";
import type { CompressedResponses } from "../irt/responses.js";
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
   * Evaluates every person's posterior over the nodes, under the weights and the items' current parameters; writes the
   * items' expected counts and the expected number of persons at each node; and returns the marginal log-likelihood:
   * the sum over persons of log Σ_q w_q Π_i P_i(x_i | θ_q).
   */
  run(): number;
  /** The expected number of persons at each node, as of the last run. */
  readonly nodeCounts: Float64Array;
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
 * @param logWeights - The log weights of the nodes, read at each run, so that a caller whose weights depend on its own
 * parameters rewrites them between runs.
 */
export function expectationStep(
  data: CompressedResponses,
  items: readonly ExpectationItem[],
  grid: Grid,
  logWeights: Float64Array,
): ExpectationStep {
  const nodeCount = grid.count;
  const { offsets, codes } = data;
  const responseItems = data.items;
  const personCount = offsets.length - 1;
  const tableStarts = new Int32Array(items.length + 1);
  items.forEach((item, i) => {
    tableStarts[i + 1] = (tableStarts[i] ?? 0) + nodeCount * item.categoryCount;
  });
  const logTable = new Float64Array(tableStarts[items.length] ?? 0);
  const posterior = new Float64Array(nodeCount);
  const nodeCounts = new Float64Array(nodeCount);

  /** Writes a person's log joint densities over the nodes into `posterior`. */
  const joint = (start: number, end: number): void => {
    posterior.set(logWeights);
    for (let r = start; r < end; r++) {
      const item = responseItems[r] ?? 0;
      addRun(posterior, 0, logTable, (tableStarts[item] ?? 0) + (codes[r] ?? 0) * nodeCount, nodeCount);
    }
  };

  /** Calls `visit` with each person who answered an item and the bounds of the person's responses. */
  const forEachRespondent = (visit: (person: number, start: number, end: number) => void): void => {
    for (let person = 0; person < personCount; person++) {
      const start = offsets[person] ?? 0;
      const end = offsets[person + 1] ?? 0;
      if (end > start) visit(person, start, end);
    }
  };

  /** Adds the posterior to the counts of the categories a person answered in and to the nodes' counts. */
  const count = (start: number, end: number): void => {
    for (let r = start; r < end; r++) {
      const item = items[responseItems[r] ?? 0];
      if (item !== undefined) addRun(item.counts, (codes[r] ?? 0) * nodeCount, posterior, 0, nodeCount);
    }
    addRun(nodeCounts, 0, posterior, 0, nodeCount);
  };

  return {
    nodeCounts,
    run() {
      tabulate(items, grid, tableStarts, logTable);
      for (const item of items) item.counts.fill(0);
      nodeCounts.fill(0);
      let logLikelihood = 0;
      forEachRespondent((_, start, end) => {
        joint(start, end);
        logLikelihood += normalizeLogs(posterior);
        count(start, end);
      });
      return logLikelihood;
    },
    forEachPosterior(visit) {
      forEachRespondent((person, start, end) => {
        joint(start, end);
        normalizeLogs(posterior);
        visit(person, posterior);
      });
    },
  };
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
        logTable[start + k * count + q] = Math.log(Math.max(probabilities[k] ?? 0, MIN_PROBABILITY));
      }
    }
  }
}

/** Adds `source[from + q]` to `out[start + q]` for each q below `length`. */
function addRun(out: Float64Array, start: number, source: Float64Array, from: number, length: number): void {
  for (let q = 0; q < length; q++) out[start + q] = (out[start + q] ?? 0) + (source[from + q] ?? 0);
}

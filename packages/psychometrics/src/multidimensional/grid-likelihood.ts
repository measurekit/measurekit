import { logProbability } from "../irt/probability.js";
import type { Grid } from "./grid.js";
import type { MultidimensionalItem } from "./model.js";

/** Adds the log-likelihood of an item's response at every node of a grid to a running sum over the nodes. */
export interface GridLikelihood {
  /** Adds ln P(response | θ_q) of `item` to `sums[q]` for every node q. */
  add(item: MultidimensionalItem, response: number, sums: Float64Array): void;
}

/**
 * Returns the log-likelihoods of items over a grid, with the log probabilities of each item at the nodes kept for the
 * next response to it, until `maxEntries` of them are kept: an item past the limit is evaluated at every call.
 *
 * @remarks
 * Scoring many persons on the same items evaluates each item at the nodes once, not once for each person. A log
 * probability is the same number kept or recomputed, so the sums do not depend on `maxEntries`.
 *
 * @param maxEntries - The most log probabilities to keep, 8 bytes each: 0 keeps none.
 */
export function gridLikelihood(grid: Grid, maxEntries: number): GridLikelihood {
  const { dimensions, count, nodes } = grid;
  const tables = new Map<MultidimensionalItem, Float64Array>();
  const theta = new Float64Array(dimensions);
  let kept = 0;

  /** Writes the probability of each category of an item at node q into `probabilities`. */
  function probabilitiesAt(item: MultidimensionalItem, q: number, probabilities: Float64Array): void {
    for (let d = 0; d < dimensions; d++) theta[d] = nodes[q * dimensions + d] ?? 0;
    item.model.probabilities(item.parameters, theta, probabilities);
  }

  /** Returns the log probabilities of every category of an item at the nodes, by category, then node. */
  function tabulate(item: MultidimensionalItem, categories: number): Float64Array {
    const table = new Float64Array(categories * count);
    const probabilities = new Float64Array(categories);
    for (let q = 0; q < count; q++) {
      probabilitiesAt(item, q, probabilities);
      for (let k = 0; k < categories; k++) table[k * count + q] = logProbability(probabilities[k] ?? 0);
    }
    return table;
  }

  function add(item: MultidimensionalItem, response: number, sums: Float64Array): void {
    const table = tables.get(item);
    if (table !== undefined) {
      const start = response * count;
      for (let q = 0; q < count; q++) sums[q] = (sums[q] ?? 0) + (table[start + q] ?? 0);
      return;
    }
    const categories = item.model.categoryCount(item.parameters);
    if (kept + categories * count <= maxEntries) {
      tables.set(item, tabulate(item, categories));
      kept += categories * count;
      add(item, response, sums);
      return;
    }
    const probabilities = new Float64Array(categories);
    for (let q = 0; q < count; q++) {
      probabilitiesAt(item, q, probabilities);
      sums[q] = (sums[q] ?? 0) + logProbability(probabilities[response] ?? 0);
    }
  }

  return { add };
}

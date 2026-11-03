import { normalizeLogs } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { type IrtItem, itemTrace, logProbability } from "../irt/internal.js";
import { checkItems, type ResidualFitOptions, residualPopulation } from "./inputs.js";

/** The deviation of an item's observed responses from its model at the nodes of the population. */
export interface ItemDeviation {
  /** The item id. */
  readonly id: string;
  /**
   * The root mean square deviation: √(Σ_q f_q Σ_k (O_qk / O_q - P_k(θ_q))² / K), from 0; `null` when no person
   * answered the item.
   */
  readonly rmsd: number | null;
  /**
   * The mean deviation of the score: Σ_q f_q Σ_k k (O_qk / O_q - P_k(θ_q)) / (K - 1), positive when the persons score
   * higher than the model expects; `null` when no person answered the item.
   */
  readonly meanDeviation: number | null;
}

/** An item's probabilities at the nodes and its posterior counts, by node, then category. */
interface ItemTable {
  readonly count: number;
  readonly trace: Float64Array;
  readonly logTrace: Float64Array;
  readonly observed: Float64Array;
}

/**
 * Returns the root mean square deviation (RMSD) and the mean deviation of each item, as TAM's `IRT.RMSD`.
 *
 * @remarks
 * Each person with a response has a posterior over the nodes θ_q of the population, from the items the person answered.
 * O_qk sums the posteriors at node q of the persons who answered the item in category k, so O_qk / O_q is the observed
 * share of category k at θ_q, which the model expects to be P_k(θ_q). The squared deviations are weighted by f_q, the
 * mean posterior of the persons with a response, which estimates the distribution of theta in the sample, and averaged
 * over the item's K categories. A node where the posteriors of the item's persons underflow to 0 adds nothing. The RMSD
 * is near 0 when the item fits.
 *
 * @param matrix - The responses of the persons.
 * @param items - The calibrated items, in the order of the matrix's items.
 * @param options - The population.
 * @returns For each item, its RMSD and mean deviation.
 * @throws `DataError` when the matrix, the items or the population are invalid, or a response is not a category of
 * its item.
 * @see {@link https://doi.org/10.1080/15305058.2013.825265 | Oliveri and von Davier (2014)}
 */
export function rootMeanSquareDeviation(
  matrix: ResponseMatrix,
  items: readonly IrtItem[],
  options: ResidualFitOptions = {},
): ItemDeviation[] {
  checkItems(matrix, items);
  const { nodes, weights } = residualPopulation(options);
  const tables = items.map((item): ItemTable => {
    const count = item.model.categoryCount(item.parameters);
    const trace = itemTrace(item, count, nodes);
    return { count, trace, logTrace: trace.map(logProbability), observed: new Float64Array(nodes.length * count) };
  });
  const sums: PosteriorSums = {
    logWeights: weights.map(Math.log),
    posterior: new Float64Array(nodes.length),
    shares: new Float64Array(nodes.length),
    persons: 0,
  };
  for (const row of matrix.responses) addPerson(row, tables, sums);
  const distribution = sums.shares.map((share) => share / sums.persons);
  return tables.map((table, i) => ({ id: matrix.itemIds[i] ?? "", ...deviation(table, distribution) }));
}

/** What the persons add up to: the summed posteriors at the nodes and the persons with a response. */
interface PosteriorSums {
  readonly logWeights: Float64Array;
  /** The posterior of the current person. */
  readonly posterior: Float64Array;
  readonly shares: Float64Array;
  persons: number;
}

/**
 * Adds a person's posterior, from the answered items, to the summed posteriors and to the posterior counts of each
 * answered item in the category of the response. A person without responses adds nothing.
 */
function addPerson(row: readonly (number | null)[], tables: readonly ItemTable[], sums: PosteriorSums): void {
  const { posterior, shares } = sums;
  posterior.set(sums.logWeights);
  if (addLogTraces(row, tables, posterior) === 0) return;
  normalizeLogs(posterior);
  sums.persons++;
  for (let q = 0; q < posterior.length; q++) shares[q] = (shares[q] ?? 0) + (posterior[q] ?? 0);
  for (let i = 0; i < row.length; i++) {
    const response = row[i] ?? null;
    const table = tables[i];
    if (response === null || table === undefined) continue;
    for (let q = 0; q < posterior.length; q++) {
      const cell = q * table.count + response;
      table.observed[cell] = (table.observed[cell] ?? 0) + (posterior[q] ?? 0);
    }
  }
}

/** Adds the log-probability of each answered item's response at each node to `posterior`; returns the items answered. */
function addLogTraces(row: readonly (number | null)[], tables: readonly ItemTable[], posterior: Float64Array): number {
  let answered = 0;
  for (let i = 0; i < row.length; i++) {
    const response = row[i] ?? null;
    const table = tables[i];
    if (response === null || table === undefined) continue;
    answered++;
    for (let q = 0; q < posterior.length; q++) {
      posterior[q] = (posterior[q] ?? 0) + (table.logTrace[q * table.count + response] ?? 0);
    }
  }
  return answered;
}

/** Returns an item's RMSD and mean deviation, weighting each node by `distribution`. */
function deviation(
  { count, trace, observed }: ItemTable,
  distribution: Float64Array,
): Pick<ItemDeviation, "rmsd" | "meanDeviation"> {
  let squares = 0;
  let mean = 0;
  let answered = false;
  distribution.forEach((weight, q) => {
    let total = 0;
    for (let k = 0; k < count; k++) total += observed[q * count + k] ?? 0;
    if (!(total > 0)) return;
    answered = true;
    for (let k = 0; k < count; k++) {
      const difference = (observed[q * count + k] ?? 0) / total - (trace[q * count + k] ?? 0);
      squares += weight * difference * difference;
      mean += weight * k * difference;
    }
  });
  if (!answered) return { rmsd: null, meanDeviation: null };
  return { rmsd: Math.sqrt(squares / count), meanDeviation: mean / (count - 1) };
}

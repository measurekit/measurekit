import { normalizeLogs } from "@measurekit/core";
import { type AbilityEstimate, type AbilityEstimator, posteriorMoments } from "./ability.js";
import { answeredItems, checkPattern } from "./items.js";
import type { IrtItem } from "./model.js";
import { logProbability } from "./probability.js";
import { quadratureRule } from "./quadrature.js";

/**
 * Returns `posteriorMeanEstimate` with its default options for the persons of one data set: it gives the same estimates,
 * but the log-probability of each response of an item at the nodes is computed once, for every person who gave it.
 *
 * @remarks
 * A person's log-likelihood at a node is the sum, in the order of the items, of the log-probabilities of the responses,
 * as `posteriorMeanEstimate` sums them; only where the terms come from differs. The tables belong to the returned
 * function, so a call sees the items as they are when it is made.
 */
export function sharedPosteriorMeanEstimator(): AbilityEstimator {
  const { nodes, weights } = quadratureRule();
  const logWeights = Float64Array.from(weights, Math.log);
  const logLikelihood = new Float64Array(nodes.length);
  const posterior = new Float64Array(nodes.length);
  const tables = new Map<IrtItem, (Float64Array | undefined)[]>();

  /** Returns the log-probability of a response to an item at each node, tabulated when first asked for. */
  const logProbabilities = (item: IrtItem, response: number): Float64Array => {
    let rows = tables.get(item);
    if (rows === undefined) {
      rows = new Array<Float64Array | undefined>(item.model.categoryCount(item.parameters));
      tables.set(item, rows);
    }
    let row = rows[response];
    if (row === undefined) {
      row = new Float64Array(nodes.length);
      const probabilities = new Float64Array(rows.length);
      for (let q = 0; q < nodes.length; q++) {
        item.model.probabilities(item.parameters, nodes[q] ?? 0, probabilities);
        row[q] = logProbability(probabilities[response] ?? 0);
      }
      rows[response] = row;
    }
    return row;
  };

  return (items, responses): AbilityEstimate => {
    checkPattern(items, responses);
    logLikelihood.fill(0);
    for (const { item, response } of answeredItems(items, responses)) {
      const row = logProbabilities(item, response);
      for (let q = 0; q < nodes.length; q++) logLikelihood[q] = (logLikelihood[q] ?? 0) + (row[q] ?? 0);
    }
    for (let q = 0; q < nodes.length; q++) posterior[q] = (logWeights[q] ?? 0) + (logLikelihood[q] ?? 0);
    normalizeLogs(posterior);
    return posteriorMoments(nodes, posterior);
  };
}

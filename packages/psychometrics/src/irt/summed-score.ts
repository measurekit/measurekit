import { checkOptions } from "@measurekit/core";
import { checkThetaPrior, type ThetaIntegrationOptions } from "./ability.js";
import { checkItemList } from "./items.js";
import type { IrtItem } from "./model.js";
import { quadratureRule } from "./quadrature.js";

/** The EAP estimate of theta from a total score of a fixed form, with the probability of the score. */
export interface SummedScoreEstimate {
  /** The total score. */
  readonly score: number;
  /** The posterior mean of theta given the total score; `null` where the score's probability underflows to 0. */
  readonly theta: number | null;
  /** The posterior standard deviation of theta given the total score; `null` with `theta`. */
  readonly standardError: number | null;
  /** The probability of the total score under the prior: Σ_q w_q L(s | θ_q). */
  readonly probability: number;
}

/**
 * Returns the summed-score conversion table of a fixed form.
 *
 * @remarks
 * For each total score from 0 to the highest, the posterior mean and standard deviation of theta given the total score,
 * and the score's probability, as mirt's `fscores(method = "EAPsum")`. The probability of each total score at each
 * quadrature node comes from the recursion of Lord and Wingersky; the nodes and weights are those of {@link
 * posteriorMeanEstimate}.
 *
 * @throws `DataError` without items, or when an item's parameters, the prior or the quadrature are invalid.
 * @see {@link https://doi.org/10.1177/014662169501900105 | Thissen, Pommerich, Billeaud and Williams (1995)}
 * @see {@link https://doi.org/10.1177/014662168400800409 | Lord and Wingersky (1984)}
 */
export function summedScoreTable(
  items: readonly IrtItem[],
  options: ThetaIntegrationOptions = {},
): SummedScoreEstimate[] {
  checkItemList(items, "A summed-score table");
  checkOptions(options);
  const { mean, sd } = checkThetaPrior(options.prior);
  const { nodes, weights } = quadratureRule(options.quadrature, mean, sd);
  const likelihoods = scoreLikelihoods(items, nodes);
  const scores = likelihoods.length / nodes.length;
  return Array.from({ length: scores }, (_, score) => {
    let probability = 0;
    let first = 0;
    for (let q = 0; q < nodes.length; q++) {
      const joint = (weights[q] ?? 0) * (likelihoods[score * nodes.length + q] ?? 0);
      probability += joint;
      first += joint * (nodes[q] ?? 0);
    }
    if (!(probability > 0)) return { score, theta: null, standardError: null, probability };
    const theta = first / probability;
    let second = 0;
    for (let q = 0; q < nodes.length; q++) {
      second += (weights[q] ?? 0) * (likelihoods[score * nodes.length + q] ?? 0) * ((nodes[q] ?? 0) - theta) ** 2;
    }
    return { score, theta, standardError: Math.sqrt(second / probability), probability };
  });
}

/**
 * Returns the probability of each total score of the items at each node, by score, then node, by the recursion of Lord
 * and Wingersky.
 */
export function scoreLikelihoods(items: readonly IrtItem[], nodes: Float64Array): Float64Array {
  const counts = items.map((item) => item.model.categoryCount(item.parameters));
  const traces = items.map((item, i) => itemTrace(item, counts[i] ?? 0, nodes));
  return summedScoreLikelihoods(traces, counts, -1, nodes.length);
}

/** Returns an item's category probabilities at each node, by node, then category. */
export function itemTrace(item: IrtItem, categories: number, nodes: Float64Array): Float64Array {
  const trace = new Float64Array(nodes.length * categories);
  nodes.forEach((theta, q) => {
    item.model.probabilities(item.parameters, theta, trace.subarray(q * categories, (q + 1) * categories));
  });
  return trace;
}

/**
 * Returns the probability of each total score of the items other than `skip` at each node, by score, then node, by the
 * recursion of Lord and Wingersky: adding an item with categories k moves the probability of score s at a node to s +
 * k, times P(k | θ).
 *
 * @param traces - Each item's category probabilities at the nodes, by node, then category.
 * @param counts - Each item's number of categories.
 * @param skip - The index of an item to leave out, or -1.
 * @see {@link https://doi.org/10.1177/014662168400800409 | Lord and Wingersky (1984)}
 */
export function summedScoreLikelihoods(
  traces: readonly Float64Array[],
  counts: readonly number[],
  skip: number,
  nodeCount: number,
): Float64Array {
  const maxScore = counts.reduce((sum, count, i) => (i === skip ? sum : sum + count - 1), 0);
  let current = new Float64Array((maxScore + 1) * nodeCount);
  let next = new Float64Array((maxScore + 1) * nodeCount);
  current.fill(1, 0, nodeCount);
  let reached = 0;
  traces.forEach((trace, i) => {
    if (i === skip) return;
    const categories = counts[i] ?? 0;
    next.fill(0);
    for (let s = 0; s <= reached; s++) {
      for (let k = 0; k < categories; k++) {
        const target = (s + k) * nodeCount;
        for (let q = 0; q < nodeCount; q++) {
          next[target + q] =
            (next[target + q] ?? 0) + (current[s * nodeCount + q] ?? 0) * (trace[q * categories + k] ?? 0);
        }
      }
    }
    reached += categories - 1;
    [current, next] = [next, current];
  });
  return current;
}

import { checkOptions } from "@measurekit/core";
import type { Quadrature } from "../math/internal.js";
import {
  checkThetaPrior,
  POSTERIOR_MEAN_OPTIONS,
  type PosteriorMeanOptions,
  type TestletEffect,
  testletMembership,
} from "./ability.js";
import { checkItemList } from "./items.js";
import type { IrtItem } from "./model.js";
import { type QuadratureOptions, quadratureRule } from "./quadrature.js";

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

/** The checked testlets of a form: each item's testlet, -1 for none, each testlet's SD and the rule over its effect. */
export interface ScoreTestlets {
  readonly testletOf: Int32Array;
  readonly sds: readonly number[];
  readonly rule: Quadrature;
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
 * With testlets, the items of a testlet answer at θ + γ, γ ~ N(0, sd²), as in the testlet response model. At each node
 * of theta the recursion runs over a testlet's items at θ + sd z_g for each node z_g of the standard rule, and the
 * testlet's score distribution is Σ_g w_g P(s | θ + sd z_g); the recursion then adds the testlets' distributions and
 * the other items' categories, as mirt's `fscores(method = "EAPsum_2.0")` for a bifactor model.
 *
 * @param items - The items of the form; with testlets, each with its id.
 * @param options - The prior, the quadrature and the testlets.
 * @returns For each summed score, the EAP estimate of theta with its standard error and the probability of the score.
 * @throws `DataError` without items, or when an item's parameters, the prior or the quadrature are invalid; with
 * testlets, when an item has no id or item ids repeat, a testlet names an id that is not a string or an item that is in
 * a testlet already, or its sd is not positive and finite.
 * @see {@link https://doi.org/10.1177/014662169501900105 | Thissen, Pommerich, Billeaud and Williams (1995)}
 * @see {@link https://doi.org/10.1177/014662168400800409 | Lord and Wingersky (1984)}
 * @see {@link https://doi.org/10.1007/s11336-014-9411-3 | Cai (2015)}
 */
export function summedScoreTable(
  items: readonly IrtItem[],
  options: PosteriorMeanOptions = {},
): readonly SummedScoreEstimate[] {
  checkItemList(items, "A summed-score table");
  checkOptions(options, POSTERIOR_MEAN_OPTIONS);
  const testlets = scoreTestlets(items, options.testlets, options.quadrature);
  const { mean, sd } = checkThetaPrior(options.prior);
  const { nodes, weights } = quadratureRule(options.quadrature, mean, sd);
  return summedScoreRows(scoreLikelihoods(items, nodes, testlets), nodes, weights);
}

/**
 * Checks testlets and returns them with the standard rule of `quadrature` over their effects, or `null` without
 * testlets.
 *
 * @throws `DataError` as {@link summedScoreTable} does for testlets and the quadrature.
 */
export function scoreTestlets(
  items: readonly IrtItem[],
  testlets: readonly TestletEffect[] = [],
  quadrature?: QuadratureOptions,
): ScoreTestlets | null {
  const testletOf = testletMembership(items, testlets);
  const rule = quadratureRule(quadrature);
  return testlets.length === 0 ? null : { testletOf, sds: testlets.map(({ sd }) => sd), rule };
}

/** Returns the rows of a summed-score table from the probability of each score at each node, by score, then node. */
function summedScoreRows(
  likelihoods: Float64Array,
  nodes: Float64Array,
  weights: Float64Array,
): readonly SummedScoreEstimate[] {
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
 * and Wingersky; with `testlets`, each testlet's effect is integrated out of its items' score distribution first.
 */
export function scoreLikelihoods(
  items: readonly IrtItem[],
  nodes: Float64Array,
  testlets: ScoreTestlets | null = null,
): Float64Array {
  const counts = items.map((item) => item.model.categoryCount(item.parameters));
  if (testlets === null) {
    const traces = items.map((item, i) => itemTrace(item, counts[i] ?? 0, nodes));
    return summedScoreLikelihoods(traces, counts, -1, nodes.length);
  }
  const { testletOf, sds, rule } = testlets;
  const traces: Float64Array[] = [];
  const unitCounts: number[] = [];
  items.forEach((item, i) => {
    if ((testletOf[i] ?? -1) >= 0) return;
    traces.push(itemTrace(item, counts[i] ?? 0, nodes));
    unitCounts.push(counts[i] ?? 0);
  });
  sds.forEach((sd, t) => {
    const members = items.filter((_, i) => testletOf[i] === t);
    if (members.length === 0) return;
    const memberCounts = members.map((item) => item.model.categoryCount(item.parameters));
    const trace = testletTrace(members, memberCounts, nodes, sd, rule);
    traces.push(trace);
    unitCounts.push(trace.length / nodes.length);
  });
  return summedScoreLikelihoods(traces, unitCounts, -1, nodes.length);
}

/**
 * Returns the probability of each summed score of a testlet's items at each node of theta, by node, then score: the
 * recursion of Lord and Wingersky at θ + sd z_g for each node z_g of the standard rule, weighted by w_g and summed, so
 * that the testlet's effect is integrated out (Cai, 2015). One node of theta at a time, the buffers serve every node.
 */
function testletTrace(
  members: readonly IrtItem[],
  counts: readonly number[],
  nodes: Float64Array,
  sd: number,
  rule: Quadrature,
): Float64Array {
  const effects = rule.nodes.length;
  const scores = counts.reduce((sum, count) => sum + count - 1, 1);
  const trace = new Float64Array(nodes.length * scores);
  const points = new Float64Array(effects);
  const traces = counts.map((count) => new Float64Array(effects * count));
  const buffers = { current: new Float64Array(scores * effects), next: new Float64Array(scores * effects) };
  nodes.forEach((theta, q) => {
    for (let g = 0; g < effects; g++) points[g] = theta + sd * (rule.nodes[g] ?? 0);
    members.forEach((item, i) => {
      fillTrace(item, counts[i] ?? 0, points, traces[i] ?? new Float64Array(0));
    });
    const likelihoods = addScores(traces, counts, -1, effects, buffers);
    for (let s = 0; s < scores; s++) {
      let sum = 0;
      for (let g = 0; g < effects; g++) sum += (rule.weights[g] ?? 0) * (likelihoods[s * effects + g] ?? 0);
      trace[q * scores + s] = sum;
    }
  });
  return trace;
}

/** Returns an item's category probabilities at each node, by node, then category. */
export function itemTrace(item: IrtItem, categories: number, nodes: Float64Array): Float64Array {
  const trace = new Float64Array(nodes.length * categories);
  fillTrace(item, categories, nodes, trace);
  return trace;
}

/** Writes an item's category probabilities at each node into `trace`, by node, then category. */
function fillTrace(item: IrtItem, categories: number, nodes: Float64Array, trace: Float64Array): void {
  nodes.forEach((theta, q) => {
    item.model.probabilities(item.parameters, theta, trace.subarray(q * categories, (q + 1) * categories));
  });
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
  const size = (maxScore + 1) * nodeCount;
  return addScores(traces, counts, skip, nodeCount, { current: new Float64Array(size), next: new Float64Array(size) });
}

/**
 * Runs the recursion of {@link summedScoreLikelihoods} in two buffers of at least (highest score + 1) × nodes entries,
 * and returns the one that holds the result, by score, then node, in its first entries.
 */
function addScores(
  traces: readonly Float64Array[],
  counts: readonly number[],
  skip: number,
  nodeCount: number,
  buffers: { current: Float64Array; next: Float64Array },
): Float64Array {
  let { current, next } = buffers;
  current.fill(0);
  current.fill(1, 0, nodeCount);
  let reached = 0;
  traces.forEach((trace, i) => {
    if (i === skip) return;
    const categories = counts[i] ?? 0;
    next.fill(0, 0, (reached + categories) * nodeCount);
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

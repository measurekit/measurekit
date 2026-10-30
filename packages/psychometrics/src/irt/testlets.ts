import { checkRecord, DataError, describe } from "@measurekit/core";
import type { Quadrature } from "../math/quadrature.js";
import { findRoot } from "../math/root.js";

/** The estimated spread of a testlet's effects. */
export interface TestletEstimate {
  /** The testlet's id, as in the `testlets` option. */
  readonly id: string;
  /** The ids of its items. */
  readonly items: readonly string[];
  /** The standard deviation of its effect γ, which adds to theta in the probabilities of its items. */
  readonly sd: number;
  /** The standard error of `sd`; `null` without standard errors. */
  readonly standardError: number | null;
}

/** A testlet during calibration. The E-step reads `weights` and writes `counts`. */
interface TestletState {
  readonly id: string;
  /** The indices of its items. */
  readonly items: readonly number[];
  /** The weight of each node of γ, the quadrature nodes, ∝ φ(γ / σ) and summing to 1. */
  readonly weights: Float64Array;
  /** The expected number of persons with responses to the testlet at each node of γ. */
  readonly counts: Float64Array;
  /** The free parameter of σ. */
  readonly sdSlot: number;
}

/** The testlets of a calibration, with the grid of θ + γ that their items' probabilities use. */
export interface TestletDesign {
  readonly testlets: readonly TestletState[];
  /** Each item's testlet index, or -1 for an item in none. */
  readonly testletOf: Int32Array;
  /** θ_q + γ_g on the rectangular rule: 2Q - 1 equally spaced points from twice the lowest node to twice the highest. */
  readonly sums: Float64Array;
}

/**
 * Checks the testlets option and builds the testlets with their starting standard deviation, 1.
 *
 * @throws `DataError` when a testlet has fewer than two items, an item id that is not an item, or an item that is in
 * another testlet, or when the rule is not rectangular.
 */
export function testletDesign(
  testlets: Readonly<Record<string, readonly string[]>>,
  itemIds: readonly string[],
  rule: Quadrature,
  rectangular: boolean,
  addFree: (lower: number, start: number) => number,
): TestletDesign {
  checkRecord(testlets, "The option testlets must be an object of item id lists by testlet id");
  if (!rectangular) throw new DataError("Testlets need the rectangular quadrature rule.");
  const testletOf = new Int32Array(itemIds.length).fill(-1);
  const nodeCount = rule.nodes.length;
  const states = Object.entries(testlets).map(([id, ids], t): TestletState => {
    if (!Array.isArray(ids) || ids.length < 2) throw new DataError(`Testlet "${id}" needs at least two items.`);
    const items = ids.map((itemId) => {
      const index = itemIds.indexOf(itemId);
      if (index < 0) throw new DataError(`Testlet "${id}" names ${describe(itemId)}, which is not an item id.`);
      if ((testletOf[index] ?? -1) >= 0) throw new DataError(`Item "${itemId}" is in more than one testlet.`);
      testletOf[index] = t;
      return index;
    });
    return {
      id,
      items,
      weights: new Float64Array(nodeCount),
      counts: new Float64Array(nodeCount),
      sdSlot: addFree(0, 1),
    };
  });
  const { nodes } = rule;
  const lowest = nodes[0] ?? 0;
  const step = (nodes[1] ?? 0) - lowest;
  const sums = Float64Array.from({ length: 2 * nodeCount - 1 }, (_, m) => 2 * lowest + m * step);
  return { testlets: states, testletOf, sums };
}

/** Sets each testlet's weights from its σ in `values`: ∝ φ(γ_g / σ) on the nodes, summing to 1. */
export function testletWeights({ testlets }: TestletDesign, values: Float64Array, nodes: Float64Array): void {
  for (const { weights, sdSlot } of testlets) normalWeightsAtZero(nodes, values[sdSlot] ?? 1, weights);
}

/**
 * Writes the weights ∝ φ(γ / σ) of the nodes into `weights`, summing to 1, and returns their second moment. The
 * largest weight is scaled to 1 before the sum, so that no σ underflows them all to 0.
 */
function normalWeightsAtZero(nodes: Float64Array, sd: number, weights: Float64Array): number {
  let smallest = Number.POSITIVE_INFINITY;
  for (const node of nodes) smallest = Math.min(smallest, node * node);
  let sum = 0;
  let moment = 0;
  nodes.forEach((node, g) => {
    const weight = Math.exp((-0.5 * (node * node - smallest)) / (sd * sd));
    weights[g] = weight;
    sum += weight;
    moment += weight * node * node;
  });
  for (let g = 0; g < nodes.length; g++) weights[g] = (weights[g] ?? 0) / sum;
  return moment / sum;
}

/** The bounds of a testlet's σ: below, the weights sit on the nodes nearest 0; above, they are flat. */
const SD_BOUNDS = [1e-4, 1e4] as const;
/** The width of the bracket of ln σ at which root finding stops: σ to about 1e-13 relative. */
const ROOT_TOLERANCE = 1e-13;

/**
 * Updates each testlet's σ from the expected counts of the last E-step to the maximum of Σ n_g ln w_g: the σ whose
 * weights on the nodes have the second moment of the counts. The moment increases with σ, so the root of their
 * difference in ln σ is bracketed by the bounds of σ, which hold a σ the counts push beyond them.
 */
export function maximizeTestlets(design: TestletDesign, values: Float64Array, nodes: Float64Array): void {
  const scratch = new Float64Array(nodes.length);
  for (const { counts, sdSlot } of design.testlets) {
    let total = 0;
    let second = 0;
    nodes.forEach((node, g) => {
      total += counts[g] ?? 0;
      second += (counts[g] ?? 0) * node * node;
    });
    const excess = (logSd: number): number => normalWeightsAtZero(nodes, Math.exp(logSd), scratch) - second / total;
    const [lower, upper] = SD_BOUNDS.map(Math.log);
    const atLower = excess(lower ?? 0);
    const atUpper = excess(upper ?? 0);
    if (atLower >= 0) values[sdSlot] = SD_BOUNDS[0];
    else if (atUpper <= 0) values[sdSlot] = SD_BOUNDS[1];
    else values[sdSlot] = Math.exp(findRoot(excess, lower ?? 0, upper ?? 0, atLower, atUpper, ROOT_TOLERANCE));
  }
  testletWeights(design, values, nodes);
}

/**
 * Returns -Σ n_g ln w_g of the testlets and subtracts its derivatives with respect to their σ from `gradient`: by
 * Fisher's identity, those of the marginal log-likelihood, Σ (n_g - N w_g) γ_g² / σ³.
 */
export function testletObjective(
  { testlets }: TestletDesign,
  values: Float64Array,
  nodes: Float64Array,
  gradient: Float64Array,
): number {
  let value = 0;
  for (const { counts, weights, sdSlot } of testlets) {
    const sd = values[sdSlot] ?? 1;
    let total = 0;
    for (const count of counts) total += count;
    let slope = 0;
    nodes.forEach((node, g) => {
      const count = counts[g] ?? 0;
      const weight = weights[g] ?? 0;
      if (count > 0) value -= count * Math.log(weight);
      slope += ((count - total * weight) * node * node) / sd ** 3;
    });
    gradient[sdSlot] = (gradient[sdSlot] ?? 0) - slope;
  }
  return value;
}

/** Returns each testlet's estimate. */
export function testletEstimates(
  { testlets }: TestletDesign,
  itemIds: readonly string[],
  values: Float64Array,
  standardError: ((slot: number) => number) | null,
): TestletEstimate[] {
  return testlets.map(({ id, items, sdSlot }) => ({
    id,
    items: items.map((item) => itemIds[item] ?? ""),
    sd: values[sdSlot] ?? 1,
    standardError: standardError === null ? null : standardError(sdSlot),
  }));
}

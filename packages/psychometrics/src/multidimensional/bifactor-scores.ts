import { MIN_PROBABILITY } from "../irt/internal.js";
import type { GroupNodes } from "./bifactor-expectation.js";
import type { ItemState } from "./calibration-setup.js";
import { itemScales } from "./m-step.js";

/** An item's scores at the nodes of its group, and where its entries sit among the group's. */
interface ItemScores {
  /** ∂ ln P_k / ∂ξ on the unbounded scale of calibration, by category, then node, then entry. */
  readonly table: Float64Array;
  /** The number of estimated entries of the item. */
  readonly size: number;
  /** The position of the item's first entry among the group's entries. */
  readonly start: number;
  readonly categories: number;
}

/** The expected number of persons at each node of a group who gave each pair of responses to two of its items. */
interface PairCounts {
  readonly first: number;
  readonly second: number;
  /** By response to the first item, response to the second, then node. */
  readonly counts: Float64Array;
}

/**
 * The scores of the estimated parameters whose derivatives depend on theta through the same nodes: the entries of the
 * items of a group, or the correlation. A score is the derivative of the log density of a response and theta.
 */
export interface GroupScores {
  /** The slot in the covariance of each of the group's entries. */
  readonly slots: Int32Array;
  /** The number of primary nodes. */
  readonly primaryNodes: number;
  /** The mean of the entries' scores given the primary node, by sub-pattern, primary node, then entry. */
  readonly means: Float64Array;
  /** The sum over persons of the expected products of the scores of every two entries, by rows. */
  readonly within: Float64Array;
  /** The scores of the group's items, in item order; none for the correlation. */
  readonly items: readonly ItemScores[];
  /** The expected persons for each pair of the group's items. */
  readonly pairs: readonly PairCounts[];
}

/**
 * Returns the scores of a group's items at its nodes, with empty means and products.
 *
 * @param slots - The slot of each of the group's entries, its items' entries in item order.
 * @param withSecond - Whether to make the tables of the persons expected for each pair of responses to two items, which
 *   only the second moments of the scores need.
 */
export function groupScores<TParameters>(
  group: GroupNodes<TParameters>,
  slots: Int32Array,
  primaryNodes: number,
  withSecond: boolean,
): GroupScores {
  let start = 0;
  const items = group.items.map((item): ItemScores => {
    const scores = {
      table: itemScoreTable(item, group),
      size: item.free.length,
      start,
      categories: item.categoryCount,
    };
    start += scores.size;
    return scores;
  });
  const pairs = (withSecond ? items : []).flatMap((first, j) =>
    items.slice(j + 1).map((second, offset) => ({
      first: j,
      second: j + 1 + offset,
      counts: new Float64Array(first.categories * second.categories * group.grid.count),
    })),
  );
  return {
    slots,
    primaryNodes,
    means: new Float64Array(group.patterns.count * primaryNodes * slots.length),
    within: new Float64Array(slots.length * slots.length),
    items,
    pairs,
  };
}

/** Returns an item's scores at the nodes of its group. */
function itemScoreTable<TParameters>(item: ItemState<TParameters>, group: GroupNodes<TParameters>): Float64Array {
  const { dimensions, count, nodes } = group.grid;
  const size = item.free.length;
  const length = item.vector.length;
  const scales = itemScales(item);
  const table = new Float64Array(item.categoryCount * count * size);
  const theta = new Float64Array(dimensions);
  const probabilities = new Float64Array(item.categoryCount);
  const derivatives = new Float64Array(item.categoryCount * length);
  for (let m = 0; m < count; m++) {
    for (let d = 0; d < dimensions; d++) theta[d] = nodes[m * dimensions + d] ?? 0;
    item.model.probabilities(item.parameters, theta, probabilities);
    item.model.parameterDerivatives(item.parameters, theta, derivatives);
    for (let k = 0; k < item.categoryCount; k++) {
      const probability = Math.max(probabilities[k] ?? 0, MIN_PROBABILITY);
      for (let l = 0; l < size; l++) {
        const derivative = derivatives[k * length + (item.free[l] ?? 0)] ?? 0;
        table[(k * count + m) * size + l] = (derivative / probability) * (scales[l] ?? 1);
      }
    }
  }
  return table;
}

/**
 * Adds a sub-pattern to its group: the mean score of each entry given each primary node, and, with `withSecond`, the
 * persons it expects at the nodes for each pair of responses to its items.
 *
 * @param conditional - The share of each node of the group among the nodes of its primary node.
 * @param mass - The persons expected at each primary node who gave the sub-pattern.
 */
export function addSubPattern<TParameters>(
  scores: GroupScores,
  nodes: GroupNodes<TParameters>,
  u: number,
  conditional: Float64Array,
  mass: Float64Array,
  withSecond: boolean,
): void {
  const { codes } = nodes.patterns;
  const itemCount = scores.items.length;
  const count = nodes.grid.count;
  scores.items.forEach((item, j) => {
    const code = codes[u * itemCount + j] ?? -1;
    if (code >= 0) addItemMeans(scores, item, u, code, count, conditional);
  });
  if (!withSecond) return;
  const expected = new Float64Array(count);
  const block = count / scores.primaryNodes;
  for (let m = 0; m < count; m++) expected[m] = (mass[Math.floor(m / block)] ?? 0) * (conditional[m] ?? 0);
  for (const pair of scores.pairs) {
    const k = codes[u * itemCount + pair.first] ?? -1;
    const l = codes[u * itemCount + pair.second] ?? -1;
    if (k >= 0 && l >= 0) {
      const categories = scores.items[pair.second]?.categories ?? 0;
      addRun(pair.counts, (k * categories + l) * count, expected, count);
    }
  }
}

/** Adds the mean scores of an item given its response `code` in sub-pattern `u`. */
function addItemMeans(
  { means, slots, primaryNodes }: GroupScores,
  { table, size, start }: ItemScores,
  u: number,
  code: number,
  count: number,
  conditional: Float64Array,
): void {
  const block = count / primaryNodes;
  for (let m = 0; m < count; m++) {
    const weight = conditional[m] ?? 0;
    const base = (u * primaryNodes + Math.floor(m / block)) * slots.length + start;
    for (let l = 0; l < size; l++) {
      means[base + l] = (means[base + l] ?? 0) + weight * (table[(code * count + m) * size + l] ?? 0);
    }
  }
}

/** Adds `source[q]` to `out[start + q]` for each q below `length`. */
function addRun(out: Float64Array, start: number, source: Float64Array, length: number): void {
  for (let q = 0; q < length; q++) out[start + q] = (out[start + q] ?? 0) + (source[q] ?? 0);
}

/**
 * Sets the sum over persons of the expected products of the scores of a group's entries, from what the sub-patterns
 * added: for two entries of one item, from its expected counts, and for two items, from the persons expected to give
 * each pair of responses at each node.
 */
export function finishProducts<TParameters>(scores: GroupScores, nodes: GroupNodes<TParameters>): void {
  const { count } = nodes.grid;
  nodes.items.forEach((item, j) => {
    const own = scores.items[j];
    if (own === undefined) return;
    for (let k = 0; k < item.categoryCount; k++) {
      for (let m = 0; m < count; m++) {
        const at = (k * count + m) * own.size;
        addProduct(scores, item.counts[k * count + m] ?? 0, [own, at], [own, at]);
      }
    }
  });
  for (const pair of scores.pairs) addPairProducts(scores, pair, count);
}

/** Adds the products of the scores of two items, weighted by the persons expected to give each pair of responses. */
function addPairProducts(scores: GroupScores, { first, second, counts }: PairCounts, count: number): void {
  const left = scores.items[first];
  const right = scores.items[second];
  if (left === undefined || right === undefined) return;
  for (let k = 0; k < left.categories; k++) {
    for (let l = 0; l < right.categories; l++) {
      for (let m = 0; m < count; m++) {
        const weight = counts[(k * right.categories + l) * count + m] ?? 0;
        if (weight === 0) continue;
        const leftAt: readonly [ItemScores, number] = [left, (k * count + m) * left.size];
        const rightAt: readonly [ItemScores, number] = [right, (l * count + m) * right.size];
        addProduct(scores, weight, leftAt, rightAt);
        addProduct(scores, weight, rightAt, leftAt);
      }
    }
  }
}

/** Adds `weight` times the outer product of two items' score vectors, each at its offset in the table, to the products. */
function addProduct(
  { slots, within }: GroupScores,
  weight: number,
  [left, leftAt]: readonly [ItemScores, number],
  [right, rightAt]: readonly [ItemScores, number],
): void {
  for (let l = 0; l < left.size; l++) {
    const scaled = weight * (left.table[leftAt + l] ?? 0);
    for (let r = 0; r < right.size; r++) {
      const at = (left.start + l) * slots.length + right.start + r;
      within[at] = (within[at] ?? 0) + scaled * (right.table[rightAt + r] ?? 0);
    }
  }
}

/** Adds the mean of a group's scores under a person's posterior, E_p[s], to `mean` at the group's slots. */
export function addPersonMeans(
  { slots, means, primaryNodes }: GroupScores,
  mean: Float64Array,
  u: number,
  posterior: Float64Array,
): void {
  slots.forEach((slot, a) => {
    let sum = 0;
    for (let q = 0; q < primaryNodes; q++) {
      sum += (posterior[q] ?? 0) * (means[(u * primaryNodes + q) * slots.length + a] ?? 0);
    }
    mean[slot] = (mean[slot] ?? 0) + sum;
  });
}

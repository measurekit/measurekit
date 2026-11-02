import { DataError } from "@measurekit/core";
import { rectangular } from "../math/internal.js";
import { type Grid, MAX_GRID_NODES } from "./grid.js";

/** The nodes and weights of the specific dimensions: one rule for them all, as they share one normal distribution. */
export interface SpecificRule {
  readonly nodes: Float64Array;
  readonly weights: Float64Array;
}

/** Returns the rectangular rule with `points` nodes of the specific dimensions, whose weights sum to 1. */
export function specificRule(points: number): SpecificRule {
  return rectangular(points);
}

/**
 * Returns the nodes of the items that share a specific dimension: every primary node with every node of the rule, the
 * specific nodes varying fastest. A node is a theta over all `dimensions`, 0 on the other specific dimensions. The items
 * without a specific dimension (`dimension` below 0) have the primary nodes alone.
 *
 * @throws `DataError` when the group has more than {@link MAX_GRID_NODES} nodes.
 */
export function groupGrid(primary: Grid, rule: SpecificRule, dimensions: number, dimension: number): Grid {
  const specific = dimension < 0 ? 1 : rule.nodes.length;
  const count = primary.count * specific;
  if (count > MAX_GRID_NODES) {
    throw new DataError(
      `A grid of ${primary.points} points in ${primary.dimensions + 1} dimensions has more than ${MAX_GRID_NODES} nodes; use fewer points.`,
    );
  }
  const nodes = new Float64Array(count * dimensions);
  for (let m = 0; m < count; m++) {
    const q = Math.floor(m / specific);
    for (let d = 0; d < primary.dimensions; d++)
      nodes[m * dimensions + d] = primary.nodes[q * primary.dimensions + d] ?? 0;
    if (dimension >= 0) nodes[m * dimensions + dimension] = rule.nodes[m % specific] ?? 0;
  }
  return { dimensions, points: primary.points, count, nodes };
}

/** Returns the weights of the specific nodes of a group: the rule's, or a single 1 for the items without a dimension. */
export function groupWeights(rule: SpecificRule, dimension: number): Float64Array {
  return dimension < 0 ? Float64Array.of(1) : rule.weights;
}

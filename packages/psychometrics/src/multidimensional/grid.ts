import { DataError, describe } from "@measurekit/core";
import { rectangular } from "../math/internal.js";

/** The most nodes a grid may have, which bounds the memory and time of an integral over it. */
export const MAX_GRID_NODES = 1_000_000;

/** The number of nodes per dimension that mirt's `quadpts` defaults to for 1 to 5 dimensions, and for more. */
const DEFAULT_POINTS = [61, 31, 15, 9, 7] as const;
const DEFAULT_POINTS_ABOVE = 3;

/** A grid of nodes over several dimensions: every combination of the nodes of one rectangular rule. */
export interface Grid {
  /** The number of dimensions. */
  readonly dimensions: number;
  /** The number of nodes per dimension. */
  readonly points: number;
  /** The number of nodes, `points ** dimensions`. */
  readonly count: number;
  /** The nodes by rows: node `q` is `nodes[q * dimensions]` to `nodes[(q + 1) * dimensions - 1]`. */
  readonly nodes: Float64Array;
}

/** Returns the number of nodes per dimension that a grid of `dimensions` dimensions has by default. */
export function defaultPoints(dimensions: number): number {
  return DEFAULT_POINTS[dimensions - 1] ?? DEFAULT_POINTS_ABOVE;
}

/**
 * Returns the grid of `points` equally spaced nodes per dimension from -6 to 6, the first dimension varying fastest, as
 * mirt's.
 *
 * @throws `DataError` when the grid has more than {@link MAX_GRID_NODES} nodes.
 */
export function productGrid(dimensions: number, points: number): Grid {
  const count = points ** dimensions;
  if (!(count <= MAX_GRID_NODES)) {
    throw new DataError(
      `A grid of ${describe(points)} points in ${dimensions} dimensions has more than ${MAX_GRID_NODES} nodes; use fewer points.`,
    );
  }
  const axis = rectangular(points).nodes;
  const nodes = new Float64Array(count * dimensions);
  for (let q = 0; q < count; q++) {
    let rest = q;
    for (let d = 0; d < dimensions; d++) {
      nodes[q * dimensions + d] = axis[rest % points] ?? 0;
      rest = Math.floor(rest / points);
    }
  }
  return { dimensions, points, count, nodes };
}

/**
 * Writes the log weights of the normal distribution with the given mean and precision matrix on the nodes into `out`,
 * normalized to sum to 1 as probabilities, and returns the log of their sum before normalization.
 *
 * @param precision - The inverse of the covariance matrix, by rows.
 */
export function normalLogWeights(
  grid: Grid,
  mean: ArrayLike<number>,
  precision: Float64Array,
  out: Float64Array,
): number {
  let largest = Number.NEGATIVE_INFINITY;
  for (let q = 0; q < out.length; q++) {
    out[q] = -0.5 * quadraticForm(grid, q, mean, precision);
    largest = Math.max(largest, out[q] ?? 0);
  }
  let total = 0;
  for (let q = 0; q < out.length; q++) total += Math.exp((out[q] ?? 0) - largest);
  const logTotal = largest + Math.log(total);
  for (let q = 0; q < out.length; q++) out[q] = (out[q] ?? 0) - logTotal;
  return logTotal;
}

/** Returns (θ_q - μ)ᵀ Ω (θ_q - μ) for the node q. */
function quadraticForm(
  { dimensions, nodes }: Grid,
  q: number,
  mean: ArrayLike<number>,
  precision: Float64Array,
): number {
  let sum = 0;
  for (let i = 0; i < dimensions; i++) {
    const row = (nodes[q * dimensions + i] ?? 0) - (mean[i] ?? 0);
    for (let j = 0; j < dimensions; j++) {
      sum += row * (precision[i * dimensions + j] ?? 0) * ((nodes[q * dimensions + j] ?? 0) - (mean[j] ?? 0));
    }
  }
  return sum;
}

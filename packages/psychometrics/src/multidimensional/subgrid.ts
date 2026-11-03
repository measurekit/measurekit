/**
 * A set of dimensions as a bit mask: bit d is set when dimension d belongs to it. An item whose probabilities depend on
 * theta through some dimensions only needs its scores on the grid over those dimensions, a subgrid, which has fewer
 * nodes than the full grid.
 */
export type DimensionSet = number;

/** Returns the number of dimensions in a set. */
function setSize(set: DimensionSet): number {
  let size = 0;
  for (let rest = set; rest > 0; rest >>= 1) size += rest & 1;
  return size;
}

/** Returns the dimensions of a set in ascending order. */
function members(set: DimensionSet): number[] {
  const dimensions: number[] = [];
  for (let d = 0; set >> d > 0; d++) if ((set >> d) & 1) dimensions.push(d);
  return dimensions;
}

/**
 * Returns the number of nodes of the subgrid over a set of dimensions: `points` to the power of its size. A node of the
 * subgrid has an index in which the first dimension of the set varies fastest, as in the full grid.
 */
export function subgridCount(set: DimensionSet, points: number): number {
  return points ** setSize(set);
}

/**
 * Returns the index in the subgrid over `to` of each node of the subgrid over `from`, a set that contains `to`: the
 * marginalization of a distribution on `from` onto `to` adds the weight of node r to the entry at `result[r]`.
 */
export function restriction(from: DimensionSet, to: DimensionSet, points: number): Int32Array {
  const source = members(from);
  const target = members(to);
  const result = new Int32Array(subgridCount(from, points));
  const positions = new Int32Array(source.reduce((most, member) => Math.max(most, member), 0) + 1);
  for (let r = 0; r < result.length; r++) {
    let rest = r;
    for (const d of source) {
      positions[d] = rest % points;
      rest = Math.floor(rest / points);
    }
    let index = 0;
    for (let k = target.length - 1; k >= 0; k--) index = index * points + (positions[target[k] ?? 0] ?? 0);
    result[r] = index;
  }
  return result;
}

/**
 * Writes the coordinates of the nodes of the subgrid over a set into `out`, by rows of `dimensions` entries, with 0 for
 * a dimension outside the set. `axis` holds the nodes of one dimension.
 */
export function subgridNodes(set: DimensionSet, axis: Float64Array, dimensions: number, out: Float64Array): void {
  const included = members(set);
  const points = axis.length;
  const count = subgridCount(set, points);
  out.fill(0, 0, count * dimensions);
  for (let r = 0; r < count; r++) {
    let rest = r;
    for (const d of included) {
      out[r * dimensions + d] = axis[rest % points] ?? 0;
      rest = Math.floor(rest / points);
    }
  }
}

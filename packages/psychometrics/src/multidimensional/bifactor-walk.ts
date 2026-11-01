import type { GroupPatterns } from "./bifactor-patterns.js";

/**
 * The number of items after which a row of products of probabilities is divided by its largest entry at each primary
 * node. A probability can be as small as 1e-19 at a node of weight that matters, and 16 of them still do not underflow.
 */
const RESCALE_INTERVAL = 16;

/** The tables a walk reads: the probabilities of each item of a group at the group's nodes. */
export interface WalkInput {
  readonly patterns: GroupPatterns;
  /** For each item, its probabilities by category, then node. The walk reads them as they stand. */
  readonly tables: readonly Float64Array[];
  /** The number of primary nodes. */
  readonly primaryNodes: number;
  /** The weights of the specific nodes: the nodes of the group are the primary nodes times these, the specific first. */
  readonly weights: Float64Array;
}

/**
 * Calls `visit` with each sub-pattern that answers an item, in order, its row and its shift: the row holds, at each
 * node, the weight of the specific node times the probabilities of the responses, divided by e raised to the shift of
 * the node's primary node.
 */
export type Walk = (visit: (subPattern: number, row: Float64Array, shift: Float64Array) => void) => void;

/**
 * Returns the walk over the sub-patterns of a group: the products of the probabilities of their responses over the
 * nodes, each shared by the sub-patterns that begin alike.
 *
 * @remarks
 * Sub-patterns are sorted, so a row is rebuilt only from the first item where a sub-pattern differs from the one
 * before it. The product over a group's items is taken in linear space, as a log would need an exponential for each
 * sub-pattern and node; the shifts keep long groups from underflowing, and the caller's logarithm of a sum over the
 * specific nodes adds them back.
 */
export function createWalk({ patterns, tables, primaryNodes, weights }: WalkInput): Walk {
  const { width, count, codes } = patterns;
  const specificNodes = weights.length;
  const nodes = primaryNodes * specificNodes;
  const rows = Array.from({ length: width + 1 }, () => new Float64Array(nodes));
  const shifts = Array.from({ length: width + 1 }, () => new Float64Array(primaryNodes));
  // The row and the shift that hold the products of the first d items: a skipped item shares those of the one before.
  const rowOf = new Int32Array(width + 1);
  const shiftOf = new Int32Array(width + 1);
  const root = rows[0] ?? new Float64Array(nodes);
  for (let m = 0; m < nodes; m++) root[m] = weights[m % specificNodes] ?? 0;

  /** Builds the row of the first d + 1 items from the row of the first d, for the response `code` to the item d. */
  const step = (d: number, code: number): void => {
    rowOf[d + 1] = rowOf[d] ?? 0;
    shiftOf[d + 1] = shiftOf[d] ?? 0;
    if (code < 0) return;
    const row = rows[d + 1] ?? root;
    multiplyRows(row, rows[rowOf[d] ?? 0] ?? root, tables[d] ?? row, code * nodes, nodes);
    rowOf[d + 1] = d + 1;
    if ((d + 1) % RESCALE_INTERVAL === 0) {
      rescale(row, shifts[shiftOf[d] ?? 0] ?? row, shifts[d + 1] ?? row, specificNodes);
      shiftOf[d + 1] = d + 1;
    }
  };

  /** Builds the rows of sub-pattern `u` from the first item where it differs from the one before it. */
  const build = (u: number): void => {
    const base = u * width;
    const first = u === 0 ? 0 : sharedPrefix(codes, base - width, base, width);
    for (let d = first; d < width; d++) step(d, codes[base + d] ?? -1);
  };

  return (visit) => {
    for (let u = 0; u < count; u++) {
      build(u);
      if (patterns.answered[u] === 1) visit(u, rows[rowOf[width] ?? 0] ?? root, shifts[shiftOf[width] ?? 0] ?? root);
    }
  };
}

/** Returns the number of entries at the start of which two rows of codes agree. */
function sharedPrefix(codes: Int32Array, a: number, b: number, width: number): number {
  let length = 0;
  while (length < width && codes[a + length] === codes[b + length]) length++;
  return length;
}

/** Writes `a[m] * table[offset + m]` into `out[m]` for each m below `length`. */
function multiplyRows(out: Float64Array, a: Float64Array, table: Float64Array, offset: number, length: number): void {
  for (let m = 0; m < length; m++) out[m] = (a[m] ?? 0) * (table[offset + m] ?? 0);
}

/**
 * Divides the entries of each primary node's block of `row` by their largest, and writes into `shift` the shift of
 * the node before plus the logarithm of that largest; a block of zeros stays so.
 */
function rescale(row: Float64Array, before: Float64Array, shift: Float64Array, block: number): void {
  for (let q = 0; q < shift.length; q++) {
    const start = q * block;
    let largest = 0;
    for (let r = 0; r < block; r++) largest = Math.max(largest, row[start + r] ?? 0);
    shift[q] = before[q] ?? 0;
    if (largest > 0) {
      for (let r = 0; r < block; r++) row[start + r] = (row[start + r] ?? 0) / largest;
      shift[q] = (shift[q] ?? 0) + Math.log(largest);
    }
  }
}

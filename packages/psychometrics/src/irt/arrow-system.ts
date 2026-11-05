import { multiplyVector } from "@measurekit/core";
import { invert, keepDescent } from "../math/internal.js";

/**
 * The Fisher-scoring system of a block of the M-step. Its Hessian is arrow-shaped: each item's own parameters meet only
 * each other and the parameters that the block's items share, such as the 1PL's `a`, the RSM's steps or the thresholds
 * of the first stage of an unfolding calibration. It is held by those parts, so that it takes memory and time that grow
 * with the items, where the whole matrix of every pair of the block's parameters grows with their square and its
 * inverse with their cube.
 */
export interface ArrowSystem {
  /** For each item, the block positions of its own parameters, in the order of its free entries. */
  readonly own: readonly Int32Array[];
  /** For each item and each of its free entries: its index among the item's own parameters, or -1 - s for shared s. */
  readonly places: readonly Int32Array[];
  /** The block positions of the shared parameters. */
  readonly shared: Int32Array;
  /** For each block position: the item that owns it, or -1 for a shared parameter. */
  readonly owners: Int32Array;
  /** For each block position: its index among its item's own parameters, or among the shared ones. */
  readonly indices: Int32Array;
  /** Each item's Hessian of its own parameters, D, by rows. */
  readonly diagonal: readonly Float64Array[];
  /** Each item's Hessian between its own parameters and the shared ones, B, by rows of its own. */
  readonly coupling: readonly Float64Array[];
  /** The Hessian of the shared parameters, C, by rows. */
  readonly corner: Float64Array;
  /** Work space: each item's D⁻¹ B, by rows of its own. */
  readonly reduced: readonly Float64Array[];
  /** Work space: each item's D⁻¹ g of its own parameters. */
  readonly reducedGradient: readonly Float64Array[];
}

/**
 * Returns the empty system of a block whose items have the free slots `slots`, each in the order of its free entries,
 * at the block positions that `parameters` lists: a slot of more than one item is shared, every other is its item's.
 */
export function arrowSystem(slots: readonly Int32Array[], parameters: Int32Array): ArrowSystem {
  const position = new Map(Array.from(parameters, (slot, l) => [slot, l]));
  const holders = new Map<number, number>();
  for (const itemSlots of slots) {
    for (const slot of new Set(itemSlots)) holders.set(slot, (holders.get(slot) ?? 0) + 1);
  }
  const isShared = (slot: number): boolean => (holders.get(slot) ?? 0) > 1;
  const owners = new Int32Array(parameters.length).fill(-1);
  const indices = new Int32Array(parameters.length);
  const shared = Int32Array.from(Array.from(parameters.keys()).filter((l) => isShared(parameters[l] ?? 0)));
  shared.forEach((l, s) => {
    indices[l] = s;
  });
  const own: Int32Array[] = [];
  const places = slots.map((itemSlots, t) => {
    const mine: number[] = [];
    const itemPlaces = Int32Array.from(itemSlots, (slot) => {
      const l = position.get(slot) ?? 0;
      if (isShared(slot)) return -1 - (indices[l] ?? 0);
      if (owners[l] !== t) {
        owners[l] = t;
        indices[l] = mine.length;
        mine.push(l);
      }
      return indices[l] ?? 0;
    });
    own.push(Int32Array.from(mine));
    return itemPlaces;
  });
  const sharedCount = shared.length;
  return {
    own,
    places,
    shared,
    owners,
    indices,
    diagonal: own.map(({ length }) => new Float64Array(length * length)),
    coupling: own.map(({ length }) => new Float64Array(length * sharedCount)),
    corner: new Float64Array(sharedCount * sharedCount),
    reduced: own.map(({ length }) => new Float64Array(length * sharedCount)),
    reducedGradient: own.map(({ length }) => new Float64Array(length)),
  };
}

/** Sets the system's Hessian to 0. */
export function clearSystem(system: ArrowSystem): void {
  for (const block of system.diagonal) block.fill(0);
  for (const block of system.coupling) block.fill(0);
  system.corner.fill(0);
}

/**
 * Adds the Hessian `local` of item t's free entries, `count` of them, by rows, to the system: the parts of its own
 * parameters, of its own with the shared ones, and of the shared ones; the shared ones with its own are B's transpose.
 */
export function addItemHessian(system: ArrowSystem, t: number, local: Float64Array, count: number): void {
  const places = system.places[t] ?? new Int32Array();
  for (let e = 0; e < count; e++) {
    for (let f = 0; f < count; f++) addCell(system, t, places[e] ?? 0, places[f] ?? 0, local[e * count + f] ?? 0);
  }
}

/** Adds `value` at the row and column places of item t: own indices from 0, shared ones as -1 - s. */
function addCell(system: ArrowSystem, t: number, row: number, column: number, value: number): void {
  const sharedCount = system.shared.length;
  if (row >= 0 && column >= 0) {
    const diagonal = system.diagonal[t] ?? new Float64Array();
    const cell = row * (system.own[t]?.length ?? 0) + column;
    diagonal[cell] = (diagonal[cell] ?? 0) + value;
  } else if (row >= 0) {
    const coupling = system.coupling[t] ?? new Float64Array();
    const cell = row * sharedCount - 1 - column;
    coupling[cell] = (coupling[cell] ?? 0) + value;
  } else if (column < 0) {
    const cell = (-1 - row) * sharedCount - 1 - column;
    system.corner[cell] = (system.corner[cell] ?? 0) + value;
  }
}

/** Adds `value` to the diagonal entry of block position l. */
export function addDiagonal(system: ArrowSystem, l: number, value: number): void {
  const owner = system.owners[l] ?? -1;
  const index = system.indices[l] ?? 0;
  if (owner < 0) {
    const cell = index * (system.shared.length + 1);
    system.corner[cell] = (system.corner[cell] ?? 0) + value;
    return;
  }
  const diagonal = system.diagonal[owner] ?? new Float64Array();
  const cell = index * ((system.own[owner]?.length ?? 0) + 1);
  diagonal[cell] = (diagonal[cell] ?? 0) + value;
}

/** Replaces the Hessian H with J H J, J the diagonal of `scales` at the block positions. */
export function scaleSystem(system: ArrowSystem, scales: Float64Array): void {
  system.own.forEach((positions, t) => {
    scaleBlock(system.diagonal[t] ?? new Float64Array(), positions, positions, scales);
    scaleBlock(system.coupling[t] ?? new Float64Array(), positions, system.shared, scales);
  });
  scaleBlock(system.corner, system.shared, system.shared, scales);
}

/** Multiplies each entry (a, b) of `matrix`, by rows, by the scales at the row position a and the column position b. */
function scaleBlock(matrix: Float64Array, rows: Int32Array, columns: Int32Array, scales: Float64Array): void {
  for (let a = 0; a < rows.length; a++) {
    const scale = scales[rows[a] ?? 0] ?? 0;
    for (let b = 0; b < columns.length; b++) {
      const cell = a * columns.length + b;
      matrix[cell] = (matrix[cell] ?? 0) * scale * (scales[columns[b] ?? 0] ?? 0);
    }
  }
}

/**
 * Writes the Newton direction -H⁻¹ g into `out`, by the Schur complement of the items' own parameters, or the negative
 * gradient where H is not positive definite.
 *
 * @remarks
 * With D each item's Hessian of its own parameters, B its coupling with the shared ones and C theirs, the shared part of
 * the direction solves (C - Σ Bᵀ D⁻¹ B) d_s = -g_s + Σ Bᵀ D⁻¹ g_o, and each item's own part is -D⁻¹ (g_o + B d_s): the
 * inverses of the items' small blocks and of the shared parameters' block, in time and memory that grow with the items.
 * A block without shared parameters takes -D⁻¹ g of its one item, as Newton's method of the whole matrix does.
 */
export function arrowDirection(system: ArrowSystem, gradient: Float64Array, out: Float64Array): void {
  keepDescent(solve(system, gradient, out), gradient, out);
}

/** Writes -H⁻¹ g into `out` and returns true, or returns false when a block to invert is singular. */
function solve(system: ArrowSystem, gradient: Float64Array, out: Float64Array): boolean {
  const { shared } = system;
  const schur = Float64Array.from(system.corner);
  const right = Float64Array.from(shared, (l) => -(gradient[l] ?? 0));
  for (let t = 0; t < system.own.length; t++) {
    if (!reduceItem(system, t, gradient, schur, right)) return false;
  }
  const sharedDirection = new Float64Array(shared.length);
  if (shared.length > 0) {
    const inverse = invert(schur, shared.length);
    if (inverse === null) return false;
    multiplyVector(inverse, right, sharedDirection);
  }
  shared.forEach((l, s) => {
    out[l] = sharedDirection[s] ?? 0;
  });
  system.own.forEach((positions, t) => {
    ownDirection(system, t, positions, sharedDirection, out);
  });
  return true;
}

/**
 * Eliminates item t's own parameters: keeps D⁻¹ B and D⁻¹ g_o, and subtracts Bᵀ D⁻¹ B from the Schur complement and adds
 * Bᵀ D⁻¹ g_o to its right side. Returns false when D is singular.
 */
function reduceItem(
  system: ArrowSystem,
  t: number,
  gradient: Float64Array,
  schur: Float64Array,
  right: Float64Array,
): boolean {
  const positions = system.own[t] ?? new Int32Array();
  const count = positions.length;
  if (count === 0) return true;
  const inverse = invert(system.diagonal[t] ?? new Float64Array(), count);
  if (inverse === null) return false;
  const sharedCount = system.shared.length;
  const coupling = system.coupling[t] ?? new Float64Array();
  const reduced = system.reduced[t] ?? new Float64Array();
  const reducedGradient = system.reducedGradient[t] ?? new Float64Array();
  multiplyVector(
    inverse,
    Float64Array.from(positions, (l) => gradient[l] ?? 0),
    reducedGradient,
  );
  multiplyBlocks(inverse, coupling, count, sharedCount, reduced);
  for (let a = 0; a < count; a++) addReducedRow(coupling, reduced, reducedGradient, a, schur, right);
  return true;
}

/** Subtracts row a's part of Bᵀ D⁻¹ B from the Schur complement and adds its part of Bᵀ D⁻¹ g_o to the right side. */
function addReducedRow(
  coupling: Float64Array,
  reduced: Float64Array,
  reducedGradient: Float64Array,
  a: number,
  schur: Float64Array,
  right: Float64Array,
): void {
  const sharedCount = right.length;
  for (let s = 0; s < sharedCount; s++) {
    const entry = coupling[a * sharedCount + s] ?? 0;
    right[s] = (right[s] ?? 0) + entry * (reducedGradient[a] ?? 0);
    for (let r = 0; r < sharedCount; r++) {
      schur[s * sharedCount + r] = (schur[s * sharedCount + r] ?? 0) - entry * (reduced[a * sharedCount + r] ?? 0);
    }
  }
}

/** Writes the product of the `count` × `count` matrix `left` and the `count` × `columns` matrix `right` into `out`. */
function multiplyBlocks(left: Float64Array, right: Float64Array, count: number, columns: number, out: Float64Array) {
  for (let a = 0; a < count; a++) {
    for (let s = 0; s < columns; s++) {
      let sum = 0;
      for (let b = 0; b < count; b++) sum += (left[a * count + b] ?? 0) * (right[b * columns + s] ?? 0);
      out[a * columns + s] = sum;
    }
  }
}

/** Writes item t's own part of the direction, -(D⁻¹ g_o + D⁻¹ B d_s), at its positions of `out`. */
function ownDirection(
  system: ArrowSystem,
  t: number,
  positions: Int32Array,
  sharedDirection: Float64Array,
  out: Float64Array,
): void {
  const sharedCount = sharedDirection.length;
  const reduced = system.reduced[t] ?? new Float64Array();
  const reducedGradient = system.reducedGradient[t] ?? new Float64Array();
  positions.forEach((l, a) => {
    let sum = reducedGradient[a] ?? 0;
    for (let s = 0; s < sharedCount; s++) sum += (reduced[a * sharedCount + s] ?? 0) * (sharedDirection[s] ?? 0);
    out[l] = -sum;
  });
}

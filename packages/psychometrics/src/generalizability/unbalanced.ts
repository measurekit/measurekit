import { DataError } from "@measurekit/core";
import { type Effect, effectsOf, facetsOf, namesOf, type Structure } from "./structure.js";

/**
 * The most facets, the object of measurement included, that an unbalanced design may have: its estimator reads the scores
 * once for each pair of the up to 2^6 - 1 effects, and a design of more facets has more effects than any study lists.
 */
const MAX_UNBALANCED_FACETS = 6;

/** One effect of a design with the quantities of its analysis of variance and its estimated variance component. */
export interface EffectEstimate {
  readonly effect: Effect;
  readonly degreesOfFreedom: number;
  readonly sumOfSquares: number;
  readonly meanSquare: number;
  readonly variance: number;
}

/** The cells of an effect: the combinations of the elements of its facets, and how many scores each holds. */
interface Cells {
  /** For each score, the number of its cell. */
  readonly ids: Int32Array;
  /** The number of scores in each cell. */
  readonly counts: Float64Array;
}

/**
 * Estimates the variance components of a design with empty cells by Henderson's Method 1, the analogous analysis of
 * variance of Brennan (2001) and the estimator of urGENOVA.
 *
 * @remarks
 * With T of an effect the sum over its cells of the squared sum of the cell's scores over their count, the expectation
 * of T less that of the grand mean's is the sum of the variance components, each multiplied by a coefficient k that
 * depends on the counts alone: k of effects α and β is the sum over the cells c of α and d of β of n_cd² / n_c, with n
 * the number of scores in a cell. The system of one equation for each effect is solved for the components. The sum of
 * squares of an effect is its T less those of the effects within it, and its degrees of freedom its number of cells
 * less 1 and less those of the effects within it.
 *
 * @param values - The scores, which need no order.
 * @param coordinates - For each score, the numbers of its elements by facet.
 * @param sizes - The most elements a facet has in any combination of the elements it is nested in.
 * @param describeCell - Names the combination of elements of a score for an error message.
 * @throws `DataError` when the design has more than 6 facets, two scores are in one cell, or an effect has no degrees
 * of freedom, which is when its variance component is confounded with others.
 */
export function unbalancedEstimates(
  values: readonly number[],
  coordinates: readonly (readonly number[])[],
  sizes: readonly number[],
  structure: Structure,
  describeCell: (cell: readonly number[]) => string,
): EffectEstimate[] {
  if (structure.names.length > MAX_UNBALANCED_FACETS) {
    throw new DataError(
      `An unbalanced design has at most ${MAX_UNBALANCED_FACETS} facets with the object of measurement, found ${structure.names.length}.`,
    );
  }
  const effects = effectsOf(structure);
  const cells = cellsOfEffects(effects, coordinates, sizes, structure);
  rejectRepeated(cells[cells.length - 1], coordinates, describeCell);
  const within = effects.map((effect) => effects.flatMap((other, k) => (isWithin(other, effect) ? [k] : [])));
  const degrees = degreesOfFreedom(cells, within);
  degrees.forEach((degree, k) => {
    if (!(degree > 0)) throw confounded(structure, effects[k]);
  });
  const centered = centeredValues(values);
  const totals = cells.map((own) => squaredSums(own, centered));
  const solution = solve(expectedValues(effects, cells, values.length), totals, effects.length);
  const sums = new Float64Array(effects.length);
  return effects.map((effect, k) => {
    sums[k] = (totals[k] ?? 0) - (within[k] ?? []).reduce((total, other) => total + (sums[other] ?? 0), 0);
    const sumOfSquares = sums[k] ?? 0;
    const degreesOfFreedom = degrees[k] ?? 0;
    return {
      effect,
      degreesOfFreedom,
      sumOfSquares,
      meanSquare: sumOfSquares / degreesOfFreedom,
      variance: solution[k] ?? 0,
    };
  });
}

/** Returns whether an effect's facets are a part of another's and not the same, so that it is within the other. */
function isWithin(inner: Effect, outer: Effect): boolean {
  return inner.mask !== outer.mask && (inner.mask & ~outer.mask) === 0;
}

/** Returns the error for a variance component that these scores cannot estimate. */
function confounded(structure: Structure, effect: Effect | undefined): DataError {
  const own = namesOf(structure, effect?.own ?? 0).join(" x ");
  const nestedIn = namesOf(structure, effect?.nestedIn ?? 0).join(" x ");
  return new DataError(
    `The variance component of "${nestedIn === "" ? own : `${own}:${nestedIn}`}" cannot be estimated from these scores: it is confounded with those of other effects, as when a facet has one element or two facets always occur together.`,
  );
}

/**
 * Returns the cells of every effect. A cell of an effect is a cell of the effect without one of its facets, the one that
 * no other facet of the effect is nested in, split by the elements of that facet.
 */
function cellsOfEffects(
  effects: readonly Effect[],
  coordinates: readonly (readonly number[])[],
  sizes: readonly number[],
  structure: Structure,
): Cells[] {
  const whole: Cells = { ids: new Int32Array(coordinates.length), counts: Float64Array.of(coordinates.length) };
  const cells: Cells[] = [];
  for (const { mask } of effects) {
    const members = facetsOf(mask);
    const leaf = members.find((facet) =>
      members.every((other) => ((structure.ancestors[other] ?? 0) >> facet) % 2 === 0),
    );
    const parent = effects.findIndex((other) => other.mask === (mask & ~(1 << (leaf ?? 0))));
    cells.push(split(cells[parent] ?? whole, coordinates, leaf ?? 0, sizes[leaf ?? 0] ?? 1));
  }
  return cells;
}

/** Splits cells by the elements of a facet, numbering the new cells in the order they appear. */
function split(parent: Cells, coordinates: readonly (readonly number[])[], facet: number, size: number): Cells {
  const ids = new Int32Array(coordinates.length);
  const known = new Map<number, number>();
  coordinates.forEach((cell, score) => {
    const key = (parent.ids[score] ?? 0) * size + (cell[facet] ?? 0);
    let id = known.get(key);
    if (id === undefined) {
      id = known.size;
      known.set(key, id);
    }
    ids[score] = id;
  });
  const counts = new Float64Array(known.size);
  for (const id of ids) counts[id] = (counts[id] ?? 0) + 1;
  return { ids, counts };
}

/** Throws for the first score whose cell, with every facet's element given, holds another score too. */
function rejectRepeated(
  full: Cells | undefined,
  coordinates: readonly (readonly number[])[],
  describeCell: (cell: readonly number[]) => string,
): void {
  const score = full?.ids.findIndex((id) => (full.counts[id] ?? 0) > 1) ?? -1;
  if (score >= 0) {
    throw new DataError(
      `There is more than one score for ${describeCell(coordinates[score] ?? [])}; an unbalanced design allows at most one for each combination of elements.`,
    );
  }
}

/** Returns the degrees of freedom of every effect: its cells less 1 and less those of the effects within it. */
function degreesOfFreedom(cells: readonly Cells[], within: readonly (readonly number[])[]): number[] {
  const degrees: number[] = [];
  cells.forEach(({ counts }, k) => {
    degrees.push(counts.length - 1 - (within[k] ?? []).reduce((total, other) => total + (degrees[other] ?? 0), 0));
  });
  return degrees;
}

/** Returns the scores less their mean, which leaves the sums of squares as they are and keeps them accurate far from 0. */
function centeredValues(values: readonly number[]): Float64Array {
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  return Float64Array.from(values, (value) => value - mean);
}

/** Returns T of an effect for scores with mean 0: the squared sum of each cell's scores over its count, summed. */
function squaredSums(cells: Cells, centered: Float64Array): number {
  const sums = new Float64Array(cells.counts.length);
  cells.ids.forEach((id, score) => {
    sums[id] = (sums[id] ?? 0) + (centered[score] ?? 0);
  });
  return sums.reduce((total, sum, id) => total + (sum * sum) / (cells.counts[id] ?? 1), 0);
}

/**
 * Returns the coefficients of the variance components in the expectations of T of the effects less that of the grand
 * mean, by rows of effects and columns of components: k of the two effects less the sum of the squared counts of the
 * component's cells over the number of scores.
 */
function expectedValues(effects: readonly Effect[], cells: readonly Cells[], count: number): Float64Array {
  const size = effects.length;
  const matrix = new Float64Array(size * size);
  effects.forEach((component, b) => {
    const grand = (cells[b]?.counts ?? new Float64Array()).reduce((total, n) => total + n * n, 0) / count;
    effects.forEach((effect, a) => {
      const contained = (component.mask & ~effect.mask) === 0;
      // The cells of an effect that holds the component's facets lie each in one cell of the component: k is the count.
      const joint = contained ? count : pairSum(cells, effects, a, b);
      matrix[a * size + b] = joint - grand;
    });
  });
  return matrix;
}

/** Returns k of two effects whose cells overlap in the cells of their union: the sum over scores of n_cd / n_c. */
function pairSum(cells: readonly Cells[], effects: readonly Effect[], a: number, b: number): number {
  const first = cells[a];
  const union = cells[effects.findIndex((other) => other.mask === ((effects[a]?.mask ?? 0) | (effects[b]?.mask ?? 0)))];
  let total = 0;
  if (first === undefined || union === undefined) return total;
  for (let score = 0; score < first.ids.length; score++) {
    total += (union.counts[union.ids[score] ?? 0] ?? 0) / (first.counts[first.ids[score] ?? 0] ?? 1);
  }
  return total;
}

/**
 * Solves a system of `size` equations, by rows, by Gaussian elimination with partial pivoting. Every effect having
 * degrees of freedom, the system has a solution: an effect without them is rejected before.
 */
function solve(matrix: Float64Array, right: ArrayLike<number>, size: number): Float64Array {
  const a = Float64Array.from(matrix);
  const x = Float64Array.from(right);
  for (let column = 0; column < size; column++) {
    swapRows(a, x, size, pivotRow(a, size, column), column);
    eliminateBelow(a, x, size, column);
  }
  for (let row = size - 1; row >= 0; row--) {
    let rest = x[row] ?? 0;
    for (let k = row + 1; k < size; k++) rest -= (a[row * size + k] ?? 0) * (x[k] ?? 0);
    x[row] = rest / (a[row * size + row] ?? 1);
  }
  return x;
}

/** Subtracts multiples of the pivot row `column` from the rows below it, to clear their coefficients in `column`. */
function eliminateBelow(a: Float64Array, x: Float64Array, size: number, column: number): void {
  const pivotValue = a[column * size + column] ?? 1;
  for (let row = column + 1; row < size; row++) {
    const factor = (a[row * size + column] ?? 0) / pivotValue;
    for (let k = column; k < size; k++) {
      a[row * size + k] = (a[row * size + k] ?? 0) - factor * (a[column * size + k] ?? 0);
    }
    x[row] = (x[row] ?? 0) - factor * (x[column] ?? 0);
  }
}

/** Returns the row, from `column` down, with the largest coefficient in `column`. */
function pivotRow(a: Float64Array, size: number, column: number): number {
  let best = column;
  for (let row = column + 1; row < size; row++) {
    if (Math.abs(a[row * size + column] ?? 0) > Math.abs(a[best * size + column] ?? 0)) best = row;
  }
  return best;
}

/** Swaps two rows of the coefficients and of the right-hand side. */
function swapRows(a: Float64Array, x: Float64Array, size: number, first: number, second: number): void {
  if (first === second) return;
  for (let k = 0; k < size; k++) {
    const value = a[first * size + k] ?? 0;
    a[first * size + k] = a[second * size + k] ?? 0;
    a[second * size + k] = value;
  }
  const value = x[first] ?? 0;
  x[first] = x[second] ?? 0;
  x[second] = value;
}

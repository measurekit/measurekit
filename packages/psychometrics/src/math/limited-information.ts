import { cholesky } from "@measurekit/core";
import { forwardSolve } from "./linear-algebra.js";
import { addOuterProduct, mirror } from "./matrix.js";
import { noncentralChiSquareCdf } from "./noncentral-chi-square.js";
import { findRoot } from "./root.js";

/**
 * The most items of a limited-information statistic: their covariance matrix has 1,830 rows, 27 MB, and its factor takes
 * 0.6 s; at 80 items, 84 MB and 3 s.
 */
export const MAX_MOMENT_ITEMS = 60;

// The moments of a limited-information statistic. Its rows are, first, one or more functions h of each item's score x,
// and then the products x_i x_j of the pairs i < j. Given a node of the population the items are independent.

/** The first-order rows: functions h of one item's score, such as the score or the indicator of a category. */
export interface FirstOrderRows {
  /** The item of each row. Different rows of one item are indicators of different categories, so h_r h_s = 0. */
  readonly items: Int32Array;
  /** E(h | node), by node, then row. */
  readonly means: Float64Array;
  /** E(h x | node), by node, then row. */
  readonly scores: Float64Array;
  /** E(h² | node), by node, then row. */
  readonly squares: Float64Array;
}

/** What the moments integrate: the items' scores given each node of the population, and the nodes' weights. */
export interface MomentTables {
  /** The weight of each node, summing to 1. */
  readonly weights: Float64Array;
  /** The number of items. */
  readonly itemCount: number;
  /** E(x | node), by node, then item. */
  readonly means: Float64Array;
  /** E(x² | node), by node, then item. */
  readonly squares: Float64Array;
  /** The first-order rows. */
  readonly first: FirstOrderRows;
}

/** The moments of a test under a model. */
export interface ModelMoments {
  /** The expected value of each row. */
  readonly expected: Float64Array;
  /** Ξ, the covariance matrix of the rows, by rows. */
  readonly covariance: Float64Array;
  /** E(row | node), by node, then row. */
  readonly conditional: Float64Array;
}

/** The first and second moments of the items' scores. */
export interface ScoreMoments {
  /** The mean of each item's score. */
  readonly means: Float64Array;
  /** The mean of x_i x_j, by item, then item; E(x_i²) on the diagonal. */
  readonly products: Float64Array;
}

/** The moments of a sample of complete persons. */
export interface SampleMoments extends ScoreMoments {
  /** The number of persons. */
  readonly persons: number;
}

/** Returns the number of rows of a statistic. */
export function rowCount(tables: MomentTables): number {
  const { itemCount, first } = tables;
  return first.items.length + (itemCount * (itemCount - 1)) / 2;
}

/** Returns the row of the pair i < j. */
export function pairRow(tables: MomentTables, i: number, j: number): number {
  const { itemCount, first } = tables;
  return first.items.length + (i * (2 * itemCount - i - 1)) / 2 + j - i - 1;
}

/**
 * Returns the moments of the rows under a model.
 *
 * @remarks
 * Two rows that share no item are independent given a node, so their product integrates the product of their
 * conditional expectations. Rows that share an item integrate the union of their items, each once: `addSharedItems`
 * adds what that changes, so that the matrix costs the rows squared times the nodes, and the shared rows the items cubed.
 */
export function modelMoments(tables: MomentTables): ModelMoments {
  const rows = rowCount(tables);
  const nodes = tables.weights.length;
  const conditional = conditionalMoments(tables, rows);
  const expected = new Float64Array(rows);
  const covariance = new Float64Array(rows * rows);
  for (let q = 0; q < nodes; q++) {
    const row = conditional.subarray(q * rows, (q + 1) * rows);
    const weight = tables.weights[q] ?? 0;
    addOuterProduct(covariance, row, weight);
    for (let r = 0; r < rows; r++) expected[r] = (expected[r] ?? 0) + weight * (row[r] ?? 0);
  }
  addSharedItems(tables, covariance, rows);
  addOuterProduct(covariance, expected, -1);
  mirror(covariance, rows);
  return { expected, covariance, conditional };
}

/** Returns E(row | node) by node, then row. */
function conditionalMoments(tables: MomentTables, rows: number): Float64Array {
  const { weights, itemCount, means, first } = tables;
  const firstRows = first.items.length;
  const conditional = new Float64Array(weights.length * rows);
  for (let q = 0; q < weights.length; q++) {
    const start = q * rows;
    for (let r = 0; r < firstRows; r++) conditional[start + r] = first.means[q * firstRows + r] ?? 0;
    let row = start + firstRows;
    for (let i = 0; i < itemCount; i++) {
      const mean = means[q * itemCount + i] ?? 0;
      for (let j = i + 1; j < itemCount; j++) conditional[row++] = mean * (means[q * itemCount + j] ?? 0);
    }
  }
  return conditional;
}

/** Returns Σ_q a_q b_q. */
function dot(a: Float64Array, b: Float64Array): number {
  let total = 0;
  for (let q = 0; q < a.length; q++) total += (a[q] ?? 0) * (b[q] ?? 0);
  return total;
}

/** Returns the sum of the values. */
function sum(values: Float64Array): number {
  return values.reduce((total, value) => total + value, 0);
}

/** Returns the columns of a table by node, then column. */
function columnsOf(table: Float64Array, columns: number, nodes: number): Float64Array[] {
  return Array.from({ length: columns }, (_, c) => {
    const column = new Float64Array(nodes);
    for (let q = 0; q < nodes; q++) column[q] = table[q * columns + c] ?? 0;
    return column;
  });
}

/** Returns the column of `columns`, which the callers' indices always find. */
function columnAt(columns: readonly Float64Array[], index: number): Float64Array {
  return columns[index] ?? new Float64Array(0);
}

/** The cross products that rows sharing an item integrate beyond the products of their expectations. */
interface SharedItems {
  readonly tables: MomentTables;
  /** The upper triangle of the matrix being built. */
  readonly covariance: Float64Array;
  readonly rows: number;
  /** E(x | node) of each item, by item, then node. */
  readonly means: readonly Float64Array[];
  /** Var(x | node) of each item, by item, then node. */
  readonly variances: readonly Float64Array[];
}

/** Adds `value` to the upper triangle of the matrix at rows r and s. */
function addAt({ covariance, rows }: SharedItems, r: number, s: number, value: number): void {
  const index = Math.min(r, s) * rows + Math.max(r, s);
  covariance[index] = (covariance[index] ?? 0) + value;
}

/** Returns the row of the pair of items i and j, in either order. */
function rowOfPair(tables: MomentTables, i: number, j: number): number {
  return i < j ? pairRow(tables, i, j) : pairRow(tables, j, i);
}

/** Adds to the product of conditional expectations what rows that share an item integrate beyond it. */
function addSharedItems(tables: MomentTables, covariance: Float64Array, rows: number): void {
  const { weights, itemCount, squares } = tables;
  const nodes = weights.length;
  const means = columnsOf(tables.means, itemCount, nodes);
  const seconds = columnsOf(squares, itemCount, nodes);
  const variances = means.map((mean, i) => mean.map((value, q) => (seconds[i]?.[q] ?? 0) - value * value));
  const shared: SharedItems = { tables, covariance, rows, means, variances };
  for (let i = 0; i < itemCount; i++) addSharedPairs(shared, i);
  for (let r = 0; r < tables.first.items.length; r++) addSharedFirstRow(shared, r);
}

/** Adds the terms of the pairs that share item i: Var(x_i | node) times the means of their other items. */
function addSharedPairs(shared: SharedItems, i: number): void {
  const { tables, means, variances } = shared;
  const { weights, itemCount } = tables;
  const weighted = columnAt(variances, i).map((value, q) => value * (weights[q] ?? 0));
  for (let j = 0; j < itemCount; j++) {
    if (j === i) continue;
    const row = rowOfPair(tables, i, j);
    const mean = columnAt(means, j);
    const scaled = weighted.map((value, q) => value * (mean[q] ?? 0));
    for (let k = 0; k <= j; k++) {
      if (k !== i) addAt(shared, row, rowOfPair(tables, i, k), dot(scaled, columnAt(means, k)));
    }
    // A pair with itself shares both items: the terms of i and of j leave out Var(x_i | node) Var(x_j | node).
    if (i < j) addAt(shared, row, row, dot(weighted, columnAt(variances, j)));
  }
}

/** Adds the terms of first-order row r with the pairs of its item and with the rows of its item up to itself. */
function addSharedFirstRow(shared: SharedItems, r: number): void {
  const { tables, means } = shared;
  const { weights, itemCount, first } = tables;
  const nodes = weights.length;
  const firstRows = first.items.length;
  const item = first.items[r] ?? 0;
  const mean = columnAt(means, item);
  const value = (table: Float64Array, q: number, row: number): number => table[q * firstRows + row] ?? 0;
  const h = weights.map((_, q) => value(first.means, q, r));
  const overlap = weights.map((w, q) => w * (value(first.scores, q, r) - (h[q] ?? 0) * (mean[q] ?? 0)));
  for (let j = 0; j < itemCount; j++) {
    if (j !== item) addAt(shared, r, rowOfPair(tables, item, j), dot(overlap, columnAt(means, j)));
  }
  const weighted = h.map((entry, q) => entry * (weights[q] ?? 0));
  addAt(shared, r, r, sum(weights.map((w, q) => w * value(first.squares, q, r))) - dot(weighted, h));
  for (let s = 0; s < r; s++) {
    if (first.items[s] !== item) continue;
    let product = 0;
    for (let q = 0; q < nodes; q++) product += (weighted[q] ?? 0) * value(first.means, q, s);
    addAt(shared, r, s, -product);
  }
}

/**
 * Adds to column `column` of the Jacobian Δ, of `columns` columns by rows, what a parameter of item i moves in the pairs
 * of the item.
 *
 * @param derivative - dE(x_i | node)/dβ at each node.
 */
export function addPairDerivatives(
  tables: MomentTables,
  jacobian: Float64Array,
  columns: number,
  column: number,
  i: number,
  derivative: Float64Array,
): void {
  const { weights, itemCount, means } = tables;
  const nodes = weights.length;
  for (let j = 0; j < itemCount; j++) {
    if (j === i) continue;
    let total = 0;
    for (let q = 0; q < nodes; q++) {
      total += (weights[q] ?? 0) * (derivative[q] ?? 0) * (means[q * itemCount + j] ?? 0);
    }
    const row = i < j ? pairRow(tables, i, j) : pairRow(tables, j, i);
    jacobian[row * columns + column] = (jacobian[row * columns + column] ?? 0) + total;
  }
}

/**
 * Adds to column `column` of the Jacobian Δ what the node weights move: Σ_q dw_q/dβ E(row | node) for every row.
 *
 * @param derivative - dw_q/dβ at each node; a node at 0 costs nothing.
 */
export function addWeightDerivatives(
  conditional: Float64Array,
  jacobian: Float64Array,
  rows: number,
  columns: number,
  column: number,
  derivative: Float64Array,
): void {
  for (let q = 0; q < derivative.length; q++) {
    const value = derivative[q] ?? 0;
    if (value === 0) continue;
    for (let r = 0; r < rows; r++) {
      const index = r * columns + column;
      jacobian[index] = (jacobian[index] ?? 0) + value * (conditional[q * rows + r] ?? 0);
    }
  }
}

/** A limited-information statistic with the rank of its Jacobian. */
export interface QuadraticForm {
  /** The statistic; `null` when Ξ is not positive definite. */
  readonly statistic: number | null;
  /** The rank of Δ: its columns less those that depend on the columns before them. */
  readonly rank: number;
}

/**
 * The part of a whitened column of Δ outside the span of the columns before it, relative to its length, below which it
 * depends on them. On the fixtures an exact dependence leaves 2e-16 to 2e-14, and the most nearly dependent column of
 * an identified model keeps 7e-6, a GRM's for M2*.
 */
const DEPENDENCE = 1e-9;

/** The share of a column's length below which Gram-Schmidt orthogonalizes it a second time: 1 / √2. */
const REORTHOGONALIZE = Math.SQRT1_2;

/**
 * Returns N e′ (Ξ⁻¹ - Ξ⁻¹ Δ (Δ′ Ξ⁻¹ Δ)⁺ Δ′ Ξ⁻¹) e and the rank of Δ.
 *
 * @remarks
 * The Cholesky factor L of Ξ whitens e and Δ, and Ξ is never inverted. The statistic is N times the squared length of
 * L⁻¹ e outside the span of L⁻¹ Δ, whose columns Gram-Schmidt orthonormalizes, a second time where the first pass
 * cancels more than 1 - 1 / √2 of a column. A column that adds nothing to
 * the span is left out, so that a Δ whose rank is below its columns, as when first- and second-order margins do not
 * identify every parameter, gives the statistic on the whole complement of its span, with as many degrees of freedom
 * as the rows less its rank.
 *
 * @param residuals - e, the observed moments less the expected ones.
 * @param covariance - Ξ, by rows.
 * @param jacobian - Δ, by rows.
 * @param columns - The number of columns of Δ.
 * @param persons - N.
 * @see {@link https://doi.org/10.1007/s11336-005-1295-9 | Maydeu-Olivares and Joe (2006)}
 * @see {@link https://doi.org/10.1090/S0025-5718-1976-0431641-8 | Daniel, Gragg, Kaufman and Stewart (1976)}
 */
export function limitedInformationStatistic(
  residuals: Float64Array,
  covariance: Float64Array,
  jacobian: Float64Array,
  columns: number,
  persons: number,
): QuadraticForm {
  const rows = residuals.length;
  const factor = cholesky(covariance, rows);
  if (factor === null) return { statistic: null, rank: columns };
  const whitened = residuals.slice();
  forwardSolve(factor, rows, whitened, 1);
  const directions = jacobian.slice();
  forwardSolve(factor, rows, directions, columns);
  const basis: Float64Array[] = [];
  let projected = 0;
  for (let c = 0; c < columns; c++) {
    const vector = Float64Array.from({ length: rows }, (_, r) => directions[r * columns + c] ?? 0);
    const length = Math.sqrt(dot(vector, vector));
    for (const unit of basis) subtractMultiple(vector, unit, dot(unit, vector));
    let rest = Math.sqrt(dot(vector, vector));
    // A second pass restores the orthogonality the first loses when it cancels much of the column (Daniel et al.).
    if (rest < REORTHOGONALIZE * length) {
      for (const unit of basis) subtractMultiple(vector, unit, dot(unit, vector));
      rest = Math.sqrt(dot(vector, vector));
    }
    if (!(rest > DEPENDENCE * length)) continue;
    for (let r = 0; r < rows; r++) vector[r] = (vector[r] ?? 0) / rest;
    basis.push(vector);
    projected += dot(vector, whitened) ** 2;
  }
  // At exact arithmetic the difference is at least 0; rounding can leave a small negative one, which mirt takes as abs.
  return { statistic: Math.abs(persons * (dot(whitened, whitened) - projected)), rank: basis.length };
}

/** Subtracts `multiple` times `unit` from `vector`. */
function subtractMultiple(vector: Float64Array, unit: Float64Array, multiple: number): void {
  for (let r = 0; r < vector.length; r++) vector[r] = (vector[r] ?? 0) - multiple * (unit[r] ?? 0);
}

/** Returns the means and the mean products of the items over complete rows. */
export function sampleMoments(rows: readonly (readonly (number | null)[])[], itemCount: number): SampleMoments {
  const means = new Float64Array(itemCount);
  const products = new Float64Array(itemCount * itemCount);
  for (const row of rows) {
    const scores = Float64Array.from(row, (score) => score ?? 0);
    addOuterProduct(products, scores);
    for (let i = 0; i < itemCount; i++) means[i] = (means[i] ?? 0) + (scores[i] ?? 0);
  }
  const persons = rows.length;
  mirror(products, itemCount);
  return { persons, means: means.map((sum) => sum / persons), products: products.map((sum) => sum / persons) };
}

/**
 * Returns the correlation of each pair of items, as {@link pairRow} orders the pairs from 0, each `NaN` when an item has no
 * variance.
 */
function pairCorrelations({ means, products }: ScoreMoments, itemCount: number): Float64Array {
  const correlations = new Float64Array((itemCount * (itemCount - 1)) / 2);
  const deviations = means.map((mean, i) => Math.sqrt((products[i * itemCount + i] ?? 0) - mean * mean));
  let pair = 0;
  for (let i = 0; i < itemCount; i++) {
    for (let j = i + 1; j < itemCount; j++) {
      const scale = (deviations[i] ?? 0) * (deviations[j] ?? 0);
      const covariance = (products[i * itemCount + j] ?? 0) - (means[i] ?? 0) * (means[j] ?? 0);
      correlations[pair++] = scale > 0 ? covariance / scale : Number.NaN;
    }
  }
  return correlations;
}

/** Returns the moments of the scores under a model, from its tables: the nodes times the pairs. */
export function tableMoments({ weights, itemCount, means, squares }: MomentTables): ScoreMoments {
  const modelMeans = new Float64Array(itemCount);
  const products = new Float64Array(itemCount * itemCount);
  for (let q = 0; q < weights.length; q++) {
    const weight = weights[q] ?? 0;
    const start = q * itemCount;
    for (let i = 0; i < itemCount; i++) {
      const mean = weight * (means[start + i] ?? 0);
      modelMeans[i] = (modelMeans[i] ?? 0) + mean;
      products[i * itemCount + i] = (products[i * itemCount + i] ?? 0) + weight * (squares[start + i] ?? 0);
      for (let j = i + 1; j < itemCount; j++) {
        products[i * itemCount + j] = (products[i * itemCount + j] ?? 0) + mean * (means[start + j] ?? 0);
      }
    }
  }
  return { means: modelMeans, products };
}

/**
 * Returns the root mean square of the residual correlations over the pairs of items, SRMSR, or `null` when an item has no
 * variance in the sample or under the model.
 *
 * @param model - The model's moments, whose products need only the upper triangle.
 */
export function residualCorrelationRoot(sample: SampleMoments, model: ScoreMoments): number | null {
  const itemCount = sample.means.length;
  const observed = pairCorrelations(sample, itemCount);
  const implied = pairCorrelations(model, itemCount);
  const total = sum(observed.map((correlation, pair) => (correlation - (implied[pair] ?? 0)) ** 2));
  return Number.isNaN(total) ? null : Math.sqrt(total / observed.length);
}

/**
 * Returns N Σ_(i<j) r_ij², M2 of the independence model, in which every item keeps its observed shares, or `null` when an
 * item has no variance.
 *
 * @remarks
 * Under independence x̄_i and the sample covariances are uncorrelated, and a covariance has variance σ_i² σ_j² / N, so
 * the full-rank form of M2, whose residuals are those covariances, is the sum of the squared correlations. It has
 * J (J - 1) / 2 degrees of freedom, as first-order rows of any number add one row and one parameter each.
 */
export function independenceStatistic(sample: SampleMoments, itemCount: number): number | null {
  const correlations = pairCorrelations(sample, itemCount);
  const total = dot(correlations, correlations);
  return Number.isNaN(total) ? null : sample.persons * total;
}

/**
 * Returns the root of F(statistic; df, λ) = probability, F decreasing in λ, searched from 0 to `limit`; 0 when the two
 * ends do not bracket it, as mirt and GDINA take it.
 */
function noncentrality(statistic: number, df: number, probability: number, limit: number): number {
  const excess = (lambda: number): number => noncentralChiSquareCdf(statistic, df, lambda) - probability;
  const atZero = excess(0);
  const atLimit = excess(limit);
  return atZero > 0 && atLimit < 0 ? findRoot(excess, 0, limit, atZero, atLimit) : 0;
}

/**
 * Returns the interval of the RMSEA at `level`: √(λ / (N df)) at the two noncentralities that place the statistic at the
 * interval's tails, as mirt's `RMSEA.CI` and GDINA's `RMSEAfun`, with the same bounds on the search.
 */
export function rmseaInterval(
  statistic: number,
  df: number,
  persons: number,
  level: number,
): readonly [lower: number, upper: number] {
  const tail = (1 - level) / 2;
  const lower = noncentrality(statistic, df, 1 - tail, statistic);
  const upper = noncentrality(statistic, df, tail, Math.max(persons, 5 * statistic));
  return [Math.sqrt(lower / (persons * df)), Math.sqrt(upper / (persons * df))];
}

import { cholesky, DataError } from "@measurekit/core";
import { addOuterProduct, mirror } from "../math/internal.js";
import { type BivariateDistribution, bivariateTable } from "./bivariate.js";
import { bivariateBasis, bivariateLogLinearPresmoothing } from "./bivariate-presmoothing.js";
import { checkDistribution } from "./observed-score.js";
import { logLinearPresmoothing, orthonormalPolynomials } from "./presmoothing.js";

/** A presmoothed table: its probabilities p, cell by cell, and C, with Cov(p̂) = C Cᵀ (von Davier et al., 2004). */
export interface SmoothedTable {
  readonly probabilities: Float64Array;
  readonly persons: number;
  /** C, by cell, then column. */
  readonly c: Float64Array;
  /** The columns of C: the terms of the log-linear model but the constant. */
  readonly width: number;
  /** The columns of a bivariate table, the scores of its second form or anchor; 1 for a univariate one. */
  readonly columns: number;
  readonly converged: boolean;
}

/**
 * A score distribution as a function of the presmoothed tables: its probabilities, the persons whose number scales its
 * variance, and for each table its Jacobian times the table's C, by score, then column.
 */
export interface Margin {
  readonly probabilities: Float64Array;
  readonly persons: number;
  readonly blocks: readonly Float64Array[];
}

/** Returns a univariate distribution presmoothed at `degree`, with its C. */
export function univariateTable(counts: readonly number[], degree: number, label: string): SmoothedTable {
  const persons = checkPersons(checkDistribution(counts, label), label);
  const fit = logLinearPresmoothing(counts, degree);
  const basis = orthonormalPolynomials(counts.length, degree);
  return smoothedTable(fit.distribution.flat(), persons, basis, degree + 1, 1, fit.converged);
}

/** Returns a bivariate distribution presmoothed at `degrees` and `crossDegree`, with its C. */
export function bivariateSmoothedTable(
  distribution: BivariateDistribution,
  degrees: readonly [number, number],
  crossDegree: number,
  label: string,
): SmoothedTable {
  const table = bivariateTable(distribution, label);
  checkPersons(table.persons, label);
  const fit = bivariateLogLinearPresmoothing(distribution, { degrees, crossDegree });
  const { basis, terms } = bivariateBasis(table.rows, table.columns, degrees[0], degrees[1], crossDegree);
  return smoothedTable(fit.distribution.flat(), table.persons, basis, terms, table.columns, fit.converged);
}

/**
 * Checks that a table has more than one person, as the variance of its scores divides by n - 1.
 *
 * @throws `DataError` when it does not.
 */
function checkPersons(persons: number, label: string): number {
  if (!(persons > 1)) throw new DataError(`Kernel equating needs more than one person in the ${label} distribution.`);
  return persons;
}

/**
 * Returns the table of fitted counts with C = D_p^½ Q / √N, Q the thin QR factor of (D_p^½ - √p pᵀ) B for the basis B
 * without its constant: kequate's `cmatrixSG`. Q comes from a Cholesky QR taken twice, which is accurate while the
 * condition number of the centered basis stays below about 1e8.
 *
 * @throws `DataError` when the centered basis is singular to working precision.
 */
function smoothedTable(
  fitted: readonly number[],
  persons: number,
  basis: Float64Array,
  terms: number,
  columns: number,
  converged: boolean,
): SmoothedTable {
  const width = terms - 1;
  const probabilities = Float64Array.from(fitted, (count) => count / persons);
  const q = centeredBasis(probabilities, basis, terms);
  orthonormalizeColumns(q, fitted.length, width);
  orthonormalizeColumns(q, fitted.length, width);
  const c = q.map((value, entry) => value * Math.sqrt((probabilities[Math.floor(entry / width)] ?? 0) / persons));
  return { probabilities, persons, c, width, columns, converged };
}

/**
 * Returns (D_p^½ - √p pᵀ) B for the basis B without its constant, by cell, then term: each cell's row of B less the
 * p-weighted mean of the rows, times √p.
 */
function centeredBasis(probabilities: Float64Array, basis: Float64Array, terms: number): Float64Array {
  const width = terms - 1;
  const term = (cell: number, k: number): number => basis[cell * terms + k + 1] ?? 0;
  const means = new Float64Array(width);
  probabilities.forEach((p, cell) => {
    for (let k = 0; k < width; k++) means[k] = (means[k] ?? 0) + p * term(cell, k);
  });
  const centered = new Float64Array(probabilities.length * width);
  probabilities.forEach((p, cell) => {
    const root = Math.sqrt(p);
    for (let k = 0; k < width; k++) centered[cell * width + k] = root * (term(cell, k) - (means[k] ?? 0));
  });
  return centered;
}

/**
 * Replaces the columns of the `rows` by `width` matrix M with those of Q = M L⁻ᵀ, L the Cholesky factor of Mᵀ M: one
 * step of a Cholesky QR.
 *
 * @throws `DataError` when Mᵀ M is singular.
 */
function orthonormalizeColumns(m: Float64Array, rows: number, width: number): void {
  const gram = new Float64Array(width * width);
  for (let i = 0; i < rows; i++) addOuterProduct(gram, m.subarray(i * width, (i + 1) * width));
  mirror(gram, width);
  const lower = cholesky(gram, width);
  if (lower === null) {
    throw new DataError(
      "The presmoothed distribution leaves its log-linear terms dependent; lower the degrees of the presmoothing.",
    );
  }
  for (let i = 0; i < rows; i++) solveLower(lower, m.subarray(i * width, (i + 1) * width));
}

/** Solves L y = v for y by forward substitution, in place: a row of M L⁻ᵀ. */
function solveLower(lower: Float64Array, row: Float64Array): void {
  const width = row.length;
  for (let j = 0; j < width; j++) {
    let value = row[j] ?? 0;
    for (let k = 0; k < j; k++) value -= (lower[j * width + k] ?? 0) * (row[k] ?? 0);
    row[j] = value / (lower[j * width + j] ?? 1);
  }
}

/** Returns a block of zeros of `scores` rows for each table, the tables having C of the given widths. */
function zeroBlocks(widths: readonly number[], scores: number): Float64Array[] {
  return widths.map((width) => new Float64Array(scores * width));
}

/** Returns a univariate table as the margin it is, its own block C. */
export function ownMargin(widths: readonly number[], table: SmoothedTable, index: number): Margin {
  const blocks = zeroBlocks(widths, table.probabilities.length);
  blocks[index] = table.c;
  return { probabilities: table.probabilities, persons: table.persons, blocks };
}

/**
 * Returns the margin of the first score of a bivariate table (its rows) or of the second (its columns), with the sums
 * of C over the other score.
 */
export function tableMargin(
  widths: readonly number[],
  table: SmoothedTable,
  index: number,
  side: "rows" | "columns",
): Margin {
  const { probabilities, columns, width, c } = table;
  const rows = probabilities.length / columns;
  const scores = side === "rows" ? rows : columns;
  const margin = new Float64Array(scores);
  const block = new Float64Array(scores * width);
  for (let x = 0; x < rows; x++) {
    for (let v = 0; v < columns; v++) {
      const cell = x * columns + v;
      const score = side === "rows" ? x : v;
      margin[score] = (margin[score] ?? 0) + (probabilities[cell] ?? 0);
      for (let k = 0; k < width; k++) {
        block[score * width + k] = (block[score * width + k] ?? 0) + (c[cell * width + k] ?? 0);
      }
    }
  }
  const blocks = zeroBlocks(widths, scores);
  blocks[index] = block;
  return { probabilities: margin, persons: table.persons, blocks };
}

/** Returns w a + (1 - w) b of two margins of the same scores, with `persons` scaling the variance. */
export function mixedMargin(first: Margin, second: Margin, weight: number, persons: number): Margin {
  const mix = (a: Float64Array, b: Float64Array | undefined): Float64Array =>
    Float64Array.from(a, (value, i) => weight * value + (1 - weight) * (b?.[i] ?? 0));
  return {
    probabilities: mix(first.probabilities, second.probabilities),
    persons,
    blocks: first.blocks.map((block, f) => mix(block, second.blocks[f])),
  };
}

/** A table of a form's scores (rows) by anchor score (columns) and its place among the tables. */
export interface IndexedTable {
  readonly table: SmoothedTable;
  readonly index: number;
}

/**
 * Returns the margin of a form in the synthetic population of post-stratification, from the table of the group that
 * took it (`own`) and the other group's: r_x = Σ_a P(x, a) (w + (1 - w) t_Q(a) / t_P(a)), w the weight of the own
 * group's population and t the anchor margins, with its Jacobians (von Davier et al., 2004, section 10.3).
 */
export function syntheticMargin(
  widths: readonly number[],
  own: IndexedTable,
  other: IndexedTable,
  weight: number,
): Margin {
  const ownAnchor = tableMargin(widths, own.table, own.index, "columns");
  const otherAnchor = tableMargin(widths, other.table, other.index, "columns");
  const { columns, probabilities } = own.table;
  const rows = probabilities.length / columns;
  const result = new Float64Array(rows);
  const blocks = zeroBlocks(widths, rows);
  const ownBlock = blocks[own.index] ?? new Float64Array(0);
  const otherBlock = blocks[other.index] ?? new Float64Array(0);
  for (let a = 0; a < columns; a++) {
    const ownShare = ownAnchor.probabilities[a] ?? 0;
    const ratio = (otherAnchor.probabilities[a] ?? 0) / ownShare;
    const factor = weight + (1 - weight) * ratio;
    for (let x = 0; x < rows; x++) {
      const p = probabilities[x * columns + a] ?? 0;
      result[x] = (result[x] ?? 0) + p * factor;
      // ∂r_x / ∂P(x, a) = factor, and through t_P(a) and t_Q(a) the two corrections.
      addRow(ownBlock, x, own.table.c, x * columns + a, factor, own.table.width);
      addRow(ownBlock, x, ownAnchor.blocks[own.index], a, ((weight - 1) * p * ratio) / ownShare, own.table.width);
      addRow(otherBlock, x, otherAnchor.blocks[other.index], a, ((1 - weight) * p) / ownShare, other.table.width);
    }
  }
  return { probabilities: result, persons: own.table.persons, blocks };
}

/** Adds `factor` times row `from` of `source` to row `to` of `target`, both of `width` columns. */
function addRow(
  target: Float64Array,
  to: number,
  source: Float64Array | undefined,
  from: number,
  factor: number,
  width: number,
): void {
  for (let k = 0; k < width; k++) {
    target[to * width + k] = (target[to * width + k] ?? 0) + factor * (source?.[from * width + k] ?? 0);
  }
}

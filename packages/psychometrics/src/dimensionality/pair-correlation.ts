import { normalCdf } from "@measurekit/core";
import { bivariateNormalCdf, bivariateNormalDensity, bivariateNormalLogDensitySlope } from "./bivariate-normal.js";

/**
 * The correlations searched are within this of ±1, where the distribution is almost a line and the cells of a table
 * nearly a function of its thresholds. A maximum beyond it is reported as ±1.
 */
const RHO_LIMIT = 1 - 1e-9;
/** The search stops when the correlation changes by less than this. */
const RHO_TOLERANCE = 1e-13;
const MAX_ITERATIONS = 100;
/**
 * The smallest cell probability in the score: a cell with a count and a probability that has underflowed is scored as
 * if its probability were this, so that the ratios of the derivatives to the probability, and their squares, stay
 * finite.
 */
const MIN_CELL_PROBABILITY = 1e-100;
/** A maximum within this of the limit, with the score still pointing outward, is on the boundary. */
const BOUNDARY_MARGIN = 1e-8;

/** A table of counts of two items with the thresholds of their categories. */
interface Table {
  /** The counts by row, then column; they may be fractional. */
  readonly counts: Float64Array;
  readonly rows: number;
  readonly columns: number;
  /** The thresholds between the categories of the row item: one fewer than `rows`. */
  readonly rowThresholds: Float64Array;
  /** The thresholds between the categories of the column item: one fewer than `columns`. */
  readonly columnThresholds: Float64Array;
}

/** An estimator of the polychoric correlation of a table, with memory allocated once. */
export interface PairEstimator {
  /**
   * Returns the maximum likelihood correlation of two standard normal variables cut at the table's thresholds into its
   * categories, given the table's counts.
   */
  estimate(table: Table): number;
}

/**
 * Returns an estimator of the polychoric correlation of a table, for tables of up to `maxRows` by `maxColumns`
 * categories.
 *
 * @remarks
 * The estimator maximizes the log-likelihood Σ n_kl ln π_kl(ρ) of the table, with the cell probabilities π_kl of the
 * bivariate normal distribution between the thresholds, by Newton's method with the exact first and second derivatives
 * (the bivariate density) safeguarded by bisection of a bracket of the root of the score. A maximum at the boundary, as
 * for a table with an empty cell where the association is perfect, gives 1 or -1.
 */
export function createPairEstimator(maxRows: number, maxColumns: number): PairEstimator {
  const size = (maxRows + 1) * (maxColumns + 1);
  const grids = {
    distribution: new Float64Array(size),
    density: new Float64Array(size),
    slope: new Float64Array(size),
  };
  return {
    estimate(table) {
      const width = table.columns + 1;
      fixBoundary(grids.distribution, table, width);
      return maximize(startingCorrelation(table), (rho, out) => {
        fillGrids(grids, table, width, rho);
        logLikelihoodDerivatives(grids, table, width, out);
      });
    },
  };
}

/**
 * Returns the ρ that maximizes a concave-enough function whose first and second derivatives `derivatives` writes to its
 * second argument, starting from `start`: Newton's method kept within a bracket that bisection narrows, and ±1 when the
 * function increases to the boundary.
 */
function maximize(start: number, derivatives: (rho: number, out: Float64Array) => void): number {
  const out = new Float64Array(2);
  let low = -RHO_LIMIT;
  let high = RHO_LIMIT;
  let rho = Math.max(low, Math.min(high, start));
  let first = 0;
  for (let iteration = 0; iteration < MAX_ITERATIONS; iteration++) {
    derivatives(rho, out);
    first = out[0] ?? 0;
    const second = out[1] ?? 0;
    if (first > 0) low = rho;
    else high = rho;
    const step = first / second;
    let next = second < 0 && Number.isFinite(step) ? rho - step : Number.NaN;
    if (!(next > low && next < high)) next = (low + high) / 2;
    const change = Math.abs(next - rho);
    rho = next;
    if (change < RHO_TOLERANCE || high - low < RHO_TOLERANCE) break;
  }
  return Math.abs(rho) > RHO_LIMIT - BOUNDARY_MARGIN && first * rho > 0 ? Math.sign(rho) : rho;
}

/**
 * Returns the Pearson correlation of the category numbers of a table, 0 when either has no variance: the starting
 * value of the search.
 */
function startingCorrelation({ counts, rows, columns }: Table): number {
  let total = 0;
  let sumX = 0;
  let sumY = 0;
  let sumXx = 0;
  let sumYy = 0;
  let sumXy = 0;
  for (let k = 0; k < rows; k++) {
    for (let l = 0; l < columns; l++) {
      const n = counts[k * columns + l] ?? 0;
      total += n;
      sumX += n * k;
      sumY += n * l;
      sumXx += n * k * k;
      sumYy += n * l * l;
      sumXy += n * k * l;
    }
  }
  const varianceX = sumXx - (sumX * sumX) / total;
  const varianceY = sumYy - (sumY * sumY) / total;
  if (!(varianceX > 0 && varianceY > 0)) return 0;
  return (sumXy - (sumX * sumY) / total) / Math.sqrt(varianceX * varianceY);
}

/** The grids of Φ₂, φ₂ and φ₂ times its slope at the thresholds, with a row and a column at each end. */
interface Grids {
  readonly distribution: Float64Array;
  readonly density: Float64Array;
  readonly slope: Float64Array;
}

/**
 * Writes the entries of the distribution grid that do not depend on ρ: entry (k, l) is Φ₂ at thresholds k and l, which
 * is 0 beyond the lowest and Φ of the other threshold beyond the highest.
 */
function fixBoundary(grid: Float64Array, table: Table, width: number): void {
  const { rows, columns, rowThresholds, columnThresholds } = table;
  for (let l = 0; l <= columns; l++) grid[l] = 0;
  for (let k = 0; k <= rows; k++) grid[k * width] = 0;
  for (let l = 1; l < columns; l++) grid[rows * width + l] = normalCdf(columnThresholds[l - 1] ?? 0);
  for (let k = 1; k < rows; k++) grid[k * width + columns] = normalCdf(rowThresholds[k - 1] ?? 0);
  grid[rows * width + columns] = 1;
}

/** Writes Φ₂, φ₂ and φ₂ times its slope at ρ to the grids, at every pair of thresholds. */
function fillGrids(grids: Grids, table: Table, width: number, rho: number): void {
  const { rows, columns, rowThresholds, columnThresholds } = table;
  for (let k = 1; k < rows; k++) {
    const x = rowThresholds[k - 1] ?? 0;
    for (let l = 1; l < columns; l++) {
      const y = columnThresholds[l - 1] ?? 0;
      const density = bivariateNormalDensity(x, y, rho);
      grids.distribution[k * width + l] = bivariateNormalCdf(x, y, rho);
      grids.density[k * width + l] = density;
      grids.slope[k * width + l] = density * bivariateNormalLogDensitySlope(x, y, rho);
    }
  }
}

/**
 * Writes to `out` the first and second derivatives of the log-likelihood of the table at the ρ of the grids.
 *
 * @remarks
 * For cell (k, l), π is the second difference of the grid of Φ₂, π' the second difference of the grid of densities φ₂,
 * and π'' that of φ₂ times its slope, all 0 on the grid's boundary.
 */
function logLikelihoodDerivatives(grids: Grids, table: Table, width: number, out: Float64Array): void {
  const { counts, rows, columns } = table;
  const { distribution } = grids;
  let first = 0;
  let second = 0;
  for (let k = 1; k <= rows; k++) {
    for (let l = 1; l <= columns; l++) {
      const n = counts[(k - 1) * columns + (l - 1)] ?? 0;
      if (n === 0) continue;
      const at = k * width + l;
      const probability = Math.max(
        (distribution[at] ?? 0) -
          (distribution[at - width] ?? 0) -
          (distribution[at - 1] ?? 0) +
          (distribution[at - width - 1] ?? 0),
        MIN_CELL_PROBABILITY,
      );
      const ratio = cornerDifference(grids.density, table, width, k, l) / probability;
      first += n * ratio;
      second += n * (cornerDifference(grids.slope, table, width, k, l) / probability - ratio * ratio);
    }
  }
  out[0] = first;
  out[1] = second;
}

/** Returns the second difference of `grid` at cell (k, l), whose entries on the boundary are 0. */
function cornerDifference(grid: Float64Array, { rows, columns }: Table, width: number, k: number, l: number): number {
  const at = k * width + l;
  const corner = (inside: boolean, index: number): number => (inside ? (grid[index] ?? 0) : 0);
  return (
    corner(k < rows && l < columns, at) -
    corner(k > 1 && l < columns, at - width) -
    corner(k < rows && l > 1, at - 1) +
    corner(k > 1 && l > 1, at - width - 1)
  );
}

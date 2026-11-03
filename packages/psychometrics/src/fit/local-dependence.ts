import { chiSquareUpperTail, DataError } from "@measurekit/core";
import { checkMatrixItems, type ResponseMatrix } from "../data/internal.js";
import { type IrtItem, itemTrace, largestCategoryCount } from "../irt/internal.js";
import { checkItems, checkPersonThetas, type ResidualFitOptions, residualPopulation, scoreMoments } from "./inputs.js";

/**
 * The most cells of the tables of the pairs of items, a cell for each pair of their categories: each cell of the expected
 * tables sums over the nodes, so 2^24 cells take about a second, and a pair's two tables at most 256 MB.
 */
const MAX_TABLE_CELLS = 2 ** 24;

/** Chen and Thissen's local-dependence statistic of a pair of items. */
export interface LocalDependencePair {
  /** The ids of the two items, in item order. */
  readonly items: readonly [first: string, second: string];
  /**
   * Σ (O - E)² / E over the cells of the pair's table of categories; `null` when no person answered both items or a
   * cell's expected count is 0.
   */
  readonly chiSquare: number | null;
  /** (K₁ - 1)(K₂ - 1) for the items' numbers of categories. */
  readonly degreesOfFreedom: number;
  /** The upper tail of the chi-square at the statistic; `null` with it. */
  readonly significance: number | null;
  /**
   * Cramér's V, √(χ² / (n (min(K₁, K₂) - 1))), positive when the observed association, by Goodman and Kruskal's gamma,
   * is at least the expected one, and negative when below it; `null` with `chiSquare`, or when the observed table has
   * no pairs of persons that order the items, so that its gamma is undefined.
   */
  readonly cramersV: number | null;
}

/**
 * Returns Chen and Thissen's local-dependence chi-square for every pair of items, as mirt's `residuals(type = "LD")`.
 *
 * @remarks
 * For each pair, the persons who answered both items fill a table of their categories, O, which is compared with the
 * table the model expects under local independence, E_kl = n Σ_q w_q P_k(θ_q) P_l(θ_q) over the nodes and weights of
 * the population: χ² = Σ (O - E)² / E, about chi-square on (K₁ - 1)(K₂ - 1) degrees of freedom when the items are
 * independent given theta. Cramér's V scales it to a size, signed by whether the items agree more than the model
 * expects. The pairs are in item order: the first item with each later one, then the second, and so on.
 *
 * @param matrix - The responses of the persons.
 * @param items - The calibrated items, in the order of the matrix's items.
 * @param options - The population.
 * @returns For each pair of items, the statistic, its degrees of freedom and significance, and the signed Cramér's V.
 * @throws `DataError` when the matrix, the items or the population are invalid, there are more than 1,000 items, or a
 * response is not a category of its item.
 * @see {@link https://doi.org/10.3102/10769986022003265 | Chen and Thissen (1997)}
 */
export function localDependenceChiSquare(
  matrix: ResponseMatrix,
  items: readonly IrtItem[],
  options: ResidualFitOptions = {},
): LocalDependencePair[] {
  checkItems(matrix, items);
  checkMatrixItems(items.length, "Local dependence");
  const { nodes, weights } = residualPopulation(options);
  const counts = items.map((item) => item.model.categoryCount(item.parameters));
  checkTableCells(counts);
  const traces = items.map((item, i) => itemTrace(item, counts[i] ?? 0, nodes));
  const pairs: LocalDependencePair[] = [];
  for (let i = 0; i < items.length; i++) {
    for (let j = i + 1; j < items.length; j++) {
      const first = { column: i, count: counts[i] ?? 0, trace: traces[i] ?? new Float64Array(0) };
      const second = { column: j, count: counts[j] ?? 0, trace: traces[j] ?? new Float64Array(0) };
      pairs.push({
        items: [matrix.itemIds[i] ?? "", matrix.itemIds[j] ?? ""],
        ...pairStatistic(matrix.responses, first, second, weights),
      });
    }
  }
  return pairs;
}

/**
 * Checks that the tables of the pairs of items have at most {@link MAX_TABLE_CELLS} cells in all.
 *
 * @throws `DataError` when they have more.
 */
function checkTableCells(counts: readonly number[]): void {
  let sum = 0;
  let squares = 0;
  for (const count of counts) {
    sum += count;
    squares += count * count;
  }
  const cells = (sum * sum - squares) / 2;
  if (cells > MAX_TABLE_CELLS) {
    throw new DataError(
      `Local dependence holds a table of the categories of every pair of items, ${cells} cells, and takes at most ${MAX_TABLE_CELLS}; test fewer items, or items of fewer categories.`,
    );
  }
}

/** An item of a pair: its column, its number of categories and its probabilities at the nodes, by node, then category. */
interface PairItem {
  readonly column: number;
  readonly count: number;
  readonly trace: Float64Array;
}

/** Returns the statistic of a pair of items from the persons who answered both. */
function pairStatistic(
  responses: ResponseMatrix["responses"],
  first: PairItem,
  second: PairItem,
  weights: Float64Array,
): Omit<LocalDependencePair, "items"> {
  const columns = second.count;
  const degreesOfFreedom = (first.count - 1) * (columns - 1);
  const observed = new Float64Array(first.count * columns);
  let persons = 0;
  for (const row of responses) {
    const k = row[first.column] ?? null;
    const l = row[second.column] ?? null;
    if (k === null || l === null) continue;
    observed[k * columns + l] = (observed[k * columns + l] ?? 0) + 1;
    persons++;
  }
  const expected = new Float64Array(first.count * columns);
  weights.forEach((weight, q) => {
    for (let k = 0; k < first.count; k++) {
      const left = weight * (first.trace[q * first.count + k] ?? 0);
      for (let l = 0; l < columns; l++) {
        expected[k * columns + l] = (expected[k * columns + l] ?? 0) + left * (second.trace[q * columns + l] ?? 0);
      }
    }
  });
  let chiSquare = 0;
  for (let cell = 0; cell < expected.length; cell++) {
    const count = persons * (expected[cell] ?? 0);
    expected[cell] = count;
    chiSquare += ((observed[cell] ?? 0) - count) ** 2 / count;
  }
  if (persons === 0 || !Number.isFinite(chiSquare)) {
    return { chiSquare: null, degreesOfFreedom, significance: null, cramersV: null };
  }
  const association = goodmanKruskalGamma(observed, columns) - goodmanKruskalGamma(expected, columns);
  const size = Math.sqrt(chiSquare / (persons * (Math.min(first.count, columns) - 1)));
  return {
    chiSquare,
    degreesOfFreedom,
    significance: chiSquareUpperTail(chiSquare, degreesOfFreedom),
    cramersV: Number.isNaN(association) ? null : association < 0 ? -size : size,
  };
}

/**
 * Returns Goodman and Kruskal's gamma of a table by rows, (C - D) / (C + D) for C the pairs of persons ordered alike on
 * both items and D those ordered oppositely; NaN when there are none. The sums of the cells below and to the right of
 * each cell come from the rows below it, summed once from the last row up, so that a table costs its cells.
 */
function goodmanKruskalGamma(table: Float64Array, columns: number): number {
  const rows = table.length / columns;
  // below[l]: the sum of the rows below the current one in the columns from l on; below[columns] is 0.
  const below = new Float64Array(columns + 1);
  let concordant = 0;
  let discordant = 0;
  for (let k = rows - 1; k >= 0; k--) {
    for (let l = 0; l < columns; l++) {
      const cell = table[k * columns + l] ?? 0;
      concordant += cell * (below[l + 1] ?? 0);
      discordant += cell * ((below[0] ?? 0) - (below[l] ?? 0));
    }
    let suffix = 0;
    for (let l = columns - 1; l >= 0; l--) {
      suffix += table[k * columns + l] ?? 0;
      below[l] = (below[l] ?? 0) + suffix;
    }
  }
  return (concordant - discordant) / (concordant + discordant);
}

/**
 * Returns Yen's Q3 for every pair of items.
 *
 * @remarks
 * The correlation over persons of the residuals x - E(x | θ) of the two items, at the persons' thetas, as mirt's
 * `residuals(type = "Q3")` computes it. Under local independence it is near -1 / (k - 1) for k items; large values flag
 * dependent pairs. Each pair uses the persons who answered both items. The result is a symmetric matrix in item order
 * with 1 on the diagonal, `null` for a pair with fewer than two such persons or a residual without variance.
 *
 * @param matrix - The responses of the persons.
 * @param items - The calibrated items, in the order of the matrix's items.
 * @param thetas - The estimate of theta of each person, in the order of the matrix.
 * @returns The matrix of Q3 by rows, with `null` for a pair that has no persons with both items.
 * @throws `DataError` when the matrix, the items or the thetas are invalid, there are more than 1,000 items, or a
 * response is not a category of its item.
 */
export function yenQ3(
  matrix: ResponseMatrix,
  items: readonly IrtItem[],
  thetas: readonly number[],
): (number | null)[][] {
  checkItems(matrix, items);
  checkMatrixItems(items.length, "Yen's Q3");
  checkPersonThetas(matrix, thetas);
  const itemCount = items.length;
  const residuals = residualTable(matrix, items, thetas);
  // The correlations of the pairs i < j, by i, then j; NaN where undefined.
  const upper = new Float64Array(itemCount * itemCount);
  for (let i = 0; i < itemCount; i++) {
    for (let j = i + 1; j < itemCount; j++) upper[i * itemCount + j] = pairwiseCorrelation(residuals, itemCount, i, j);
  }
  return items.map((_, i) =>
    items.map((__, j) => {
      if (i === j) return 1;
      const value = upper[Math.min(i, j) * itemCount + Math.max(i, j)] ?? Number.NaN;
      return Number.isNaN(value) ? null : value;
    }),
  );
}

/** Returns the residuals x - E(x | θ) by person, then item, NaN where a response is missing. */
function residualTable(matrix: ResponseMatrix, items: readonly IrtItem[], thetas: readonly number[]): Float64Array {
  const itemCount = items.length;
  const residuals = new Float64Array(matrix.responses.length * itemCount).fill(Number.NaN);
  const probabilities = new Float64Array(largestCategoryCount(items));
  const moments = new Float64Array(3);
  matrix.responses.forEach((row, person) => {
    row.forEach((response, column) => {
      const item = items[column];
      if (response === null || item === undefined) return;
      scoreMoments(item, thetas[person] ?? 0, probabilities, moments);
      residuals[person * itemCount + column] = response - (moments[0] ?? 0);
    });
  });
  return residuals;
}

/**
 * Returns the correlation of columns i and j over the rows where both are numbers, by two passes, as R's `cor`; NaN
 * with fewer than two such rows or a column without variance.
 */
function pairwiseCorrelation(residuals: Float64Array, itemCount: number, i: number, j: number): number {
  let count = 0;
  let sumI = 0;
  let sumJ = 0;
  for (let index = 0; index < residuals.length; index += itemCount) {
    const x = residuals[index + i] ?? Number.NaN;
    const y = residuals[index + j] ?? Number.NaN;
    if (Number.isNaN(x) || Number.isNaN(y)) continue;
    count++;
    sumI += x;
    sumJ += y;
  }
  if (count < 2) return Number.NaN;
  const meanI = sumI / count;
  const meanJ = sumJ / count;
  let squaresI = 0;
  let squaresJ = 0;
  let products = 0;
  for (let index = 0; index < residuals.length; index += itemCount) {
    const x = (residuals[index + i] ?? Number.NaN) - meanI;
    const y = (residuals[index + j] ?? Number.NaN) - meanJ;
    if (Number.isNaN(x) || Number.isNaN(y)) continue;
    squaresI += x * x;
    squaresJ += y * y;
    products += x * y;
  }
  return squaresI > 0 && squaresJ > 0 ? products / Math.sqrt(squaresI * squaresJ) : Number.NaN;
}

import { checkArray, checkOptions, DataError, describe, type FieldNames } from "@measurekit/core";
import { type BivariateDistribution, bivariateTable } from "./bivariate.js";
import { MAX_LOG_LINEAR_DEGREE, orthonormalPolynomials, poissonFit } from "./presmoothing.js";

/** Options of {@link bivariateLogLinearPresmoothing}. */
export interface BivariatePresmoothingOptions {
  /**
   * The degrees C of the polynomials of the total and the anchor scores: integers from 1 to one less than the number of
   * scores of each, and at most 20.
   */
  readonly degrees: readonly [total: number, anchor: number];
  /**
   * The highest power of each score in the cross products: the terms are x^i v^j for i and j from 1 to it. An integer
   * from 0, for none, to the smaller of the degrees.
   *
   * @defaultValue `1`
   */
  readonly crossDegree?: number;
}

const BIVARIATE_PRESMOOTHING_OPTIONS: FieldNames<BivariatePresmoothingOptions> = { degrees: true, crossDegree: true };

/** A bivariate distribution smoothed by a log-linear model. */
export interface BivariateLogLinearPresmoothing {
  /** The fitted number of persons at each total score and anchor score. */
  readonly distribution: readonly (readonly number[])[];
  /** Whether Newton's method reached the maximum of the likelihood within its iterations. */
  readonly converged: boolean;
  /** Newton iterations run. */
  readonly iterations: number;
}

/**
 * Returns a bivariate distribution of total and anchor scores smoothed by a log-linear model.
 *
 * @remarks
 * The maximum likelihood fit of the Poisson regression ln m_xv = β0 + Σ_{i ≤ C_x} β_i x^i + Σ_{j ≤ C_v} γ_j v^j + Σ_{i,
 * j ≤ C_xv} δ_ij x^i v^j to the counts, as the equate package's `presmoothing(smoothmethod = "loglinear", degrees =
 * list(c(C_x, C_v), C_xv))`. The fit keeps the number of persons, the first C_x moments of the total scores, the first
 * C_v of the anchor scores and, for the cross products, the moments x^i v^j. Every cell has a positive fitted count, so
 * the result suits frequency estimation. Newton's method works on polynomials of the scores made orthonormal, which
 * give the same fit as the powers without their ill-conditioning. The distribution is the persons at each total score
 * and anchor score.
 *
 * @param distribution - The persons at each total score and anchor score.
 * @param options - The degrees of the log-linear model.
 * @returns The smoothed distribution, whether the fit converged and the iterations run.
 * @throws `DataError` when the distribution is invalid, or a degree is outside its range.
 * @see {@link https://doi.org/10.3102/10769986025002133 | Holland and Thayer (2000)}
 */
export function bivariateLogLinearPresmoothing(
  distribution: BivariateDistribution,
  options: BivariatePresmoothingOptions,
): BivariateLogLinearPresmoothing {
  checkOptions(options, BIVARIATE_PRESMOOTHING_OPTIONS);
  const table = bivariateTable(distribution, "presmoothed");
  const { degrees, crossDegree = 1 } = options;
  checkArray(degrees, "The option degrees must be an array of the degrees of the total and the anchor scores");
  if (degrees.length !== 2) {
    throw new DataError(
      `The option degrees must have the degrees of the total and the anchor scores, found ${degrees.length}.`,
    );
  }
  const [totalDegree, anchorDegree] = degrees;
  checkDegree(totalDegree, table.rows, "the total scores");
  checkDegree(anchorDegree, table.columns, "the anchor scores");
  const highestCross = Math.min(totalDegree, anchorDegree);
  if (!(Number.isSafeInteger(crossDegree) && crossDegree >= 0 && crossDegree <= highestCross)) {
    throw new DataError(
      `The option crossDegree must be an integer from 0 to ${highestCross}, the smaller of the degrees, found ${describe(crossDegree)}.`,
    );
  }
  const { basis, terms } = bivariateBasis(table.rows, table.columns, totalDegree, anchorDegree, crossDegree);
  const fit = poissonFit(table.counts, basis, terms, table.persons);
  return {
    distribution: Array.from({ length: table.rows }, (_, x) =>
      fit.distribution.slice(x * table.columns, (x + 1) * table.columns),
    ),
    converged: fit.converged,
    iterations: fit.iterations,
  };
}

/**
 * Checks the degree of one score's polynomial.
 *
 * @throws `DataError` unless it is an integer from 1 to one less than the number of scores, and at most 20.
 */
function checkDegree(degree: unknown, scores: number, label: string): void {
  const highest = Math.min(scores - 1, MAX_LOG_LINEAR_DEGREE);
  if (!(typeof degree === "number" && Number.isSafeInteger(degree) && degree >= 1 && degree <= highest)) {
    throw new DataError(
      `The degree of ${label} must be an integer from 1 to ${highest}, one less than the number of scores and at most ${MAX_LOG_LINEAR_DEGREE}, found ${describe(degree)}.`,
    );
  }
}

/**
 * Returns the basis of the bivariate log-linear model, by cell (row, then column), then term, and its number of terms:
 * the products of orthonormal polynomials of the two scores, the constant first, then the main terms and the cross
 * products, which span the powers xⁱ vʲ of the model.
 */
export function bivariateBasis(
  rows: number,
  columns: number,
  totalDegree: number,
  anchorDegree: number,
  crossDegree: number,
): { readonly basis: Float64Array; readonly terms: number } {
  const powers = termPowers(totalDegree, anchorDegree, crossDegree);
  const totalBasis = orthonormalPolynomials(rows, totalDegree);
  const anchorBasis = orthonormalPolynomials(columns, anchorDegree);
  const basis = new Float64Array(rows * columns * powers.length);
  for (let x = 0; x < rows; x++) {
    for (let v = 0; v < columns; v++) {
      const cell = (x * columns + v) * powers.length;
      powers.forEach(([i, j], term) => {
        basis[cell + term] =
          (totalBasis[x * (totalDegree + 1) + i] ?? 0) * (anchorBasis[v * (anchorDegree + 1) + j] ?? 0);
      });
    }
  }
  return { basis, terms: powers.length };
}

/** Returns the powers [i, j] of the total and the anchor score of each term: the constant first, then the main terms and the cross products. */
function termPowers(totalDegree: number, anchorDegree: number, crossDegree: number): [number, number][] {
  const terms: [number, number][] = [[0, 0]];
  for (let i = 1; i <= totalDegree; i++) terms.push([i, 0]);
  for (let j = 1; j <= anchorDegree; j++) terms.push([0, j]);
  for (let i = 1; i <= crossDegree; i++) for (let j = 1; j <= crossDegree; j++) terms.push([i, j]);
  return terms;
}

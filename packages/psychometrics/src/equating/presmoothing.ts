import { DataError, describe, dot } from "@measurekit/core";
import { newton } from "../math/internal.js";
import { checkDistribution } from "./observed-score.js";

/** A score distribution smoothed by a log-linear model. */
export interface LogLinearPresmoothing {
  /** The fitted number of persons at each score, in the order of the scores. */
  readonly distribution: readonly number[];
  /** Whether Newton's method reached the maximum of the likelihood within its iterations. */
  readonly converged: boolean;
  /** Newton iterations run. */
  readonly iterations: number;
}

const MAX_ITERATIONS = 200;
const STEP_TOLERANCE = 1e-6;
/**
 * The highest degree of a log-linear model, twice the 1 to 10 among which Kolen and Brennan choose: a fit holds a row
 * of its terms for each score and takes the cube of the terms in each Newton step, so a degree near a long distribution's
 * length would cost minutes and gigabytes.
 */
export const MAX_LOG_LINEAR_DEGREE = 20;

/**
 * Returns a score distribution smoothed by the log-linear model of degree C.
 *
 * @remarks
 * The maximum likelihood fit of the Poisson regression ln m_x = β0 + β1 x + ... + βC x^C to the counts, as the equate
 * package's `presmoothing(smoothmethod = "loglinear")`. The fit keeps the number of persons and the first C moments of
 * the scores. Newton's method works on polynomials of the scores made orthonormal, which give the same fit as the
 * powers without their ill-conditioning. The distribution is the number of persons at each score, from 0 to the
 * highest.
 *
 * @param distribution - The persons at each score, from 0.
 * @param degree - The degree of the log-linear model: the number of moments of the distribution that the fit keeps.
 * @returns The smoothed distribution, whether the fit converged and the iterations run.
 * @throws `DataError` when the distribution is invalid, or the degree is not an integer from 1 to the number of scores
 * less 1 and at most 20.
 * @see {@link https://doi.org/10.3102/10769986025002133 | Holland and Thayer (2000)}
 */
export function logLinearPresmoothing(distribution: readonly number[], degree: number): LogLinearPresmoothing {
  const total = checkDistribution(distribution, "presmoothed");
  const highest = Math.min(distribution.length - 1, MAX_LOG_LINEAR_DEGREE);
  if (!(Number.isSafeInteger(degree) && degree >= 1 && degree <= highest)) {
    throw new DataError(
      `A log-linear degree must be an integer from 1 to ${highest}, one less than the number of scores and at most ${MAX_LOG_LINEAR_DEGREE}, found ${describe(degree)}.`,
    );
  }
  const basis = orthonormalPolynomials(distribution.length, degree);
  return poissonFit(Float64Array.from(distribution), basis, degree + 1, total);
}

/**
 * Returns the maximum likelihood fit of the Poisson regression ln m = Q β to the counts, `basis` being the matrix Q by
 * cell, then term, whose first column is the constant.
 */
export function poissonFit(
  counts: Float64Array,
  basis: Float64Array,
  terms: number,
  total: number,
): LogLinearPresmoothing {
  const fitted = new Float64Array(counts.length);
  const start = new Float64Array(terms);
  start[0] = Math.log(total / counts.length) * Math.sqrt(counts.length);
  const result = newton(
    (beta, gradient, hessian) => poissonObjective(counts, basis, beta, fitted, gradient, hessian),
    start,
    // Newton's method converges quadratically: a step below 1e-6 leaves an error near 1e-12, below the rounding of
    // the likelihood's value, which the line search would need to see.
    { tolerance: 0, stepTolerance: STEP_TOLERANCE, maxIterations: MAX_ITERATIONS },
  );
  return { distribution: Array.from(fitted), converged: result.converged, iterations: result.iterations };
}

/**
 * Returns the negative Poisson log-likelihood Σ (m_x - f_x ln m_x), up to a constant, of the counts f at the fitted
 * means m = exp(Q β), which it writes into `fitted`, with its gradient Qᵀ (m - f) and Hessian Qᵀ diag(m) Q.
 */
function poissonObjective(
  counts: Float64Array,
  basis: Float64Array,
  beta: Float64Array,
  fitted: Float64Array,
  gradient: Float64Array,
  hessian: Float64Array,
): number {
  const terms = beta.length;
  gradient.fill(0);
  hessian.fill(0);
  let value = 0;
  for (let cell = 0; cell < counts.length; cell++) {
    const offset = cell * terms;
    let eta = 0;
    for (let j = 0; j < terms; j++) eta += (basis[offset + j] ?? 0) * (beta[j] ?? 0);
    const mean = Math.exp(eta);
    const count = counts[cell] ?? 0;
    fitted[cell] = mean;
    value += mean - count * eta;
    addCell(basis, offset, mean - count, mean, gradient, hessian);
  }
  for (let j = 0; j < terms; j++) for (let k = 0; k < j; k++) hessian[k * terms + j] = hessian[j * terms + k] ?? 0;
  return value;
}

/**
 * Adds the terms of the cell whose basis row starts at `offset` to the gradient, Q_j (m - f), and to the lower triangle
 * of the Hessian, m Q_j Q_k.
 */
function addCell(
  basis: Float64Array,
  offset: number,
  residual: number,
  mean: number,
  gradient: Float64Array,
  hessian: Float64Array,
): void {
  const terms = gradient.length;
  for (let j = 0; j < terms; j++) {
    const termJ = basis[offset + j] ?? 0;
    gradient[j] = (gradient[j] ?? 0) + residual * termJ;
    for (let k = 0; k <= j; k++) {
      hessian[j * terms + k] = (hessian[j * terms + k] ?? 0) + mean * termJ * (basis[offset + k] ?? 0);
    }
  }
}

/**
 * Returns the polynomials of degree 0 to `degree` of the scores 0 to `count - 1`, orthonormal over them, by scores,
 * then degree: Gram-Schmidt on the powers of the standardized scores, twice for precision.
 */
export function orthonormalPolynomials(count: number, degree: number): Float64Array {
  const terms = degree + 1;
  const center = (count - 1) / 2;
  const scale = Math.max(center, 1);
  const columns = Array.from({ length: terms }, (_, j) =>
    Float64Array.from({ length: count }, (_, x) => ((x - center) / scale) ** j),
  );
  columns.forEach((column, j) => {
    for (let pass = 0; pass < 2; pass++)
      for (const previous of columns.slice(0, j)) subtractProjection(column, previous);
    const norm = Math.sqrt(dot(column, column));
    column.forEach((value, x) => {
      column[x] = value / norm;
    });
  });
  const basis = new Float64Array(count * terms);
  columns.forEach((column, j) => {
    column.forEach((value, x) => {
      basis[x * terms + j] = value;
    });
  });
  return basis;
}

/** Subtracts from `column` its projection on the unit vector `unit`. */
function subtractProjection(column: Float64Array, unit: Float64Array): void {
  const projection = dot(column, unit);
  column.forEach((value, x) => {
    column[x] = value - projection * (unit[x] ?? 0);
  });
}

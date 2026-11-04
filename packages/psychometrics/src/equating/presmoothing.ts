import { DataError, describe } from "@measurekit/core";
import { orthonormalBasis, poissonFit } from "../math/internal.js";
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
 * Returns the polynomials of degree 0 to `degree` of the scores 0 to `count - 1`, orthonormal over them, by scores,
 * then degree.
 */
export function orthonormalPolynomials(count: number, degree: number): Float64Array {
  return orthonormalBasis(
    Float64Array.from({ length: count }, (_, x) => x),
    degree,
  );
}

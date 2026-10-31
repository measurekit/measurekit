import { checkArray, checkFinite, checkRecord, DataError, isRecord } from "@measurekit/core";
import { invertPositiveDefinite } from "../math/linear-algebra.js";
import { identity } from "./correlation.js";

/** A multivariate normal prior distribution of theta. */
export interface MultivariateNormalPrior {
  /** The mean of each dimension. */
  readonly mean: readonly number[];
  /** The covariance matrix by rows: symmetric and positive definite. */
  readonly covariance: readonly (readonly number[])[];
}

/** The prior as the estimators use it. */
export interface CheckedPrior {
  /** The mean of each dimension. */
  readonly mean: Float64Array;
  /** The covariance matrix by rows. */
  readonly covariance: Float64Array;
  /** The inverse of the covariance matrix by rows. */
  readonly precision: Float64Array;
}

/**
 * Returns the number of dimensions a prior claims, the length of its mean, or `undefined` when it has no mean of at
 * least one value: for estimates without items, where the prior gives the dimensions.
 */
export function priorDimensions(prior: MultivariateNormalPrior | undefined): number | undefined {
  return isRecord(prior) && Array.isArray(prior.mean) && prior.mean.length > 0 ? prior.mean.length : undefined;
}

/**
 * Checks a multivariate normal prior of the given dimensions, N(0, I) by default, and returns it with its precision.
 *
 * @throws `DataError` unless the prior is an object with a finite mean and a symmetric positive definite covariance of
 * the dimensions.
 */
export function checkedPrior(prior: MultivariateNormalPrior | undefined, dimensions: number): CheckedPrior {
  if (prior === undefined) {
    const covariance = identity(dimensions);
    return { mean: new Float64Array(dimensions), covariance, precision: covariance };
  }
  checkRecord(prior, "The option prior must be an object { mean, covariance }");
  const { mean, covariance } = prior;
  checkArray(mean, "The mean of the prior must be an array");
  checkArray(covariance, "The covariance of the prior must be an array of rows");
  if (mean.length !== dimensions || covariance.length !== dimensions) {
    throw new DataError(`The prior needs a mean of ${dimensions} values and a covariance of ${dimensions} rows.`);
  }
  mean.forEach((value, d) => {
    checkFinite(value, `The mean of dimension ${d + 1} of the prior`);
  });
  const matrix = new Float64Array(dimensions * dimensions);
  covariance.forEach((row, i) => {
    checkArray(row, `Row ${i + 1} of the covariance of the prior must be an array`);
    if (row.length !== dimensions) {
      throw new DataError(
        `Row ${i + 1} of the covariance of the prior needs ${dimensions} values, found ${row.length}.`,
      );
    }
    row.forEach((value, j) => {
      checkFinite(value, `Entry (${i + 1}, ${j + 1}) of the covariance of the prior`);
      matrix[i * dimensions + j] = value;
    });
  });
  const precision = invertPositiveDefinite(matrix, dimensions);
  const symmetric = covariance.every((row, i) => row.every((value, j) => value === covariance[j]?.[i]));
  if (precision === null || !symmetric) {
    throw new DataError("The covariance of the prior must be a symmetric positive definite matrix.");
  }
  return { mean: Float64Array.from(mean), covariance: matrix, precision };
}

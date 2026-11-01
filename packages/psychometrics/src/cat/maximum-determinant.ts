import { cholesky } from "@measurekit/core";
import {
  checkedPrior,
  type MultidimensionalEstimate,
  type MultivariateNormalPrior,
  priorDimensions,
} from "../multidimensional/internal.js";
import type { MultidimensionalPoolItem } from "./multidimensional-items.js";
import type { ItemSelectionRule } from "./selection.js";

/**
 * Returns D-optimality selection: scores each item by the determinant of the posterior precision of theta after it,
 * det(Σ⁻¹ + Σ_administered I_j(θ̂) + I_k(θ̂)), with Σ the covariance of the prior, I_j the information matrix of item j
 * and θ̂ the current estimate.
 *
 * @remarks
 * The item that gives the largest determinant shrinks the volume of the confidence ellipsoid of theta the most: the
 * determinant of the posterior covariance is its inverse. The covariance of the prior only enters through its inverse,
 * which matters while few items are administered; pass the covariance of the estimator's prior, such as a calibration's
 * `correlations`, to both. The cost of a selection is a Cholesky factorization of a matrix of the dimensions per eligible
 * item.
 *
 * @param prior - The prior of the thetas; N(0, I) without one.
 * @returns The selection rule.
 * @defaultValue `prior` is N(0, I) of the items' dimensions
 * @throws `DataError` when `prior` is not an object with a finite mean and a symmetric positive definite covariance of
 * as many rows as the mean has values, or, in a test, when the items have another number of dimensions.
 * @see {@link https://doi.org/10.1007/BF02294343 | Segall (1996)}
 */
export function maximumDeterminant(
  prior?: MultivariateNormalPrior,
): ItemSelectionRule<MultidimensionalPoolItem, MultidimensionalEstimate> {
  if (prior !== undefined) checkedPrior(prior, priorDimensions(prior) ?? 0);
  return {
    scores({ pool, administered, estimate, eligible }, out) {
      const dimensions = estimate.theta.length;
      const { precision } = checkedPrior(prior, dimensions);
      const theta = Float64Array.from(estimate.theta);
      const base = Float64Array.from(precision);
      const information = new Float64Array(dimensions * dimensions);
      for (const index of administered) {
        const item = pool[index];
        if (item === undefined) continue;
        item.model.information(item.parameters, theta, information);
        add(base, information);
      }
      const total = new Float64Array(dimensions * dimensions);
      pool.forEach((item, index) => {
        if (eligible[index] !== 1) return;
        item.model.information(item.parameters, theta, information);
        total.set(base);
        add(total, information);
        out[index] = determinant(total, dimensions);
      });
    },
  };
}

function add(sum: Float64Array, matrix: Float64Array): void {
  for (let k = 0; k < sum.length; k++) sum[k] = (sum[k] ?? 0) + (matrix[k] ?? 0);
}

/** Returns the determinant of a symmetric positive definite matrix, the squared product of its Cholesky diagonal. */
function determinant(matrix: Float64Array, size: number): number {
  const lower = cholesky(matrix, size);
  if (lower === null) return 0;
  let product = 1;
  for (let d = 0; d < size; d++) product *= lower[d * size + d] ?? 0;
  return product * product;
}

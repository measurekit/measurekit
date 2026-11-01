import {
  checkArray,
  checkPositiveInteger,
  checkRandom,
  DataError,
  describe,
  type Random,
  type ThetaDistribution,
} from "@measurekit/core";
import { checkThetaDistribution } from "../irt/internal.js";
import { symmetricEigen } from "../math/eigen.js";
import { normalCdf } from "../math/normal.js";

/** An eigenvalue of the correlation with an absolute value below this is 0, so that rounding does not part correlation 1. */
const EIGENVALUE_TOLERANCE = 1e-10;
/** The asymmetry or distance of a diagonal entry from 1 that a correlation matrix may have. */
const MATRIX_TOLERANCE = 1e-8;

/** A posterior as a distribution function that is linear within the cell of each node. */
export interface Margin {
  /** The edges of the cells: node k owns [`edges[k]`, `edges[k + 1]`], and a one-node posterior has equal edges. */
  readonly edges: Float64Array;
  /** The probability at or below the upper edge of each cell, the weights summed and divided by their total. */
  readonly cumulative: Float64Array;
  /** The index of the last cell with a positive weight. */
  readonly last: number;
}

/**
 * Returns the margin of a posterior: each node owns the cell between the midpoints with its neighbors, the outer cells
 * mirrored about their nodes.
 *
 * @param subject - Names the posterior in error messages.
 * @throws `DataError` when the posterior is invalid or its nodes do not strictly ascend.
 */
export function marginOf(posterior: ThetaDistribution, subject: string): Margin {
  const { nodes, weights } = checkThetaDistribution(posterior, subject);
  const size = nodes.length;
  const edges = new Float64Array(size + 1);
  for (let k = 1; k < size; k++) {
    const below = nodes[k - 1] ?? 0;
    const node = nodes[k] ?? 0;
    if (!(node > below)) {
      throw new DataError(
        `The nodes of the ${subject} must ascend strictly, found ${describe(node)} after ${describe(below)} at ${k}.`,
      );
    }
    edges[k] = (below + node) / 2;
  }
  const first = nodes[0] ?? 0;
  const end = nodes[size - 1] ?? 0;
  edges[0] = size === 1 ? first : 2 * first - (edges[1] ?? 0);
  edges[size] = size === 1 ? end : 2 * end - (edges[size - 1] ?? 0);
  // The weights sum to 1 only within a tolerance, so the sum is divided out; the last sum is then exactly 1.
  const cumulative = new Float64Array(size);
  let sum = 0;
  weights.forEach((weight, k) => {
    sum += weight;
    cumulative[k] = sum;
  });
  let last = size - 1;
  while ((weights[last] ?? 0) === 0) last--;
  for (let k = 0; k < size; k++) cumulative[k] = (cumulative[k] ?? 0) / sum;
  return { edges, cumulative, last };
}

/**
 * Returns the theta at probability `u` of a margin: the first cell whose cumulative probability exceeds `u`, and the
 * position within the cell that the distribution function, linear there, gives. A cell with no weight is never reached, `u`
 * of 1 gives the upper edge of the last cell with weight, and a posterior of one node gives that node.
 */
export function quantile({ edges, cumulative, last }: Margin, u: number): number {
  if (u >= 1) return edges[last + 1] ?? 0;
  let low = 0;
  let high = last;
  while (low < high) {
    const middle = (low + high) >> 1;
    if ((cumulative[middle] ?? 0) > u) high = middle;
    else low = middle + 1;
  }
  const below = low === 0 ? 0 : (cumulative[low - 1] ?? 0);
  const lower = edges[low] ?? 0;
  return lower + ((edges[low + 1] ?? 0) - lower) * ((u - below) / ((cumulative[low] ?? 0) - below));
}

/** Returns the thetas of correlated normal scores, one for each margin: the margin's quantile at Φ of the score. */
export function thetasOf(margins: readonly Margin[], scores: Float64Array): number[] {
  return margins.map((margin, i) => quantile(margin, normalCdf(scores[i] ?? 0)));
}

/**
 * Returns the upper-triangle entry (`i`, `j`) of the correlation, checked against its mirror.
 *
 * @throws `DataError` when it is not a number from -1 to 1, is not within 1e-8 of 1 on the diagonal, or is not within
 * 1e-8 of its mirror entry.
 */
function checkedEntry(correlation: readonly (readonly number[])[], i: number, j: number): number {
  const value: unknown = correlation[i]?.[j];
  const mirror: unknown = correlation[j]?.[i];
  if (!(typeof value === "number" && Math.abs(value) <= 1)) {
    throw new DataError(
      `Entry (${i + 1}, ${j + 1}) of the correlation must be a number from -1 to 1, found ${describe(value)}.`,
    );
  }
  if (i === j && !(Math.abs(value - 1) <= MATRIX_TOLERANCE)) {
    throw new DataError(`Entry (${i + 1}, ${i + 1}) of the correlation must be 1, found ${value}.`);
  }
  if (!(typeof mirror === "number" && Math.abs(value - mirror) <= MATRIX_TOLERANCE)) {
    throw new DataError(
      `The correlation must be symmetric, but entry (${i + 1}, ${j + 1}) is ${value} and entry (${j + 1}, ${i + 1}) is ${describe(mirror)}.`,
    );
  }
  return value;
}

/**
 * Returns the correlation as a symmetric matrix by rows, from the upper triangle.
 *
 * @throws `DataError` when it is not an array of `size` arrays of `size` numbers from -1 to 1, symmetric within 1e-8,
 * with 1 on the diagonal within 1e-8.
 */
function symmetricMatrix(correlation: readonly (readonly number[])[], size: number): Float64Array {
  checkArray(correlation, "The correlation must be an array of rows");
  if (correlation.length !== size) {
    throw new DataError(`The correlation needs ${size} rows, one for each posterior, found ${correlation.length}.`);
  }
  correlation.forEach((row, i) => {
    checkArray(row, `Row ${i + 1} of the correlation must be an array`);
    if (row.length !== size) {
      throw new DataError(`Row ${i + 1} of the correlation needs ${size} entries, found ${row.length}.`);
    }
  });
  const matrix = new Float64Array(size * size);
  for (let i = 0; i < size; i++) {
    for (let j = i; j < size; j++) {
      const value = checkedEntry(correlation, i, j);
      matrix[i * size + j] = value;
      matrix[j * size + i] = value;
    }
  }
  return matrix;
}

/**
 * Returns the factor F of a correlation matrix, R = F Fᵀ, by rows, from its eigendecomposition R = V Λ Vᵀ: F_ij = V_ij
 * √λ_j, with an eigenvalue below 1e-10 in absolute value taken as 0.
 *
 * @throws `DataError` when the matrix is not a valid correlation matrix of `size` rows or has an eigenvalue below
 * -1e-10.
 */
export function correlationFactor(correlation: readonly (readonly number[])[], size: number): Float64Array {
  const { values, vectors } = symmetricEigen(symmetricMatrix(correlation, size), size);
  const factor = new Float64Array(size * size);
  values.forEach((value, j) => {
    if (value < -EIGENVALUE_TOLERANCE) {
      throw new DataError(
        `The correlation must be positive semidefinite, but has the eigenvalue ${value}. Pass it through smoothCorrelations of the dimensionality module.`,
      );
    }
    const root = Math.abs(value) < EIGENVALUE_TOLERANCE ? 0 : Math.sqrt(value);
    for (let i = 0; i < size; i++) factor[i * size + j] = (vectors[j * size + i] ?? 0) * root;
  });
  return factor;
}

/**
 * Draws thetas for several people or sections jointly from separate posteriors, coupled by a Gaussian copula.
 *
 * @remarks
 * Each draw takes independent standard normals e, correlated scores z = F e with F Fᵀ = `correlation`, and for each
 * posterior the theta at probability Φ(z) of its distribution function. Each posterior's draws follow it, and the
 * posteriors move together as the correlation says (Cario and Nelson, 1997). A posterior is read as a distribution
 * function that is linear within the cell of each node, which spans from the midpoint with its lower neighbor to that
 * with its upper one, the outer cells mirrored about their nodes; so a cell adds h² / 12 to the variance for a spacing
 * h: 0.0033 on the 61 points of spacing 0.2 that `thetaPosterior` returns, and 0.0002 on 241 points. Pass 241 or more
 * points for a posterior with a standard deviation under 0.5. A node without weight is never drawn.
 *
 * `correlation` is that of the normal scores z, which is also the correlation of the thetas when the posteriors are near
 * normal. A correlation of 1 gives one quantile of every posterior, the largest variance of any sum of the thetas; 0 gives
 * independent draws. Separate posteriors carry nothing of what one section says about another, so a forecast that
 * needs a correlation should show the results at the bounds 0 and 1 until one is estimated. A multidimensional model
 * gives the joint posterior of its dimensions, with its covariance, through the estimates of the multidimensional
 * module; this function serves posteriors that were computed apart.
 *
 * A draw uses as many standard normals from `random` as there are posteriors, in order, whatever the correlation, so a
 * seeded generator gives the same draws for the same posteriors.
 *
 * @param posteriors - The distribution of theta of each person or section, such as `thetaPosterior` returns: finite
 * strictly ascending nodes and non-negative weights that sum to 1 within 1e-6.
 * @param correlation - The correlation of the normal scores: a symmetric matrix by rows, with 1 on the diagonal, entries
 * from -1 to 1, positive semidefinite, with one row for each posterior.
 * @param draws - The number of draws, a positive integer.
 * @param random - The source of the normals.
 * @returns The draws: one array of thetas for each draw, in the order of `posteriors`.
 * @throws `DataError` when `posteriors` is empty or holds an invalid distribution, `correlation` is not a valid
 * correlation matrix of the posteriors' number or has an eigenvalue below -1e-10, `draws` is not a positive integer, or
 * `random` is not a generator.
 * @see {@link https://doi.org/10.2307/2290687 | Cario and Nelson (1997)}
 * @see {@link https://doi.org/10.1016/S0167-6687(02)00135-X | Dhaene et al. (2002)}
 * @example
 * ```ts
 * import { createRandom } from "@measurekit/psychometrics";
 * import { drawJointThetas } from "@measurekit/psychometrics/scaling";
 *
 * const reading = { nodes: [-1, 0, 1], weights: [0.25, 0.5, 0.25] };
 * const writing = { nodes: [-0.5, 0.5], weights: [0.5, 0.5] };
 * const correlation = [
 *   [1, 0.7],
 *   [0.7, 1],
 * ];
 * const draws = drawJointThetas([reading, writing], correlation, 1000, createRandom(1));
 * draws.length; // 1000: each draw holds a theta for reading and one for writing
 * ```
 */
export function drawJointThetas(
  posteriors: readonly ThetaDistribution[],
  correlation: readonly (readonly number[])[],
  draws: number,
  random: Random,
): number[][] {
  checkArray(posteriors, "The posteriors must be an array of distributions");
  if (posteriors.length === 0) throw new DataError("Drawing jointly needs at least one posterior, found none.");
  const margins = posteriors.map((posterior, k) => marginOf(posterior, `posterior ${k + 1}`));
  const size = margins.length;
  const factor = correlationFactor(correlation, size);
  checkPositiveInteger(draws, "The number of draws");
  checkRandom(random);
  const normals = new Float64Array(size);
  const scores = new Float64Array(size);
  return Array.from({ length: draws }, () => {
    for (let j = 0; j < size; j++) normals[j] = random.normal();
    for (let i = 0; i < size; i++) {
      let score = 0;
      for (let j = 0; j < size; j++) score += (factor[i * size + j] ?? 0) * (normals[j] ?? 0);
      scores[i] = score;
    }
    return thetasOf(margins, scores);
  });
}

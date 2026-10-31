import { checkOptions, checkPositiveInteger, DataError, describe, normalizeLogs } from "@measurekit/core";
import { checkFunction } from "../checks.js";
import type { ResponseMatrix } from "../data/response-matrix.js";
import type { ResponseRecords } from "../data/response-records.js";
import { compressResponses } from "../irt/responses.js";
import { invertPositiveDefinite } from "../math/linear-algebra.js";
import { defaultPoints, type Grid, MAX_GRID_NODES, normalLogWeights, productGrid } from "./grid.js";
import { checkMultidimensionalPattern, indexMultidimensionalItems } from "./items.js";
import type { MultidimensionalItem, MultidimensionalTestItem } from "./model.js";
import { answeredPattern } from "./pattern.js";
import { posteriorMode } from "./posterior-mode.js";
import { type CheckedPrior, checkedPrior, type MultivariateNormalPrior, priorDimensions } from "./prior.js";

/** An estimate of a person's thetas with their standard errors and covariance. */
export interface MultidimensionalEstimate {
  /** The estimate of theta in each dimension. */
  readonly theta: readonly number[];
  /** The standard error of each dimension: the square root of the diagonal of `covariance`. */
  readonly standardErrors: readonly number[];
  /**
   * The covariance of the estimate: for the posterior mean the covariance of the posterior, and for the posterior mode
   * the inverse of the sum of the test information and the inverse of the prior covariance.
   */
  readonly covariance: readonly (readonly number[])[];
}

/** A function that estimates thetas from items and one response per item, `null` when missing. */
export type MultidimensionalEstimator = (
  items: readonly MultidimensionalItem[],
  responses: readonly (number | null)[],
) => MultidimensionalEstimate;

/** Options of {@link multidimensionalPosteriorMeanEstimate}. */
export interface MultidimensionalPosteriorMeanOptions {
  /**
   * The prior of theta: pass the `correlations` of a calibration as its covariance.
   *
   * @defaultValue N(0, I) of the items' dimensions
   */
  readonly prior?: MultivariateNormalPrior;
  /**
   * The number of nodes per dimension of the rectangular grid placed on the prior, at least 2: nodes `mean + sd * z`
   * with z from -6 to 6.
   *
   * @defaultValue `61`, `31`, `15`, `9` and `7` for 1 to 5 dimensions, `3` above
   */
  readonly points?: number;
}

/** Options of {@link multidimensionalPosteriorModeEstimate}. */
export interface MultidimensionalPosteriorModeOptions {
  /**
   * The prior of theta: pass the `correlations` of a calibration as its covariance.
   *
   * @defaultValue N(0, I) of the items' dimensions
   */
  readonly prior?: MultivariateNormalPrior;
}

/** A person's id with the estimate of the thetas, `null` for a person without responses. */
export interface MultidimensionalPersonEstimate {
  /** The person's id. */
  readonly id: string;
  /** The estimate, or `null` when the person has no responses. */
  readonly estimate: MultidimensionalEstimate | null;
}

/** The grids of the standard rule computed so far, by dimensions and points: scoring asks for one per person. */
const STANDARD_GRIDS = new Map<string, Grid>();

/**
 * Returns the expected a posteriori estimate (EAP) of the thetas under a multivariate normal prior.
 *
 * @remarks
 * The posterior mean, with the posterior covariance, by quadrature on the rectangular grid placed on the prior: nodes
 * `mean + sd * z`, weights proportional to the prior density at the nodes times the likelihood. Missing responses are
 * skipped; without responses the estimate is the prior, also without items, when the `prior` gives the dimensions. The
 * grid has `points ** dimensions` nodes: more than 10⁶ throw.
 *
 * @throws `DataError` when the items, responses, prior or points are invalid, there are no items and no prior, or the
 * grid has more than 10⁶ nodes: use {@link multidimensionalPosteriorModeEstimate}, which works in any number of
 * dimensions.
 */
export function multidimensionalPosteriorMeanEstimate(
  items: readonly MultidimensionalItem[],
  responses: readonly (number | null)[],
  options: MultidimensionalPosteriorMeanOptions = {},
): MultidimensionalEstimate {
  checkOptions(options);
  const dimensions = checkMultidimensionalPattern(items, responses, priorDimensions(options.prior));
  const prior = checkedPrior(options.prior, dimensions);
  const points = checkedPoints(options.points, dimensions);
  const pattern = answeredPattern(items, responses, dimensions);
  const grid = priorGrid(standardGrid(dimensions, points), prior);
  const posterior = new Float64Array(grid.count);
  normalLogWeights(grid, prior.mean, prior.precision, posterior);
  const theta = new Float64Array(dimensions);
  for (let q = 0; q < grid.count; q++) {
    for (let d = 0; d < dimensions; d++) theta[d] = grid.nodes[q * dimensions + d] ?? 0;
    posterior[q] = (posterior[q] ?? 0) + pattern.logLikelihood(theta);
  }
  normalizeLogs(posterior);
  return posteriorMoments(grid, posterior);
}

/**
 * Checks the number of nodes per dimension of the posterior mean's grid.
 *
 * @throws `DataError` when it is not an integer from 2, or the grid would have more than {@link MAX_GRID_NODES} nodes.
 */
function checkedPoints(points: number | undefined, dimensions: number): number {
  const checked = points === undefined ? defaultPoints(dimensions) : points;
  checkPositiveInteger(checked, "The option points");
  if (checked < 2) throw new DataError(`The option points must be at least 2, found ${describe(checked)}.`);
  if (!(checked ** dimensions <= MAX_GRID_NODES)) {
    throw new DataError(
      `A grid of ${checked} points in ${dimensions} dimensions has more than ${MAX_GRID_NODES} nodes; use fewer points or multidimensionalPosteriorModeEstimate.`,
    );
  }
  return checked;
}

/** Returns the standard grid placed on the prior: the nodes `mean + sd * z`. */
function priorGrid(standard: Grid, prior: CheckedPrior): Grid {
  const { dimensions } = standard;
  const sds = Float64Array.from(prior.mean, (_, d) => Math.sqrt(prior.covariance[d * dimensions + d] ?? 0));
  const nodes = standard.nodes.map((z, k) => (prior.mean[k % dimensions] ?? 0) + (sds[k % dimensions] ?? 0) * z);
  return { dimensions, count: standard.count, nodes };
}

/** Returns the mean and the covariance of the nodes under the posterior weights. */
function posteriorMoments(grid: Grid, posterior: Float64Array): MultidimensionalEstimate {
  const { dimensions, count, nodes } = grid;
  const mean = new Float64Array(dimensions);
  for (let q = 0; q < count; q++) addScaled(mean, nodes, q * dimensions, posterior[q] ?? 0);
  const covariance = new Float64Array(dimensions * dimensions);
  const centered = new Float64Array(dimensions);
  for (let q = 0; q < count; q++) {
    for (let d = 0; d < dimensions; d++) centered[d] = (nodes[q * dimensions + d] ?? 0) - (mean[d] ?? 0);
    addOuterProduct(covariance, centered, posterior[q] ?? 0);
  }
  return estimateOf(mean, covariance, dimensions);
}

/** Adds `weight` times `values[from + d]` to `out[d]` for each d. */
function addScaled(out: Float64Array, values: Float64Array, from: number, weight: number): void {
  for (let d = 0; d < out.length; d++) out[d] = (out[d] ?? 0) + weight * (values[from + d] ?? 0);
}

/** Adds `weight` times x xᵀ to the matrix `out`, by rows. */
function addOuterProduct(out: Float64Array, x: Float64Array, weight: number): void {
  const size = x.length;
  for (let i = 0; i < size; i++) {
    const row = weight * (x[i] ?? 0);
    for (let j = 0; j < size; j++) out[i * size + j] = (out[i * size + j] ?? 0) + row * (x[j] ?? 0);
  }
}

/**
 * Returns the maximum a posteriori estimate (MAP, Bayes modal) of the thetas under a multivariate normal prior.
 *
 * @remarks
 * A scan of the log posterior on a grid of 5 points per dimension (3 above three dimensions) from -3 to 3 standard
 * deviations of the prior picks the start of Fisher scoring with backtracking, which runs until the gradient is below
 * 1e-10. The covariance is the inverse of the test information plus the inverse of the prior covariance, at the
 * estimate. Missing responses are skipped; without responses the estimate is the prior mean, also without items, when
 * the `prior` gives the dimensions.
 *
 * @throws `DataError` when the items, responses or prior are invalid, or there are no items and no prior.
 */
export function multidimensionalPosteriorModeEstimate(
  items: readonly MultidimensionalItem[],
  responses: readonly (number | null)[],
  options: MultidimensionalPosteriorModeOptions = {},
): MultidimensionalEstimate {
  checkOptions(options);
  const dimensions = checkMultidimensionalPattern(items, responses, priorDimensions(options.prior));
  const prior = checkedPrior(options.prior, dimensions);
  const { theta, information } = posteriorMode(answeredPattern(items, responses, dimensions), prior, dimensions);
  const covariance =
    invertPositiveDefinite(information, dimensions) ??
    new Float64Array(dimensions * dimensions).fill(Number.POSITIVE_INFINITY);
  return estimateOf(theta, covariance, dimensions);
}

/**
 * Returns the estimate of the thetas of every person, in the order of the data's person ids, from the person's answered
 * items only, so that sparse data, such as adaptive tests on a large pool, need no row per person over all items.
 * Items are matched to the data by id, as a calibration's items are. Persons without responses get `null` estimates.
 *
 * @param estimator - The estimator of each person.
 * @defaultValue `estimator` is `multidimensionalPosteriorModeEstimate`
 * @throws `DataError` when the data are malformed, an item is invalid, item ids repeat, an item of the data has no item
 * with its id, a response is not a category of its item, the estimator is not a function, or it throws it.
 */
export function multidimensionalEstimates(
  data: ResponseMatrix | ResponseRecords,
  items: readonly MultidimensionalTestItem[],
  estimator: MultidimensionalEstimator = multidimensionalPosteriorModeEstimate,
): MultidimensionalPersonEstimate[] {
  checkFunction(
    estimator,
    "The estimator must be a function of items and responses, such as multidimensionalPosteriorModeEstimate",
  );
  const responses = compressResponses(data);
  const indices = indexMultidimensionalItems(items);
  const columns = responses.itemIds.map((id) => {
    const item = items[indices.get(id) ?? -1];
    if (item === undefined) throw new DataError(`The data have item "${id}", which is not among the items.`);
    return item;
  });
  return responses.personIds.map((id, person) => {
    const start = responses.offsets[person] ?? 0;
    const end = responses.offsets[person + 1] ?? 0;
    if (end === start) return { id, estimate: null };
    const answered = Array.from(responses.items.subarray(start, end)).flatMap((column) => columns[column] ?? []);
    return { id, estimate: estimator(answered, Array.from(responses.codes.subarray(start, end))) };
  });
}

function standardGrid(dimensions: number, points: number): Grid {
  const key = `${dimensions} ${points}`;
  let grid = STANDARD_GRIDS.get(key);
  if (grid === undefined) {
    grid = productGrid(dimensions, points);
    STANDARD_GRIDS.set(key, grid);
  }
  return grid;
}

function estimateOf(theta: Float64Array, covariance: Float64Array, dimensions: number): MultidimensionalEstimate {
  return {
    theta: Array.from(theta),
    standardErrors: Array.from(theta, (_, d) => Math.sqrt(covariance[d * dimensions + d] ?? 0)),
    covariance: Array.from({ length: dimensions }, (_, i) =>
      Array.from(covariance.subarray(i * dimensions, (i + 1) * dimensions)),
    ),
  };
}

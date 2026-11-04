import {
  checkIntegerFrom,
  checkOptions,
  DataError,
  type FieldNames,
  invertPositiveDefinite,
  normalizeLogs,
} from "@measurekit/core";
import { checkFunction } from "../checks.js";
import type { ResponseMatrix, ResponseRecords } from "../data/internal.js";
import { checkRange, estimatePersons, RANGE_OPTIONS, type RangeOptions } from "../irt/internal.js";
import { addOuterProduct, mirror } from "../math/internal.js";
import { defaultPoints, type Grid, MAX_GRID_NODES, normalLogWeights, productGrid } from "./grid.js";
import { type GridLikelihood, gridLikelihood } from "./grid-likelihood.js";
import { checkMultidimensionalPattern, indexMultidimensionalItems } from "./items.js";
import { maximumLikelihoodMode, weightedLikelihoodMode } from "./likelihood-mode.js";
import type { MultidimensionalItem, MultidimensionalTestItem } from "./model.js";
import { type AnsweredPattern, answeredPattern } from "./pattern.js";
import { posteriorMode } from "./posterior-mode.js";
import { type CheckedPrior, checkedPrior, type MultivariateNormalPrior, priorDimensions } from "./prior.js";

/** An estimate of a person's thetas with their standard errors and covariance. */
export interface MultidimensionalEstimate {
  /** The estimate of theta in each dimension. */
  readonly theta: readonly number[];
  /** The standard error of each dimension: the square root of the diagonal of `covariance`. */
  readonly standardErrors: readonly number[];
  /**
   * The covariance of the estimate: for the posterior mean the covariance of the posterior, for the posterior mode the
   * inverse of the sum of the test information and the inverse of the prior covariance, and for the maximum and
   * weighted likelihood estimates the inverse of the test information.
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

const MULTIDIMENSIONAL_POSTERIOR_MEAN_OPTIONS: FieldNames<MultidimensionalPosteriorMeanOptions> = {
  prior: true,
  points: true,
};

/** Options of {@link multidimensionalPosteriorModeEstimate}. */
export interface MultidimensionalPosteriorModeOptions {
  /**
   * The prior of theta: pass the `correlations` of a calibration as its covariance.
   *
   * @defaultValue N(0, I) of the items' dimensions
   */
  readonly prior?: MultivariateNormalPrior;
}

const MULTIDIMENSIONAL_POSTERIOR_MODE_OPTIONS: FieldNames<MultidimensionalPosteriorModeOptions> = { prior: true };

/** A person's id with the estimate of the thetas, `null` for a person without responses. */
export interface MultidimensionalPersonEstimate {
  /** The person's id. */
  readonly id: string;
  /** The estimate, or `null` when the person has no responses. */
  readonly estimate: MultidimensionalEstimate | null;
}

/** The most grids kept for reuse, the latest asked for, so that grids of many sizes do not accumulate. */
const MAX_CACHED_GRIDS = 8;
/**
 * The most coordinates of nodes that the grids kept for reuse hold, 8 bytes each: 64 MiB. The grid asked for last stays
 * whatever its size, at most 80 MB within the limit of nodes.
 */
const MAX_CACHED_COORDINATES = 2 ** 23;

/**
 * The grids of the standard rule computed lately, by dimensions and points, in the order they were last asked for:
 * scoring asks for one per person.
 */
export const STANDARD_GRIDS = new Map<string, Grid>();

/**
 * The most log probabilities of items over a grid that the posterior means of the persons of one call keep, 8 bytes
 * each: 64 MiB.
 */
const MAX_KEPT_LOG_PROBABILITIES = 2 ** 23;

/** What the posterior mean of a person needs of the prior and the grid, which persons with one prior share. */
interface PosteriorSetup {
  readonly grid: Grid;
  /** The log weights of the prior at the nodes, normalized. */
  readonly logPrior: Float64Array;
  readonly likelihood: GridLikelihood;
}

/**
 * Returns the expected a posteriori estimate (EAP) of the thetas under a multivariate normal prior.
 *
 * @remarks
 * The posterior mean, with the posterior covariance, by quadrature on the rectangular grid placed on the prior: nodes
 * `mean + sd * z`, weights proportional to the prior density at the nodes times the likelihood. Missing responses are
 * skipped; without responses the estimate is the prior, also without items, when the `prior` gives the dimensions. The
 * grid has `points ** dimensions` nodes: more than 10⁶ throw.
 *
 * @param items - The items the person answered, with their parameters.
 * @param responses - The person's response to each item; `null` when missing.
 * @param options - The prior and the number of points of the grid.
 * @returns The thetas, their standard errors and their posterior covariance.
 * @throws `DataError` when the items, responses, prior or points are invalid, there are no items and no prior, or the
 * grid has more than 10⁶ nodes: use {@link multidimensionalPosteriorModeEstimate}, which works in any number of
 * dimensions.
 */
export function multidimensionalPosteriorMeanEstimate(
  items: readonly MultidimensionalItem[],
  responses: readonly (number | null)[],
  options: MultidimensionalPosteriorMeanOptions = {},
): MultidimensionalEstimate {
  checkOptions(options, MULTIDIMENSIONAL_POSTERIOR_MEAN_OPTIONS);
  const dimensions = checkMultidimensionalPattern(items, responses, priorDimensions(options.prior));
  const prior = checkedPrior(options.prior, dimensions);
  const points = checkedPoints(options.points, dimensions);
  const grid = priorGrid(standardGrid(dimensions, points), prior);
  return posteriorMean(posteriorSetup(grid, prior, 0), items, responses);
}

/** Returns the grid, the prior's log weights on it and the likelihoods of the items over it. */
function posteriorSetup(grid: Grid, prior: CheckedPrior, maxKept: number): PosteriorSetup {
  const logPrior = new Float64Array(grid.count);
  normalLogWeights(grid, prior.mean, prior.precision, logPrior);
  return { grid, logPrior, likelihood: gridLikelihood(grid, maxKept) };
}

/** Returns the posterior mean and covariance of the thetas given checked items and their responses. */
function posteriorMean(
  { grid, logPrior, likelihood }: PosteriorSetup,
  items: readonly MultidimensionalItem[],
  responses: readonly (number | null)[],
): MultidimensionalEstimate {
  const sums = new Float64Array(grid.count);
  items.forEach((item, index) => {
    const response = responses[index];
    if (response !== null && response !== undefined) likelihood.add(item, response, sums);
  });
  const posterior = new Float64Array(grid.count);
  for (let q = 0; q < grid.count; q++) posterior[q] = (logPrior[q] ?? 0) + (sums[q] ?? 0);
  normalizeLogs(posterior);
  return posteriorMoments(grid, posterior);
}

/**
 * Returns {@link multidimensionalPosteriorMeanEstimate} with its default options, for the persons of one call of
 * {@link multidimensionalEstimates}: the prior's grid and the log probabilities of each item over it are computed once,
 * not for each person.
 */
function sharedPosteriorMean(): MultidimensionalEstimator {
  const setups = new Map<number, PosteriorSetup>();
  return (items, responses) => {
    const dimensions = checkMultidimensionalPattern(items, responses);
    let setup = setups.get(dimensions);
    if (setup === undefined) {
      const prior = checkedPrior(undefined, dimensions);
      const grid = priorGrid(standardGrid(dimensions, checkedPoints(undefined, dimensions)), prior);
      setup = posteriorSetup(grid, prior, MAX_KEPT_LOG_PROBABILITIES);
      setups.set(dimensions, setup);
    }
    return posteriorMean(setup, items, responses);
  };
}

/**
 * Checks the number of nodes per dimension of the posterior mean's grid.
 *
 * @throws `DataError` when it is not an integer from 2, or the grid would have more than {@link MAX_GRID_NODES} nodes.
 */
function checkedPoints(points: number | undefined, dimensions: number): number {
  const checked = points === undefined ? defaultPoints(dimensions) : points;
  checkIntegerFrom(checked, "The option points", 2);
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
  return { dimensions, points: standard.points, count: standard.count, nodes };
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
  mirror(covariance, dimensions);
  return estimateOf(mean, covariance, dimensions);
}

/** Adds `weight` times `values[from + d]` to `out[d]` for each d. */
function addScaled(out: Float64Array, values: Float64Array, from: number, weight: number): void {
  for (let d = 0; d < out.length; d++) out[d] = (out[d] ?? 0) + weight * (values[from + d] ?? 0);
}

/**
 * Returns the maximum a posteriori estimate (MAP, Bayes modal) of the thetas under a multivariate normal prior.
 *
 * @remarks
 * A scan of the log posterior on a grid of 5 points per dimension (3 above three dimensions) from -3 to 3 standard
 * deviations of the prior, or above five dimensions at the prior mean and 3 standard deviations either side of it on
 * each axis, picks the start of Fisher scoring with backtracking, which runs until the gradient is below 1e-10. The
 * covariance is the inverse of the test information plus the inverse of the prior covariance, at the estimate. Missing
 * responses are skipped; without responses the estimate is the prior mean, also without items, when the `prior` gives
 * the dimensions.
 *
 * @param items - The items the person answered, with their parameters.
 * @param responses - The person's response to each item; `null` when missing.
 * @param options - The prior.
 * @returns The thetas, their standard errors and their covariance.
 * @throws `DataError` when the items, responses or prior are invalid, or there are no items and no prior.
 */
export function multidimensionalPosteriorModeEstimate(
  items: readonly MultidimensionalItem[],
  responses: readonly (number | null)[],
  options: MultidimensionalPosteriorModeOptions = {},
): MultidimensionalEstimate {
  checkOptions(options, MULTIDIMENSIONAL_POSTERIOR_MODE_OPTIONS);
  const dimensions = checkMultidimensionalPattern(items, responses, priorDimensions(options.prior));
  const prior = checkedPrior(options.prior, dimensions);
  const { theta, information } = posteriorMode(answeredPattern(items, responses, dimensions), prior, dimensions);
  const covariance =
    invertPositiveDefinite(information, dimensions) ??
    new Float64Array(dimensions * dimensions).fill(Number.POSITIVE_INFINITY);
  return estimateOf(theta, covariance, dimensions);
}

/**
 * Returns the maximum likelihood estimate (MLE) of the thetas within `range` in every dimension.
 *
 * @remarks
 * A scan of the likelihood on a grid of 5 points per dimension (3 above three dimensions) of the box of `range`, or
 * above five dimensions at its center and its faces on each axis, picks the start of Fisher scoring with the steps
 * projected onto the box, which runs until the score of the coordinates inside it is below 1e-10. Where the likelihood
 * grows without limit, as for an extreme pattern or a dimension whose items were all answered in their lowest or highest
 * categories, the estimate is at that bound of `range`, as that of `maximumLikelihoodEstimate` is. The covariance is the
 * inverse of the test information at the estimate. Missing responses are skipped.
 *
 * @param items - The items the person answered, with their parameters.
 * @param responses - The person's response to each item; `null` when missing.
 * @param options - The range of each theta.
 * @returns The thetas, their standard errors and their covariance.
 * @throws `DataError` when the items, responses or range are invalid, there is no response, or the slopes of the
 * answered items do not span every dimension, along which the likelihood is then flat.
 * @see {@link https://doi.org/10.1007/978-0-387-89976-3 | Reckase (2009)}
 */
export function multidimensionalMaximumLikelihoodEstimate(
  items: readonly MultidimensionalItem[],
  responses: readonly (number | null)[],
  options: RangeOptions = {},
): MultidimensionalEstimate {
  const { dimensions, range } = checkLikelihoodPattern(items, responses, options, "Maximum likelihood");
  const pattern = answeredPattern(items, responses, dimensions);
  return informationEstimate(pattern, maximumLikelihoodMode(pattern, dimensions, range), dimensions);
}

/**
 * Returns Warm's weighted likelihood estimate (WLE) of the thetas within `range` in every dimension.
 *
 * @remarks
 * The root of the score plus Warm's correction, ½ tr(I⁻¹ J_r) for dimension r, with I the test information and J_r
 * the sum over the answered items and their categories of ∂P/∂θ_r ∇²P / P: the score less the information times the
 * first-order bias of the MLE, as `weightedLikelihoodEstimate` solves it on one dimension. It is the mode of the
 * likelihood times √det I(θ), Jeffreys' prior, for the M2PL and the graded model, and differs from it for the M3PL. From
 * the posterior mode under N(0, I), Fisher scoring's steps, or Newton's with the Jacobian by central differences where
 * scoring's do not halve the weighted score, projected onto the box, run until the weighted score of the coordinates
 * inside it is below 1e-9. The WLE of an extreme pattern is finite where the correction balances the score; where the
 * weighted score still pushes outward, the estimate is at that bound of `range`. The covariance is the inverse of the
 * test information at the estimate. Missing responses are skipped.
 *
 * @param items - The items the person answered, with their parameters.
 * @param responses - The person's response to each item; `null` when missing.
 * @param options - The range of each theta.
 * @returns The thetas, their standard errors and their covariance.
 * @throws `DataError` when the items, responses or range are invalid, there is no response, or the slopes of the
 * answered items do not span every dimension.
 * @see {@link https://doi.org/10.1007/BF02294627 | Warm (1989)}
 * @see {@link https://doi.org/10.1007/s11336-013-9399-0 | Wang (2015)}
 */
export function multidimensionalWeightedLikelihoodEstimate(
  items: readonly MultidimensionalItem[],
  responses: readonly (number | null)[],
  options: RangeOptions = {},
): MultidimensionalEstimate {
  const { dimensions, range } = checkLikelihoodPattern(items, responses, options, "Weighted likelihood");
  const pattern = answeredPattern(items, responses, dimensions);
  return informationEstimate(pattern, weightedLikelihoodMode(pattern, dimensions, range), dimensions);
}

/**
 * Checks the options, the items and responses of a likelihood estimate, and that the slopes of the answered items span
 * every dimension; returns the dimensions and the range.
 *
 * @throws `DataError` when they are invalid, there is no response, or the sum of the outer products of the slopes is
 * singular.
 */
function checkLikelihoodPattern(
  items: readonly MultidimensionalItem[],
  responses: readonly (number | null)[],
  options: RangeOptions,
  method: string,
): { readonly dimensions: number; readonly range: readonly [number, number] } {
  checkOptions(options, RANGE_OPTIONS);
  const noResponse = `${method} estimation needs at least one response.`;
  if (Array.isArray(items) && items.length === 0) throw new DataError(noResponse);
  const dimensions = checkMultidimensionalPattern(items, responses);
  const range = checkRange(options.range);
  const span = new Float64Array(dimensions * dimensions);
  let answered = 0;
  items.forEach((item, i) => {
    if (responses[i] === null || responses[i] === undefined) return;
    answered++;
    addOuterProduct(span, item.model.toVector(item.parameters).subarray(0, dimensions));
  });
  if (answered === 0) throw new DataError(noResponse);
  mirror(span, dimensions);
  if (invertPositiveDefinite(span, dimensions) === null) {
    throw new DataError(
      `${method} estimation needs answered items whose slopes span the ${dimensions} dimensions: along a dimension they leave out, the likelihood is flat.`,
    );
  }
  return { dimensions, range };
}

/** Returns the estimate at theta with the inverse of the test information there as its covariance. */
function informationEstimate(
  pattern: AnsweredPattern,
  theta: Float64Array,
  dimensions: number,
): MultidimensionalEstimate {
  const information = new Float64Array(dimensions * dimensions);
  pattern.information(theta, information);
  const covariance =
    invertPositiveDefinite(information, dimensions) ??
    new Float64Array(dimensions * dimensions).fill(Number.POSITIVE_INFINITY);
  return estimateOf(theta, covariance, dimensions);
}

/** Options of {@link multidimensionalEstimates}. */
export interface MultidimensionalEstimatesOptions {
  /**
   * The estimator of each person.
   *
   * @defaultValue `multidimensionalPosteriorModeEstimate`
   */
  readonly estimator?: MultidimensionalEstimator;
}

const MULTIDIMENSIONAL_ESTIMATES_OPTIONS: FieldNames<MultidimensionalEstimatesOptions> = { estimator: true };

/**
 * Returns the estimate of the thetas of every person, in the order of the data's person ids, from the person's answered
 * items only, so that sparse data, such as adaptive tests on a large pool, need no row per person over all items.
 * Items are matched to the data by id, as a calibration's items are. Persons without responses get `null` estimates.
 *
 * @param data - The responses of the persons, as a matrix or as records.
 * @param items - The items, matched to the data by id.
 * @param options - The estimator of each person.
 * @returns The estimate of each person, with `null` for a person without responses.
 * @throws `DataError` when the data are malformed, an item is invalid, item ids repeat, an item of the data has no item
 * with its id, a response is not a category of its item, the options are not an object, the estimator is not a
 * function, or it throws it.
 */
export function multidimensionalEstimates(
  data: ResponseMatrix | ResponseRecords,
  items: readonly MultidimensionalTestItem[],
  options: MultidimensionalEstimatesOptions = {},
): readonly MultidimensionalPersonEstimate[] {
  checkOptions(options, MULTIDIMENSIONAL_ESTIMATES_OPTIONS);
  const { estimator = multidimensionalPosteriorModeEstimate } = options;
  checkFunction(
    estimator,
    "The estimator must be a function of items and responses, such as multidimensionalPosteriorModeEstimate",
  );
  const posteriorMeans = estimator === multidimensionalPosteriorMeanEstimate;
  return estimatePersons(data, items, indexMultidimensionalItems, posteriorMeans ? sharedPosteriorMean() : estimator);
}

/** Returns the grid of the standard rule, kept for reuse with the grids asked for last within the limits above. */
function standardGrid(dimensions: number, points: number): Grid {
  const key = `${dimensions} ${points}`;
  const grid = STANDARD_GRIDS.get(key) ?? productGrid(dimensions, points);
  // Moved to the end of the order, as the latest asked for; the earliest asked for leave past the limits.
  STANDARD_GRIDS.delete(key);
  STANDARD_GRIDS.set(key, grid);
  let coordinates = 0;
  for (const cached of STANDARD_GRIDS.values()) coordinates += cached.nodes.length;
  for (const [oldest, cached] of STANDARD_GRIDS) {
    if (oldest === key || (STANDARD_GRIDS.size <= MAX_CACHED_GRIDS && coordinates <= MAX_CACHED_COORDINATES)) break;
    STANDARD_GRIDS.delete(oldest);
    coordinates -= cached.nodes.length;
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

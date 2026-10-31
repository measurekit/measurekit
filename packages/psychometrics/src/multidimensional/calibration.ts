import { checkChoice, checkOptions, checkPositiveInteger, checkStopping, DataError, describe } from "@measurekit/core";
import type { ResponseMatrix } from "../data/response-matrix.js";
import type { ResponseRecords } from "../data/response-records.js";
import type { StandardErrorMethod } from "../irt/calibration-options.js";
import { compressResponses } from "../irt/responses.js";
import type { MultidimensionalCalibrationOptions } from "./calibration-options.js";
import { type ItemState, setUp } from "./calibration-setup.js";
import { type Latent, latentDistribution } from "./correlation.js";
import { designOf } from "./design.js";
import { checkEmDimensions, expectationMaximization } from "./em.js";
import { expectationStep } from "./expectation.js";
import { defaultPoints, productGrid } from "./grid.js";
import { standardizedLoadings } from "./loadings.js";
import { itemMaximization } from "./m-step.js";
import type { MultidimensionalTestItem } from "./model.js";
import { crossProductCovariance, type Fit, hessianCovariance, parameterCount } from "./standard-errors.js";

/** An item with its calibrated parameters. */
export interface CalibratedMultidimensionalItem<TParameters> extends MultidimensionalTestItem<TParameters> {
  /**
   * Standard errors of the estimated parameters by name: `a1` to `aD`, `intercept` or `intercept1` upward, and `c`.
   * Slopes fixed at 0 by the pattern of loadings are not listed. `null` without standard errors: with the option
   * `standardErrors` `"none"`, and when the information matrix cannot be inverted or gives a variance that is not
   * positive.
   */
  readonly standardErrors: Readonly<Record<string, number>> | null;
}

/** Item parameters and correlations of dimensions estimated by marginal maximum likelihood, with fit statistics. */
export interface MultidimensionalCalibration<TParameters> {
  /** The names of the dimensions, in the order of the slopes. */
  readonly dimensions: readonly string[];
  /** The items, in item order. */
  readonly items: readonly CalibratedMultidimensionalItem<TParameters>[];
  /** The correlations between dimensions, by rows: the identity unless a confirmatory model estimates them. */
  readonly correlations: readonly (readonly number[])[];
  /**
   * The standard errors of the correlations, by rows with 0 on the diagonal; `null` when the correlations are not
   * estimated or the calibration has no standard errors.
   */
  readonly correlationStandardErrors: readonly (readonly number[])[] | null;
  /** The standardized loadings, items by dimensions: λ = (a / 1.702) / √(1 + (a / 1.702)ᵀ Φ (a / 1.702)). */
  readonly loadings: readonly (readonly number[])[];
  /** The communality of each item: λᵀ Φ λ, with Φ the correlations. */
  readonly communalities: readonly number[];
  /** The marginal log-likelihood at the estimates, without the priors. */
  readonly logLikelihood: number;
  /** Akaike's information criterion: -2 logLikelihood + 2p, with p the number of estimated item parameters and correlations. */
  readonly aic: number;
  /** The Bayesian information criterion: -2 logLikelihood + p ln n, with n the number of persons with at least one response. */
  readonly bic: number;
  /** Whether EM reached `tolerance` within `maxIterations`. */
  readonly converged: boolean;
  /** EM iterations run. */
  readonly iterations: number;
}

const DEFAULT_TOLERANCE = 1e-4;
const DEFAULT_MAX_ITERATIONS = 500;
const STANDARD_ERROR_METHODS: readonly StandardErrorMethod[] = ["none", "hessian", "cross-product"];

/**
 * Estimates the item parameters of a multidimensional model, and the correlations between its dimensions, by marginal
 * maximum likelihood with the Bock-Aitkin EM algorithm, theta integrated by quadrature over a grid.
 *
 * @remarks
 * Exploratory models (`dimensions`) have every item load on every dimension, and confirmatory ones (`loadings`) only
 * on the dimensions named for them; the dimensions have mean 0 and variance 1. The grid has `points` equally spaced
 * nodes per dimension from -6 to 6, weighted by the normal density of the correlations, as mirt. The M-step takes a
 * Fisher-scoring step for each item and maximizes the correlations, which keep a unit diagonal and stay positive
 * definite, with BFGS on the weights of the grid, so that it maximizes the reported likelihood. With priors, the
 * estimates are posterior modes. Standard errors come from the Hessian of the marginal log-likelihood plus the log
 * priors, by central differences of its exact gradient, or from the cross-product of the persons' score vectors.
 * Missing responses are skipped; persons without responses change nothing.
 *
 * An exploratory solution is determined up to the signs of its dimensions; {@link rotateLoadings} rotates it.
 *
 * @throws `DataError` when the data or an option is invalid, there are no responses, neither or both of `dimensions`
 * and `loadings` are given, there are more than 3 dimensions, an item is not in `loadings` or lacks responses in a
 * category up to its highest or has more categories than its model allows, or the grid has more than 10⁶ nodes.
 */
export function calibrateMultidimensional<TParameters>(
  data: ResponseMatrix | ResponseRecords,
  options: MultidimensionalCalibrationOptions<TParameters>,
): MultidimensionalCalibration<TParameters> {
  checkOptions(options);
  const responses = compressResponses(data);
  const { tolerance = DEFAULT_TOLERANCE, maxIterations = DEFAULT_MAX_ITERATIONS, standardErrors = "none" } = options;
  checkStopping(tolerance, maxIterations);
  checkChoice(standardErrors, "The option standardErrors", STANDARD_ERROR_METHODS);
  const respondents = Array.from(responses.personIds, (_, person) => {
    return (responses.offsets[person + 1] ?? 0) > (responses.offsets[person] ?? 0);
  }).filter(Boolean).length;
  if (respondents === 0) throw new DataError("Calibration needs at least one response.");
  const design = designOf(options, responses.itemIds);
  const dimensions = design.dimensionNames.length;
  checkEmDimensions(dimensions);
  const points = options.points === undefined ? defaultPoints(dimensions) : options.points;
  checkPoints(points);
  const grid = productGrid(dimensions, points);
  const items = setUp(responses, options, design, grid.count);
  const latent = latentDistribution(grid);
  const fit: Fit<TParameters> = {
    items,
    grid,
    latent,
    correlated: design.correlated,
    maximization: itemMaximization(items, grid),
    eStep: expectationStep(responses, items, grid, latent.logWeights),
    responses,
  };

  const { converged, iterations } = expectationMaximization(fit, tolerance, maxIterations);
  const logLikelihood = fit.eStep.run();
  const size = parameterCount(fit);
  const covariance = size > 0 ? covarianceOf(standardErrors, fit) : null;
  const { loadings, communalities } = standardizedLoadings(
    items.map((item) => Array.from(item.vector.subarray(0, dimensions))),
    matrixRows(latent),
  );
  return {
    dimensions: design.dimensionNames,
    items: items.map((item, i) => ({
      id: item.id,
      model: item.model,
      parameters: item.parameters,
      standardErrors: covariance === null ? null : itemErrors(item, i, items, covariance, size),
    })),
    correlations: matrixRows(latent),
    correlationStandardErrors:
      covariance === null || !design.correlated ? null : correlationErrors(latent, covariance, size),
    loadings,
    communalities,
    logLikelihood,
    aic: -2 * logLikelihood + 2 * size,
    bic: -2 * logLikelihood + size * Math.log(respondents),
    converged,
    iterations,
  };
}

function checkPoints(points: unknown): asserts points is number {
  checkPositiveInteger(points, "The option points");
  if (points < 2) throw new DataError(`The option points must be at least 2, found ${describe(points)}.`);
}

/** Returns the covariance matrix of the estimated parameters by the method, or `null` without standard errors. */
function covarianceOf<TParameters>(method: StandardErrorMethod, fit: Fit<TParameters>): Float64Array | null {
  if (method === "hessian") return hessianCovariance(fit);
  if (method === "cross-product") return crossProductCovariance(fit);
  return null;
}

/** Returns the standard errors of an item's estimated entries by name. */
function itemErrors<TParameters>(
  item: ItemState<TParameters>,
  index: number,
  items: readonly ItemState<TParameters>[],
  covariance: Float64Array,
  size: number,
): Record<string, number> {
  let start = 0;
  for (let i = 0; i < index; i++) start += items[i]?.free.length ?? 0;
  return Object.fromEntries(
    Array.from(item.free, (j, l) => [item.names[j] ?? "", Math.sqrt(covariance[(start + l) * size + start + l] ?? 0)]),
  );
}

/** Returns the standard errors of the correlations as a matrix with a zero diagonal. */
function correlationErrors(latent: Latent, covariance: Float64Array, size: number): number[][] {
  const { dimensions, pairs } = latent;
  const first = size - pairs.length;
  const errors = Array.from({ length: dimensions }, () => new Array<number>(dimensions).fill(0));
  pairs.forEach(([i, j], p) => {
    const error = Math.sqrt(covariance[(first + p) * size + first + p] ?? 0);
    const row = errors[i];
    const column = errors[j];
    if (row !== undefined) row[j] = error;
    if (column !== undefined) column[i] = error;
  });
  return errors;
}

function matrixRows({ matrix, dimensions }: Latent): number[][] {
  return Array.from({ length: dimensions }, (_, i) =>
    Array.from(matrix.subarray(i * dimensions, (i + 1) * dimensions)),
  );
}

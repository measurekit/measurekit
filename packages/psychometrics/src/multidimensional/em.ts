import { checkPositiveInteger, DataError, describe } from "@measurekit/core";
import type { CompressedResponses } from "../irt/responses.js";
import type { MultidimensionalCalibrationOptions } from "./calibration-options.js";
import { setUp } from "./calibration-setup.js";
import { latentDistribution } from "./correlation.js";
import type { Design } from "./design.js";
import { expectationStep } from "./expectation.js";
import { defaultPoints, productGrid } from "./grid.js";
import { itemMaximization } from "./m-step.js";
import type { Settings, Solution } from "./solution.js";
import { crossProductCovariance, type Fit, louisCovariance, parameterCount } from "./standard-errors.js";

/** The most dimensions that EM calibrates: its grid grows as the number of points to the power of the dimensions. */
export const MAX_EM_DIMENSIONS = 3;

/**
 * Checks that EM can calibrate the number of dimensions.
 *
 * @throws `DataError` when it is above {@link MAX_EM_DIMENSIONS}.
 */
function checkEmDimensions(dimensions: number): void {
  if (dimensions > MAX_EM_DIMENSIONS) {
    throw new DataError(`EM calibrates up to ${MAX_EM_DIMENSIONS} dimensions, found ${dimensions}.`);
  }
}

/** Runs EM cycles of the E-step and M-step until no parameter changes by more than `tolerance` or `maxIterations`. */
function expectationMaximization<TParameters>(
  fit: Fit<TParameters>,
  tolerance: number,
  maxIterations: number,
): { readonly converged: boolean; readonly iterations: number } {
  const { items, latent, correlated, maximization, eStep } = fit;
  const current = new Float64Array(parameterCount(fit));
  const previous = new Float64Array(current.length);
  let iterations = 0;
  let converged = false;
  while (!converged && iterations < maxIterations) {
    iterations++;
    eStep.run();
    estimatesOf(fit, previous);
    for (const item of items) maximization.maximize(item);
    if (correlated) latent.maximize(eStep.nodeCounts);
    estimatesOf(fit, current);
    converged = largestChange(previous, current) <= tolerance;
  }
  return { converged, iterations };
}

/** Writes the estimated parameters, on their own scale, into `out`: the items' estimated entries, then the correlations. */
function estimatesOf<TParameters>({ items, latent, correlated }: Fit<TParameters>, out: Float64Array): void {
  let slot = 0;
  for (const item of items) for (const j of item.free) out[slot++] = item.vector[j] ?? 0;
  if (correlated) for (const [i, j] of latent.pairs) out[slot++] = latent.matrix[i * latent.dimensions + j] ?? 0;
}

function largestChange(previous: Float64Array, current: Float64Array): number {
  let largest = 0;
  for (let j = 0; j < current.length; j++)
    largest = Math.max(largest, Math.abs((current[j] ?? 0) - (previous[j] ?? 0)));
  return largest;
}

/**
 * Calibrates by EM: the grid, the E-step and M-step until convergence, the log-likelihood at the estimates and the
 * covariance of the estimates by the method of the settings.
 *
 * @throws `DataError` when there are more dimensions than EM calibrates, `points` is not an integer from 2, or the
 * grid has too many nodes.
 */
export function calibrateByEm<TParameters>(
  responses: CompressedResponses,
  options: MultidimensionalCalibrationOptions<TParameters>,
  design: Design,
  settings: Settings,
): Solution<TParameters> {
  const dimensions = design.dimensionNames.length;
  checkEmDimensions(dimensions);
  const points = settings.points ?? defaultPoints(dimensions);
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
  const { converged, iterations } = expectationMaximization(fit, settings.tolerance, settings.maxIterations);
  const logLikelihood = fit.eStep.run();
  const covariance = parameterCount(fit) > 0 ? covarianceOf(settings.standardErrors, fit) : null;
  return {
    items,
    correlations: latent.matrix,
    correlated: design.correlated,
    covariance,
    logLikelihood,
    converged,
    iterations,
  };
}

function checkPoints(points: unknown): asserts points is number {
  checkPositiveInteger(points, "The option points");
  if (points < 2) throw new DataError(`The option points must be at least 2, found ${describe(points)}.`);
}

/** Returns the covariance matrix of the estimated parameters by the method, or `null` without standard errors. */
function covarianceOf<TParameters>(method: Settings["standardErrors"], fit: Fit<TParameters>): Float64Array | null {
  if (method === "hessian") return louisCovariance(fit);
  if (method === "cross-product") return crossProductCovariance(fit);
  return null;
}

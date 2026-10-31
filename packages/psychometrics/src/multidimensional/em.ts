import { checkIntegerFrom, DataError, largestChange } from "@measurekit/core";
import type { CompressedResponses } from "../irt/responses.js";
import { type Iteration, squarem } from "../irt/squarem.js";
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

/** Runs EM, with the extrapolation SQUAREM, until no parameter changes by more than `tolerance` or `maxIterations`. */
function expectationMaximization<TParameters>(
  fit: Fit<TParameters>,
  tolerance: number,
  maxIterations: number,
): { readonly converged: boolean; readonly iterations: number } {
  return squarem(emIteration(fit), { tolerance, maxIterations });
}

/**
 * Returns EM as an iteration on the unbounded coordinates of the items' estimated entries, then the parameters of the
 * correlations, which any values keep a correlation matrix when it is positive definite.
 */
function emIteration<TParameters>(fit: Fit<TParameters>): Iteration {
  const { items, latent, correlated, maximization, eStep } = fit;
  const itemSize = items.reduce((sum, item) => sum + item.free.length, 0);
  const current = new Float64Array(parameterCount(fit));
  const previous = new Float64Array(current.length);
  return {
    step() {
      const logLikelihood = eStep.run();
      estimatesOf(fit, previous);
      for (const item of items) maximization.maximize(item);
      if (correlated) latent.maximize(eStep.nodeCounts);
      estimatesOf(fit, current);
      return { logLikelihood, change: largestChange(previous, current) };
    },
    get() {
      const state = new Float64Array(itemSize + (correlated ? latent.pairs.length : 0));
      let slot = 0;
      for (const item of items) {
        state.set(maximization.coordinates(item), slot);
        slot += item.free.length;
      }
      if (correlated) state.set(latent.parameters(), itemSize);
      return state;
    },
    set(state) {
      // Correlations that are not positive definite stay as they were: the log-likelihood at the jump decides.
      if (correlated) latent.assign(state.subarray(itemSize));
      let slot = 0;
      for (const item of items) {
        maximization.assign(item, state.subarray(slot, slot + item.free.length));
        slot += item.free.length;
      }
    },
  };
}

/** Writes the estimated parameters, on their own scale, into `out`: the items' estimated entries, then the correlations. */
function estimatesOf<TParameters>({ items, latent, correlated }: Fit<TParameters>, out: Float64Array): void {
  let slot = 0;
  for (const item of items) for (const j of item.free) out[slot++] = item.vector[j] ?? 0;
  if (correlated) for (const [i, j] of latent.pairs) out[slot++] = latent.matrix[i * latent.dimensions + j] ?? 0;
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
  checkIntegerFrom(points, "The option points", 2);
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

/** Returns the covariance matrix of the estimated parameters by the method, or `null` without standard errors. */
function covarianceOf<TParameters>(method: Settings["standardErrors"], fit: Fit<TParameters>): Float64Array | null {
  if (method === "hessian") return louisCovariance(fit);
  if (method === "cross-product") return crossProductCovariance(fit);
  return null;
}

import { checkIntegerFrom, DataError } from "@measurekit/core";
import { type CompressedResponses, checkStandardErrorSize, type Iteration, squarem } from "../irt/internal.js";
import { largestChange } from "../math/internal.js";
import type { MultidimensionalCalibrationOptions } from "./calibration-options.js";
import { setUp } from "./calibration-setup.js";
import type { Latent } from "./correlation.js";
import type { Design } from "./design.js";
import { expectationStep } from "./expectation.js";
import { defaultPoints, type Grid, productGrid } from "./grid.js";
import { population } from "./groups.js";
import { itemMaximization } from "./m-step.js";
import type { GroupMoments, Settings, Solution } from "./solution.js";
import { crossProductCovariance, type Fit, louisCovariance, parameterCount } from "./standard-errors.js";

/** The label of the group of all persons when fixed items come without the option `groups`. */
const SINGLE_GROUP = "all";

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
 * distributions of the groups, which any values keep a distribution when the correlations are positive definite.
 */
function emIteration<TParameters>(fit: Fit<TParameters>): Iteration {
  const { items, latents, maximization, eStep } = fit;
  const itemSize = items.reduce((sum, item) => sum + item.free.length, 0);
  const current = new Float64Array(parameterCount(fit));
  const previous = new Float64Array(current.length);
  return {
    step() {
      const logLikelihood = eStep.run();
      estimatesOf(fit, previous);
      for (const item of items) maximization.maximize(item);
      latents.forEach((latent, g) => {
        latent.maximize(eStep.groupCounts[g] ?? eStep.nodeCounts);
      });
      estimatesOf(fit, current);
      return { logLikelihood, change: largestChange(previous, current) };
    },
    get() {
      const state = new Float64Array(parameterCount(fit));
      let slot = 0;
      for (const item of items) {
        state.set(maximization.coordinates(item), slot);
        slot += item.free.length;
      }
      for (const latent of latents) {
        state.set(latent.parameters(), slot);
        slot += latent.size;
      }
      return state;
    },
    set(state) {
      // Correlations that are not positive definite stay as they were: the log-likelihood at the jump decides.
      let slot = itemSize;
      for (const latent of latents) {
        latent.assign(state.subarray(slot, slot + latent.size));
        slot += latent.size;
      }
      slot = 0;
      for (const item of items) {
        maximization.assign(item, state.subarray(slot, slot + item.free.length));
        slot += item.free.length;
      }
    },
  };
}

/**
 * Writes the estimated parameters, on their own scale, into `out`: the items' estimated entries, then the entries of
 * each group's distribution, the standard deviations on the logarithmic scale.
 */
function estimatesOf<TParameters>({ items, latents }: Fit<TParameters>, out: Float64Array): void {
  let slot = 0;
  for (const item of items) for (const j of item.free) out[slot++] = item.vector[j] ?? 0;
  for (const latent of latents) {
    out.set(latent.entries(), slot);
    slot += latent.size;
  }
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
  const { labels, groupOf, latents } = population(options, responses, items, design, grid);
  const fit: Fit<TParameters> = {
    items,
    grid,
    latents,
    groupOf,
    maximization: itemMaximization(items, grid),
    eStep: expectationStep(
      responses,
      items,
      grid,
      latents.map((latent) => latent.logWeights),
      groupOf,
    ),
    responses,
  };
  if (settings.standardErrors !== "none") checkStandardErrorSize(parameterCount(fit));
  const { converged, iterations } = expectationMaximization(fit, settings.tolerance, settings.maxIterations);
  const logLikelihood = fit.eStep.run();
  const covariance = parameterCount(fit) > 0 ? covarianceOf(settings.standardErrors, fit) : null;
  const [first] = latents;
  return {
    items,
    correlations: first === undefined ? new Float64Array(0) : correlationsOf(first),
    correlated: design.correlated,
    latentSize: latents.reduce((sum, latent) => sum + latent.size, 0),
    groups:
      labels === null && !items.some((item) => item.fixed)
        ? null
        : latents.map(
            (latent, g): GroupMoments => ({
              group: labels?.[g] ?? SINGLE_GROUP,
              means: latent.mean.slice(),
              covariance: latent.matrix.slice(),
            }),
          ),
    covariance,
    logLikelihood,
    converged: converged && !latents.some((latent) => runsAway(latent, grid)),
    iterations,
  };
}

/** Returns the correlation matrix of a distribution: its covariance over the product of the standard deviations. */
function correlationsOf({ matrix, standardDeviations, dimensions }: Latent): Float64Array {
  return Float64Array.from(matrix, (value, k) => {
    const i = Math.floor(k / dimensions);
    const j = k % dimensions;
    return i === j ? 1 : value / ((standardDeviations[i] ?? 1) * (standardDeviations[j] ?? 1));
  });
}

/**
 * Returns whether an estimated distribution has run to a boundary where the likelihood has no maximum: a mean outside
 * the grid, or a standard deviation below the spacing of its nodes or above their range, of a dimension that is not
 * pinned.
 */
export function runsAway({ free, mean, standardDeviations }: Latent, { nodes, points }: Grid): boolean {
  const low = nodes[0] ?? 0;
  const high = nodes[nodes.length - 1] ?? 0;
  const spacing = (high - low) / (points - 1);
  const estimated = (dimension: number): boolean => free.pinned?.includes(dimension) !== true;
  return (
    (free.means && mean.some((value, d) => estimated(d) && (value < low || value > high))) ||
    (free.variances && standardDeviations.some((value, d) => estimated(d) && (value < spacing || value > high - low)))
  );
}

/** Returns the covariance matrix of the estimated parameters by the method, or `null` without standard errors. */
function covarianceOf<TParameters>(method: Settings["standardErrors"], fit: Fit<TParameters>): Float64Array | null {
  if (method === "hessian") return louisCovariance(fit);
  if (method === "cross-product") return crossProductCovariance(fit);
  return null;
}

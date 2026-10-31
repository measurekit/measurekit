import { unboundedDerivative } from "../irt/bounds.js";
import type { CompressedResponses } from "../irt/responses.js";
import { invert, invertPositiveDefinite } from "../math/linear-algebra.js";
import { symmetrized } from "../math/matrix.js";
import { type ItemState, UNBOUNDED } from "./calibration-setup.js";
import type { Latent } from "./correlation.js";
import type { ExpectationStep } from "./expectation.js";
import type { Grid } from "./grid.js";
import type { ItemMaximization } from "./m-step.js";
import { parameterCount, posteriorScoreMoments } from "./posterior-scores.js";

export { parameterCount };

/** What a calibration's standard errors need: the fitted items, the latent distribution and the E-step. */
export interface Fit<TParameters> {
  readonly items: readonly ItemState<TParameters>[];
  readonly grid: Grid;
  readonly latent: Latent;
  /** Whether the correlations are estimated, and so part of the parameters. */
  readonly correlated: boolean;
  readonly maximization: ItemMaximization<TParameters>;
  readonly eStep: ExpectationStep;
  readonly responses: CompressedResponses;
}

/** The relative step of the central differences of the complete-data gradient, about the cube root of epsilon. */
const STEP = 1e-5;

/**
 * Returns the covariance matrix of the estimated parameters on their own scale, by rows, in the order of the items'
 * estimated entries and then the correlations; `null` when the information cannot be inverted or gives a variance that
 * is not positive.
 *
 * @remarks
 * The observed information of the marginal log-posterior by Louis' formula: the expected complete-data information
 * less the posterior variance of the complete-data scores. The first is the Hessian of the M-step's objective at the
 * counts of the last E-step, by central differences of its exact gradient on the unbounded scale of calibration; the
 * second sums over persons the variance of the scores under each person's posterior, from the marginals of the
 * posterior over the dimensions that the scores depend on. The covariance is converted to the parameters' own scale by
 * the delta method. The fit is left at its estimates, and the E-step's last run is the one the counts come from.
 *
 * @see {@link https://doi.org/10.1111/j.2517-6161.1982.tb01203.x | Louis (1982)}
 */
export function louisCovariance<TParameters>(fit: Fit<TParameters>): Float64Array | null {
  const size = parameterCount(fit);
  const { second, outer } = posteriorScoreMoments(fit, true, true);
  const information = completeInformation(fit, size);
  for (let k = 0; k < information.length; k++) {
    information[k] = (information[k] ?? 0) - ((second?.[k] ?? 0) - (outer[k] ?? 0));
  }
  return covarianceOfInformation(symmetrized(information, size), fit.items, size);
}

/**
 * Returns the covariance matrix of the estimated parameters, as {@link louisCovariance} lays it out, from the
 * cross-product information.
 *
 * @remarks
 * The information is Σ_p s_p s_pᵀ over persons, with s_p the derivative of a person's log marginal likelihood, Σ_q
 * post_p(q) ∂ ln P(x_p | θ_q) / ∂ξ, plus minus the second derivatives of the log priors, as mirt's `SE.type =
 * "crossprod"`. `null` when the information is not positive definite. Uses the posteriors of the E-step's last run.
 */
export function crossProductCovariance<TParameters>(fit: Fit<TParameters>): Float64Array | null {
  const size = parameterCount(fit);
  const { outer } = posteriorScoreMoments(fit, false, false);
  let slot = 0;
  for (const item of fit.items) {
    item.free.forEach((j, l) => {
      const prior = item.entries[l]?.prior;
      const index = (slot + l) * size + slot + l;
      if (prior !== undefined)
        outer[index] = (outer[index] ?? 0) - prior.logDensitySecondDerivative(item.vector[j] ?? 0);
    });
    slot += item.free.length;
  }
  return invertPositiveDefinite(outer, size);
}

/**
 * Returns the expected complete-data information of the estimated parameters on the unbounded scale: the Hessian of
 * the negative expected complete-data log-posterior of each item, and of the correlations, at the last E-step's counts.
 * It is block diagonal, as the items and the correlations share no parameters.
 */
function completeInformation<TParameters>(fit: Fit<TParameters>, size: number): Float64Array {
  const { items, latent, correlated, maximization, eStep } = fit;
  const information = new Float64Array(size * size);
  let slot = 0;
  for (const item of items) {
    const eta = maximization.coordinates(item);
    const length = item.free.length;
    const up = new Float64Array(length);
    const down = new Float64Array(length);
    const moved = Float64Array.from(eta);
    for (let l = 0; l < length; l++) {
      const step = STEP * Math.max(1, Math.abs(eta[l] ?? 0));
      moved[l] = (eta[l] ?? 0) + step;
      maximization.objective(item, moved, up);
      moved[l] = (eta[l] ?? 0) - step;
      maximization.objective(item, moved, down);
      moved[l] = eta[l] ?? 0;
      for (let m = 0; m < length; m++)
        information[(slot + m) * size + slot + l] = ((up[m] ?? 0) - (down[m] ?? 0)) / (2 * step);
    }
    maximization.assign(item, eta);
    slot += length;
  }
  if (correlated && latent.pairs.length > 0) latentInformation(latent, eStep.nodeCounts, information, size, slot);
  return information;
}

/** Writes the Hessian of the negative expected log-likelihood of the correlations into the block at `slot`. */
function latentInformation(
  latent: Latent,
  nodeCounts: Float64Array,
  information: Float64Array,
  size: number,
  slot: number,
): void {
  const pairs = latent.pairs.length;
  const start = latent.matrix.slice();
  const up = new Float64Array(pairs);
  const down = new Float64Array(pairs);
  for (let a = 0; a < pairs; a++) {
    const [i, j] = latent.pairs[a] ?? [0, 0];
    const at = i * latent.dimensions + j;
    const step = STEP * Math.max(1, Math.abs(start[at] ?? 0));
    latent.set(shifted(start, latent.dimensions, i, j, step));
    latent.gradient(nodeCounts, up);
    latent.set(shifted(start, latent.dimensions, i, j, -step));
    latent.gradient(nodeCounts, down);
    for (let b = 0; b < pairs; b++)
      information[(slot + b) * size + slot + a] = ((up[b] ?? 0) - (down[b] ?? 0)) / (2 * step);
  }
  latent.set(start);
}

/** Returns the correlation matrix with the correlation of dimensions i and j moved by `step`. */
function shifted(matrix: Float64Array, dimensions: number, i: number, j: number, step: number): Float64Array {
  const next = matrix.slice();
  next[i * dimensions + j] = (next[i * dimensions + j] ?? 0) + step;
  next[j * dimensions + i] = (next[j * dimensions + i] ?? 0) + step;
  return next;
}

/**
 * Returns the covariance of the parameters on their own scale from the information on the unbounded scale of the items'
 * entries: its inverse scaled by dx/dη; `null` when the information cannot be inverted or a variance is not positive.
 */
export function covarianceOfInformation<TParameters>(
  information: Float64Array,
  items: readonly ItemState<TParameters>[],
  size: number,
): Float64Array | null {
  const inverse = invert(information, size);
  if (inverse === null) return null;
  const scales = new Float64Array(size).fill(1);
  let slot = 0;
  for (const item of items) {
    item.free.forEach((j, l) => {
      const { lower, upper } = item.entries[l] ?? UNBOUNDED;
      scales[slot + l] = unboundedDerivative(item.vector[j] ?? 0, lower, upper);
    });
    slot += item.free.length;
  }
  for (let a = 0; a < size; a++) {
    if (!((inverse[a * size + a] ?? 0) > 0)) return null;
    for (let b = 0; b < size; b++)
      inverse[a * size + b] = (inverse[a * size + b] ?? 0) * (scales[a] ?? 1) * (scales[b] ?? 1);
  }
  return inverse;
}

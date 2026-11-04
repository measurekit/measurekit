import {
  type CompressedResponses,
  type Iteration,
  type Maximization,
  type Setup,
  squarem,
  unboundedCoordinates,
} from "../irt/internal.js";
import { largestChange } from "../math/internal.js";
import type { Latent } from "./correlation.js";
import { runsAway } from "./em.js";
import type { Grid } from "./grid.js";
import type { OccasionExpectation } from "./longitudinal-expectation.js";

/** What a longitudinal calibration estimates: the items, the distribution of the occasions' traits and the E-step. */
export interface LongitudinalFit<TParameters> {
  /** The items, their free parameters and their bounds, as the calibration of one trait sets them up. */
  readonly setup: Setup<TParameters>;
  /** The M-step of the items. */
  readonly maximization: Maximization<TParameters>;
  /** The joint distribution of the occasions' traits on the grid, the first occasion pinned to N(0, 1). */
  readonly latent: Latent;
  readonly eStep: OccasionExpectation;
  readonly grid: Grid;
  /** The nodes of one axis of the grid, at which the items' probabilities enter. */
  readonly axis: Float64Array;
  /** The responses of each occasion. */
  readonly occasions: readonly CompressedResponses[];
}

/**
 * Runs EM with the extrapolation SQUAREM until no estimated parameter changes by more than `tolerance` in a step or
 * `maxIterations` have run. Converged also needs the distribution to stay inside the grid.
 */
export function expectationMaximization<TParameters>(
  fit: LongitudinalFit<TParameters>,
  tolerance: number,
  maxIterations: number,
): { readonly converged: boolean; readonly iterations: number } {
  const { converged, iterations } = squarem(emIteration(fit), { tolerance, maxIterations });
  return { converged: converged && !runsAway(fit.latent, fit.grid), iterations };
}

/**
 * Returns EM as an iteration on the items' free parameters on their unbounded scale, then the parameters of the
 * distribution, which any values keep a distribution when the correlations are positive definite.
 */
function emIteration<TParameters>({ setup, maximization, latent, eStep }: LongitudinalFit<TParameters>): Iteration {
  const { values, free } = setup;
  const previous = new Float64Array(values.length + latent.size);
  const current = new Float64Array(previous.length);
  /** Writes the free parameters of the items on their own scale and the entries of the distribution into `out`. */
  const estimates = (out: Float64Array): void => {
    out.set(values);
    out.set(latent.entries(), values.length);
  };
  return {
    step() {
      const logLikelihood = eStep.run();
      estimates(previous);
      for (const block of maximization.blocks) maximization.maximize(block);
      latent.maximize(eStep.nodeCounts);
      estimates(current);
      return { logLikelihood, change: largestChange(previous, current) };
    },
    get() {
      const state = new Float64Array(values.length + latent.size);
      state.set(unboundedCoordinates(values, free));
      state.set(latent.parameters(), values.length);
      return state;
    },
    set(state) {
      // Correlations that are not positive definite stay as they were: the log-likelihood at the jump decides.
      latent.assign(state.subarray(values.length));
      maximization.assign(maximization.all, state.subarray(0, values.length));
    },
  };
}

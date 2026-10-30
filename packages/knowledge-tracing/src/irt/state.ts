import { DataError } from "@measurekit/core";
import { checkLearnerState } from "../data/learner.js";
import { indexOfMax } from "../math/extremes.js";

/**
 * A learner's state in an IRT model: the log-density of the posterior of θ at each point of the model's grid, up to a
 * constant, with its first and second derivatives with respect to θ, which give the posterior mode between grid points.
 * Plain data, which JSON serializes; its size depends only on the grid.
 */
export interface IrtState {
  /** The learner's id. */
  readonly learner: string;
  /** ln of the posterior density at each grid point, less its maximum. */
  readonly logDensity: readonly number[];
  /** Its first derivative at each grid point. */
  readonly slope: readonly number[];
  /** Its second derivative at each grid point. */
  readonly curvature: readonly number[];
}

/** Checks a state, which may come from storage, against the number of grid points. */
export function checkState(state: IrtState, points: number): void {
  const learner = checkLearnerState(state, "IRT");
  for (const name of ["logDensity", "slope", "curvature"] as const) {
    const values = state[name];
    if (!(Array.isArray(values) && values.length === points && values.every(Number.isFinite))) {
      throw new DataError(
        `The IRT state of learner "${learner}" needs ${name}, an array of ${points} finite numbers, one per grid point.`,
      );
    }
  }
  if (state.logDensity[indexOfMax(state.logDensity)] !== 0) {
    throw new DataError(
      `The IRT state of learner "${learner}" needs logDensity less its maximum, so that its largest value is 0.`,
    );
  }
}

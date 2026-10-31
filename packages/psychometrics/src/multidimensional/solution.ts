import type { ItemState } from "./calibration-setup.js";

/** What a method of estimation hands back: the items at their estimates and the fit of the whole calibration. */
export interface Solution<TParameters> {
  /** The items, with their parameters at the estimates. */
  readonly items: readonly ItemState<TParameters>[];
  /** The correlations between the dimensions, by rows. */
  readonly correlations: Float64Array;
  /** Whether the correlations are estimated, and so part of the covariance. */
  readonly correlated: boolean;
  /**
   * The covariance of the estimated parameters, by rows, in the order of the items' estimated entries and then the
   * correlations; `null` without standard errors.
   */
  readonly covariance: Float64Array | null;
  readonly logLikelihood: number;
  readonly converged: boolean;
  readonly iterations: number;
}

/** Settings of estimation that every method takes, after their defaults are filled in. */
export interface Settings {
  readonly tolerance: number;
  readonly maxIterations: number;
  readonly seed: number;
  readonly standardErrors: "none" | "hessian" | "cross-product";
  readonly points: number | undefined;
}

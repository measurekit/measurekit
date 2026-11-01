import type { ItemState } from "./calibration-setup.js";

/** The estimated distribution of one group of persons. */
export interface GroupMoments {
  /** The group's label. */
  readonly group: string;
  /** The mean of each dimension. */
  readonly means: Float64Array;
  /** The covariance matrix by rows. */
  readonly covariance: Float64Array;
}

/** What a method of estimation hands back: the items at their estimates and the fit of the whole calibration. */
export interface Solution<TParameters> {
  /** The items, with their parameters at the estimates. */
  readonly items: readonly ItemState<TParameters>[];
  /** The correlations between the dimensions in the first group, by rows. */
  readonly correlations: Float64Array;
  /** Whether the correlations are estimated, and so part of the covariance. */
  readonly correlated: boolean;
  /** The number of estimated entries of the distributions of theta, the correlations of the first group first. */
  readonly latentSize: number;
  /** The distribution of each group; `null` unless groups or fixed items were given. */
  readonly groups: readonly GroupMoments[] | null;
  /**
   * The covariance of the estimated parameters, by rows, in the order of the items' estimated entries and then the
   * entries of the distributions; `null` without standard errors.
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

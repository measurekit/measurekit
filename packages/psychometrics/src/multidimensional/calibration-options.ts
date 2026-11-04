import type { FieldNames } from "@measurekit/core";
import type { ParameterPrior, StandardErrorMethod } from "../irt/internal.js";
import type { LoadingPattern, MultidimensionalModel } from "./model.js";

/** How a calibration estimates: `"em"`, by quadrature over a grid, or `"mhrm"`, by Metropolis-Hastings Robbins-Monro. */
export type CalibrationMethod = "em" | "mhrm";

/** Options of {@link calibrateMultidimensional}. */
export interface MultidimensionalCalibrationOptions<TParameters> {
  /**
   * The model of every item, or one model per item, in item order. A list of models with different parameters needs
   * its type written out, such as `MultidimensionalModel<MultidimensionalLogisticParameters |
   * MultidimensionalGradedParameters>[]`.
   */
  readonly model: MultidimensionalModel<TParameters> | readonly MultidimensionalModel<TParameters>[];
  /**
   * How to estimate: `"em"`, which integrates theta by quadrature over a grid and calibrates up to 3 dimensions, or
   * `"mhrm"`, the Metropolis-Hastings Robbins-Monro algorithm, which draws theta and has no limit of dimensions.
   *
   * @defaultValue `"em"` up to 3 dimensions, `"mhrm"` above
   */
  readonly method?: CalibrationMethod;
  /**
   * The number of dimensions of an exploratory model, a positive integer, at most 3 for EM: every item loads on every
   * dimension. The slopes of the first D - 1 items on later dimensions are fixed at 0 (item j loads on the first j
   * dimensions), which fixes the rotation, and the dimensions are uncorrelated. One of `dimensions` and `loadings` is
   * required.
   *
   * @defaultValue none
   */
  readonly dimensions?: number;
  /**
   * The dimensions each item loads on, in a confirmatory model: the dimension names by item id, such as `{ q1:
   * ["verbal"], q2: ["verbal", "quantitative"] }`, every item of the data included. The dimensions, at most 3 for EM,
   * are named in order of first appearance among the items in item order; the slopes of the others are fixed at 0. One
   * of `dimensions` and `loadings` is required.
   *
   * @defaultValue none
   */
  readonly loadings?: LoadingPattern;
  /**
   * Whether a confirmatory model estimates the correlations between dimensions; their variances are 1 and their means
   * 0. Ignored by an exploratory model, whose dimensions are uncorrelated.
   *
   * @defaultValue `true`
   */
  readonly correlated?: boolean;
  /**
   * Each person's group, aligned with the persons (the rows of a matrix, the `personIds` of records), for a calibration
   * of several groups on one scale: every item has the same parameters in every group, and each group has its own means
   * and covariance. The first group, in order of first appearance, has means 0 and variances 1 unless `fixed` sets the
   * scale. Its correlations, and every group's, are estimated when `correlated`. EM only.
   *
   * @defaultValue one group
   */
  readonly groups?: readonly string[];
  /**
   * Parameters of anchor items by item id: they stay fixed, and the other items are estimated on their scale. The
   * anchors fix the scale, so the means and the covariance of every group, the first included, are estimated; every
   * dimension needs an anchor that loads on it in each set of groups that share items. EM only.
   *
   * @defaultValue none: every item is estimated
   */
  readonly fixed?: Readonly<Record<string, TParameters>>;
  /**
   * Priors by parameter name: `"a"` for every slope, `"intercept"` for every intercept, and `"c"`, applied to that
   * parameter of every item whose model estimates it. Estimates with priors are posterior modes.
   *
   * @defaultValue none
   */
  readonly priors?: Readonly<Record<string, ParameterPrior>>;
  /**
   * The number of nodes per dimension of the grid over the dimensions, at least 2, from -6 to 6 standard deviations.
   * EM only: MH-RM has no grid, and rejects the option.
   *
   * @defaultValue `61` for one dimension, `31` for two and `15` for three
   */
  readonly points?: number;
  /**
   * The method of the standard errors. `"hessian"` is Louis' formula for the observed information: exact, from the
   * posterior over the grid, for EM, and from 400 cycles of draws at the estimates for MH-RM, which makes it stochastic;
   * `"cross-product"` is the sum over persons of the outer products of their scores, for EM only.
   *
   * @defaultValue `"none"`
   */
  readonly standardErrors?: StandardErrorMethod;
  /**
   * The seed of the draws of MH-RM, of the persons' thetas and of the importance sampling of the log-likelihood. A seed
   * gives the same calibration. EM ignores it.
   *
   * @defaultValue `0`
   */
  readonly seed?: number;
  /**
   * The calibration has converged when no estimated parameter changes by more than this: in an EM iteration, or, for
   * MH-RM, in three successive cycles of the last stage.
   *
   * @defaultValue `1e-4` for EM, `1e-3` for MH-RM
   */
  readonly tolerance?: number;
  /**
   * The most iterations to run: EM iterations, or the cycles of the last stage of MH-RM, after the 250 of its first two
   * stages.
   *
   * @defaultValue `500` for EM, `2000` for MH-RM
   */
  readonly maxIterations?: number;
}

export const MULTIDIMENSIONAL_CALIBRATION_OPTIONS: FieldNames<MultidimensionalCalibrationOptions<unknown>> = {
  model: true,
  method: true,
  dimensions: true,
  loadings: true,
  correlated: true,
  groups: true,
  fixed: true,
  priors: true,
  points: true,
  standardErrors: true,
  seed: true,
  tolerance: true,
  maxIterations: true,
};

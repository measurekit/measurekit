import type { StandardErrorMethod } from "../irt/calibration-options.js";
import type { ParameterPrior } from "../irt/prior.js";
import type { LoadingPattern, MultidimensionalModel } from "./model.js";

/** Options of {@link calibrateMultidimensional}. */
export interface MultidimensionalCalibrationOptions<TParameters> {
  /**
   * The model of every item, or one model per item, in item order. A list of models with different parameters needs
   * its type written out, such as `MultidimensionalModel<MultidimensionalLogisticParameters |
   * MultidimensionalGradedParameters>[]`.
   */
  readonly model: MultidimensionalModel<TParameters> | readonly MultidimensionalModel<TParameters>[];
  /**
   * The number of dimensions of an exploratory model, from 1 to 3: every item loads on every dimension. The slopes of
   * the first D - 1 items on later dimensions are fixed at 0 (item j loads on the first j dimensions), which fixes the
   * rotation, and the dimensions are uncorrelated. One of `dimensions` and `loadings` is required.
   *
   * @defaultValue none
   */
  readonly dimensions?: number;
  /**
   * The dimensions each item loads on, in a confirmatory model: the dimension names by item id, such as `{ q1:
   * ["verbal"], q2: ["verbal", "quantitative"] }`, every item of the data included. The dimensions, up to 3, are named
   * in order of first appearance among the items in item order; the slopes of the others are fixed at 0. One of
   * `dimensions` and `loadings` is required.
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
   * Priors by parameter name: `"a"` for every slope, `"intercept"` for every intercept, and `"c"`, applied to that
   * parameter of every item whose model estimates it. Estimates with priors are posterior modes.
   *
   * @defaultValue none
   */
  readonly priors?: Readonly<Record<string, ParameterPrior>>;
  /**
   * The number of nodes per dimension of the grid over the dimensions, at least 2, from -6 to 6 standard deviations.
   *
   * @defaultValue `61` for one dimension, `31` for two and `15` for three
   */
  readonly points?: number;
  /**
   * The method of the standard errors. `"hessian"` runs two E-steps per estimated parameter; `"cross-product"` one pass
   * over the persons.
   *
   * @defaultValue `"none"`
   */
  readonly standardErrors?: StandardErrorMethod;
  /**
   * EM has converged when no estimated parameter changes by more than this in an iteration.
   *
   * @defaultValue `1e-4`
   */
  readonly tolerance?: number;
  /**
   * The most EM iterations to run.
   *
   * @defaultValue `500`
   */
  readonly maxIterations?: number;
}

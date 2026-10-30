import type { LatentDistribution } from "./latent.js";
import type { IrtModel } from "./model.js";
import type { ParameterPrior } from "./prior.js";
import type { QuadratureOptions } from "./quadrature.js";

/**
 * How calibration computes standard errors: `"none"`; `"hessian"`, the observed information, the Hessian of the
 * log-posterior by central differences of its exact gradient, as mirt's `SE.type = "Richardson"`; `"cross-product"`,
 * the sum over persons of the outer products of their score vectors, as mirt's `SE.type = "crossprod"`.
 */
export type StandardErrorMethod = "none" | "hessian" | "cross-product";

/** Options of {@link calibrate}. */
export interface CalibrationOptions<TParameters> {
  /**
   * The model of every item, or one model per item, in item order. A list of models with different parameters needs
   * its type written out, such as `IrtModel<LogisticParameters | ThresholdParameters>[]`.
   */
  readonly model: IrtModel<TParameters> | readonly IrtModel<TParameters>[];
  /** Parameters of anchor items by item id: they stay fixed, and the other items are estimated on their scale. */
  readonly fixed?: Readonly<Record<string, TParameters>>;
  /**
   * Priors by parameter name, such as `{ c: betaPrior(5, 17) }`, applied to that parameter of every estimated item
   * whose model has it.
   */
  readonly priors?: Readonly<Record<string, ParameterPrior>>;
  /**
   * The quadrature rule over theta.
   *
   * @defaultValue 61 rectangular points
   */
  readonly quadrature?: QuadratureOptions;
  /**
   * Each person's group, for a calibration of several groups on one scale: every item has the same parameters in every
   * group, and each group has its own distribution of theta.
   *
   * @defaultValue one group
   */
  readonly groups?: readonly string[];
  /**
   * The distribution of theta: one for every group, or one by group label, the groups not listed taking the standard
   * normal. A distribution other than the standard normal needs the rectangular rule, and fixed items or a standard
   * normal group to fix the scale.
   *
   * @defaultValue `"standard-normal"`
   */
  readonly latent?: LatentDistribution | Readonly<Record<string, LatentDistribution>>;
  /**
   * The method of the standard errors. `"hessian"` runs two E-steps per estimated parameter; `"cross-product"` one pass
   * over the persons, so it suits large pools.
   *
   * @defaultValue `"none"`
   */
  readonly standardErrors?: StandardErrorMethod;
  /**
   * Testlets by id, each the ids of its items, such as the items of a reading passage, for the testlet response model:
   * an item of a testlet answers at θ + γ, with γ ~ N(0, σ²) per person and testlet. Testlets need one group with the
   * standard normal, the rectangular rule, and standard errors `"none"` or `"hessian"`.
   */
  readonly testlets?: Readonly<Record<string, readonly string[]>>;
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

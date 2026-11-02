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
   * its type written out, such as `IrtModel<LogisticItem | ThresholdItem>[]`.
   */
  readonly model: IrtModel<TParameters> | readonly IrtModel<TParameters>[];
  /**
   * Parameters of anchor items by item id: they stay fixed, and the other items are estimated on their scale.
   *
   * @defaultValue none: every item is estimated
   */
  readonly fixed?: Readonly<Record<string, TParameters>>;
  /**
   * Priors by parameter name, such as `{ c: betaPrior(5, 17) }`, applied to that parameter of every estimated item
   * whose model has it.
   *
   * @defaultValue none
   */
  readonly priors?: Readonly<Record<string, ParameterPrior>>;
  /**
   * Priors by item id, then parameter name, such as `{ new1: { b: normalPrior(0.4, 0.8) } }`, for the items whose
   * parameters have information beyond their responses, such as a regression on their features. An item's prior of a
   * parameter replaces the prior of `priors` for that parameter. The items must be estimated, and the parameters must
   * be ones its model does not share among items.
   *
   * @defaultValue none
   */
  readonly itemPriors?: Readonly<Record<string, Readonly<Record<string, ParameterPrior>>>>;
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
   * Each person's covariates by name, aligned with the persons, for a latent regression: theta ~ N(xᵀβ, σ²) with no
   * intercept, so a covariate with a mean other than 0 moves the scale and is better centered. The coefficients β, and σ
   * when fixed items fix the scale, are estimated with the items; otherwise the items' slopes carry the scale and σ
   * is 1. Covariates replace the distribution of theta, so they need one group, no `latent` other than the standard
   * normal, the rectangular rule, no testlets, and `standardErrors` `"none"` or `"hessian"`.
   *
   * @defaultValue none: theta follows `latent`
   */
  readonly covariates?: Readonly<Record<string, readonly number[]>>;
  /**
   * The method of the standard errors. `"hessian"` runs two E-steps per estimated parameter; `"cross-product"` one pass
   * over the persons, so it suits large pools.
   *
   * @defaultValue `"none"`
   */
  readonly standardErrors?: StandardErrorMethod;
  /**
   * Testlets by id, each the ids of its items, such as the items of a reading passage, for the testlet response model:
   * an item of a testlet answers at θ + γ, with γ ~ N(0, σ²) per person and testlet, and one σ per testlet in all groups.
   * Testlets need the rectangular rule, a standard normal group or fixed items to fix the scale, and standard errors
   * `"none"` or `"hessian"`.
   *
   * @defaultValue none: the items answer independently given theta
   */
  readonly testlets?: Readonly<Record<string, readonly string[]>>;
  /**
   * Each person's survey weight, above 0, aligned with the persons. The weights are rescaled to sum to the persons with
   * a response, as mirt asks of `survey.weights`, so the log-likelihood, the BIC and the standard errors keep the
   * sample's size. They are not design-based: strata and clusters stay with the caller.
   *
   * @defaultValue 1 for every person
   */
  readonly personWeights?: readonly number[];
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

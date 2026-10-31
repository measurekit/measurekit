import {
  checkArray,
  checkBoolean,
  checkOptions,
  checkStopping,
  DataError,
  describe,
  isNonEmptyString,
  isRecord,
} from "@measurekit/core";
import type { CalibratedItem } from "../irt/calibration.js";
import { checkItem } from "../irt/items.js";
import { checkFullRank, designMatrix, type FeatureEffect, type ItemFeatures } from "./features.js";
import { remlRegression } from "./reml.js";

/** Options of {@link itemFeatureRegression}. */
export interface ItemFeatureRegressionOptions {
  /**
   * The name of the item parameter regressed on the features, such as `"b"` or `"a"`, as in the items' `standardErrors`.
   * The slope `a` is regressed as its logarithm, since it is positive; every other parameter on its own scale.
   *
   * @defaultValue `"b"`
   */
  readonly parameter?: string;
  /**
   * Whether the design has an intercept.
   *
   * @defaultValue `true`
   */
  readonly intercept?: boolean;
  /**
   * The Fisher scoring has converged when the variance of the residual changes by less than this in a step.
   *
   * @defaultValue `1e-10`
   */
  readonly tolerance?: number;
  /**
   * The most Fisher scoring steps.
   *
   * @defaultValue `100`
   */
  readonly maxIterations?: number;
}

/** The regression of an item parameter on the items' features. */
export interface ItemFeatureRegression {
  /** The regressed parameter's name. */
  readonly parameter: string;
  /** `"log"` for the slope `a`, regressed as its logarithm; `"identity"` for any other parameter. */
  readonly scale: "identity" | "log";
  /** The ids of the items in the regression, in item order. */
  readonly itemIds: readonly string[];
  /** The effects: the intercept, if any, then the features in order of their first appearance. */
  readonly effects: readonly FeatureEffect[];
  /** The covariance matrix of the estimates of the effects, by rows, in the order of `effects`. */
  readonly covariance: readonly (readonly number[])[];
  /** The standard deviation τ of the residual that the features leave in the parameter, from item to item. */
  readonly residualSd: number;
  /** The restricted log-likelihood at the estimates. */
  readonly logLikelihood: number;
  /** Whether the Fisher scoring reached `tolerance` within `maxIterations`. */
  readonly converged: boolean;
  /** The Fisher scoring steps taken. */
  readonly iterations: number;
}

const DEFAULT_PARAMETER = "b";
const DEFAULT_TOLERANCE = 1e-10;
const DEFAULT_MAX_ITERATIONS = 100;
/** The parameter regressed as its logarithm. */
const SLOPE = "a";

/**
 * Regresses an item parameter of a calibration on the features of the items, by the random-item model estimated in two
 * stages.
 *
 * @remarks
 * The calibration's estimate of the parameter for item i is y_i with sampling variance v_i, the square of its standard
 * error. The model is y_i = x_iᵀη + u_i + e_i with u_i ~ N(0, τ²), the variation between items that the features do
 * not explain, and e_i ~ N(0, v_i). η and τ² come from restricted maximum likelihood (REML) with the covariance of η̂
 * (XᵀWX)⁻¹, W = diag(1 / (v_i + τ²)); τ² is held at 0 where the features explain the variation. The slope `a` is
 * regressed as ln a, with the variance (SE / a)² by the delta method. {@link predictItemParameter} turns the regression
 * into the prior of a new item.
 *
 * Items left out of the regression: those without standard errors, such as anchors fixed in the calibration, and
 * those whose model has no parameter of that name. `itemIds` lists the others.
 *
 * Valid input: items of a calibration with `standardErrors`; features that are finite numbers; more items than
 * effects; a parameter that varies by item, not one the items of a model share, such as the `a` of the 1PL; for the
 * slope, positive estimates.
 *
 * @example
 * ```ts
 * import { itemFeatureRegression } from "@measurekit/psychometrics/explanatory";
 * import { twoParameterLogistic } from "@measurekit/psychometrics/irt";
 *
 * const item = (id: string, b: number, se: number) => ({
 *   id,
 *   model: twoParameterLogistic,
 *   parameters: { a: 1, b, c: 0, d: 1 },
 *   standardErrors: { a: 0.1, b: se },
 * });
 * const items = [item("q1", -1.1, 0.12), item("q2", -0.4, 0.1), item("q3", 0.2, 0.11), item("q4", 0.9, 0.1), item("q5", 1.6, 0.13)];
 * const features = { q1: { steps: 1 }, q2: { steps: 2 }, q3: { steps: 3 }, q4: { steps: 4 }, q5: { steps: 5 } };
 * const regression = itemFeatureRegression(items, features);
 * regression.effects; // [{ name: "(intercept)", estimate: -1.76, ... }, { name: "steps", estimate: 0.67, ... }]
 * ```
 *
 * @throws `DataError` when the items, features or options are invalid, no item has the parameter or standard errors
 * for it, the parameter is shared by the items of its model, a standard error is not positive, a slope is not
 * positive, there are not more items than effects, or an effect is a linear combination of those before it.
 * @see {@link https://doi.org/10.1007/978-1-4757-3990-9_6 | Janssen, Schepers and Peres (2004)}
 * @see {@link https://doi.org/10.1007/s11336-008-9092-x | De Boeck (2008)}
 * @see {@link https://doi.org/10.3102/10769986030003261 | Viechtbauer (2005)}
 */
export function itemFeatureRegression<TParameters>(
  items: readonly CalibratedItem<TParameters>[],
  features: ItemFeatures,
  options: ItemFeatureRegressionOptions = {},
): ItemFeatureRegression {
  checkArray(items, "The items must be an array of calibrated items");
  checkOptions(options);
  const {
    parameter = DEFAULT_PARAMETER,
    intercept = true,
    tolerance = DEFAULT_TOLERANCE,
    maxIterations = DEFAULT_MAX_ITERATIONS,
  } = options;
  if (!isNonEmptyString(parameter)) {
    throw new DataError(`The option parameter must be the name of a parameter, found ${describe(parameter)}.`);
  }
  checkBoolean(intercept, "The option intercept");
  checkStopping(tolerance, maxIterations);
  const scale = parameter === SLOPE ? "log" : "identity";
  const observed = observations(items, parameter, scale);
  const design = designMatrix(
    observed.map(({ id }) => id),
    features,
    intercept,
  );
  const { names, matrix } = design;
  const n = observed.length;
  if (n <= names.length) {
    throw new DataError(
      `Regressing "${parameter}" needs more items with standard errors than effects, found ${n} items for ${names.length} effects.`,
    );
  }
  checkFullRank(design);
  const fit = remlRegression(
    matrix,
    n,
    names.length,
    Float64Array.from(observed, ({ y }) => y),
    Float64Array.from(observed, ({ v }) => v),
    tolerance,
    maxIterations,
  );
  const size = names.length;
  return {
    parameter,
    scale,
    itemIds: observed.map(({ id }) => id),
    effects: names.map((name, j) => ({
      name,
      estimate: fit.coefficients[j] ?? 0,
      standardError: Math.sqrt(fit.covariance[j * size + j] ?? 0),
    })),
    covariance: Array.from({ length: size }, (_, j) => Array.from(fit.covariance.subarray(j * size, (j + 1) * size))),
    residualSd: Math.sqrt(fit.residualVariance),
    logLikelihood: fit.logLikelihood,
    converged: fit.converged,
    iterations: fit.iterations,
  };
}

/** One item's estimate of the regressed parameter on the regression's scale, with its sampling variance. */
interface Observation {
  readonly id: string;
  readonly y: number;
  readonly v: number;
}

/**
 * Returns the estimates and sampling variances of the parameter for the items that have both, on the scale of the
 * regression.
 *
 * @throws `DataError` when no item has the parameter, the parameter is shared by the items of its model, ids repeat, a
 * standard error is not positive, or a slope is not positive.
 */
function observations<TParameters>(
  items: readonly CalibratedItem<TParameters>[],
  parameter: string,
  scale: "identity" | "log",
): Observation[] {
  const ids = new Set<string>();
  const results = items.map((item, index) => {
    if (!isRecord(item) || !isNonEmptyString(item.id)) {
      throw new DataError(`Item ${index + 1} needs an id, found ${describe(isRecord(item) ? item.id : item)}.`);
    }
    if (ids.has(item.id)) throw new DataError(`Item ids must be unique; "${item.id}" appears twice.`);
    ids.add(item.id);
    checkItem(item, `Item "${item.id}"`);
    return observe(item, parameter, scale);
  });
  if (results.every((result) => result === undefined)) throw new DataError(`No item has the parameter "${parameter}".`);
  return results.filter((result) => result !== undefined && result !== null);
}

/**
 * Returns an item's observation of the parameter, `null` for an item without standard errors, and `undefined` for an
 * item whose model has no parameter of that name.
 */
function observe<TParameters>(
  item: CalibratedItem<TParameters>,
  parameter: string,
  scale: "identity" | "log",
): Observation | null | undefined {
  const { id } = item;
  const specifications = item.model.parameters(item.model.categoryCount(item.parameters));
  const position = specifications.findIndex(({ name }) => name === parameter);
  if (position < 0) return undefined;
  if (specifications[position]?.shared === true) {
    throw new DataError(
      `The parameter "${parameter}" of item "${id}" is shared by the items of its ${item.model.name} model, so it does not vary by item.`,
    );
  }
  const errors: unknown = item.standardErrors;
  if (!isRecord(errors)) return null;
  const standardError = errors[parameter];
  if (!(typeof standardError === "number" && standardError > 0 && Number.isFinite(standardError))) {
    throw new DataError(
      `The standard error of "${parameter}" of item "${id}" must be a positive number, found ${describe(standardError)}.`,
    );
  }
  const estimate = item.model.toVector(item.parameters)[position] ?? 0;
  if (scale === "identity") return { id, y: estimate, v: standardError ** 2 };
  if (!(estimate > 0)) {
    throw new DataError(
      `The parameter "${parameter}" of item "${id}" must be positive to be regressed as its logarithm, found ${estimate}.`,
    );
  }
  return { id, y: Math.log(estimate), v: (standardError / estimate) ** 2 };
}

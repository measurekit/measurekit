import {
  checkBoolean,
  checkChoice,
  checkOptions,
  checkStopping,
  chiSquareUpperTail,
  DataError,
  type FieldNames,
  type LogisticItem,
  type ThresholdItem,
} from "@measurekit/core";
import type { ResponseMatrix, ResponseRecords } from "../data/internal.js";
import {
  type CompressedResponses,
  categoryTallies,
  checkCategories,
  checkLinearRaschSize,
  compressResponses,
  fitLinearRasch,
  type LinearRaschFit,
  type LocationRow,
  partialCredit,
  type QuadratureOptions,
  quadratureRule,
  type TestItem,
  twoParameterLogistic,
} from "../irt/internal.js";
import { checkFullRank, type Design, designMatrix, type FeatureEffect, type ItemFeatures } from "./features.js";

/** How the standard errors of the effects of an LLTM are computed. */
export type LltmStandardErrors = "none" | "hessian";

/**
 * The responses an LLTM explains: `"dichotomous"`, items scored 0 or 1; `"partial-credit"`, items of any number of
 * categories, each with thresholds of its own; `"rating-scale"`, items that share one number of categories and one set
 * of thresholds.
 */
export type LltmModel = "dichotomous" | "partial-credit" | "rating-scale";

/** Options of {@link lltm}. */
export interface LltmOptions {
  /**
   * The model of the responses. With `"partial-credit"` and `"rating-scale"` the features explain the location of each
   * item, the mean of its step difficulties, and the thresholds around it are estimated.
   *
   * @defaultValue `"dichotomous"`
   */
  readonly model?: LltmModel;
  /**
   * Whether the difficulties have an intercept: the difficulty of an item whose features are all 0.
   *
   * @defaultValue `true`
   */
  readonly intercept?: boolean;
  /**
   * The quadrature rule over theta.
   *
   * @defaultValue 61 rectangular points
   */
  readonly quadrature?: QuadratureOptions;
  /**
   * `"hessian"` computes the standard errors from the Hessian of the marginal log-likelihood, by central differences
   * of its exact gradient; `"none"` leaves them out.
   *
   * @defaultValue `"hessian"`
   */
  readonly standardErrors?: LltmStandardErrors;
  /**
   * Estimation has converged when no parameter changes by more than this in an EM iteration.
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

const LLTM_OPTIONS: FieldNames<LltmOptions> = {
  model: true,
  intercept: true,
  quadrature: true,
  standardErrors: true,
  tolerance: true,
  maxIterations: true,
};

/** The likelihood-ratio test of an LLTM against the Rasch model that gives every item a difficulty of its own. */
export interface RaschComparison {
  /** The marginal log-likelihood of the Rasch model. */
  readonly raschLogLikelihood: number;
  /** 2 (ℓ_Rasch - ℓ_LLTM), the likelihood-ratio statistic: how much the features' difficulties lose. */
  readonly chiSquare: number;
  /** The number of items less the number of effects: the difficulties that the features do not use. */
  readonly degreesOfFreedom: number;
  /** The probability of a chi-square at least as large if the features explain the difficulties. */
  readonly significance: number;
}

/**
 * A linear logistic test model estimated by marginal maximum likelihood.
 *
 * @typeParam TParameters - The parameters of its items: those of the 2PL for the dichotomous model, and of the partial
 * credit model for the others.
 */
export interface Lltm<TParameters = LogisticItem> {
  /** The effects on difficulty: the intercept, if any, then the features in order of their first appearance. */
  readonly effects: readonly FeatureEffect[];
  /** The standard deviation of theta, whose mean is 0 on the scale of the effects. */
  readonly sd: number;
  /**
   * The items with the difficulties the effects give them. For `"dichotomous"`, 2PL items with a = 1, b = the intercept
   * plus xᵀη, c = 0 and d = 1. For the other models, partial credit items with a = 1 and the steps b_j = the intercept
   * plus xᵀη plus the threshold τ_j of the item.
   */
  readonly items: readonly TestItem<TParameters>[];
  /** The test of the features' difficulties against the Rasch model of the same thresholds. */
  readonly comparison: RaschComparison;
  /** The marginal log-likelihood at the estimates. */
  readonly logLikelihood: number;
  /**
   * Akaike's information criterion: -2 logLikelihood + 2p, with p the effects, the standard deviation of theta and the
   * free thresholds.
   */
  readonly aic: number;
  /** The Bayesian information criterion: -2 logLikelihood + p ln n, with n the number of persons with a response. */
  readonly bic: number;
  /** Whether EM reached `tolerance` within `maxIterations` for the LLTM and for the Rasch model of the comparison. */
  readonly converged: boolean;
  /** EM iterations run for the LLTM. */
  readonly iterations: number;
}

const DEFAULT_TOLERANCE = 1e-4;
const DEFAULT_MAX_ITERATIONS = 500;
const STANDARD_ERRORS: readonly LltmStandardErrors[] = ["none", "hessian"];
const MODELS: readonly LltmModel[] = ["dichotomous", "partial-credit", "rating-scale"];
/** The model in the messages of the limits of its size. */
const RASCH_MODEL_NAME = "The Rasch model of an LLTM's comparison";

/**
 * Estimates the linear logistic test model (LLTM): the difficulty of each dichotomous item is a sum of the effects of
 * its features, by marginal maximum likelihood, and tests it against the Rasch model.
 *
 * @remarks
 * Item i gives a person at theta the probability of a correct response logistic(θ - b_i), with b_i = x_iᵀη the sum
 * of the effects of its features, plus an intercept if there is one. Theta is N(0, σ²), σ estimated, so the effects and
 * σ are on the Rasch scale with discrimination 1. With an intercept, the model estimates the mean μ of theta and the
 * effects of the features, and reports -μ as the intercept: the difficulty of an item with all features 0, the mean
 * of theta being 0. The Rasch model of the comparison gives every item a difficulty of its own, with the same
 * distribution of theta, by the same EM algorithm as `manyFacetRasch`: the LLTM is nested in it, so
 * χ² = 2(ℓ_Rasch - ℓ_LLTM) with the items less the effects as degrees of freedom tests whether the features explain
 * the difficulties. Items need responses in both categories, as in calibration. Persons without responses change
 * nothing.
 *
 * Valid input: dichotomous responses with a feature record for every item, finite numbers by feature; more items than
 * effects; effects that no combination of the others reproduces, so a constant feature with an intercept throws
 * `DataError` naming it. An LLTM explains difficulty without a residual between items: items with the same features
 * get the same difficulty, so when the comparison rejects it, `itemFeatureRegression` of a calibration gives
 * predictions with the residual between items.
 *
 * @example
 * ```ts
 * import {
 *   createRandom,
 *   lltm,
 *   simulateResponses,
 *   simulateThetas,
 *   twoParameterLogistic,
 * } from "@measurekit/psychometrics";
 *
 * // Six items whose difficulty is -1.5 + 0.5 per step, answered by 500 simulated persons.
 * const random = createRandom(3);
 * const items = [1, 2, 3, 4, 5, 6].map((steps) => ({
 *   model: twoParameterLogistic,
 *   parameters: { a: 1, b: -1.5 + 0.5 * steps, c: 0, d: 1 },
 * }));
 * const responses = simulateResponses(items, simulateThetas(500, random), random);
 * const features = {
 *   item1: { steps: 1 },
 *   item2: { steps: 2 },
 *   item3: { steps: 3 },
 *   item4: { steps: 4 },
 *   item5: { steps: 5 },
 *   item6: { steps: 6 },
 * };
 * const model = lltm(responses, features);
 * model.effects.map(({ name, estimate }) => [name, estimate]); // => [["(intercept)", -1.56], ["steps", 0.50]]
 * model.comparison; // => { chiSquare: 2.43, degreesOfFreedom: 4, significance: 0.66, ... }
 * ```
 *
 * @param data - The responses: a response matrix or response records, dichotomous.
 * @param features - The features of each item by item id, each a finite number by feature name.
 * @param options - How the model is estimated.
 * @returns The effects of the features, the standard deviation of theta, the items with their difficulties and the test
 * against the Rasch model.
 * @throws `DataError` when the responses or the options are invalid, an item has a response above 1 or responses in one
 * category only, an item lacks features, there are not more items than effects, an effect is a linear combination of
 * those before it, or the Rasch model of the comparison has more than 1,000 parameters, its items and free thresholds,
 * or more than 2^24 expected counts, its items × quadrature points × categories.
 * @see {@link https://doi.org/10.1016/0001-6918(73)90003-6 | Fischer (1973)}
 * @see {@link https://doi.org/10.1007/BF02293801 | Bock and Aitkin (1981)}
 */
export function lltm(
  data: ResponseMatrix | ResponseRecords,
  features: ItemFeatures,
  options?: LltmOptions & { readonly model?: "dichotomous" },
): Lltm;
/**
 * Estimates the linear logistic test model of items with any number of categories, by the partial credit or the rating
 * scale model: the features explain the location of each item and its thresholds are estimated.
 *
 * @remarks
 * Item i gives a person at theta category k with probability P_k ∝ exp(Σ_(j ≤ k) (θ - b_i - τ_j)), with b_i = x_iᵀη the
 * location of the item, plus an intercept if there is one, and τ the thresholds, which sum to 0: those of each item
 * for `"partial-credit"`, and one set that all items share for `"rating-scale"`. Everything else is as for the
 * dichotomous model, and the Rasch model of the comparison gives each item a location of its own with the same
 * thresholds. Every item needs responses in every category from 0 to its highest, at least two; for the rating scale,
 * every item has the same highest category.
 *
 * @param data - The responses: a response matrix or response records, scored from 0.
 * @param features - The features of each item by item id, each a finite number by feature name.
 * @param options - The model and how it is estimated.
 * @returns The effects of the features on the locations, the standard deviation of theta, the items as partial credit
 * items and the test against the Rasch model.
 * @throws `DataError` as the dichotomous model does, and when an item has responses in fewer than two categories or
 * skips one, or, for the rating scale, the items differ in their highest category.
 * @see {@link https://doi.org/10.1007/978-1-4612-4230-7_19 | Fischer and Ponocny (1995)}
 */
export function lltm(
  data: ResponseMatrix | ResponseRecords,
  features: ItemFeatures,
  options: LltmOptions & { readonly model: "partial-credit" | "rating-scale" },
): Lltm<ThresholdItem>;
/**
 * Estimates the linear logistic test model of the model that `options.model` names, when it is not known to the compiler
 * which: the items are those of the dichotomous or the partial credit model.
 *
 * @param data - The responses: a response matrix or response records, scored from 0.
 * @param features - The features of each item by item id, each a finite number by feature name.
 * @param options - The model and how it is estimated.
 * @returns As the call for the model that `options.model` names.
 * @throws `DataError` as the call for that model does.
 */
export function lltm(
  data: ResponseMatrix | ResponseRecords,
  features: ItemFeatures,
  options?: LltmOptions,
): Lltm<LogisticItem | ThresholdItem>;
export function lltm(
  data: ResponseMatrix | ResponseRecords,
  features: ItemFeatures,
  options: LltmOptions = {},
): Lltm<LogisticItem | ThresholdItem> {
  checkOptions(options, LLTM_OPTIONS);
  const {
    model = "dichotomous",
    intercept = true,
    quadrature = {},
    standardErrors = "hessian",
    tolerance = DEFAULT_TOLERANCE,
    maxIterations = DEFAULT_MAX_ITERATIONS,
  } = options;
  checkChoice(model, "The option model", MODELS);
  checkBoolean(intercept, "The option intercept");
  checkChoice(standardErrors, "The option standardErrors", STANDARD_ERRORS);
  checkStopping(tolerance, maxIterations);
  const rule = quadratureRule(quadrature);
  const responses = compressResponses(data);
  const scale = scaleOf(responses, model);
  const itemIds = responses.itemIds;
  const common = { data: responses, setOf: scale.setOf, categoryCounts: scale.categoryCounts };
  // The Rasch model of the comparison has a location for each item, more parameters than the LLTM's, so it bounds both.
  const raschModel = { ...common, locations: ownRows(itemIds.length), locationCount: itemIds.length, fixedMean: 0 };
  checkLinearRaschSize(raschModel, rule.nodes.length, RASCH_MODEL_NAME);
  const design = designMatrix(itemIds, features, intercept);
  const effectCount = design.names.length;
  if (itemIds.length <= effectCount) {
    throw new DataError(
      `An LLTM needs more items than effects, found ${itemIds.length} items for ${effectCount} effects.`,
    );
  }
  checkFullRank(design);
  const featureCount = effectCount - (intercept ? 1 : 0);
  const fit = fitLinearRasch(
    {
      ...common,
      locations: featureRows(design, intercept ? 1 : 0),
      locationCount: featureCount,
      ...(intercept ? {} : { fixedMean: 0 }),
    },
    { rule, tolerance, maxIterations, standardErrors: standardErrors === "hessian" },
  );
  const rasch = fitLinearRasch(raschModel, { rule, tolerance, maxIterations, standardErrors: false });
  const effects = effectsOf(fit, design.names, intercept);
  const parameterCount = effectCount + 1 + scale.freeThresholds;
  const persons = responses.offsets.reduce(
    (count, end, p) => count + (end > (responses.offsets[p - 1] ?? 0) ? 1 : 0),
    0,
  );
  return {
    effects,
    sd: fit.sd,
    items: itemIds.map((id, i) =>
      itemOf(id, difficulty(design, effects, i), model, fit.thresholds[scale.setOf[i] ?? 0]),
    ),
    comparison: comparison(fit.logLikelihood, rasch.logLikelihood, itemIds.length - effectCount),
    logLikelihood: fit.logLikelihood,
    aic: -2 * fit.logLikelihood + 2 * parameterCount,
    bic: -2 * fit.logLikelihood + parameterCount * Math.log(persons),
    converged: fit.converged && rasch.converged,
    iterations: fit.iterations,
  };
}

/** How the categories of the items are grouped into sets of thresholds, as the engine takes them. */
interface Scale {
  /** The set of thresholds of each item. */
  readonly setOf: Int32Array;
  /** The number of categories of each set. */
  readonly categoryCounts: Int32Array;
  /** The thresholds the sets leave free: the sum of each set's categories less 2, as they sum to 0. */
  readonly freeThresholds: number;
}

/**
 * Checks the responses against the model and returns the sets of thresholds: one set of two categories for
 * `"dichotomous"`, one of the items' common number of categories for `"rating-scale"`, and one for each item for
 * `"partial-credit"`.
 *
 * @throws `DataError` naming the first item that has a response above the model's categories, responses in fewer than
 * two categories, or no response in a category below its highest; and, for the rating scale, the first item whose
 * categories are not the first item's.
 */
function scaleOf(responses: CompressedResponses, model: LltmModel): Scale {
  const items = responses.itemIds.length;
  const counts = categoryTallies(responses).map(
    (tally, item) =>
      checkCategories(
        responses.itemIds[item] ?? "",
        model === "dichotomous" ? twoParameterLogistic : partialCredit,
        tally,
      ).length,
  );
  if (model === "partial-credit") {
    return {
      setOf: Int32Array.from({ length: items }, (_, i) => i),
      categoryCounts: Int32Array.from(counts),
      freeThresholds: counts.reduce((sum, categories) => sum + categories - 2, 0),
    };
  }
  const shared = counts[0] ?? 2;
  const odd = counts.findIndex((categories) => categories !== shared);
  if (odd >= 0) {
    throw new DataError(
      `The rating scale needs the same number of categories for every item, found ${shared} for item "${responses.itemIds[0]}" and ${counts[odd]} for item "${responses.itemIds[odd]}".`,
    );
  }
  return { setOf: new Int32Array(items), categoryCounts: Int32Array.of(shared), freeThresholds: shared - 2 };
}

/** Returns the location of each item as its row of the design, from column `first`: the feature effects. */
function featureRows({ names, matrix }: Design, first: number): LocationRow[] {
  const columns = names.length;
  const indices = Int32Array.from({ length: columns - first }, (_, j) => j);
  return Array.from({ length: matrix.length / columns }, (_, i) => ({
    indices,
    coefficients: Float64Array.from(indices, (j) => matrix[i * columns + first + j] ?? 0),
  }));
}

/**
 * Returns the item with difficulty `difficulty`: a 2PL item for the dichotomous model, and otherwise a partial credit
 * item whose steps are the difficulty plus the item's thresholds.
 */
function itemOf(
  id: string,
  difficulty: number,
  model: LltmModel,
  thresholds: Float64Array | undefined,
): TestItem<LogisticItem> | TestItem<ThresholdItem> {
  if (model === "dichotomous") {
    return { id, model: twoParameterLogistic, parameters: { a: 1, b: difficulty, c: 0, d: 1 } };
  }
  return { id, model: partialCredit, parameters: { a: 1, b: Array.from(thresholds ?? [], (tau) => difficulty + tau) } };
}

/** Returns a location of its own for each of `count` items. */
function ownRows(count: number): LocationRow[] {
  return Array.from({ length: count }, (_, i) => ({ indices: Int32Array.of(i), coefficients: Float64Array.of(1) }));
}

/** Returns the effects of the fit: the intercept is minus the mean of theta, which the model estimates in its place. */
function effectsOf(fit: LinearRaschFit, names: readonly string[], intercept: boolean): FeatureEffect[] {
  const { covariance, layout } = fit;
  const standardError = (index: number): number | null =>
    covariance === null ? null : Math.sqrt(covariance[index * layout.size + index] ?? 0);
  return names.map((name, j) => {
    if (intercept && j === 0) {
      return { name, estimate: -fit.mean, standardError: standardError(layout.meanIndex) };
    }
    const index = j - (intercept ? 1 : 0);
    return { name, estimate: fit.parameters[index] ?? 0, standardError: standardError(index) };
  });
}

/** Returns the difficulty of item i: its row of the design times the effects, the intercept's column being 1. */
function difficulty({ names, matrix }: Design, effects: readonly FeatureEffect[], item: number): number {
  let sum = 0;
  effects.forEach(({ estimate }, j) => {
    sum += (matrix[item * names.length + j] ?? 0) * estimate;
  });
  return sum;
}

function comparison(logLikelihood: number, raschLogLikelihood: number, degreesOfFreedom: number): RaschComparison {
  // The LLTM is nested in the Rasch model, so a negative statistic is the rounding of two maxima.
  const chiSquare = Math.max(0, 2 * (raschLogLikelihood - logLikelihood));
  return {
    raschLogLikelihood,
    chiSquare,
    degreesOfFreedom,
    significance: chiSquareUpperTail(chiSquare, degreesOfFreedom),
  };
}

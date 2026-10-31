import { checkBoolean, checkChoice, checkOptions, checkStopping, DataError } from "@measurekit/core";
import type { ResponseMatrix } from "../data/response-matrix.js";
import type { ResponseRecords } from "../data/response-records.js";
import { fitLinearRasch, type LinearRaschFit, type LocationRow } from "../irt/linear-rasch.js";
import { type LogisticParameters, twoParameterLogistic } from "../irt/logistic.js";
import type { TestItem } from "../irt/model.js";
import { type QuadratureOptions, quadratureRule } from "../irt/quadrature.js";
import { type CompressedResponses, categoryCounts, checkCategories, compressResponses } from "../irt/responses.js";
import { chiSquareUpperTail } from "../math/chi-square.js";
import { checkFullRank, type Design, designMatrix, type FeatureEffect, type ItemFeatures } from "./features.js";

/** How the standard errors of the effects of an LLTM are computed. */
export type LltmStandardErrors = "none" | "hessian";

/** Options of {@link lltm}. */
export interface LltmOptions {
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

/** A linear logistic test model estimated by marginal maximum likelihood. */
export interface Lltm {
  /** The effects on difficulty: the intercept, if any, then the features in order of their first appearance. */
  readonly effects: readonly FeatureEffect[];
  /** The standard deviation of theta, whose mean is 0 on the scale of the effects. */
  readonly sd: number;
  /** The items with the difficulties the effects give them: a = 1, b = the intercept plus xᵀη, c = 0 and d = 1. */
  readonly items: readonly TestItem<LogisticParameters>[];
  /** The test of the features' difficulties against the Rasch model. */
  readonly comparison: RaschComparison;
  /** The marginal log-likelihood at the estimates. */
  readonly logLikelihood: number;
  /** Akaike's information criterion: -2 logLikelihood + 2p, with p the effects and the standard deviation of theta. */
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
 * model.effects.map(({ name, estimate }) => [name, estimate]); // [["(intercept)", -1.56], ["steps", 0.50]]
 * model.comparison; // { chiSquare: 2.43, degreesOfFreedom: 4, significance: 0.66, ... }
 * ```
 *
 * @param data - The responses: a response matrix or response records, dichotomous.
 * @param features - The features of each item by item id, each a finite number by feature name.
 * @param options - How the model is estimated.
 * @returns The effects of the features, the standard deviation of theta, the items with their difficulties and the test
 * against the Rasch model.
 * @throws `DataError` when the responses or the options are invalid, an item has a response above 1 or responses in one
 * category only, an item lacks features, there are not more items than effects, or an effect is a linear combination of
 * those before it.
 * @see {@link https://doi.org/10.1016/0001-6918(73)90003-6 | Fischer (1973)}
 * @see {@link https://doi.org/10.1007/BF02293801 | Bock and Aitkin (1981)}
 */
export function lltm(data: ResponseMatrix | ResponseRecords, features: ItemFeatures, options: LltmOptions = {}): Lltm {
  checkOptions(options);
  const {
    intercept = true,
    quadrature = {},
    standardErrors = "hessian",
    tolerance = DEFAULT_TOLERANCE,
    maxIterations = DEFAULT_MAX_ITERATIONS,
  } = options;
  checkBoolean(intercept, "The option intercept");
  checkChoice(standardErrors, "The option standardErrors", STANDARD_ERRORS);
  checkStopping(tolerance, maxIterations);
  const rule = quadratureRule(quadrature);
  const responses = compressResponses(data);
  checkDichotomous(responses);
  const itemIds = responses.itemIds;
  const design = designMatrix(itemIds, features, intercept);
  const effectCount = design.names.length;
  if (itemIds.length <= effectCount) {
    throw new DataError(
      `An LLTM needs more items than effects, found ${itemIds.length} items for ${effectCount} effects.`,
    );
  }
  checkFullRank(design);
  const common = { data: responses, setOf: new Int32Array(itemIds.length), categoryCounts: Int32Array.of(2) };
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
  const rasch = fitLinearRasch(
    { ...common, locations: ownRows(itemIds.length), locationCount: itemIds.length, fixedMean: 0 },
    { rule, tolerance, maxIterations, standardErrors: false },
  );
  const effects = effectsOf(fit, design.names, intercept);
  const parameterCount = effectCount + 1;
  const persons = responses.offsets.reduce(
    (count, end, p) => count + (end > (responses.offsets[p - 1] ?? 0) ? 1 : 0),
    0,
  );
  return {
    effects,
    sd: fit.sd,
    items: itemIds.map((id, i) => ({
      id,
      model: twoParameterLogistic,
      parameters: { a: 1, b: difficulty(design, effects, i), c: 0, d: 1 },
    })),
    comparison: comparison(fit.logLikelihood, rasch.logLikelihood, itemIds.length - effectCount),
    logLikelihood: fit.logLikelihood,
    aic: -2 * fit.logLikelihood + 2 * parameterCount,
    bic: -2 * fit.logLikelihood + parameterCount * Math.log(persons),
    converged: fit.converged && rasch.converged,
    iterations: fit.iterations,
  };
}

/**
 * Checks that every item has responses in both categories and none above 1.
 *
 * @throws `DataError` naming the first item that does not.
 */
function checkDichotomous(responses: CompressedResponses): void {
  categoryCounts(responses).forEach((counts, item) => {
    checkCategories(responses.itemIds[item] ?? "", twoParameterLogistic, counts);
  });
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

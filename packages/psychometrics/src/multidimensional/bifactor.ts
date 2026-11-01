import { checkChoice, checkIntegerFrom, checkOptions, checkStopping } from "@measurekit/core";
import type { ResponseMatrix, ResponseRecords } from "../data/internal.js";
import { compressResponses, type ParameterPrior, respondentsOf, type StandardErrorMethod } from "../irt/internal.js";
import { bifactorFit, solveBifactor } from "./bifactor-fit.js";
import { structureOf } from "./bifactor-structure.js";
import { assemble, type MultidimensionalCalibration } from "./calibration.js";
import { type MultidimensionalLogisticParameters, multidimensionalTwoParameterLogistic } from "./logistic.js";
import type { LoadingPattern, MultidimensionalModel } from "./model.js";

/**
 * Options of {@link calibrateBifactor}.
 *
 * @typeParam TParameters - The parameters of the items' model.
 */
export interface BifactorOptions<TParameters = MultidimensionalLogisticParameters> {
  /**
   * Each item's specific dimension, by item id, every item of the data included: a name, or `null` for an item with
   * none. A specific dimension has at least two items, and an item loads on at most one. Required.
   */
  readonly specific: Readonly<Record<string, string | null>>;
  /**
   * The primary dimensions each item loads on, by item id, every item of the data included: one dimension name for a
   * bifactor model, and up to two in all for a two-tier model, whose primary dimensions are correlated.
   *
   * @defaultValue one dimension, `"general"`, for every item
   */
  readonly primary?: LoadingPattern;
  /**
   * The model of every item, such as `multidimensionalThreeParameterLogistic` or `multidimensionalGradedResponse`.
   *
   * @defaultValue `multidimensionalTwoParameterLogistic`
   */
  readonly model?: MultidimensionalModel<TParameters>;
  /**
   * Priors by parameter name: `"a"` for every slope, `"intercept"` for every intercept, and `"c"`, applied to that
   * parameter of every item whose model estimates it. Estimates with priors are posterior modes. The asymptotes of the
   * 3PL make the likelihood flat, so that a prior on `"c"` is the usual way to estimate them.
   *
   * @defaultValue none
   */
  readonly priors?: Readonly<Record<string, ParameterPrior>>;
  /**
   * The number of nodes per integrated dimension, at least 2, from -6 to 6 standard deviations: the primary dimensions
   * and each specific dimension in turn, so that the cost grows with the primary dimensions plus one, not with the
   * number of specific dimensions.
   *
   * @defaultValue `31` with one primary dimension, as mirt's `bfactor`, and `21` with two
   */
  readonly points?: number;
  /**
   * The method of the standard errors: `"hessian"` is Louis' formula for the observed information, and
   * `"cross-product"` the sum over persons of the outer products of their scores.
   *
   * @defaultValue `"none"`
   */
  readonly standardErrors?: StandardErrorMethod;
  /**
   * The calibration has converged when no estimated parameter changes by more than this in an EM iteration.
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

/** The defaults of EM, as `calibrateMultidimensional`'s. */
const DEFAULT_TOLERANCE = 1e-4;
const DEFAULT_MAX_ITERATIONS = 500;
/** The nodes per dimension: mirt's `bfactor` uses 31, and a second primary dimension makes a three-dimensional grid. */
const DEFAULT_POINTS = [31, 21] as const;
const STANDARD_ERROR_METHODS: readonly StandardErrorMethod[] = ["none", "hessian", "cross-product"];

/**
 * Estimates the item parameters of a bifactor or two-tier model by marginal maximum likelihood, with the specific
 * dimensions integrated one group of items at a time.
 *
 * @remarks
 * Each item loads on the primary dimension, or on two in a two-tier model, and on at most one specific dimension. The
 * primary dimensions have mean 0 and variance 1, and are correlated in a two-tier model; the specific dimensions are
 * uncorrelated with every other dimension. Given the primary dimensions, the groups of items of the specific dimensions
 * are independent, so the integral over all dimensions is a sum over the primary nodes of a product of sums over each
 * specific dimension's nodes, and the cost follows the primary dimensions plus one (Gibbons and Hedeker, 1992). The sum
 * of a group depends only on the responses to its own items, so it is computed once for each distinct pattern of them,
 * and once for each distinct pattern of all responses. The M-step and SQUAREM are those of
 * {@link calibrateMultidimensional}, and the grid is the rectangular rule of mirt, so the estimates are mirt's `bfactor`.
 *
 * The calibration lists the primary dimensions and then the specific ones, with a slope of 0 on those an item does not
 * load on. {@link multidimensionalEstimates} scores persons from it; the posterior mean integrates over the grid of all
 * dimensions, so it needs a small `points` with many specific dimensions, and the posterior mode does not. Missing
 * responses are skipped; persons without responses change nothing.
 *
 * @param data - The responses of the persons, as a matrix or as records.
 * @param options - The specific and primary dimensions of each item, the priors, the quadrature, the standard errors
 *   and the stopping rules; each option has its default.
 * @returns The items with their parameters, the correlations of the primary dimensions, the standardized loadings and
 *   communalities, the log-likelihood, AIC and BIC, whether the EM converged and the iterations run.
 * @see {@link https://doi.org/10.1177/014662169201600302 | Gibbons and Hedeker (1992)}
 * @see {@link https://doi.org/10.1007/s11336-010-9178-0 | Cai (2010)}
 * @throws `DataError` when the data or an option is invalid, there are no responses, `specific` or `primary` does not
 * name every item of the data and no other, an item has two specific dimensions, a specific dimension has one item,
 * there are more than two primary dimensions, a name is both primary and specific, an item lacks responses in a
 * category up to its highest or has more categories than its model allows, or the grid has more than 10⁶ nodes.
 */
export function calibrateBifactor(
  data: ResponseMatrix | ResponseRecords,
  options: BifactorOptions & { readonly model?: undefined },
): MultidimensionalCalibration<MultidimensionalLogisticParameters>;
export function calibrateBifactor<TParameters>(
  data: ResponseMatrix | ResponseRecords,
  options: BifactorOptions<TParameters> & { readonly model: MultidimensionalModel<TParameters> },
): MultidimensionalCalibration<TParameters>;
export function calibrateBifactor(
  data: ResponseMatrix | ResponseRecords,
  options: BifactorOptions<unknown>,
): MultidimensionalCalibration<unknown> {
  checkOptions(options);
  const responses = compressResponses(data);
  const { standardErrors = "none", model = multidimensionalTwoParameterLogistic, priors } = options;
  checkChoice(standardErrors, "The option standardErrors", STANDARD_ERROR_METHODS);
  const { respondents } = respondentsOf(responses);
  const structure = structureOf(options, responses.itemIds);
  const { points = DEFAULT_POINTS[structure.primaryCount - 1] ?? 21 } = options;
  checkIntegerFrom(points, "The option points", 2);
  const { tolerance = DEFAULT_TOLERANCE, maxIterations = DEFAULT_MAX_ITERATIONS } = options;
  checkStopping(tolerance, maxIterations);
  const fit = bifactorFit(responses, { model, priors }, structure, points);
  const solution = solveBifactor(fit, structure.design.dimensionNames.length, {
    tolerance,
    maxIterations,
    standardErrors,
  });
  return assemble(structure.design, solution, respondents);
}

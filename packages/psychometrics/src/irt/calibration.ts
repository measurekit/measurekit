import {
  checkChoice,
  checkOptions,
  checkStopping,
  DataError,
  iterateToConvergence,
  largestChange,
} from "@measurekit/core";
import type { ResponseMatrix } from "../data/response-matrix.js";
import type { ResponseRecords } from "../data/response-records.js";
import { toUnbounded } from "./bounds.js";
import type { CalibrationOptions, StandardErrorMethod } from "./calibration-options.js";
import { type Setup, setUp } from "./calibration-setup.js";
import { crossProductCovariance } from "./cross-product.js";
import { type ExpectationStep, expectationStep } from "./expectation.js";
import { hessianCovariance } from "./hessian-covariance.js";
import {
  type GroupDistribution,
  groupDistributions,
  hasDegenerateGroup,
  histogramChange,
  histogramParameters,
  maximizeDistributions,
} from "./latent.js";
import { type Maximization, maximization } from "./m-step.js";
import type { TestItem } from "./model.js";
import { quadratureRule } from "./quadrature.js";
import { type CompressedResponses, compressResponses } from "./responses.js";
import { squarem } from "./squarem.js";
import { maximizeTestlets, type TestletEstimate, testletEstimates } from "./testlets.js";

/** An item with its calibrated parameters. */
export interface CalibratedItem<TParameters> extends TestItem<TParameters> {
  /**
   * Standard errors of the estimated parameters by name. `null` for fixed items, without standard errors, and for
   * every item when the information matrix cannot be inverted or gives a variance that is not positive.
   */
  readonly standardErrors: Readonly<Record<string, number>> | null;
}

/** Item parameters estimated by marginal maximum likelihood, with fit statistics. */
export interface Calibration<TParameters> {
  /** The items, in item order. */
  readonly items: readonly CalibratedItem<TParameters>[];
  /** The distribution of theta in each group, in order of first appearance in `groups`; one without groups. */
  readonly groups: readonly GroupDistribution[];
  /** The standard deviation of each testlet's effect, in the order of the `testlets` option; none without testlets. */
  readonly testlets: readonly TestletEstimate[];
  /** The marginal log-likelihood at the estimates, without the priors. */
  readonly logLikelihood: number;
  /**
   * Akaike's information criterion: -2 logLikelihood + 2p, with p the number of estimated parameters, those of the
   * distributions of theta included: 2 for a normal one, the number of nodes less 1 for a histogram.
   */
  readonly aic: number;
  /**
   * The Bayesian information criterion: -2 logLikelihood + p ln n, with n the number of persons with at least one
   * response.
   */
  readonly bic: number;
  /**
   * Whether EM reached `tolerance` within `maxIterations` and no estimated normal distribution of theta ran to a
   * degenerate boundary: a mean outside the nodes, or an SD below their spacing or above their range.
   */
  readonly converged: boolean;
  /** EM iterations run. */
  readonly iterations: number;
}

const DEFAULT_TOLERANCE = 1e-4;
const DEFAULT_MAX_ITERATIONS = 500;

/**
 * Estimates item parameters by marginal maximum likelihood with the Bock-Aitkin EM algorithm, theta integrated by
 * quadrature over its distribution.
 *
 * @remarks
 * The distribution is N(0, 1) by default; in groups of persons, each with its own distribution; and normal
 * distributions and histograms estimated with the items, on the scale that fixed items or a standard normal group fix.
 * With priors, the estimates are posterior modes. Standard errors come from the Hessian of the marginal log-likelihood
 * plus the log priors, by central differences of its exact gradient. Missing responses are skipped; persons without
 * responses change nothing.
 *
 * @throws `DataError` when the matrix or an option is invalid, there are no responses, a fixed item's parameters are
 * invalid or do not fit its responses, an estimated item lacks responses in a category up to its highest or has more
 * categories than its model allows, estimated items of a model that share parameters, as RSM items share the steps,
 * have different numbers of categories, a group has no person with a response, or distributions are estimated without
 * the rectangular rule or without fixed items or a standard normal group.
 *
 * @example
 * ```ts
 * import {
 *   calibrate,
 *   createRandom,
 *   simulateItems,
 *   simulateResponses,
 *   simulateThetas,
 *   twoParameterLogistic,
 * } from "@measurekit/psychometrics";
 *
 * const random = createRandom(7);
 * const items = simulateItems(twoParameterLogistic, 5, random);
 * const responses = simulateResponses(items, simulateThetas(200, random), random);
 * const fit = calibrate(responses, { model: twoParameterLogistic });
 * fit.items[0]?.parameters; // { a: 1.43, b: 0.16, c: 0, d: 1 }
 * fit.converged; // true
 * ```
 */
export function calibrate<TParameters>(
  data: ResponseMatrix | ResponseRecords,
  options: CalibrationOptions<TParameters>,
): Calibration<TParameters> {
  return calibrateWithCovariance(data, options).calibration;
}

/**
 * The covariance matrix of the estimated parameters of a calibration's items, labeled by item id and parameter name.
 * Plain data, which JSON serializes.
 */
export interface ParameterCovariance {
  /**
   * The item and the parameter name of each row and column: the estimated items in item order, each with the names of
   * its `standardErrors`. Items whose model shares a parameter, as the rating scale model shares its steps, repeat the
   * shared parameter, with the same variance and a correlation of 1.
   */
  readonly parameters: readonly {
    /** The item id. */
    readonly item: string;
    /** The parameter's name in the item's parameter vector. */
    readonly name: string;
  }[];
  /**
   * The covariance matrix by rows: `matrix[i][j]` is the covariance of `parameters[i]` and `parameters[j]`, on the
   * parameters' own scale.
   */
  readonly matrix: readonly (readonly number[])[];
}

/** A calibration with the covariance matrix of its estimated item parameters. */
export interface CovarianceCalibration<TParameters> {
  /** The calibration. */
  readonly calibration: Calibration<TParameters>;
  /**
   * The covariance of the estimated item parameters, including the covariances between items; `null` where the
   * calibration has no standard errors: with `standardErrors` `"none"`, with no estimated item, and when the
   * information matrix cannot be inverted or gives a variance that is not positive. The parameters of the
   * distributions of theta and of the testlets are estimated with the items but not listed.
   */
  readonly covariance: ParameterCovariance | null;
}

const STANDARD_ERROR_METHODS: readonly StandardErrorMethod[] = ["none", "hessian", "cross-product"];

/**
 * Calibrates as {@link calibrate} does and also returns the covariance matrix of the estimated item parameters.
 *
 * @remarks
 * The covariance comes from the method of the option `standardErrors`, and the standard errors of the calibration are
 * the square roots of its diagonal. It is the part for the item parameters of the covariance matrix of all the
 * estimated parameters, so it includes the uncertainty of the distributions of theta estimated with the items. A
 * delta method for a function of the item parameters, such as an equated score, is `gᵀ Σ g` with `g` the gradient.
 *
 * @throws `DataError` as {@link calibrate} does.
 */
export function calibrateWithCovariance<TParameters>(
  data: ResponseMatrix | ResponseRecords,
  options: CalibrationOptions<TParameters>,
): CovarianceCalibration<TParameters> {
  checkOptions(options);
  const responses = compressResponses(data);
  const { tolerance = DEFAULT_TOLERANCE, maxIterations = DEFAULT_MAX_ITERATIONS, standardErrors = "none" } = options;
  checkStopping(tolerance, maxIterations);
  checkChoice(standardErrors, "The option standardErrors", STANDARD_ERROR_METHODS);
  const answered = Array.from(responses.personIds, (_, person) => {
    return (responses.offsets[person + 1] ?? 0) > (responses.offsets[person] ?? 0);
  });
  const respondents = answered.filter(Boolean).length;
  if (respondents === 0) throw new DataError("Calibration needs at least one response.");
  const rule = quadratureRule(options.quadrature);
  const { nodes } = rule;
  const setup = setUp(responses, options, answered, rule);
  const { items, values, population: latent, design } = setup;
  const mStep = maximization(setup, nodes);
  const eStep = expectationStep(responses, items, nodes, latent, design);

  const { converged: stopped, iterations } = iterate(setup, mStep, eStep, nodes, tolerance, maxIterations);
  const converged = stopped && !hasDegenerateGroup(latent, values, nodes);
  const logLikelihood = eStep.run();
  const size = values.length;
  const covariance = size > 0 ? covarianceOf(standardErrors, setup, mStep, eStep, responses, nodes) : null;
  const standardError = covariance === null ? null : (slot: number) => Math.sqrt(covariance[slot * size + slot] ?? 0);

  const parameterCount = size + histogramParameters(latent);
  return {
    calibration: {
      items: items.map((item) => ({
        id: item.id,
        model: item.model,
        parameters: item.parameters,
        standardErrors:
          standardError === null || !item.estimated
            ? null
            : Object.fromEntries(item.names.map((name, p) => [name, standardError(item.slots[p] ?? 0)])),
      })),
      groups: groupDistributions(latent, values, nodes, standardError),
      testlets: design === null ? [] : testletEstimates(design, responses.itemIds, values, standardError),
      logLikelihood,
      aic: -2 * logLikelihood + 2 * parameterCount,
      bic: -2 * logLikelihood + parameterCount * Math.log(respondents),
      converged,
      iterations,
    },
    covariance: parameterCovariance(items, covariance, size),
  };
}

/**
 * Runs EM cycles of the E-step and M-step until no parameter changes by more than `tolerance` or `maxIterations`: with
 * the extrapolation SQUAREM, except for a histogram of theta, whose weights it does not extrapolate.
 */
function iterate<TParameters>(
  setup: Setup<TParameters>,
  mStep: Maximization<TParameters>,
  eStep: ExpectationStep,
  nodes: Float64Array,
  tolerance: number,
  maxIterations: number,
): { readonly converged: boolean; readonly iterations: number } {
  const { values, population: latent, design, free } = setup;
  const previous = new Float64Array(values.length);
  const histograms = new Float64Array(histogramParameters(latent) + latent.groups.length);
  histogramChange(latent, histograms);
  const step = (): { readonly logLikelihood: number; readonly change: number } => {
    const logLikelihood = eStep.run();
    previous.set(values);
    for (const block of mStep.blocks) mStep.maximize(block);
    maximizeDistributions(latent, values, nodes);
    if (design !== null) maximizeTestlets(design, values, nodes);
    return { logLikelihood, change: Math.max(largestChange(previous, values), histogramChange(latent, histograms)) };
  };
  const stopping = { tolerance, maxIterations };
  if (histogramParameters(latent) > 0) return iterateToConvergence(() => step().change, stopping);
  return squarem(
    {
      step,
      get: () => Float64Array.from(values, (x, j) => toUnbounded(x, free[j]?.lower ?? 0, free[j]?.upper ?? 0)),
      set: (state) => mStep.assign(mStep.all, state),
    },
    stopping,
  );
}

/**
 * Returns the part of the covariance matrix of the free parameters at the estimated items' parameters, labeled and
 * symmetrized.
 */
function parameterCovariance<TParameters>(
  items: Setup<TParameters>["items"],
  covariance: Float64Array | null,
  size: number,
): ParameterCovariance | null {
  const estimated = items.filter((item) => item.estimated);
  if (covariance === null || estimated.length === 0) return null;
  const slots = estimated.flatMap((item) => Array.from(item.slots));
  return {
    parameters: estimated.flatMap((item) => item.names.map((name) => ({ item: item.id, name }))),
    // The inverse of the Hessian is symmetric only to rounding.
    matrix: slots.map((row) =>
      slots.map((column) => ((covariance[row * size + column] ?? 0) + (covariance[column * size + row] ?? 0)) / 2),
    ),
  };
}

/** Returns the covariance matrix of the free parameters by the method, or `null` without standard errors. */
function covarianceOf<TParameters>(
  method: StandardErrorMethod,
  setup: Setup<TParameters>,
  mStep: Maximization<TParameters>,
  eStep: ExpectationStep,
  responses: CompressedResponses,
  nodes: Float64Array,
): Float64Array | null {
  if (method === "hessian") return hessianCovariance(setup, mStep, eStep);
  if (method === "cross-product") return crossProductCovariance(setup, eStep, responses, nodes);
  return null;
}

import { checkChoice, checkOptions, checkStopping, DataError } from "@measurekit/core";
import type { ResponseMatrix } from "../data/response-matrix.js";
import type { ResponseRecords } from "../data/response-records.js";
import type { CalibrationOptions, StandardErrorMethod } from "./calibration-options.js";
import { type Setup, setUp } from "./calibration-setup.js";
import { crossProductCovariance } from "./cross-product.js";
import { type ExpectationStep, expectationStep } from "./expectation.js";
import { hessianCovariance } from "./hessian-covariance.js";
import { largestChange } from "./iteration.js";
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

/** A calibration with the covariance matrix of each item's estimated parameters. */
export interface CovarianceCalibration<TParameters> {
  /** The calibration. */
  readonly calibration: Calibration<TParameters>;
  /**
   * For each item, the covariance matrix of its vector entries by rows, in the order of its `standardErrors`; `null`
   * where its standard errors are `null`.
   */
  readonly covariances: readonly (Float64Array | null)[];
}

const STANDARD_ERROR_METHODS: readonly StandardErrorMethod[] = ["none", "hessian", "cross-product"];

/** Calibrates as {@link calibrate} does and also returns the covariance matrix of each item's estimates. */
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
  const itemCovariances = covariancesOfItems(items, covariance, size);

  const parameterCount = size + histogramParameters(latent);
  return {
    calibration: {
      items: items.map((item, i) => {
        const block = itemCovariances[i] ?? null;
        const count = item.slots.length;
        return {
          id: item.id,
          model: item.model,
          parameters: item.parameters,
          standardErrors:
            block === null
              ? null
              : Object.fromEntries(item.names.map((name, p) => [name, Math.sqrt(block[p * count + p] ?? 0)])),
        };
      }),
      groups: groupDistributions(latent, values, nodes, standardError),
      testlets: design === null ? [] : testletEstimates(design, responses.itemIds, values, standardError),
      logLikelihood,
      aic: -2 * logLikelihood + 2 * parameterCount,
      bic: -2 * logLikelihood + parameterCount * Math.log(respondents),
      converged,
      iterations,
    },
    covariances: itemCovariances,
  };
}

/** Runs EM cycles of the E-step and M-step until no parameter changes by more than `tolerance` or `maxIterations`. */
function iterate<TParameters>(
  setup: Setup<TParameters>,
  mStep: Maximization<TParameters>,
  eStep: ExpectationStep,
  nodes: Float64Array,
  tolerance: number,
  maxIterations: number,
): { readonly converged: boolean; readonly iterations: number } {
  const { values, population: latent, design } = setup;
  const previous = new Float64Array(values.length);
  const histograms = new Float64Array(histogramParameters(latent) + latent.groups.length);
  histogramChange(latent, histograms);
  let iterations = 0;
  let converged = false;
  while (!converged && iterations < maxIterations) {
    iterations++;
    eStep.run();
    previous.set(values);
    for (const block of mStep.blocks) mStep.maximize(block);
    maximizeDistributions(latent, values, nodes);
    if (design !== null) maximizeTestlets(design, values, nodes);
    converged = Math.max(largestChange(previous, values), histogramChange(latent, histograms)) <= tolerance;
  }
  return { converged, iterations };
}

/** Returns, for each item, the part of the covariance matrix of the free parameters at its slots; `null` for a fixed item. */
function covariancesOfItems<TParameters>(
  items: Setup<TParameters>["items"],
  covariance: Float64Array | null,
  size: number,
): (Float64Array | null)[] {
  return items.map((item) => {
    if (!item.estimated || covariance === null) return null;
    const { slots } = item;
    return Float64Array.from({ length: slots.length ** 2 }, (_, index) => {
      const row = slots[Math.floor(index / slots.length)] ?? 0;
      return covariance[row * size + (slots[index % slots.length] ?? 0)] ?? 0;
    });
  });
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

import {
  checkChoice,
  checkFinite,
  checkOptions,
  checkPositive,
  checkRecord,
  checkStopping,
  DataError,
  describe,
  normalizeLogs,
} from "@measurekit/core";
import type { ResponseMatrix, ResponseRecords } from "../data/internal.js";
import {
  type CalibratedItem,
  type CalibrationOptions,
  type IrtModel,
  indexItems,
  logProbability,
  type ParameterPrior,
  type QuadratureOptions,
  quadratureRule,
  respondentsOf,
  setUp,
} from "../irt/internal.js";
import { calibrateResponseTimes } from "./lognormal.js";
import { expectation, iterate, jointModel } from "./speed-accuracy-em.js";
import { jointCovariance } from "./speed-accuracy-errors.js";
import {
  type Conditional,
  conditional,
  emptySums,
  fillLogTerms,
  fillSums,
  type PersonSums,
  type SpeedMoments,
  speedMoments,
} from "./speed-accuracy-time.js";
import { type TimedResponses, timedResponses } from "./timed-responses.js";

/** How the standard errors of {@link calibrateSpeedAccuracy} are computed. */
export type SpeedAccuracyStandardErrors = "none" | "hessian";

/** Options of {@link calibrateSpeedAccuracy}. */
export interface SpeedAccuracyOptions<TParameters> {
  /** The model of every item, or one model per item, in item order. */
  readonly model: IrtModel<TParameters> | readonly IrtModel<TParameters>[];
  /**
   * Priors by parameter name, such as `{ c: betaPrior(5, 17) }`, applied to that parameter of every item whose model
   * has it, as in `calibrate`.
   *
   * @defaultValue none
   */
  readonly priors?: Readonly<Record<string, ParameterPrior>>;
  /**
   * The quadrature rule over theta.
   *
   * @defaultValue 61 rectangular points
   */
  readonly quadrature?: QuadratureOptions;
  /**
   * `"hessian"` computes the standard errors from the Hessian of the marginal log-likelihood, by central differences of
   * its exact gradient; it runs two E-steps per parameter.
   *
   * @defaultValue `"none"`
   */
  readonly standardErrors?: SpeedAccuracyStandardErrors;
  /**
   * EM has converged when no parameter changes by more than this in an iteration.
   *
   * @defaultValue `1e-6`
   */
  readonly tolerance?: number;
  /**
   * The most EM iterations to run.
   *
   * @defaultValue `1000`
   */
  readonly maxIterations?: number;
}

/** An item calibrated with its response model and its time parameters. */
export interface SpeedAccuracyItem<TParameters> extends CalibratedItem<TParameters> {
  /** β, the time intensity: the mean log time of a person of speed 0. */
  readonly intensity: number;
  /** α, the time discrimination: 1 / α is the SD of the log times at a given speed. */
  readonly discrimination: number;
  /** The standard errors of the time parameters; `null` without standard errors. */
  readonly timeStandardErrors: {
    /** The standard error of the intensity. */
    readonly intensity: number;
    /** The standard error of the discrimination. */
    readonly discrimination: number;
  } | null;
}

/** The hierarchical model of speed and accuracy calibrated by marginal maximum likelihood. */
export interface SpeedAccuracyCalibration<TParameters> {
  /** The items, in the data's item order. */
  readonly items: readonly SpeedAccuracyItem<TParameters>[];
  /** σ_τ, the standard deviation of the persons' speed; their mean is 0, and theta is N(0, 1). */
  readonly speedSd: number;
  /** ρ, the correlation between theta and speed. */
  readonly correlation: number;
  /** The standard error of `correlation`; `null` without standard errors. */
  readonly correlationStandardError: number | null;
  /** The marginal log-likelihood of the responses and the times. */
  readonly logLikelihood: number;
  /** -2 logLikelihood + 2p, with p the item parameters, two time parameters for each item, σ_τ and ρ. */
  readonly aic: number;
  /** -2 logLikelihood + p ln n, with n the persons with responses. */
  readonly bic: number;
  /** Whether EM reached `tolerance` within `maxIterations`. */
  readonly converged: boolean;
  /** EM iterations run. */
  readonly iterations: number;
}

/** A person's ability and speed, estimated from the responses and the times together. */
export interface SpeedAccuracyEstimate {
  /** The person's id. */
  readonly id: string;
  /** The posterior mean of theta (EAP); `null` for a person without responses. */
  readonly theta: number | null;
  /** The posterior SD of theta; `null` with `theta`. */
  readonly thetaStandardError: number | null;
  /** The posterior mean of speed τ; `null` for a person without times. */
  readonly speed: number | null;
  /** The posterior SD of speed; `null` with `speed`. */
  readonly speedStandardError: number | null;
}

const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_MAX_ITERATIONS = 1000;
const STANDARD_ERRORS: readonly SpeedAccuracyStandardErrors[] = ["none", "hessian"];

/**
 * Calibrates the hierarchical model of speed and accuracy: an IRT model for the responses, ln T_ij ~ N(β_i - τ_j,
 * 1 / α_i²) for the times, and (θ, τ) bivariate normal with θ ~ N(0, 1), τ ~ N(0, σ_τ²) and correlation ρ, by marginal
 * maximum likelihood with EM.
 *
 * @remarks
 * Given theta, τ is normal with mean ρ σ_τ θ and variance σ_τ² (1 - ρ²), and the log times are linear in τ, so τ
 * integrates out of the time density in closed form: a person's log times given theta are multivariate normal with
 * mean β - ρ σ_τ θ and covariance diag(1 / α²) + σ_τ² (1 - ρ²) 11ᵀ, whose determinant and inverse take one pass over
 * the person's times. The marginal likelihood needs only the quadrature over theta, and the estimates are those of
 * maximum likelihood with the item parameters as fixed effects. The E-step is that of `calibrate`, with each person's
 * time density at each node added; the items' M-step is `calibrate`'s, and β, α, σ_τ and ρ follow in closed form from
 * the expected moments of τ.
 *
 * The times carry information on theta through ρ, so the items' parameters differ from those of `calibrate` on the
 * responses alone. A response without a time counts for the responses only, and a person without times for theta only.
 * The item population level of the hierarchical model, a multivariate normal distribution of the item parameters, is
 * not part of it.
 *
 * @param data - The responses of the persons with their times.
 * @param options - The model, priors and quadrature of the responses, the standard errors and the stopping rules; each
 *   option has its default.
 * @returns The items with their parameters, the standard deviation of the speed, the correlation of speed and ability
 *   with its standard error, the log-likelihood, AIC and BIC, whether the estimation converged and the iterations run.
 * @throws `DataError` when the data are malformed or have no times, an item has fewer than two times or times that do
 * not differ, an item cannot be calibrated as `calibrate` requires, or an option is invalid.
 * @see {@link https://doi.org/10.3102/1076998607300422 | van der Linden (2007)}
 * @see {@link https://doi.org/10.1080/00273171.2014.1003701 | Molenaar, Tuerlinckx and van der Maas (2015)}
 */
export function calibrateSpeedAccuracy<TParameters>(
  data: ResponseMatrix | ResponseRecords,
  options: SpeedAccuracyOptions<TParameters>,
): SpeedAccuracyCalibration<TParameters> {
  checkOptions(options);
  const {
    model,
    priors,
    quadrature,
    standardErrors = "none",
    tolerance = DEFAULT_TOLERANCE,
    maxIterations = DEFAULT_MAX_ITERATIONS,
  } = options;
  checkChoice(standardErrors, "The option standardErrors", STANDARD_ERRORS);
  checkStopping(tolerance, maxIterations);
  const timed = timedResponses(data, "Speed-accuracy calibration");
  const start = calibrateResponseTimes(data);
  const { answered, respondents } = respondentsOf(timed);
  const rule = quadratureRule(quadrature);
  const irtOptions: CalibrationOptions<TParameters> = {
    model,
    ...(priors === undefined ? {} : { priors }),
    ...(quadrature === undefined ? {} : { quadrature }),
  };
  const setup = setUp(timed, irtOptions, answered, rule);
  const itemCount = timed.itemIds.length;
  const parameters = new Float64Array(2 * itemCount + 2);
  start.items.forEach((item, i) => {
    parameters[i] = item.intensity;
    parameters[itemCount + i] = item.discrimination;
  });
  parameters[2 * itemCount] = start.speedSd;
  const joint = jointModel(timed, setup, Float64Array.from(rule.nodes), parameters);
  const { converged, iterations } = iterate(joint, tolerance, maxIterations);
  const logLikelihood = expectation(joint);
  const covariance = standardErrors === "hessian" ? jointCovariance(joint) : null;
  const itemSize = setup.values.length;
  const size = itemSize + parameters.length;
  const error = (slot: number): number | null =>
    covariance === null ? null : Math.sqrt(covariance[slot * size + slot] ?? 0);
  return {
    items: setup.items.map((item, i) => ({
      id: item.id,
      model: item.model,
      parameters: item.parameters,
      standardErrors:
        covariance === null
          ? null
          : Object.fromEntries(item.names.map((name, p) => [name, error(item.slots[p] ?? 0) ?? 0])),
      intensity: parameters[i] ?? 0,
      discrimination: parameters[itemCount + i] ?? 0,
      timeStandardErrors:
        covariance === null
          ? null
          : { intensity: error(itemSize + i) ?? 0, discrimination: error(itemSize + itemCount + i) ?? 0 },
    })),
    speedSd: parameters[2 * itemCount] ?? 0,
    correlation: parameters[2 * itemCount + 1] ?? 0,
    correlationStandardError: error(itemSize + 2 * itemCount + 1),
    logLikelihood,
    aic: -2 * logLikelihood + 2 * size,
    bic: -2 * logLikelihood + size * Math.log(respondents),
    converged,
    iterations,
  };
}

/**
 * Returns the estimate of theta and speed of every person, from the responses and the times together, in the order of
 * the data's person ids.
 *
 * @remarks
 * Theta is the expected a posteriori estimate over 61 rectangular points under N(0, 1), with the density of the
 * person's times, which depends on theta through the correlation, in the likelihood: the times sharpen theta, the
 * collateral information of van der Linden, Klein Entink and Fox. Speed is the posterior mean and SD of τ, whose
 * conditional distribution given theta is normal. A person with responses and no times has theta from the responses
 * only and no speed. Items are matched to the calibration by id.
 *
 * @param data - The responses of the persons with their times.
 * @param calibration - The calibrated hierarchical model.
 * @returns The ability and the speed of each person with their standard errors; `null` for a person without responses.
 * @throws `DataError` when the data are malformed or have no times, the calibration is not one, an item of the data is
 * not among its items, or a response is not a category of its item.
 * @see {@link https://doi.org/10.1177/0146621609344844 | van der Linden, Klein Entink and Fox (2010)}
 */
export function speedAccuracyEstimates<TParameters>(
  data: ResponseMatrix | ResponseRecords,
  calibration: SpeedAccuracyCalibration<TParameters>,
): SpeedAccuracyEstimate[] {
  const timed = timedResponses(data, "Speed-accuracy estimation");
  checkCalibration(calibration);
  const indices = indexItems(calibration.items);
  const items = timed.itemIds.map((id) => {
    const item = calibration.items[indices.get(id) ?? -1];
    if (item === undefined)
      throw new DataError(`The data have item "${id}", which is not among the calibration's items.`);
    return item;
  });
  const { nodes, weights } = quadratureRule();
  const sums = emptySums(timed.personIds.length);
  fillSums(
    timed,
    items.map(({ intensity }) => intensity),
    items.map(({ discrimination }) => discrimination),
    sums,
  );
  const second = conditional(calibration.speedSd, calibration.correlation);
  const logTerms = new Float64Array(timed.personIds.length * nodes.length);
  fillLogTerms(sums, second, nodes, logTerms);
  const context: EstimationContext<TParameters> = {
    timed,
    items,
    tables: logProbabilityTables(items, nodes),
    nodes,
    logWeights: Float64Array.from(weights, Math.log),
    logTerms,
    sums,
    second,
    posterior: new Float64Array(nodes.length),
  };
  return timed.personIds.map((id, person) => estimateOf(context, id, person));
}

/** What the estimates of the persons share: the data, the items' log probabilities and the time terms. */
interface EstimationContext<TParameters> {
  readonly timed: TimedResponses;
  readonly items: readonly SpeedAccuracyItem<TParameters>[];
  readonly tables: readonly { categories: number; table: Float64Array }[];
  readonly nodes: Float64Array;
  readonly logWeights: Float64Array;
  readonly logTerms: Float64Array;
  readonly sums: PersonSums;
  readonly second: Conditional;
  /** Scratch space for a person's posterior over the nodes. */
  readonly posterior: Float64Array;
}

/** Returns one person's estimates: nothing without responses, theta only without times. */
function estimateOf<TParameters>(
  context: EstimationContext<TParameters>,
  id: string,
  person: number,
): SpeedAccuracyEstimate {
  const { timed, nodes, sums, second, posterior } = context;
  if ((timed.offsets[person + 1] ?? 0) === (timed.offsets[person] ?? 0)) {
    return { id, theta: null, thetaStandardError: null, speed: null, speedStandardError: null };
  }
  posteriorOfTheta(context, person);
  let thetaMean = 0;
  let thetaSquare = 0;
  for (let q = 0; q < nodes.length; q++) {
    thetaMean += (posterior[q] ?? 0) * (nodes[q] ?? 0);
    thetaSquare += (posterior[q] ?? 0) * (nodes[q] ?? 0) ** 2;
  }
  const thetaStandardError = Math.sqrt(Math.max(thetaSquare - thetaMean * thetaMean, 0));
  if ((sums.count[person] ?? 0) === 0) {
    return { id, theta: thetaMean, thetaStandardError, speed: null, speedStandardError: null };
  }
  const speed: SpeedMoments = { mean: 0, variance: 0, square: 0, product: 0 };
  speedMoments(sums, person, second, thetaMean, thetaSquare, speed);
  return {
    id,
    theta: thetaMean,
    thetaStandardError,
    speed: speed.mean,
    speedStandardError: Math.sqrt(Math.max(speed.variance, 0)),
  };
}

/** Writes a person's posterior over the nodes into `context.posterior`: the prior, the responses and the times. */
function posteriorOfTheta<TParameters>(context: EstimationContext<TParameters>, person: number): void {
  const { timed, items, tables, nodes, logWeights, logTerms, posterior } = context;
  for (let q = 0; q < nodes.length; q++) {
    posterior[q] = (logWeights[q] ?? 0) + (logTerms[person * nodes.length + q] ?? 0);
  }
  for (let r = timed.offsets[person] ?? 0; r < (timed.offsets[person + 1] ?? 0); r++) {
    checkResponse(timed, items, person, r);
    const { categories, table } = tables[timed.items[r] ?? 0] ?? { categories: 0, table: new Float64Array(0) };
    for (let q = 0; q < nodes.length; q++) {
      posterior[q] = (posterior[q] ?? 0) + (table[q * categories + (timed.codes[r] ?? 0)] ?? 0);
    }
  }
  normalizeLogs(posterior);
}

/** Returns each item's log probabilities at the nodes, by node, then category, with its number of categories. */
function logProbabilityTables<TParameters>(
  items: readonly SpeedAccuracyItem<TParameters>[],
  nodes: Float64Array,
): { categories: number; table: Float64Array }[] {
  return items.map((item) => {
    const categories = item.model.categoryCount(item.parameters);
    const table = new Float64Array(nodes.length * categories);
    const probabilities = new Float64Array(categories);
    nodes.forEach((theta, q) => {
      item.model.probabilities(item.parameters, theta, probabilities);
      for (let k = 0; k < categories; k++) {
        table[q * categories + k] = logProbability(probabilities[k] ?? 0);
      }
    });
    return { categories, table };
  });
}

/** Throws unless response `r` of the person is a category of its item. */
function checkResponse<TParameters>(
  timed: TimedResponses,
  items: readonly SpeedAccuracyItem<TParameters>[],
  person: number,
  r: number,
): void {
  const item = items[timed.items[r] ?? 0];
  const code = timed.codes[r] ?? 0;
  const categories = item === undefined ? 0 : item.model.categoryCount(item.parameters);
  if (code >= categories) {
    throw new DataError(
      `The response of person "${timed.personIds[person]}" to item "${timed.itemIds[timed.items[r] ?? 0]}" is ${code}, but the item has categories 0 to ${categories - 1}.`,
    );
  }
}

/** Throws unless the calibration has items with time parameters, a positive speed SD and a correlation in (-1, 1). */
function checkCalibration<TParameters>(calibration: SpeedAccuracyCalibration<TParameters>): void {
  checkRecord(calibration, "The calibration must be a speed and accuracy calibration");
  checkFinite(calibration.speedSd, "The calibration's speedSd");
  checkPositive(calibration.speedSd, "The calibration's speedSd");
  checkFinite(calibration.correlation, "The calibration's correlation");
  if (!(calibration.correlation > -1 && calibration.correlation < 1)) {
    throw new DataError(
      `The calibration's correlation must be between -1 and 1, found ${describe(calibration.correlation)}.`,
    );
  }
  const items: unknown = calibration.items;
  if (!Array.isArray(items)) {
    throw new DataError(`The calibration needs items, an array, found ${describe(items)}.`);
  }
  for (const item of calibration.items) {
    checkFinite(item.intensity, `The intensity of item ${describe(item.id)}`);
    checkPositive(item.discrimination, `The time discrimination of item ${describe(item.id)}`);
  }
}

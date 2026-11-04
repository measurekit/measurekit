import {
  checkOptions,
  checkStopping,
  DataError,
  type FieldNames,
  iterateToConvergence,
  largestChange,
} from "@measurekit/core";
import type { ResponseMatrix, ResponseRecords } from "../data/internal.js";
import { withinStandardErrorSize } from "../irt/internal.js";
import { invertPositiveDefinite } from "../math/internal.js";
import { forEachTime, type TimedResponses, type TimeItem, timedResponses } from "./timed-responses.js";

/** Options of {@link calibrateResponseTimes}. */
export interface ResponseTimeCalibrationOptions {
  /**
   * EM has converged when no parameter changes by more than this in an iteration.
   *
   * @defaultValue `1e-6`
   */
  readonly tolerance?: number;
  /**
   * The most EM iterations.
   *
   * @defaultValue `1000`
   */
  readonly maxIterations?: number;
}

const RESPONSE_TIME_CALIBRATION_OPTIONS: FieldNames<ResponseTimeCalibrationOptions> = {
  tolerance: true,
  maxIterations: true,
};

/** A calibrated time item. */
export interface CalibratedTimeItem extends TimeItem {
  /**
   * The standard error of the intensity; `null` when its information cannot be inverted, or for more than 2,000 items,
   * whose information holds every pair of them.
   */
  readonly standardError: number | null;
}

/** The lognormal model of response times calibrated by marginal maximum likelihood. */
export interface ResponseTimeCalibration {
  /** The items, in the data's item order. */
  readonly items: readonly CalibratedTimeItem[];
  /** σ_τ, the standard deviation of the persons' speed; their mean is 0. */
  readonly speedSd: number;
  /** The marginal log-likelihood of the log times. */
  readonly logLikelihood: number;
  /** -2 logLikelihood + 2p, p = 2 per item plus 1. */
  readonly aic: number;
  /** -2 logLikelihood + p ln n, n the persons with times. */
  readonly bic: number;
  /** Whether EM reached `tolerance` within `maxIterations`. */
  readonly converged: boolean;
  /** EM iterations run. */
  readonly iterations: number;
}

const DEFAULT_TOLERANCE = 1e-6;
const DEFAULT_MAX_ITERATIONS = 1000;
/** The starting standard deviation of the speed. */
const START_SPEED_SD = 0.5;

/**
 * Calibrates the lognormal model of response times: ln T_ij ~ N(β_i - τ_j, 1 / α_i²), with the speed τ_j ~ N(0, σ_τ²),
 * by marginal maximum likelihood with EM.
 *
 * @remarks
 * τ integrates out in closed form, so the log-likelihood is exact; it is the linear mixed model of nlme's `lme(log(t) ~
 * 0 + item, random = ~ 1 | person, weights = varIdent(form = ~ 1 | item), method = "ML")`. The E-step gives each
 * person's normal posterior of τ, the M-step each β_i, α_i and σ_τ in closed form. The intensities' standard errors
 * come from their information, Σ_j X_jᵀ V_j⁻¹ X_j, for at most 2,000 items, as it holds every pair of them. Uses the
 * responses with times; persons and items without them change nothing.
 *
 * @param data - The responses of the persons with their times.
 * @param options - The stopping rules.
 * @returns The parameters of each item, the standard deviation of the speed, the log-likelihood, AIC and BIC, whether
 *   the estimation converged and the iterations run.
 * @throws `DataError` when the data are malformed or have no times, an item has fewer than two times or all its times
 * equal, which leaves its variance at 0 and the likelihood unbounded, or an option is invalid.
 * @see {@link https://doi.org/10.3102/10769986031002181 | van der Linden (2006)}
 */
export function calibrateResponseTimes(
  data: ResponseMatrix | ResponseRecords,
  options: ResponseTimeCalibrationOptions = {},
): ResponseTimeCalibration {
  checkOptions(options, RESPONSE_TIME_CALIBRATION_OPTIONS);
  const { tolerance = DEFAULT_TOLERANCE, maxIterations = DEFAULT_MAX_ITERATIONS } = options;
  checkStopping(tolerance, maxIterations);
  const timed = timedResponses(data, "Response time calibration");
  const itemCount = timed.itemIds.length;
  const counts = new Float64Array(itemCount);
  const firstTimes = new Float64Array(itemCount);
  const varies = new Uint8Array(itemCount);
  forEachTime(timed, (_, item, logTime) => {
    counts[item] = (counts[item] ?? 0) + 1;
    if (counts[item] === 1) firstTimes[item] = logTime;
    else if (logTime !== firstTimes[item]) varies[item] = 1;
  });
  counts.forEach((count, item) => {
    if (count < 2) throw new DataError(`Item "${timed.itemIds[item]}" needs at least two response times.`);
    if (varies[item] !== 1) throw new DataError(`Item "${timed.itemIds[item]}" needs response times that differ.`);
  });
  const state = startingValues(timed, counts);
  const previous = new Float64Array(state.values.length);
  const { converged, iterations } = iterateToConvergence(
    () => {
      previous.set(state.values);
      emCycle(timed, state, counts);
      return largestChange(previous, state.values);
    },
    { tolerance, maxIterations },
  );
  const { intensity, discrimination } = state;
  const speedSd = state.values[2 * itemCount] ?? 1;
  const persons = new Set<number>();
  forEachTime(timed, (person) => persons.add(person));
  const logLikelihood = marginalLogLikelihood(timed, state);
  // The intensities' information holds every pair of the items, 32 MB at the bound, and its inversion takes their cube.
  const errors = withinStandardErrorSize(itemCount) ? intensityErrors(timed, state) : null;
  const parameters = 2 * itemCount + 1;
  return {
    items: timed.itemIds.map((id, i) => ({
      id,
      intensity: intensity[i] ?? 0,
      discrimination: discrimination[i] ?? 0,
      standardError: errors === null ? null : Math.sqrt(errors[i * itemCount + i] ?? 0),
    })),
    speedSd,
    logLikelihood,
    aic: -2 * logLikelihood + 2 * parameters,
    bic: -2 * logLikelihood + parameters * Math.log(persons.size),
    converged,
    iterations,
  };
}

/** The parameters during EM: intensities, discriminations and σ_τ, in one array for the convergence check. */
interface LognormalState {
  /** β, α and σ_τ, in this order. */
  readonly values: Float64Array;
  readonly intensity: Float64Array;
  readonly discrimination: Float64Array;
}

/** Starts at each item's mean and SD of the log times, and at the starting σ_τ. */
function startingValues(timed: TimedResponses, counts: Float64Array): LognormalState {
  const itemCount = counts.length;
  const values = new Float64Array(2 * itemCount + 1);
  const intensity = values.subarray(0, itemCount);
  const discrimination = values.subarray(itemCount, 2 * itemCount);
  const squares = new Float64Array(itemCount);
  forEachTime(timed, (_, item, logTime) => {
    intensity[item] = (intensity[item] ?? 0) + logTime / (counts[item] ?? 1);
  });
  forEachTime(timed, (_, item, logTime) => {
    squares[item] = (squares[item] ?? 0) + (logTime - (intensity[item] ?? 0)) ** 2;
  });
  squares.forEach((square, item) => {
    discrimination[item] = Math.sqrt((counts[item] ?? 1) / square);
  });
  values[2 * itemCount] = START_SPEED_SD;
  return { values, intensity, discrimination };
}

/**
 * Runs one EM iteration. E-step: a person's τ is normal with precision P = 1 / σ_τ² + Σ α_i² and mean
 * Σ α_i² (β_i - y_i) / P over the person's log times y. M-step: β_i the mean of y + E(τ), 1 / α_i² the mean of
 * (y - β_i + E(τ))² + Var(τ), σ_τ² the mean of E(τ)² + Var(τ).
 */
function emCycle(timed: TimedResponses, state: LognormalState, counts: Float64Array): void {
  const { values, intensity, discrimination } = state;
  const itemCount = counts.length;
  const variance = (values[2 * itemCount] ?? 1) ** 2;
  const means = new Float64Array(timed.personIds.length);
  const variances = new Float64Array(timed.personIds.length);
  posteriors(timed, intensity, discrimination, variance, means, variances);
  const sums = new Float64Array(itemCount);
  forEachTime(timed, (person, item, logTime) => {
    sums[item] = (sums[item] ?? 0) + logTime + (means[person] ?? 0);
  });
  sums.forEach((sum, item) => {
    intensity[item] = sum / (counts[item] ?? 1);
  });
  const squares = new Float64Array(itemCount);
  forEachTime(timed, (person, item, logTime) => {
    squares[item] =
      (squares[item] ?? 0) + (logTime - (intensity[item] ?? 0) + (means[person] ?? 0)) ** 2 + (variances[person] ?? 0);
  });
  squares.forEach((square, item) => {
    discrimination[item] = Math.sqrt((counts[item] ?? 1) / square);
  });
  let speedSquares = 0;
  let persons = 0;
  variances.forEach((personVariance, person) => {
    if (personVariance > 0) {
      speedSquares += (means[person] ?? 0) ** 2 + personVariance;
      persons++;
    }
  });
  values[2 * itemCount] = Math.sqrt(speedSquares / persons);
}

/**
 * Writes each person's posterior mean and variance of τ under N(0, `variance`), or 0 and 0 for a person without times;
 * with `variance` infinite, the maximum likelihood estimate and its sampling variance.
 */
export function posteriors(
  timed: TimedResponses,
  intensity: Float64Array,
  discrimination: Float64Array,
  variance: number,
  means: Float64Array,
  variances: Float64Array,
): void {
  const precisions = new Float64Array(means.length);
  means.fill(0);
  forEachTime(timed, (person, item, logTime) => {
    const weight = (discrimination[item] ?? 0) ** 2;
    precisions[person] = (precisions[person] ?? 0) + weight;
    means[person] = (means[person] ?? 0) + weight * ((intensity[item] ?? 0) - logTime);
  });
  precisions.forEach((precision, person) => {
    if (precision === 0) {
      variances[person] = 0;
      return;
    }
    const total = precision + 1 / variance;
    means[person] = (means[person] ?? 0) / total;
    variances[person] = 1 / total;
  });
}

/**
 * Returns the marginal log-likelihood of the log times: each person's y ~ N(β, D + σ_τ² 11ᵀ), D = diag(1 / α²), with
 * the determinant by the matrix determinant lemma and the inverse by Sherman and Morrison.
 */
function marginalLogLikelihood(timed: TimedResponses, state: LognormalState): number {
  const { values, intensity, discrimination } = state;
  const variance = (values[intensity.length * 2] ?? 1) ** 2;
  const personCount = timed.personIds.length;
  const count = new Float64Array(personCount);
  const logDeterminant = new Float64Array(personCount);
  const weights = new Float64Array(personCount);
  const quadratic = new Float64Array(personCount);
  const weighted = new Float64Array(personCount);
  forEachTime(timed, (person, item, logTime) => {
    const precision = (discrimination[item] ?? 0) ** 2;
    const residual = logTime - (intensity[item] ?? 0);
    count[person] = (count[person] ?? 0) + 1;
    logDeterminant[person] = (logDeterminant[person] ?? 0) - Math.log(precision);
    weights[person] = (weights[person] ?? 0) + precision;
    quadratic[person] = (quadratic[person] ?? 0) + precision * residual * residual;
    weighted[person] = (weighted[person] ?? 0) + precision * residual;
  });
  let total = 0;
  count.forEach((n, person) => {
    if (n === 0) return;
    const shrink = 1 + variance * (weights[person] ?? 0);
    const form = (quadratic[person] ?? 0) - (variance * (weighted[person] ?? 0) ** 2) / shrink;
    total -= 0.5 * (n * Math.log(2 * Math.PI) + (logDeterminant[person] ?? 0) + Math.log(shrink) + form);
  });
  return total;
}

/**
 * Returns the covariance matrix of the intensities, by rows: the inverse of Σ_j (D_j⁻¹ - σ_τ² a_j a_jᵀ / (1 + σ_τ²
 * Σ α²)), a_j the α² of person j's items; `null` when it is not positive definite.
 */
function intensityErrors(timed: TimedResponses, state: LognormalState): Float64Array | null {
  const { values, intensity, discrimination } = state;
  const itemCount = intensity.length;
  const variance = (values[2 * itemCount] ?? 1) ** 2;
  const information = new Float64Array(itemCount * itemCount);
  const { offsets, items, logTimes } = timed;
  for (let person = 0; person + 1 < offsets.length; person++) {
    const answered: number[] = [];
    for (let r = offsets[person] ?? 0; r < (offsets[person + 1] ?? 0); r++) {
      if (!Number.isNaN(logTimes[r] ?? Number.NaN)) answered.push(items[r] ?? 0);
    }
    const weights = answered.map((item) => (discrimination[item] ?? 0) ** 2);
    const shrink = variance / (1 + variance * weights.reduce((sum, weight) => sum + weight, 0));
    answered.forEach((i, a) => {
      information[i * itemCount + i] = (information[i * itemCount + i] ?? 0) + (weights[a] ?? 0);
      answered.forEach((j, b) => {
        information[i * itemCount + j] =
          (information[i * itemCount + j] ?? 0) - shrink * (weights[a] ?? 0) * (weights[b] ?? 0);
      });
    });
  }
  return invertPositiveDefinite(information, itemCount);
}

import {
  checkArray,
  checkFinite,
  checkOptions,
  checkPositive,
  checkRecord,
  DataError,
  describe,
  normalizeLogs,
  type ThetaDistribution,
} from "@measurekit/core";
import { findRoot, type Quadrature } from "../math/internal.js";
import { answeredItems, checkPattern, indexItems, largestCategoryCount } from "./items.js";
import type { IrtItem } from "./model.js";
import { logProbability, MIN_PROBABILITY } from "./probability.js";
import { type QuadratureOptions, quadratureRule } from "./quadrature.js";

/** An estimate of a person's theta with its standard error. */
export interface AbilityEstimate {
  /** The estimate of theta. */
  readonly theta: number;
  /** Its standard error: `Infinity` when the items give no information. */
  readonly standardError: number;
}

/** The interval that maximum likelihood, weighted likelihood and MAP estimation search. */
export interface RangeOptions {
  /**
   * The lowest and the highest theta.
   *
   * @defaultValue `[-6, 6]`
   */
  readonly range?: readonly [lower: number, upper: number];
}

/** A normal prior distribution of theta. */
export interface ThetaPrior {
  /** The mean. */
  readonly mean: number;
  /** The standard deviation. */
  readonly sd: number;
}

/** Options of {@link posteriorModeEstimate}. */
export interface PosteriorModeOptions extends RangeOptions {
  /**
   * The prior of theta.
   *
   * @defaultValue `{ mean: 0, sd: 1 }`
   */
  readonly prior?: ThetaPrior;
}

/**
 * A testlet in ability estimation: the ids of its items and the SD of its effect, as a calibration's `testlets` gives
 * them.
 */
export interface TestletEffect {
  /**
   * The ids of its items. Ids that none of the items estimated from has, such as those of the items a person did not
   * answer, which `abilityEstimates` leaves out, are skipped.
   */
  readonly items: readonly string[];
  /** The standard deviation of the testlet effect γ, as calibration estimates it. */
  readonly sd: number;
}

/** How an integral over theta is taken: under a normal prior, by a quadrature rule placed on it. */
export interface ThetaIntegrationOptions {
  /**
   * The prior of theta.
   *
   * @defaultValue `{ mean: 0, sd: 1 }`
   */
  readonly prior?: ThetaPrior;
  /**
   * The quadrature rule, placed on the prior: nodes `mean + sd * z`.
   *
   * @defaultValue 61 rectangular points
   */
  readonly quadrature?: QuadratureOptions;
}

/** Options of {@link posteriorMeanEstimate}. */
export interface PosteriorMeanOptions extends ThetaIntegrationOptions {
  /**
   * Testlets, whose effects the estimate integrates out (the testlet response model), each over the quadrature rule
   * placed on N(0, sd²).
   *
   * @defaultValue `[]`
   */
  readonly testlets?: readonly TestletEffect[];
}

/**
 * A function that estimates theta from items and one response per item, `null` when missing.
 *
 * @remarks
 * One of `maximumLikelihoodEstimate`, `weightedLikelihoodEstimate`, `posteriorModeEstimate` and
 * `posteriorMeanEstimate`, one of them with options, such as `(items, responses) => posteriorModeEstimate(items,
 * responses, { prior })`, or `untilMixed` of two of them.
 */
export type AbilityEstimator = (items: readonly IrtItem[], responses: readonly (number | null)[]) => AbilityEstimate;

/** The interval of theta that estimation and information peaks search. */
export const DEFAULT_THETA_RANGE = [-6, 6] as const;
const STANDARD_PRIOR: ThetaPrior = { mean: 0, sd: 1 };
/** Spacing of the grid on which modal estimation looks for maxima before refining them, unless a caller asks for finer. */
const SCAN_STEP = 0.25;

/**
 * Returns the maximum likelihood estimate (MLE) of theta within `range`, with the standard error 1 / √I(θ) from
 * the test information of the answered items. Missing responses are skipped. When every response is in its item's
 * lowest or highest category, the likelihood grows toward -∞ or +∞ and the estimate is that end of `range`.
 *
 * @param items - The items the person answered, with their parameters.
 * @param responses - The person's response to each item; `null` when missing.
 * @param options - The range of theta to search.
 * @returns The estimate and its standard error.
 * @throws `DataError` when the items, responses or range are invalid, or no item was answered.
 */
export function maximumLikelihoodEstimate(
  items: readonly IrtItem[],
  responses: readonly (number | null)[],
  options: RangeOptions = {},
): AbilityEstimate {
  checkOptions(options);
  const pattern = answered(items, responses, "Maximum likelihood");
  const theta = modalTheta((t) => pattern.slope(t), checkRange(options.range));
  return { theta, standardError: 1 / Math.sqrt(pattern.information(theta)) };
}

/**
 * Returns Warm's weighted likelihood estimate (WLE) of theta within `range`.
 *
 * @remarks
 * The root of d log L / dθ + J(θ) / (2 I(θ)), where I is the test information and J the sum over items and categories
 * of P'(θ) P''(θ) / P(θ); of several roots, the one with the largest weighted likelihood. The standard error is 1 /
 * √I(θ). Missing responses are skipped. The WLE is finite for the 1PL, the 2PL and the polytomous models when every
 * response is in the lowest or highest category; where no root exists, the estimate is the end of `range` toward which
 * the weighted likelihood grows.
 *
 * @param items - The items the person answered, with their parameters.
 * @param responses - The person's response to each item; `null` when missing.
 * @param options - The range of theta to search.
 * @returns The estimate and its standard error.
 * @throws `DataError` when the items, responses or range are invalid, or no item was answered.
 */
export function weightedLikelihoodEstimate(
  items: readonly IrtItem[],
  responses: readonly (number | null)[],
  options: RangeOptions = {},
): AbilityEstimate {
  checkOptions(options);
  const pattern = answered(items, responses, "Weighted likelihood");
  const theta = modalTheta((t) => pattern.slope(t) + pattern.warmCorrection(t), checkRange(options.range));
  return { theta, standardError: 1 / Math.sqrt(pattern.information(theta)) };
}

/**
 * Returns the maximum a posteriori estimate (MAP, Bayes modal) of theta under a normal prior, within `range`, with
 * the standard error 1 / √(I(θ) + 1 / sd²). Missing responses are skipped; without responses the estimate is the
 * prior mean.
 *
 * @param items - The items the person answered, with their parameters.
 * @param responses - The person's response to each item; `null` when missing.
 * @param options - The prior and the range of theta to search.
 * @returns The estimate and its standard error.
 * @throws `DataError` when the items, responses, range or prior are invalid.
 */
export function posteriorModeEstimate(
  items: readonly IrtItem[],
  responses: readonly (number | null)[],
  options: PosteriorModeOptions = {},
): AbilityEstimate {
  checkOptions(options);
  const pattern = answered(items, responses);
  const { mean, sd } = checkThetaPrior(options.prior);
  const precision = 1 / (sd * sd);
  const theta = modalTheta((t) => pattern.slope(t) + (mean - t) * precision, checkRange(options.range));
  return { theta, standardError: 1 / Math.sqrt(pattern.information(theta) + precision) };
}

/**
 * Returns the expected a posteriori estimate (EAP) of theta under a normal prior.
 *
 * @remarks
 * The posterior mean, with the posterior standard deviation as its standard error, by quadrature. With testlets, an
 * item of a testlet answers at θ + γ, γ ~ N(0, σ²) per testlet, and the likelihood integrates each testlet's γ out at
 * each theta, Σ_g w_g Π_j P_j(x_j | θ + σ z_g); testlets name their items by id. Missing responses are skipped; without
 * responses the estimate is the prior.
 *
 * @param items - The items the person answered, with their parameters.
 * @param responses - The person's response to each item; `null` when missing.
 * @param options - The testlets, the prior and the quadrature.
 * @returns The estimate and its standard error, the standard deviation of the posterior.
 * @throws `DataError` when the items, responses, prior or quadrature are invalid; with testlets, when an item has no id
 * or item ids repeat, a testlet names an id that is not a string or an item that is in a testlet already, or its sd is
 * not positive and finite.
 */
export function posteriorMeanEstimate(
  items: readonly IrtItem[],
  responses: readonly (number | null)[],
  options: PosteriorMeanOptions = {},
): AbilityEstimate {
  checkOptions(options);
  const pattern = answered(items, responses);
  const { testlets = [] } = options;
  const testletOf = testletMembership(items, testlets);
  const { mean, sd } = checkThetaPrior(options.prior);
  const { nodes, weights } = quadratureRule(options.quadrature, mean, sd);
  const logLikelihood =
    testlets.length === 0
      ? (theta: number): number => pattern.logLikelihood(theta)
      : testletLogLikelihood(items, responses, testlets, testletOf, quadratureRule(options.quadrature));
  return posteriorMoments(nodes, posteriorWeights(nodes, weights, logLikelihood));
}

/** Returns the mean of a distribution on the nodes, with its standard deviation as the standard error. */
export function posteriorMoments(nodes: Float64Array, posterior: Float64Array): AbilityEstimate {
  let theta = 0;
  for (let q = 0; q < nodes.length; q++) theta += (posterior[q] ?? 0) * (nodes[q] ?? 0);
  let variance = 0;
  for (let q = 0; q < nodes.length; q++) variance += (posterior[q] ?? 0) * ((nodes[q] ?? 0) - theta) ** 2;
  return { theta, standardError: Math.sqrt(variance) };
}

/**
 * Returns the posterior distribution of theta on the quadrature nodes placed on the prior.
 *
 * @remarks
 * The weight of the node θ_q is w_q L(θ_q) / Σ w L, with w_q the prior's quadrature weight and L the likelihood of the
 * answered responses; the nodes are `mean + sd * z_q`, as for `posteriorMeanEstimate`, whose estimate is the mean of
 * this distribution. Missing responses are skipped; without responses the posterior is the prior.
 *
 * @param items - The items the person answered, with their parameters.
 * @param responses - The person's response to each item; `null` when missing.
 * @param options - The prior and the quadrature.
 * @returns The nodes and their posterior weights.
 * @throws `DataError` when the items, responses, prior or quadrature are invalid.
 */
export function thetaPosterior(
  items: readonly IrtItem[],
  responses: readonly (number | null)[],
  options: ThetaIntegrationOptions = {},
): ThetaDistribution {
  checkOptions(options);
  const pattern = answered(items, responses);
  const { mean, sd } = checkThetaPrior(options.prior);
  const { nodes, weights } = quadratureRule(options.quadrature, mean, sd);
  const posterior = posteriorWeights(nodes, weights, (theta) => pattern.logLikelihood(theta));
  return { nodes: Array.from(nodes), weights: Array.from(posterior) };
}

/** Returns w_q L(θ_q) at each node, normalized to sum to 1, from the prior weights w and the log-likelihood ln L. */
function posteriorWeights(
  nodes: Float64Array,
  weights: Float64Array,
  logLikelihood: (theta: number) => number,
): Float64Array {
  const posterior = Float64Array.from(nodes, (theta, q) => Math.log(weights[q] ?? 0) + logLikelihood(theta));
  normalizeLogs(posterior);
  return posterior;
}

/**
 * Checks testlets and returns each item's testlet index, -1 for an item in none; items are matched by id.
 *
 * @throws `DataError` when an item has no id or item ids repeat, a testlet names an id that is not a string or an item
 * that is in a testlet already, or its sd is not positive and finite.
 */
function testletMembership(items: readonly IrtItem[], testlets: readonly TestletEffect[]): Int32Array {
  checkArray(testlets, "The option testlets must be an array of testlets");
  if (testlets.length === 0) return new Int32Array(items.length).fill(-1);
  const indices = indexItems(items);
  const testletOf = new Int32Array(items.length).fill(-1);
  testlets.forEach((testlet, t) => {
    checkRecord(testlet, `Testlet ${t + 1} must be an object { items, sd }`);
    const { items: ids, sd } = testlet;
    checkPositive(sd, `The sd of testlet ${t + 1}`);
    checkArray(ids, `The items of testlet ${t + 1} must be an array of item ids`);
    for (const id of ids) {
      if (typeof id !== "string") throw new DataError(`Testlet ${t + 1} needs item ids, found ${describe(id)}.`);
      const index = indices.get(id);
      if (index === undefined) continue;
      if ((testletOf[index] ?? -1) >= 0) throw new DataError(`Item "${id}" is in a testlet twice.`);
      testletOf[index] = t;
    }
  });
  return testletOf;
}

/**
 * Returns the log-likelihood of theta with the testlets' effects integrated out: that of the items in no testlet plus,
 * for each testlet, ln Σ_g w_g L(θ + σ z_g) of its responses over the standard rule.
 */
function testletLogLikelihood(
  items: readonly IrtItem[],
  responses: readonly (number | null)[],
  testlets: readonly TestletEffect[],
  testletOf: Int32Array,
  standard: Quadrature,
): (theta: number) => number {
  const free = answered(
    items,
    responses.map((response, i) => ((testletOf[i] ?? -1) >= 0 ? null : response)),
  );
  const testletPatterns = testlets.map((testlet, t) => {
    const members = items.flatMap((_, i) => (testletOf[i] === t ? [i] : []));
    return {
      sd: testlet.sd,
      pattern: answered(
        members.flatMap((i) => items[i] ?? []),
        members.map((i) => responses[i] ?? null),
      ),
    };
  });
  const values = new Float64Array(standard.nodes.length);
  return (theta) => {
    let sum = free.logLikelihood(theta);
    for (const testlet of testletPatterns) {
      sum += integratedLogLikelihood(testlet.pattern, theta, testlet.sd, standard, values);
    }
    return sum;
  };
}

/**
 * Returns ln Σ_g w_g exp(ln L(θ + σ z_g)) of a testlet's responses, over the nodes z and weights w of the rule.
 * `values` is scratch space with one entry per node.
 */
function integratedLogLikelihood(
  pattern: AnsweredPattern,
  theta: number,
  sd: number,
  { nodes, weights }: Quadrature,
  values: Float64Array,
): number {
  let largest = Number.NEGATIVE_INFINITY;
  for (let g = 0; g < nodes.length; g++) {
    const value = Math.log(weights[g] ?? 0) + pattern.logLikelihood(theta + sd * (nodes[g] ?? 0));
    values[g] = value;
    largest = Math.max(largest, value);
  }
  let sum = 0;
  for (const value of values) sum += Math.exp(value - largest);
  return largest + Math.log(sum);
}

/** The likelihood of a response pattern and its derivatives, over the answered items. */
interface AnsweredPattern {
  /** Returns log L(θ). */
  logLikelihood(theta: number): number;
  /** Returns d log L / dθ. */
  slope(theta: number): number;
  /** Returns Warm's correction J(θ) / (2 I(θ)), or 0 where I(θ) is 0. */
  warmCorrection(theta: number): number;
  /** Returns the test information I(θ) of the answered items. */
  information(theta: number): number;
}

/**
 * Checks a response pattern and returns its likelihood over the answered items. With `method`, at least one item
 * must have been answered.
 */
function answered(items: readonly IrtItem[], responses: readonly (number | null)[], method?: string): AnsweredPattern {
  checkPattern(items, responses);
  const answers = answeredItems(items, responses);
  if (method !== undefined && answers.length === 0) {
    throw new DataError(`${method} estimation needs at least one response.`);
  }
  const size = largestCategoryCount(items);
  const probabilities = new Float64Array(size);
  const first = new Float64Array(size);
  const second = new Float64Array(size);
  return {
    logLikelihood(theta) {
      let sum = 0;
      for (const { item, response } of answers) {
        item.model.probabilities(item.parameters, theta, probabilities);
        sum += logProbability(probabilities[response] ?? 0);
      }
      return sum;
    },
    slope(theta) {
      let sum = 0;
      for (const { item, response } of answers) {
        item.model.probabilities(item.parameters, theta, probabilities);
        item.model.thetaDerivatives(item.parameters, theta, first, second);
        sum += (first[response] ?? 0) / Math.max(probabilities[response] ?? 0, MIN_PROBABILITY);
      }
      return sum;
    },
    warmCorrection(theta) {
      let j = 0;
      let information = 0;
      for (const { item } of answers) {
        const count = item.model.categoryCount(item.parameters);
        item.model.probabilities(item.parameters, theta, probabilities);
        item.model.thetaDerivatives(item.parameters, theta, first, second);
        for (let k = 0; k < count; k++) {
          j += ((first[k] ?? 0) * (second[k] ?? 0)) / Math.max(probabilities[k] ?? 0, MIN_PROBABILITY);
        }
        information += item.model.information(item.parameters, theta);
      }
      return information > 0 ? j / (2 * information) : 0;
    },
    information(theta) {
      let sum = 0;
      for (const { item } of answers) sum += item.model.information(item.parameters, theta);
      return sum;
    },
  };
}

/**
 * Returns the theta in [lower, upper] at which a function is largest, given its derivative `slope`.
 *
 * @remarks
 * The slope is evaluated on a grid of spacing `step`; each change of sign from positive to negative brackets a local
 * maximum, refined by root finding. A maximum and the minimum beside it that lie within one step of each other are not
 * seen, so a function whose maxima can be that close, such as a polytomous item's information, needs a finer step.
 * The ends are candidates where the function decreases into the interval. Candidates are compared by `objective`, the
 * function itself, when it is given; otherwise by the integral of the slope from `lower` by the trapezoid rule, so that
 * maxima can be compared when only the slope is known, as for the weighted likelihood.
 */
export function modalTheta(
  slope: (theta: number) => number,
  [lower, upper]: readonly [number, number],
  objective?: (theta: number) => number,
  step = SCAN_STEP,
): number {
  const valueAt = (theta: number, integral: number): number => (objective === undefined ? integral : objective(theta));
  const intervals = Math.ceil((upper - lower) / step);
  let previousTheta = lower;
  let previousSlope = slope(lower);
  let value = 0;
  let best = lower;
  let bestValue = previousSlope <= 0 ? valueAt(lower, 0) : Number.NEGATIVE_INFINITY;
  for (let j = 1; j <= intervals; j++) {
    const theta = j === intervals ? upper : lower + ((upper - lower) * j) / intervals;
    const thetaSlope = slope(theta);
    if (previousSlope > 0 && thetaSlope <= 0) {
      const root = thetaSlope === 0 ? theta : findRoot(slope, previousTheta, theta, previousSlope, thetaSlope);
      const rootValue = valueAt(root, value + (previousSlope * (root - previousTheta)) / 2);
      if (rootValue > bestValue) {
        best = root;
        bestValue = rootValue;
      }
    }
    value += ((previousSlope + thetaSlope) * (theta - previousTheta)) / 2;
    previousTheta = theta;
    previousSlope = thetaSlope;
  }
  return previousSlope > 0 && valueAt(upper, value) > bestValue ? upper : best;
}

/**
 * Checks the interval of theta that the estimators search, `[-6, 6]` by default, and returns it.
 *
 * @throws `DataError` unless it is an array of two finite numbers with the lower below the upper.
 */
export function checkRange(range: readonly [number, number] = DEFAULT_THETA_RANGE): readonly [number, number] {
  checkArray(range, "The option range must be an array [lower, upper]");
  const [lower, upper] = range;
  checkFinite(lower, "The lower bound of the range");
  checkFinite(upper, "The upper bound of the range");
  if (!(lower < upper)) {
    throw new DataError(`The range needs the lower bound below the upper, found [${lower}, ${upper}].`);
  }
  return range;
}

/**
 * Checks a normal prior of theta, the standard normal by default, and returns it.
 *
 * @throws `DataError` unless the prior is an object with a finite mean and a positive finite sd.
 */
export function checkThetaPrior(prior: ThetaPrior = STANDARD_PRIOR): ThetaPrior {
  checkRecord(prior, "The option prior must be an object { mean, sd }");
  checkFinite(prior.mean, "The mean of the prior");
  checkPositive(prior.sd, "The sd of the prior");
  return prior;
}

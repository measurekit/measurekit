import { checkOptions, DataError, type FieldNames } from "@measurekit/core";
import {
  checkThetaPrior,
  type IrtItem,
  largestCategoryCount,
  logProbability,
  type ThetaPrior,
} from "../irt/internal.js";
import type { PoolItem } from "./items.js";
import type { AdaptiveTestContext, ItemSelectionRule } from "./selection.js";

// catR's default integration grid, parInt = c(-4, 4, 33): equally spaced nodes, which the trapezoid rule weights.
const GRID_LOWER = -4;
const GRID_UPPER = 4;
/** The number of nodes of the grid. */
export const GRID_POINTS = 33;
const GRID_STEP = (GRID_UPPER - GRID_LOWER) / (GRID_POINTS - 1);
/** The nodes of the grid. */
export const GRID_NODES = Float64Array.from({ length: GRID_POINTS }, (_, q) => GRID_LOWER + q * GRID_STEP);
const TRAPEZOID = Float64Array.from({ length: GRID_POINTS }, (_, q) =>
  q === 0 || q === GRID_POINTS - 1 ? GRID_STEP / 2 : GRID_STEP,
);

/** Options of the selection rules that weight theta by the posterior. */
export interface WeightedRuleOptions {
  /**
   * The normal prior of theta, whose density weights the grid.
   *
   * @defaultValue `{ mean: 0, sd: 1 }`
   */
  readonly prior?: ThetaPrior;
}

const WEIGHTED_RULE_OPTIONS: FieldNames<WeightedRuleOptions> = { prior: true };

/** ln √(2π), of the normal density. */
const LOG_SQRT_TWO_PI = 0.5 * Math.log(2 * Math.PI);

/** How a rule weights the nodes of the grid. */
interface GridWeighting {
  /** The logarithm of the prior's density at each node, or `null` for the likelihood alone. */
  readonly prior: Float64Array | null;
  /** Whether the trapezoid rule's weights enter, for an integral, or not, for a sum. */
  readonly trapezoid: boolean;
}

/**
 * Returns the logarithm of the density of the checked prior of the options at each node of the grid.
 *
 * @throws `DataError` when the options are not an object or the prior is not `{ mean, sd }` with a finite mean and a
 * positive sd.
 */
export function priorOnGrid(options: WeightedRuleOptions): Float64Array {
  checkOptions(options, WEIGHTED_RULE_OPTIONS);
  const { mean, sd } = checkThetaPrior(options.prior);
  return GRID_NODES.map((theta) => -0.5 * ((theta - mean) / sd) ** 2 - Math.log(sd) - LOG_SQRT_TWO_PI);
}

/**
 * Returns the weight of each node of the grid after the responses so far: the likelihood of the responses, times the
 * prior's density where the weighting has a prior, scaled to a maximum of 1, which leaves the order of the items
 * unchanged and keeps a weight from underflowing at every node; then times the trapezoid weight where the weighting asks
 * for it. `probabilities` is scratch space for the largest item of the pool.
 */
export function gridWeights(
  { pool, administered, responses }: AdaptiveTestContext,
  { prior, trapezoid }: GridWeighting,
  probabilities: Float64Array,
): Float64Array {
  const logWeights = new Float64Array(GRID_POINTS);
  administered.forEach((index, j) => {
    const item = pool[index];
    const response = responses[j] ?? 0;
    if (item === undefined) return;
    for (let q = 0; q < GRID_POINTS; q++) {
      item.model.probabilities(item.parameters, GRID_NODES[q] ?? 0, probabilities);
      logWeights[q] = (logWeights[q] ?? 0) + logProbability(probabilities[response] ?? 0);
    }
  });
  if (prior !== null) {
    for (let q = 0; q < GRID_POINTS; q++) logWeights[q] = (logWeights[q] ?? 0) + (prior[q] ?? 0);
  }
  const largest = logWeights.reduce((most, value) => Math.max(most, value), Number.NEGATIVE_INFINITY);
  return logWeights.map((value, q) => Math.exp(value - largest) * (trapezoid ? (TRAPEZOID[q] ?? 0) : 1));
}

/** Builds a rule that scores each eligible item by the sum of its information at the nodes times their weights. */
function weightedInformation(prior: Float64Array | null): ItemSelectionRule {
  return {
    scores(context, out) {
      const { pool, eligible } = context;
      const weights = gridWeights(context, { prior, trapezoid: true }, new Float64Array(largestCategoryCount(pool)));
      pool.forEach(({ model, parameters }, index) => {
        if (eligible[index] !== 1) return;
        let score = 0;
        for (let q = 0; q < GRID_POINTS; q++) {
          score += (weights[q] ?? 0) * model.information(parameters, GRID_NODES[q] ?? 0);
        }
        out[index] = score;
      });
    },
  };
}

/**
 * Maximum likelihood weighted information (MLWI), catR's `MLWI`: scores each item by its Fisher information integrated
 * over theta, weighted by the likelihood of the responses so far.
 *
 * @remarks
 * ∫ L(θ) I(θ) dθ by the trapezoid rule on 33 equally spaced points from -4 to 4, catR's default, with the likelihood
 * scaled to a maximum of 1 on them, which leaves the order of the items unchanged. Before the first response the
 * likelihood is 1, and the score is the item's information over the grid.
 *
 * @see {@link https://doi.org/10.3102/10769986022002203 | Veerkamp and Berger (1997)}
 */
export const likelihoodWeightedInformation: ItemSelectionRule = weightedInformation(null);

/**
 * Returns maximum posterior weighted information (MPWI), catR's `MPWI`: each item scored by its Fisher information
 * integrated over theta, weighted by the posterior of the responses so far.
 *
 * @remarks
 * ∫ L(θ) π(θ) I(θ) dθ, π the density of the normal prior, by the trapezoid rule on 33 equally spaced points from -4 to
 * 4, with L π scaled to a maximum of 1 on them, which leaves the order of the items unchanged.
 *
 * @param options - The prior.
 * @returns The selection rule.
 * @throws `DataError` when the options are not an object or the prior is not `{ mean, sd }` with a finite mean and a
 * positive sd.
 * @see {@link https://doi.org/10.1007/BF02294775 | van der Linden (1998)}
 */
export function posteriorWeightedInformation(options: WeightedRuleOptions = {}): ItemSelectionRule {
  return weightedInformation(priorOnGrid(options));
}

/**
 * Returns the minimum expected posterior variance rule (MEPV), catR's `MEPV`: each item scored by minus the variance of
 * the posterior after its response, expected over the item's categories at the current estimate.
 *
 * @remarks
 * The score is -Σ_k P_k(θ̂) Var(θ | the responses so far and k), where the posterior after a response k is the prior's
 * density times the likelihood times P_k(θ), and its mean and variance are integrals by the trapezoid rule on 33 equally
 * spaced points from -4 to 4, as catR's `eapEst` and `eapSem` with `parInt` take them. The highest score is the smallest
 * expected variance.
 *
 * @param options - The prior.
 * @returns The selection rule.
 * @throws `DataError` when the options are not an object or the prior is not `{ mean, sd }` with a finite mean and a
 * positive sd.
 * @see {@link https://doi.org/10.3102/10769986022002203 | Veerkamp and Berger (1997)}
 * @see {@link https://doi.org/10.1007/BF02294775 | van der Linden (1998)}
 */
export function expectedPosteriorVariance(options: WeightedRuleOptions = {}): ItemSelectionRule {
  const prior = priorOnGrid(options);
  return {
    scores(context, out) {
      const { pool, eligible, estimate } = context;
      const size = largestCategoryCount(pool);
      const probabilities = new Float64Array(size);
      const atNodes = new Float64Array(size * GRID_POINTS);
      const weights = gridWeights(context, { prior, trapezoid: true }, probabilities);
      pool.forEach((item, index) => {
        if (eligible[index] === 1) {
          out[index] = -expectedVariance(item, estimate.theta, weights, probabilities, atNodes);
        }
      });
    },
  };
}

/**
 * Returns Σ_k P_k(θ̂) Var(θ | k) of an item: the posterior variance after each of its responses, expected at the
 * estimate. `probabilities` and `atNodes`, the probability of each category at each node by category, are scratch space.
 */
function expectedVariance(
  { model, parameters }: IrtItem,
  theta: number,
  weights: Float64Array,
  probabilities: Float64Array,
  atNodes: Float64Array,
): number {
  const count = model.categoryCount(parameters);
  for (let q = 0; q < GRID_POINTS; q++) {
    model.probabilities(parameters, GRID_NODES[q] ?? 0, probabilities);
    for (let k = 0; k < count; k++) atNodes[k * GRID_POINTS + q] = probabilities[k] ?? 0;
  }
  model.probabilities(parameters, theta, probabilities);
  let expected = 0;
  for (let k = 0; k < count; k++) {
    expected += (probabilities[k] ?? 0) * posteriorVariance(weights, atNodes, k * GRID_POINTS);
  }
  return expected;
}

/**
 * Returns the variance of theta on the grid under the weights times the probabilities of a response, which start at
 * `offset` of `atNodes`: about the mean, in a second pass, as catR's `eapSem` takes it about `eapEst`. It is 0 where the
 * probabilities vanish at every node.
 */
function posteriorVariance(weights: Float64Array, atNodes: Float64Array, offset: number): number {
  let mass = 0;
  let first = 0;
  for (let q = 0; q < GRID_POINTS; q++) {
    const weight = (weights[q] ?? 0) * (atNodes[offset + q] ?? 0);
    mass += weight;
    first += weight * (GRID_NODES[q] ?? 0);
  }
  if (!(mass > 0)) return 0;
  const mean = first / mass;
  let second = 0;
  for (let q = 0; q < GRID_POINTS; q++) {
    second += (weights[q] ?? 0) * (atNodes[offset + q] ?? 0) * ((GRID_NODES[q] ?? 0) - mean) ** 2;
  }
  return second / mass;
}

/** Builds a rule that scores each eligible item by the variance of its probability of success over the weights. */
function discrimination(prior: Float64Array | null): ItemSelectionRule {
  return {
    scores(context, out) {
      const { pool, eligible } = context;
      const probabilities = new Float64Array(largestCategoryCount(pool));
      const weights = gridWeights(context, { prior, trapezoid: false }, probabilities);
      const total = weights.reduce((sum, weight) => sum + weight, 0);
      const success = new Float64Array(GRID_POINTS);
      pool.forEach((item, index) => {
        if (eligible[index] === 1) out[index] = successVariance(item, weights, total, probabilities, success);
      });
    },
  };
}

/**
 * Returns the variance of an item's probability of success over the nodes, each weighted by its weight over `total`.
 * `probabilities` and `success` are scratch space.
 *
 * @throws `DataError` when the item does not have two categories.
 */
function successVariance(
  item: PoolItem,
  weights: Float64Array,
  total: number,
  probabilities: Float64Array,
  success: Float64Array,
): number {
  const { model, parameters } = item;
  const count = model.categoryCount(parameters);
  if (count !== 2) {
    throw new DataError(`The discrimination index takes items of two categories; item "${item.id}" has ${count}.`);
  }
  let mean = 0;
  for (let q = 0; q < GRID_POINTS; q++) {
    model.probabilities(parameters, GRID_NODES[q] ?? 0, probabilities);
    success[q] = probabilities[1] ?? 0;
    mean += ((weights[q] ?? 0) / total) * (success[q] ?? 0);
  }
  let variance = 0;
  for (let q = 0; q < GRID_POINTS; q++) variance += ((weights[q] ?? 0) / total) * ((success[q] ?? 0) - mean) ** 2;
  return variance;
}

/**
 * The global discrimination index (GDI), catR's `GDI`: scores each item of two categories by the variance of its
 * probability of success over theta, weighted by the likelihood of the responses so far.
 *
 * @remarks
 * Σ_q w_q (P(θ_q) - P̄)², with w_q = L(θ_q) / Σ L on 33 equally spaced points from -4 to 4 and P̄ = Σ_q w_q P(θ_q): a
 * weighted sum, not the trapezoid rule, as catR's. It is the G-DINA discrimination index of diagnostic tests with
 * theta in place of the latent groups.
 *
 * @throws `DataError` from `scores` when an eligible item does not have two categories.
 * @see {@link https://doi.org/10.1177/0146621614554650 | Kaplan, de la Torre and Barrada (2015)}
 * @see {@link https://doi.org/10.18637/jss.v076.c01 | Magis and Barrada (2017)}
 */
export const discriminationIndex: ItemSelectionRule = discrimination(null);

/**
 * Returns the posterior global discrimination index (GDIP), catR's `GDIP`: each item of two categories scored by the
 * variance of its probability of success over theta, weighted by the posterior of the responses so far.
 *
 * @remarks
 * As {@link discriminationIndex}, with w_q ∝ L(θ_q) π(θ_q), π the density of the normal prior.
 *
 * @param options - The prior.
 * @returns The selection rule, whose `scores` throws `DataError` when an eligible item does not have two categories.
 * @throws `DataError` when the options are not an object or the prior is not `{ mean, sd }` with a finite mean and a
 * positive sd.
 * @see {@link https://doi.org/10.1177/0146621614554650 | Kaplan, de la Torre and Barrada (2015)}
 * @see {@link https://doi.org/10.18637/jss.v076.c01 | Magis and Barrada (2017)}
 */
export function posteriorDiscriminationIndex(options: WeightedRuleOptions = {}): ItemSelectionRule {
  return discrimination(priorOnGrid(options));
}

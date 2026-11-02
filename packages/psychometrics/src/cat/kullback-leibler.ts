import { type IrtItem, largestCategoryCount, MIN_PROBABILITY } from "../irt/internal.js";
import type { ItemSelectionRule } from "./selection.js";
import { GRID_NODES, GRID_POINTS, gridWeights, priorOnGrid, type WeightedRuleOptions } from "./weighted-rules.js";

/** Builds the KL rule, whose weights are the likelihood alone or, with a prior's log-density, the posterior. */
function divergenceRule(prior: Float64Array | null): ItemSelectionRule {
  return {
    scores(context, out) {
      const { pool, estimate, eligible } = context;
      const size = largestCategoryCount(pool);
      const atEstimate = new Float64Array(size);
      const atNode = new Float64Array(size);
      const weights = gridWeights(context, { prior, trapezoid: true }, atNode);
      pool.forEach((item, index) => {
        if (eligible[index] === 1) out[index] = divergence(item, estimate.theta, weights, atEstimate, atNode);
      });
    },
  };
}

/**
 * Returns Σ_q w_q Σ_k P_k(θ̂) ln(P_k(θ̂) / P_k(θ_q)) of an item over the grid. `atEstimate` and `atNode` are scratch
 * space.
 */
function divergence(
  { model, parameters }: IrtItem,
  theta: number,
  weights: Float64Array,
  atEstimate: Float64Array,
  atNode: Float64Array,
): number {
  const count = model.categoryCount(parameters);
  model.probabilities(parameters, theta, atEstimate);
  let score = 0;
  for (let q = 0; q < GRID_POINTS; q++) {
    model.probabilities(parameters, GRID_NODES[q] ?? 0, atNode);
    let sum = 0;
    for (let k = 0; k < count; k++) {
      const p = atEstimate[k] ?? 0;
      if (p > 0) sum += p * Math.log(p / Math.max(atNode[k] ?? 0, MIN_PROBABILITY));
    }
    score += (weights[q] ?? 0) * sum;
  }
  return score;
}

/**
 * Kullback-Leibler information (KL), catR's `KL`.
 *
 * @remarks
 * Scores each item by its Kullback-Leibler divergence Σ_k P_k(θ̂) ln(P_k(θ̂) / P_k(θ)) between the current estimate θ̂
 * and theta, integrated over theta weighted by the likelihood of the responses so far. The integral is the trapezoid
 * rule on 33 equally spaced points from -4 to 4, catR's default, with the likelihood scaled to a maximum of 1 on them,
 * which leaves the order of the items unchanged.
 */
export const kullbackLeibler: ItemSelectionRule = divergenceRule(null);

/**
 * Returns posterior Kullback-Leibler information (KLP), catR's `KLP`: each item scored by its Kullback-Leibler
 * divergence between the current estimate and theta, integrated over theta weighted by the posterior.
 *
 * @remarks
 * As {@link kullbackLeibler}, with the likelihood times the density of the normal prior as the weight, scaled to a
 * maximum of 1 on the grid.
 *
 * @param options - The prior.
 * @returns The selection rule.
 * @throws `DataError` when the options are not an object or the prior is not `{ mean, sd }` with a finite mean and a
 * positive sd.
 * @see {@link https://doi.org/10.1177/014662169602000303 | Chang and Ying (1996)}
 */
export function posteriorKullbackLeibler(options: WeightedRuleOptions = {}): ItemSelectionRule {
  return divergenceRule(priorOnGrid(options));
}

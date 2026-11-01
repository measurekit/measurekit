import { largestCategoryCount, logProbability, MIN_PROBABILITY } from "../irt/internal.js";
import type { PoolItem } from "./items.js";
import type { ItemSelectionRule } from "./selection.js";

// catR's default integration grid for KL, parInt = c(-4, 4, 33): equally spaced nodes, trapezoid weights.
const KL_LOWER = -4;
const KL_UPPER = 4;
const KL_POINTS = 33;
const KL_STEP = (KL_UPPER - KL_LOWER) / (KL_POINTS - 1);
const KL_NODES = Float64Array.from({ length: KL_POINTS }, (_, q) => KL_LOWER + q * KL_STEP);

/**
 * Kullback-Leibler information (KL), catR's `KL`.
 *
 * @remarks
 * Scores each item by its Kullback-Leibler divergence Σ_k P_k(θ̂) ln(P_k(θ̂) / P_k(θ)) between the current estimate θ̂
 * and theta, integrated over theta weighted by the likelihood of the responses so far. The integral is the trapezoid
 * rule on 33 equally spaced points from -4 to 4, catR's default, with the likelihood scaled to a maximum of 1 on them,
 * which leaves the order of the items unchanged.
 */
export const kullbackLeibler: ItemSelectionRule = {
  scores({ pool, administered, responses, estimate, eligible }, out) {
    const size = largestCategoryCount(pool);
    const atEstimate = new Float64Array(size);
    const atNode = new Float64Array(size);
    const weights = likelihoodWeights(pool, administered, responses, atNode);
    pool.forEach((item, index) => {
      if (eligible[index] !== 1) return;
      const { model, parameters } = item;
      const count = model.categoryCount(parameters);
      model.probabilities(parameters, estimate.theta, atEstimate);
      let score = 0;
      for (let q = 0; q < KL_POINTS; q++) {
        model.probabilities(parameters, KL_NODES[q] ?? 0, atNode);
        let divergence = 0;
        for (let k = 0; k < count; k++) {
          const p = atEstimate[k] ?? 0;
          if (p > 0) divergence += p * Math.log(p / Math.max(atNode[k] ?? 0, MIN_PROBABILITY));
        }
        score += (weights[q] ?? 0) * divergence;
      }
      out[index] = score;
    });
  },
};

/**
 * Returns the trapezoid weight of each node times the likelihood of the responses there, scaled to a maximum
 * likelihood of 1. `probabilities` is scratch space.
 */
function likelihoodWeights(
  pool: readonly PoolItem[],
  administered: readonly number[],
  responses: readonly number[],
  probabilities: Float64Array,
): Float64Array {
  const logLikelihood = new Float64Array(KL_POINTS);
  administered.forEach((index, j) => {
    const item = pool[index];
    const response = responses[j] ?? 0;
    if (item === undefined) return;
    for (let q = 0; q < KL_POINTS; q++) {
      item.model.probabilities(item.parameters, KL_NODES[q] ?? 0, probabilities);
      logLikelihood[q] = (logLikelihood[q] ?? 0) + logProbability(probabilities[response] ?? 0);
    }
  });
  const largest = logLikelihood.reduce((most, value) => Math.max(most, value), Number.NEGATIVE_INFINITY);
  return logLikelihood.map((value, q) => {
    const end = q === 0 || q === KL_POINTS - 1;
    return (end ? KL_STEP / 2 : KL_STEP) * Math.exp(value - largest);
  });
}

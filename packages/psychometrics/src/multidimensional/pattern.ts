import { MIN_PROBABILITY } from "../irt/probability.js";
import type { MultidimensionalItem } from "./model.js";

/** The likelihood of a response pattern and its derivatives, over the answered items. */
export interface AnsweredPattern {
  /** Returns ln L(θ). */
  logLikelihood(theta: Float64Array): number;
  /** Writes the gradient of ln L with respect to theta into `out`. */
  score(theta: Float64Array, out: Float64Array): void;
  /** Writes the test information matrix Σ I_i(θ) of the answered items into `out`, by rows. */
  information(theta: Float64Array, out: Float64Array): void;
}

/**
 * Returns the likelihood of the answered responses of checked items on `dimensions` dimensions. `responses` has one
 * response per item, `null` when missing.
 */
export function answeredPattern(
  items: readonly MultidimensionalItem[],
  responses: readonly (number | null)[],
  dimensions: number,
): AnsweredPattern {
  const answers = items.flatMap((item, index) => {
    const response = responses[index];
    return response === null || response === undefined ? [] : [{ item, response }];
  });
  const categories = answers.reduce((most, { item }) => Math.max(most, item.model.categoryCount(item.parameters)), 0);
  const probabilities = new Float64Array(categories);
  const gradients = new Float64Array(categories * dimensions);
  const matrix = new Float64Array(dimensions * dimensions);
  return {
    logLikelihood(theta) {
      let sum = 0;
      for (const { item, response } of answers) {
        item.model.probabilities(item.parameters, theta, probabilities);
        sum += Math.log(Math.max(probabilities[response] ?? 0, MIN_PROBABILITY));
      }
      return sum;
    },
    score(theta, out) {
      out.fill(0, 0, dimensions);
      for (const { item, response } of answers) {
        item.model.probabilities(item.parameters, theta, probabilities);
        item.model.thetaGradients(item.parameters, theta, gradients);
        const probability = Math.max(probabilities[response] ?? 0, MIN_PROBABILITY);
        for (let d = 0; d < dimensions; d++)
          out[d] = (out[d] ?? 0) + (gradients[response * dimensions + d] ?? 0) / probability;
      }
    },
    information(theta, out) {
      out.fill(0, 0, dimensions * dimensions);
      for (const { item } of answers) {
        item.model.information(item.parameters, theta, matrix);
        for (let k = 0; k < out.length; k++) out[k] = (out[k] ?? 0) + (matrix[k] ?? 0);
      }
    },
  };
}

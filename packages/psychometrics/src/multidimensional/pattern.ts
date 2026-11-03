import { answeredItems, largestCategoryCount, logProbability, MIN_PROBABILITY } from "../irt/internal.js";
import type { MultidimensionalItem } from "./model.js";

/** The likelihood of a response pattern and its derivatives, over the answered items. */
export interface AnsweredPattern {
  /** Returns ln L(θ). */
  logLikelihood(theta: Float64Array): number;
  /** Writes the gradient of ln L with respect to theta into `out`. */
  score(theta: Float64Array, out: Float64Array): void;
  /** Writes the test information matrix Σ I_i(θ) of the answered items into `out`, by rows. */
  information(theta: Float64Array, out: Float64Array): void;
  /**
   * Writes Warm's correction of the score into `out`, given the inverse of the test information at theta: entry r is
   * ½ Σ over the answered items and their categories of ∂P/∂θ_r tr(I⁻¹ ∇²P) / P, the second derivatives by central
   * differences of the gradients of the probabilities.
   */
  warmCorrection(theta: Float64Array, inverseInformation: Float64Array, out: Float64Array): void;
}

/** The relative step of the central differences of the gradients of the probabilities, about the cube root of epsilon. */
const STEP = 1e-5;

/**
 * Returns the likelihood of the answered responses of checked items on `dimensions` dimensions. `responses` has one
 * response per item, `null` when missing.
 */
export function answeredPattern(
  items: readonly MultidimensionalItem[],
  responses: readonly (number | null)[],
  dimensions: number,
): AnsweredPattern {
  const answers = answeredItems(items, responses);
  const categories = largestCategoryCount(items);
  const probabilities = new Float64Array(categories);
  const gradients = new Float64Array(categories * dimensions);
  const matrix = new Float64Array(dimensions * dimensions);
  const scratch: CurvatureScratch = {
    moved: new Float64Array(dimensions),
    up: new Float64Array(categories * dimensions),
    down: new Float64Array(categories * dimensions),
    traces: new Float64Array(categories),
  };
  return {
    logLikelihood(theta) {
      let sum = 0;
      for (const { item, response } of answers) {
        item.model.probabilities(item.parameters, theta, probabilities);
        sum += logProbability(probabilities[response] ?? 0);
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
    warmCorrection(theta, inverseInformation, out) {
      out.fill(0, 0, dimensions);
      for (const { item } of answers) {
        const count = item.model.categoryCount(item.parameters);
        item.model.probabilities(item.parameters, theta, probabilities);
        item.model.thetaGradients(item.parameters, theta, gradients);
        curvatureTraces(item, theta, inverseInformation, scratch);
        for (let k = 0; k < count; k++) {
          const probability = probabilities[k] ?? 0;
          // A category of probability 0 adds nothing: its gradient is 0 with it.
          if (probability > 0)
            addScaled(out, gradients, k * dimensions, (0.5 * (scratch.traces[k] ?? 0)) / probability);
        }
      }
    },
  };
}

/** The arrays that {@link curvatureTraces} works in. */
interface CurvatureScratch {
  readonly moved: Float64Array;
  readonly up: Float64Array;
  readonly down: Float64Array;
  /** tr(I⁻¹ ∇²P_k) of each category k. */
  readonly traces: Float64Array;
}

/**
 * Writes tr(I⁻¹ ∇²P_k) of each category k of an item at theta into `scratch.traces`, the second derivatives by central
 * differences of the gradients of its probabilities.
 */
function curvatureTraces(
  item: MultidimensionalItem,
  theta: Float64Array,
  inverseInformation: Float64Array,
  { moved, up, down, traces }: CurvatureScratch,
): void {
  const dimensions = theta.length;
  const count = item.model.categoryCount(item.parameters);
  traces.fill(0, 0, count);
  for (let u = 0; u < dimensions; u++) {
    const step = STEP * Math.max(1, Math.abs(theta[u] ?? 0));
    moved.set(theta);
    moved[u] = (theta[u] ?? 0) + step;
    item.model.thetaGradients(item.parameters, moved, up);
    moved[u] = (theta[u] ?? 0) - step;
    item.model.thetaGradients(item.parameters, moved, down);
    for (let k = 0; k < count; k++) {
      let sum = 0;
      for (let t = 0; t < dimensions; t++) {
        const second = ((up[k * dimensions + t] ?? 0) - (down[k * dimensions + t] ?? 0)) / (2 * step);
        sum += (inverseInformation[t * dimensions + u] ?? 0) * second;
      }
      traces[k] = (traces[k] ?? 0) + sum;
    }
  }
}

/** Adds `factor` times `values[from + r]` to `out[r]` for each r. */
function addScaled(out: Float64Array, values: Float64Array, from: number, factor: number): void {
  for (let r = 0; r < out.length; r++) out[r] = (out[r] ?? 0) + factor * (values[from + r] ?? 0);
}

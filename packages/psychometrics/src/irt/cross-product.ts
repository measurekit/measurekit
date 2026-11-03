import { invertPositiveDefinite } from "../math/internal.js";
import type { ExpectationStep } from "./expectation.js";
import type { Population } from "./latent.js";
import type { IrtModel } from "./model.js";
import type { ParameterPrior } from "./prior.js";
import { MIN_PROBABILITY } from "./probability.js";
import type { CompressedResponses } from "./responses.js";

/** The parts of a calibration that the cross-product information needs. */
interface ScoredCalibration {
  readonly items: readonly {
    readonly model: IrtModel;
    readonly parameters: unknown;
    readonly categoryCount: number;
    readonly estimated: boolean;
    /** The free parameter behind each vector entry. */
    readonly slots: Int32Array;
    /** The vector entries that a free parameter stands behind. */
    readonly freeEntries: Int32Array;
  }[];
  readonly population: Population;
  readonly free: readonly { readonly prior: ParameterPrior | undefined }[];
  readonly values: Float64Array;
}

/**
 * Returns the covariance matrix of the free parameters, by rows, from the cross-product information.
 *
 * @remarks
 * The information is Σ_p w_p s_p s_pᵀ over persons, with s_p the derivative of a person's log marginal likelihood, Σ_q
 * post_p(q) ∂ ln P(x_p | θ_q) / ∂ξ, and w_p the person's weight, 1 without person weights, plus minus the second
 * derivatives of the log priors, as mirt's `SE.type = "crossprod"`. `null` when the information is not positive
 * definite. Uses the posteriors of the E-step's last run.
 */
export function crossProductCovariance(
  calibration: ScoredCalibration,
  eStep: ExpectationStep,
  responses: CompressedResponses,
  nodes: Float64Array,
): Float64Array | null {
  const { items, population, free, values } = calibration;
  const size = values.length;
  const ratios = items.map((item) => (item.estimated ? derivativeRatios(item, nodes) : null));
  const latentSlopes = population.groups.map((group) => {
    if (group.distribution !== "normal") return null;
    return normalSlopes(group.logWeights, values[group.meanSlot] ?? 0, values[group.sdSlot] ?? 1, nodes);
  });
  const information = new Float64Array(size * size);
  const score = sparseScore(size);
  eStep.forEachPosterior((person, posterior) => {
    score.start(person);
    for (let r = responses.offsets[person] ?? 0; r < (responses.offsets[person + 1] ?? 0); r++) {
      const itemIndex = responses.items[r] ?? 0;
      addItemScore(score, items[itemIndex], ratios[itemIndex], responses.codes[r] ?? 0, posterior);
    }
    const g = population.groupOf[person] ?? 0;
    const group = population.groups[g];
    const slopes = latentSlopes[g];
    if (group !== undefined && slopes !== null && slopes !== undefined) {
      score.add(group.meanSlot, nodeSum(posterior, slopes.mean, 0, 1));
      score.add(group.sdSlot, nodeSum(posterior, slopes.sd, 0, 1));
    }
    score.addOuterProduct(information, population.weights?.[person] ?? 1);
  });
  free.forEach(({ prior }, j) => {
    if (prior !== undefined) {
      information[j * size + j] = (information[j * size + j] ?? 0) - prior.logDensitySecondDerivative(values[j] ?? 0);
    }
  });
  return invertPositiveDefinite(information, size);
}

/**
 * Adds the derivatives of ln P(x | θ) of an estimated item's response x with respect to its free parameters, averaged
 * over the posterior, to the score.
 */
function addItemScore(
  score: SparseScore,
  item: ScoredCalibration["items"][number] | undefined,
  ratios: Float64Array | null | undefined,
  response: number,
  posterior: Float64Array,
): void {
  if (item === undefined || ratios === null || ratios === undefined) return;
  const length = item.slots.length;
  for (const l of item.freeEntries) {
    score.add(item.slots[l] ?? 0, nodeSum(posterior, ratios, response * length + l, item.categoryCount * length));
  }
}

/** Returns Σ_q posterior[q] table[start + q stride]: a posterior mean of a function tabulated at the nodes. */
function nodeSum(posterior: Float64Array, table: Float64Array, start: number, stride: number): number {
  let sum = 0;
  for (let q = 0, index = start; q < posterior.length; q++, index += stride)
    sum += (posterior[q] ?? 0) * (table[index] ?? 0);
  return sum;
}

/** A person's score vector over the free parameters, stored at the slots the person's responses touch. */
interface SparseScore {
  /** Starts the score vector of a person at 0. */
  start(person: number): void;
  /** Adds `value` to the entry of a slot. */
  add(slot: number, value: number): void;
  /**
   * Adds the outer product of the score vector with itself, times `weight`, to `information`, a matrix of all slots by
   * rows.
   */
  addOuterProduct(information: Float64Array, weight: number): void;
}

function sparseScore(size: number): SparseScore {
  const score = new Float64Array(size);
  const owner = new Int32Array(size).fill(-1);
  const touched = new Int32Array(size);
  let person = -1;
  let count = 0;
  return {
    start(next) {
      person = next;
      count = 0;
    },
    add(slot, value) {
      if (owner[slot] !== person) {
        owner[slot] = person;
        score[slot] = 0;
        touched[count++] = slot;
      }
      score[slot] = (score[slot] ?? 0) + value;
    },
    addOuterProduct(information, weight) {
      for (let a = 0; a < count; a++) {
        const row = touched[a] ?? 0;
        const value = (score[row] ?? 0) * weight;
        for (let b = 0; b < count; b++) {
          const index = row * size + (touched[b] ?? 0);
          information[index] = (information[index] ?? 0) + value * (score[touched[b] ?? 0] ?? 0);
        }
      }
    },
  };
}

/**
 * Returns dP_k/dξ_l / P_k of an item at each node, category and vector entry, by node, then category, then entry: the
 * derivatives of ln P_k.
 */
export function derivativeRatios(
  item: {
    readonly model: IrtModel;
    readonly parameters: unknown;
    readonly categoryCount: number;
    readonly slots: Int32Array;
  },
  nodes: Float64Array,
): Float64Array {
  const { model, parameters, categoryCount } = item;
  const length = item.slots.length;
  const probabilities = new Float64Array(categoryCount);
  const derivatives = new Float64Array(categoryCount * length);
  const table = new Float64Array(nodes.length * categoryCount * length);
  nodes.forEach((theta, q) => {
    model.probabilities(parameters, theta, probabilities);
    model.parameterDerivatives(parameters, theta, derivatives);
    for (let k = 0; k < categoryCount; k++) {
      const probability = Math.max(probabilities[k] ?? 0, MIN_PROBABILITY);
      for (let l = 0; l < length; l++) {
        table[(q * categoryCount + k) * length + l] = (derivatives[k * length + l] ?? 0) / probability;
      }
    }
  });
  return table;
}

/**
 * Returns the derivatives of ln w_q with respect to the mean and the standard deviation of a normal distribution on
 * the nodes, its weights w_q normalized over them: (θ_q - μ) / σ² and (θ_q - μ)² / σ³, less their means under w.
 */
function normalSlopes(
  logWeights: Float64Array,
  mean: number,
  sd: number,
  nodes: Float64Array,
): { mean: Float64Array; sd: Float64Array } {
  const meanSlopes = Float64Array.from(nodes, (theta) => (theta - mean) / (sd * sd));
  const sdSlopes = Float64Array.from(nodes, (theta) => (theta - mean) ** 2 / sd ** 3);
  let meanCenter = 0;
  let sdCenter = 0;
  logWeights.forEach((logWeight, q) => {
    const weight = Math.exp(logWeight);
    meanCenter += weight * (meanSlopes[q] ?? 0);
    sdCenter += weight * (sdSlopes[q] ?? 0);
  });
  return { mean: meanSlopes.map((slope) => slope - meanCenter), sd: sdSlopes.map((slope) => slope - sdCenter) };
}

import { unboundedDerivative } from "../irt/bounds.js";
import { MIN_PROBABILITY } from "../irt/probability.js";
import type { CompressedResponses } from "../irt/responses.js";
import { numericalHessian } from "../math/hessian.js";
import { invert, invertPositiveDefinite } from "../math/linear-algebra.js";
import { type ItemState, UNBOUNDED } from "./calibration-setup.js";
import type { Latent } from "./correlation.js";
import type { ExpectationStep } from "./expectation.js";
import type { Grid } from "./grid.js";
import type { ItemMaximization } from "./m-step.js";

/** What a calibration's standard errors need: the fitted items, the latent distribution and the E-step. */
export interface Fit<TParameters> {
  readonly items: readonly ItemState<TParameters>[];
  readonly grid: Grid;
  readonly latent: Latent;
  /** Whether the correlations are estimated, and so part of the parameters. */
  readonly correlated: boolean;
  readonly maximization: ItemMaximization<TParameters>;
  readonly eStep: ExpectationStep;
  readonly responses: CompressedResponses;
}

/** Returns the number of estimated parameters: the items' estimated entries, then the correlations. */
export function parameterCount<TParameters>({ items, latent, correlated }: Fit<TParameters>): number {
  return items.reduce((sum, item) => sum + item.free.length, 0) + (correlated ? latent.pairs.length : 0);
}

/**
 * Returns the covariance matrix of the estimated parameters on their own scale, by rows, in the order of the items'
 * estimated entries and then the correlations; `null` when the Hessian cannot be inverted or gives a variance that is
 * not positive.
 *
 * @remarks
 * It is the inverse of the Hessian of the negative log-posterior in unbounded coordinates, by the delta method. Each
 * gradient evaluation runs its own E-step: by Fisher's identity, the gradient of the expected complete-data
 * log-likelihood at the parameters of the E-step is the gradient of the marginal one. The fit is left at its estimates.
 */
export function hessianCovariance<TParameters>(fit: Fit<TParameters>): Float64Array | null {
  const { items, latent, correlated, maximization, eStep } = fit;
  const size = parameterCount(fit);
  const pairs = correlated ? latent.pairs.length : 0;
  const itemSize = size - pairs;
  const estimates = new Float64Array(size);
  let start = 0;
  for (const item of items) {
    estimates.set(maximization.coordinates(item), start);
    start += item.free.length;
  }
  const correlations = latent.matrix.slice();
  estimates.set(correlationValues(latent).subarray(0, pairs), itemSize);
  const itemGradient = new Float64Array(Math.max(0, ...items.map((item) => item.free.length)));
  const gradientOf = (point: Float64Array, gradient: Float64Array): number => {
    let offset = 0;
    for (const item of items) {
      maximization.assign(item, point.subarray(offset, offset + item.free.length));
      offset += item.free.length;
    }
    if (pairs > 0 && !latent.set(withCorrelations(latent, point.subarray(itemSize)))) {
      gradient.fill(0);
      return 0;
    }
    eStep.run();
    let value = 0;
    offset = 0;
    for (const item of items) {
      const eta = point.subarray(offset, offset + item.free.length);
      value += maximization.objective(item, eta, itemGradient);
      gradient.set(itemGradient.subarray(0, item.free.length), offset);
      offset += item.free.length;
    }
    if (pairs > 0) latent.gradient(eStep.nodeCounts, gradient.subarray(itemSize));
    return value;
  };
  const hessian = numericalHessian(gradientOf, estimates);
  latent.set(correlations);
  let offset = 0;
  for (const item of items) {
    maximization.assign(item, estimates.subarray(offset, offset + item.free.length));
    offset += item.free.length;
  }
  const inverse = invert(hessian, size);
  if (inverse === null) return null;
  const scales = new Float64Array(size).fill(1);
  offset = 0;
  for (const item of items) {
    item.free.forEach((j, l) => {
      const { lower, upper } = item.entries[l] ?? UNBOUNDED;
      scales[offset + l] = unboundedDerivative(item.vector[j] ?? 0, lower, upper);
    });
    offset += item.free.length;
  }
  for (let a = 0; a < size; a++) {
    if (!((inverse[a * size + a] ?? 0) > 0)) return null;
    for (let b = 0; b < size; b++)
      inverse[a * size + b] = (inverse[a * size + b] ?? 0) * (scales[a] ?? 1) * (scales[b] ?? 1);
  }
  return inverse;
}

/**
 * Returns the covariance matrix of the estimated parameters, as {@link hessianCovariance} lays it out, from the
 * cross-product information.
 *
 * @remarks
 * The information is Σ_p s_p s_pᵀ over persons, with s_p the derivative of a person's log marginal likelihood, Σ_q
 * post_p(q) ∂ ln P(x_p | θ_q) / ∂ξ, plus minus the second derivatives of the log priors, as mirt's `SE.type =
 * "crossprod"`. `null` when the information is not positive definite. Uses the posteriors of the E-step's last run.
 */
export function crossProductCovariance<TParameters>(fit: Fit<TParameters>): Float64Array | null {
  const { items, grid, latent, correlated, eStep } = fit;
  const size = parameterCount(fit);
  const pairs = correlated ? latent.pairs.length : 0;
  const offsets = new Int32Array(items.length + 1);
  items.forEach((item, i) => {
    offsets[i + 1] = (offsets[i] ?? 0) + item.free.length;
  });
  const ratios = items.map((item) => derivativeRatios(item, grid));
  const latentScores = new Float64Array(pairs * grid.count);
  if (pairs > 0) latent.scores(latentScores);
  const information = new Float64Array(size * size);
  const score = new Float64Array(size);
  eStep.forEachPosterior((person, posterior) => {
    score.fill(0);
    addItemScores(fit, person, posterior, ratios, offsets, score);
    for (let p = 0; p < pairs; p++) score[size - pairs + p] = weightedSum(posterior, latentScores, p * grid.count);
    addOuterProduct(information, score);
  });
  items.forEach((item, i) => {
    item.free.forEach((j, l) => {
      const prior = item.entries[l]?.prior;
      if (prior === undefined) return;
      const slot = (offsets[i] ?? 0) + l;
      information[slot * size + slot] =
        (information[slot * size + slot] ?? 0) - prior.logDensitySecondDerivative(item.vector[j] ?? 0);
    });
  });
  return invertPositiveDefinite(information, size);
}

/** Adds the derivatives of ln P(x | θ) of a person's responses, averaged over the posterior, to the score vector. */
function addItemScores<TParameters>(
  { items, grid, responses }: Fit<TParameters>,
  person: number,
  posterior: Float64Array,
  ratios: readonly Float64Array[],
  offsets: Int32Array,
  score: Float64Array,
): void {
  for (let r = responses.offsets[person] ?? 0; r < (responses.offsets[person + 1] ?? 0); r++) {
    const i = responses.items[r] ?? 0;
    const free = items[i]?.free.length ?? 0;
    const table = ratios[i];
    if (table === undefined) continue;
    const response = responses.codes[r] ?? 0;
    for (let l = 0; l < free; l++) {
      const slot = (offsets[i] ?? 0) + l;
      score[slot] = (score[slot] ?? 0) + weightedSum(posterior, table, (response * free + l) * grid.count);
    }
  }
}

/** Adds the outer product of the score vector with itself to `information`. */
function addOuterProduct(information: Float64Array, score: Float64Array): void {
  const size = score.length;
  for (let a = 0; a < size; a++) {
    for (let b = 0; b < size; b++)
      information[a * size + b] = (information[a * size + b] ?? 0) + (score[a] ?? 0) * (score[b] ?? 0);
  }
}

/** Returns Σ_q posterior[q] table[start + q]. */
function weightedSum(posterior: Float64Array, table: Float64Array, start: number): number {
  let sum = 0;
  for (let q = 0; q < posterior.length; q++) sum += (posterior[q] ?? 0) * (table[start + q] ?? 0);
  return sum;
}

/**
 * Returns dP_k/dξ_l / P_k of an item at each node, for each category and estimated entry, by category, then entry, then
 * node: the derivatives of ln P_k.
 */
function derivativeRatios<TParameters>(item: ItemState<TParameters>, { dimensions, count, nodes }: Grid): Float64Array {
  const { model, parameters, categoryCount, free } = item;
  const length = item.vector.length;
  const theta = new Float64Array(dimensions);
  const probabilities = new Float64Array(categoryCount);
  const derivatives = new Float64Array(categoryCount * length);
  const table = new Float64Array(categoryCount * free.length * count);
  for (let q = 0; q < count; q++) {
    for (let d = 0; d < dimensions; d++) theta[d] = nodes[q * dimensions + d] ?? 0;
    model.probabilities(parameters, theta, probabilities);
    model.parameterDerivatives(parameters, theta, derivatives);
    for (let k = 0; k < categoryCount; k++) {
      const probability = Math.max(probabilities[k] ?? 0, MIN_PROBABILITY);
      for (let l = 0; l < free.length; l++) {
        table[(k * free.length + l) * count + q] = (derivatives[k * length + (free[l] ?? 0)] ?? 0) / probability;
      }
    }
  }
  return table;
}

/** Returns the correlations below the diagonal in the order of the pairs. */
function correlationValues({ pairs, matrix, dimensions }: Latent): Float64Array {
  return Float64Array.from(pairs, ([i, j]) => matrix[i * dimensions + j] ?? 0);
}

/** Returns the correlation matrix with the correlations below the diagonal set to `values`. */
function withCorrelations({ pairs, matrix, dimensions }: Latent, values: Float64Array): Float64Array {
  const next = matrix.slice();
  pairs.forEach(([i, j], p) => {
    next[i * dimensions + j] = values[p] ?? 0;
    next[j * dimensions + i] = values[p] ?? 0;
  });
  return next;
}

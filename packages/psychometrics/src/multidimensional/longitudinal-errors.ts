import { invertPositiveDefinite } from "@measurekit/core";
import {
  type CompressedResponses,
  derivativeRatios,
  type StandardErrorMethod,
  unboundedCoordinates,
  unboundedScales,
} from "../irt/internal.js";
import { addOuterProduct, invert, mirror, numericalHessian } from "../math/internal.js";
import type { LongitudinalFit } from "./longitudinal-em.js";

/**
 * Returns the standard errors of the estimated parameters on their own scale: the items' free parameters in the order
 * of the calibration's setup, then the correlations, the standard deviations and the means of the distribution; `null`
 * without the method, when the information cannot be inverted or gives a variance that is not positive.
 *
 * @remarks
 * `"hessian"` is the observed information: the Hessian of the negative marginal log-likelihood by central differences
 * of its exact gradient, which by Fisher's identity is the gradient of the expected complete-data log-likelihood at
 * the counts of an E-step run at each point. `"cross-product"` is the sum over persons of the outer products of their
 * score vectors, as mirt's `SE.type = "crossprod"`. The fit is left at its estimates, with the E-step run there.
 */
export function standardErrorsOf<TParameters>(
  method: StandardErrorMethod,
  fit: LongitudinalFit<TParameters>,
): Float64Array | null {
  if (method === "none") return null;
  const covariance = method === "hessian" ? hessianCovariance(fit) : crossProductCovariance(fit);
  if (covariance === null) return null;
  const size = covariance.length ** 0.5;
  const errors = new Float64Array(size);
  for (let j = 0; j < size; j++) {
    const variance = covariance[j * size + j] ?? 0;
    if (!(variance > 0)) return null;
    errors[j] = Math.sqrt(variance);
  }
  return errors;
}

/** Returns the multipliers that turn a parameter's own scale into the reported scale: dx/dη, the sd for its logarithm. */
function reportScales<TParameters>({ setup, latent }: LongitudinalFit<TParameters>, unbounded: boolean): Float64Array {
  const { values, free } = setup;
  const scales = new Float64Array(values.length + latent.size).fill(1);
  if (unbounded) scales.set(unboundedScales(values, free));
  if (latent.free.variances) {
    const first = values.length + (latent.free.correlations ? latent.pairs.length : 0);
    latent.standardDeviations.forEach((deviation, d) => {
      const slot = estimatedSlot(latent.free.pinned, d);
      if (slot >= 0) scales[first + slot] = deviation;
    });
  }
  return scales;
}

/** Returns the position of dimension `d` among the dimensions that are not pinned, or -1 when it is pinned. */
function estimatedSlot(pinned: readonly number[] | undefined, d: number): number {
  if (pinned?.includes(d) === true) return -1;
  return d - (pinned?.filter((p) => p < d).length ?? 0);
}

/** Scales a covariance matrix of the parameters' own scales to the reported ones, by rows. */
function rescale(covariance: Float64Array, scales: Float64Array): Float64Array {
  const size = scales.length;
  return covariance.map((value, k) => value * (scales[Math.floor(k / size)] ?? 1) * (scales[k % size] ?? 1));
}

/** Returns the covariance from the Hessian by central differences of the gradient; see {@link standardErrorsOf}. */
function hessianCovariance<TParameters>(fit: LongitudinalFit<TParameters>): Float64Array | null {
  const { setup, maximization, latent, eStep } = fit;
  const { values, free } = setup;
  const n = values.length;
  const start = new Float64Array(n + latent.size);
  start.set(unboundedCoordinates(values, free));
  start.set(latent.entries(), n);
  const hessian = numericalHessian((point, gradient) => {
    const eta = point.subarray(0, n);
    maximization.assign(maximization.all, eta);
    latent.assignEntries(point.subarray(n));
    eStep.run();
    const value = maximization.objective(maximization.all, eta, gradient.subarray(0, n));
    latent.gradient(eStep.nodeCounts, gradient.subarray(n));
    return value;
  }, start);
  maximization.assign(maximization.all, start.subarray(0, n));
  latent.assignEntries(start.subarray(n));
  eStep.run();
  const inverse = invert(hessian, start.length);
  return inverse === null ? null : rescale(inverse, reportScales(fit, true));
}

/** Returns the covariance from the cross-product of the persons' scores; see {@link standardErrorsOf}. */
function crossProductCovariance<TParameters>(fit: LongitudinalFit<TParameters>): Float64Array | null {
  const { setup, latent, eStep, grid } = fit;
  const n = setup.values.length;
  const size = n + latent.size;
  const ratios = setup.items.map((item) => derivativeRatios(item, fit.axis));
  const nodeScores = new Float64Array(latent.size * grid.count);
  latent.scores(nodeScores);
  const information = new Float64Array(size * size);
  const score = new Float64Array(size);
  eStep.forEachPosterior((person, { joint, marginals }) => {
    score.fill(0);
    fit.occasions.forEach((occasion, t) => {
      addItemScores(score, occasion, person, marginals[t], fit, ratios);
    });
    for (let e = 0; e < latent.size; e++) {
      score[n + e] = expectation(joint, nodeScores.subarray(e * grid.count, (e + 1) * grid.count), 0, 1);
    }
    addOuterProduct(information, score);
  });
  mirror(information, size);
  const inverse = invertPositiveDefinite(information, size);
  return inverse === null ? null : rescale(inverse, reportScales(fit, false));
}

/**
 * Adds to the score the derivatives of the log-probability of each response a person gave on an occasion, averaged over
 * the person's marginal posterior there, at the slots of the item's parameters.
 */
function addItemScores<TParameters>(
  score: Float64Array,
  occasion: CompressedResponses,
  person: number,
  marginal: Float64Array | undefined,
  { setup }: LongitudinalFit<TParameters>,
  ratios: readonly Float64Array[],
): void {
  for (let r = occasion.offsets[person] ?? 0; r < (occasion.offsets[person + 1] ?? 0); r++) {
    const index = occasion.items[r] ?? 0;
    const item = setup.items[index];
    const table = ratios[index];
    if (item === undefined || table === undefined) continue;
    const length = item.slots.length;
    item.slots.forEach((slot, l) => {
      const start = (occasion.codes[r] ?? 0) * length + l;
      score[slot] = (score[slot] ?? 0) + expectation(marginal, table, start, item.categoryCount * length);
    });
  }
}

/** Returns Σ_j weights[j] table[start + j stride]: the mean of a function tabulated at the nodes under a posterior. */
function expectation(weights: Float64Array | undefined, table: Float64Array, start: number, stride: number): number {
  let sum = 0;
  weights?.forEach((weight, j) => {
    sum += weight * (table[start + j * stride] ?? 0);
  });
  return sum;
}

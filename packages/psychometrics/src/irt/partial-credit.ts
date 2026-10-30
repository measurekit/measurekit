import { divideByTotalThetaDerivatives, softmax } from "./divide-by-total.js";
import type { IrtModel, ModelParameter } from "./model.js";
import {
  type ThresholdParameters,
  thresholdParameterList,
  thresholdsFromVector,
  thresholdVector,
} from "./thresholds.js";

// The partial credit family gives category k the logit z_k = Σ_(j ≤ k) a (θ - δ_j), z_0 = 0, with the thresholds
// δ_j = location + steps[j - 1]: the steps are the GPCM's and PCM's thresholds b with location 0, and the RSM's steps
// with its location b.

/**
 * The generalized partial credit model (GPCM, Muraki): an item with K categories has a slope `a` and step thresholds
 * b1 to b(K-1), and P(k) ∝ exp(Σ_(j ≤ k) a (θ - b_j)). The thresholds need not be ordered. This is mirt's `gpcm`
 * item type with `IRTpars = TRUE`.
 */
export const generalizedPartialCredit: IrtModel<ThresholdParameters> = partialCreditModel("GPCM", { name: "a" });

/**
 * The partial credit model (PCM, Masters): the GPCM with one slope `a` for all its items, estimated so that theta
 * keeps the N(0, 1) scale. It is the Rasch model for ordered categories up to scale: the Rasch thresholds are `a * b`
 * and the Rasch ability `a * θ`. With two categories it is the 1PL.
 */
export const partialCredit: IrtModel<ThresholdParameters> = partialCreditModel("PCM", { name: "a", shared: true });

function partialCreditModel(name: string, slope: ModelParameter): IrtModel<ThresholdParameters> {
  return {
    name,
    maxCategories: Number.POSITIVE_INFINITY,
    categoryCount: ({ b }) => b.length + 1,
    parameters: (categoryCount) => thresholdParameterList(slope, categoryCount),
    startingValues: (proportions) => ({ a: 1, b: stepStarts(proportions) }),
    toVector: (parameters) => thresholdVector(name, parameters),
    fromVector: thresholdsFromVector,
    probabilities: ({ a, b }, theta, out) => partialCreditProbabilities(a, 0, b, theta, out),
    thetaDerivatives: ({ a, b }, theta, first, second) => partialCreditThetaDerivatives(a, 0, b, theta, first, second),
    parameterDerivatives: ({ a, b }, theta, out) => partialCreditDerivatives(a, 0, b, theta, out),
    information: ({ a, b }, theta) => partialCreditInformation(a, 0, b, theta),
  };
}

/**
 * Returns the thresholds ln(p_(k-1) / p_k) at which an item with a = 1 gives a person at θ = 0 the shares p of persons
 * in its categories.
 */
export function stepStarts(proportions: Float64Array): number[] {
  return Array.from({ length: proportions.length - 1 }, (_, j) =>
    Math.log((proportions[j] ?? 0) / (proportions[j + 1] ?? 0)),
  );
}

/** Writes the logits z_k of a partial credit item at `out[k * stride]`. */
function writeLogits(
  a: number,
  location: number,
  steps: ArrayLike<number>,
  theta: number,
  out: Float64Array,
  stride: number,
): void {
  let z = 0;
  out[0] = 0;
  for (let k = 1; k <= steps.length; k++) {
    z += a * (theta - location - (steps[k - 1] ?? 0));
    out[k * stride] = z;
  }
}

/** Writes the category probabilities of a partial credit item with thresholds `location + steps[j]`. */
export function partialCreditProbabilities(
  a: number,
  location: number,
  steps: ArrayLike<number>,
  theta: number,
  out: Float64Array,
): void {
  writeLogits(a, location, steps, theta, out, 1);
  softmax(out, steps.length + 1);
}

/** Writes the first and second theta derivatives of a partial credit item's category probabilities. */
export function partialCreditThetaDerivatives(
  a: number,
  location: number,
  steps: ArrayLike<number>,
  theta: number,
  first: Float64Array,
  second: Float64Array,
): void {
  const count = steps.length + 1;
  writeLogits(a, location, steps, theta, first, 1);
  softmax(first, count);
  for (let k = 0; k < count; k++) second[k] = k * a;
  divideByTotalThetaDerivatives(first, second, count);
}

/**
 * Writes the derivatives of a partial credit item's category probabilities with respect to a and the thresholds
 * δ_1 to δ_(K-1): category k's at `out[k * K]` to `out[k * K + K - 1]`. S_m is P(X ≥ m).
 */
export function partialCreditDerivatives(
  a: number,
  location: number,
  steps: ArrayLike<number>,
  theta: number,
  out: Float64Array,
): void {
  const count = steps.length + 1;
  writeLogits(a, location, steps, theta, out, count);
  softmax(out, count, count);
  // Column 0 holds the probabilities until the derivatives with respect to a replace them last.
  writeThresholdDerivatives(a, count, out);
  writeSlopeDerivatives(location, steps, theta, out);
}

/** Writes dP_k/dδ_m = a P_k (S_m - [k ≥ m]) into columns 1 to K - 1, given the probabilities in column 0. */
function writeThresholdDerivatives(a: number, count: number, out: Float64Array): void {
  let tail = 0;
  for (let m = count - 1; m >= 1; m--) {
    tail += out[m * count] ?? 0;
    for (let k = 0; k < count; k++) out[k * count + m] = a * (out[k * count] ?? 0) * (tail - (k >= m ? 1 : 0));
  }
}

/** Replaces the probabilities in column 0 with dP_k/da = P_k (A_k - Σ_i P_i A_i), A_k = Σ_(j ≤ k) (θ - δ_j). */
function writeSlopeDerivatives(location: number, steps: ArrayLike<number>, theta: number, out: Float64Array): void {
  const count = steps.length + 1;
  let mean = 0;
  let distance = 0;
  for (let k = 0; k < count; k++) {
    if (k > 0) distance += theta - location - (steps[k - 1] ?? 0);
    mean += (out[k * count] ?? 0) * distance;
  }
  distance = 0;
  for (let k = 0; k < count; k++) {
    if (k > 0) distance += theta - location - (steps[k - 1] ?? 0);
    out[k * count] = (out[k * count] ?? 0) * (distance - mean);
  }
}

/**
 * Returns the information of a partial credit item, a² times the variance of the category under its probabilities,
 * by West's weighted update, which never goes below 0.
 */
export function partialCreditInformation(a: number, location: number, steps: ArrayLike<number>, theta: number): number {
  let z = 0;
  let largest = 0;
  for (let k = 1; k <= steps.length; k++) {
    z += a * (theta - location - (steps[k - 1] ?? 0));
    largest = Math.max(largest, z);
  }
  let total = 0;
  let mean = 0;
  let squares = 0;
  z = 0;
  for (let k = 0; k <= steps.length; k++) {
    if (k > 0) z += a * (theta - location - (steps[k - 1] ?? 0));
    const weight = Math.exp(z - largest);
    total += weight;
    const deviation = k - mean;
    mean += (deviation * weight) / total;
    squares += weight * deviation * (k - mean);
  }
  return (a * a * squares) / total;
}

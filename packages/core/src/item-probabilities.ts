import { isUnitInterval } from "./checks.js";
import { logistic, logisticPair } from "./logistic.js";

/**
 * The parameters of a dichotomous logistic item in the IRT form with D = 1: the probability of a correct response is
 * P(θ) = c + (d - c) logistic(a (θ - b)). The 1PL has `c = 0` and `d = 1` and the same `a` for every item, the 2PL
 * `c = 0` and `d = 1`, and the 3PL `d = 1`.
 */
export interface LogisticItem {
  /** Discrimination: the slope. */
  readonly a: number;
  /** Difficulty: the location. */
  readonly b: number;
  /** Lower asymptote (guessing). */
  readonly c: number;
  /** Upper asymptote. */
  readonly d: number;
}

/**
 * The parameters of a polytomous item with a slope and one threshold per boundary between adjacent categories, in the
 * IRT form with D = 1: an item of the graded response model or of the (generalized) partial credit model. An item with
 * K categories has K - 1 thresholds.
 */
export interface ThresholdItem {
  /** Discrimination: the slope. */
  readonly a: number;
  /** The thresholds b1 to b(K-1), one per boundary between adjacent categories. */
  readonly b: readonly number[];
}

/**
 * Writes the probabilities of the two categories of a logistic item at the logit `z`: `out[0]` is P(0) and `out[1]` is
 * P(1) = c + (d - c) logistic(z).
 *
 * @remarks
 * `z` is a (θ - b) for one dimension and a · θ + intercept for several. P(0) = (1 - d) + (d - c) logistic(-z) is taken
 * from the tail it is near, so it keeps its precision where P(1) rounds to 1.
 *
 * @param z - The logit, a · θ + intercept or a (θ - b).
 * @param c - The lower asymptote, from 0 to below `d`.
 * @param d - The upper asymptote, above `c` and at most 1.
 * @param out - Receives P(0) at index 0 and P(1) at index 1.
 */
export function logisticProbabilities(z: number, c: number, d: number, out: Float64Array): void {
  logisticPair(z, out);
  const upper = out[0] ?? 0;
  const lower = out[1] ?? 0;
  out[0] = 1 - d + (d - c) * lower;
  out[1] = c + (d - c) * upper;
}

/**
 * Writes the probabilities of the K = `b.length + 1` categories of a graded response item at `theta`:
 * P(k) = P(X ≥ k) - P(X ≥ k + 1), with P(X ≥ k) = logistic(a (θ - b_k)).
 *
 * @remarks
 * Each category is written as P(X ≥ k) P(X < k + 1) (1 - exp(-a (b_(k+1) - b_k))), with P(X ≥ 0) = P(X < K) = 1, which
 * equals the difference and keeps the precision of small probabilities in both tails.
 *
 * @param a - The slope of the item.
 * @param b - The thresholds, in the order of the slope's sign.
 * @param theta - The ability.
 * @param out - Receives the probability of each category, from index 0.
 */
export function gradedProbabilities(a: number, b: ArrayLike<number>, theta: number, out: Float64Array): void {
  let above = 1;
  for (let k = 0; k < b.length; k++) {
    const z = a * (theta - (b[k] ?? 0));
    const gap = k === 0 ? 1 : -Math.expm1(-a * ((b[k] ?? 0) - (b[k - 1] ?? 0)));
    out[k] = above * logistic(-z) * gap;
    above = logistic(z);
  }
  out[b.length] = above;
}

/**
 * Writes the probabilities of the K = `steps.length + 1` categories of a partial credit item at `theta`:
 * P(k) ∝ exp(Σ_(j ≤ k) a (θ - δ_j)), with the thresholds δ_j = `location` + `steps[j - 1]`.
 *
 * @remarks
 * The generalized partial credit model has `location` 0 and its thresholds as `steps`, and the rating scale model its
 * location and the steps around it.
 *
 * @param a - The slope of the item.
 * @param location - The location that is added to every step; 0 for the generalized partial credit model.
 * @param steps - The thresholds of the K - 1 steps.
 * @param theta - The ability.
 * @param out - Receives the probability of each category, from index 0.
 */
export function partialCreditProbabilities(
  a: number,
  location: number,
  steps: ArrayLike<number>,
  theta: number,
  out: Float64Array,
): void {
  let z = 0;
  out[0] = 0;
  for (let k = 1; k <= steps.length; k++) {
    z += a * (theta - location - (steps[k - 1] ?? 0));
    out[k] = z;
  }
  softmax(out, steps.length + 1);
}

/**
 * Turns the first `count` logits of `values` into probabilities in place: exp(z_k) / Σ_j exp(z_j).
 *
 * @remarks
 * Subtracting the largest logit keeps exp from overflowing. The partial credit and nominal response models use it.
 *
 * @param values - The logits, which the probabilities replace.
 * @param count - The number of logits.
 */
export function softmax(values: Float64Array, count: number): void {
  stridedSoftmax(values, count, 1);
}

/**
 * Turns the logits at `values[k * stride]`, k < `count`, into probabilities in place, as {@link softmax} does: the
 * logits of one column of a matrix by rows, such as the nominal response model's table of derivatives.
 *
 * @param values - The logits, which the probabilities replace.
 * @param count - The number of logits.
 * @param stride - The distance between two logits in `values`.
 */
export function stridedSoftmax(values: Float64Array, count: number, stride: number): void {
  let largest = Number.NEGATIVE_INFINITY;
  for (let k = 0; k < count; k++) largest = Math.max(largest, values[k * stride] ?? 0);
  let total = 0;
  for (let k = 0; k < count; k++) {
    const weight = Math.exp((values[k * stride] ?? 0) - largest);
    values[k * stride] = weight;
    total += weight;
  }
  for (let k = 0; k < count; k++) values[k * stride] = (values[k * stride] ?? 0) / total;
}

/**
 * Returns whether the asymptotes of a logistic item are valid: numbers with 0 ≤ c < d ≤ 1.
 *
 * @param c - The lower asymptote to test.
 * @param d - The upper asymptote to test.
 * @returns Whether both are numbers with 0 ≤ c < d ≤ 1.
 */
export function hasValidAsymptotes(c: unknown, d: unknown): boolean {
  return isUnitInterval(c) && isUnitInterval(d) && c < d;
}

/**
 * Returns whether the thresholds of a graded response item run strictly with the sign of its slope: increasing when
 * a > 0 and decreasing when a < 0. An item with one threshold is always ordered.
 *
 * @param a - The slope.
 * @param b - The thresholds.
 * @returns Whether the thresholds are strictly increasing for a positive slope and strictly decreasing for a negative
 *   one.
 */
export function hasOrderedThresholds(a: number, b: readonly number[]): boolean {
  return b.every((threshold, k) => k === 0 || a * (threshold - (b[k - 1] ?? 0)) > 0);
}

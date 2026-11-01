import { DataError, describe, isRecord, partialCreditProbabilities } from "@measurekit/core";
import type { IrtModel } from "./model.js";
import {
  partialCreditDerivatives,
  partialCreditInformation,
  partialCreditThetaDerivatives,
  stepStarts,
} from "./partial-credit.js";
import { formatList } from "./thresholds.js";

/**
 * Parameters of a rating scale item: a partial credit item whose thresholds are its location plus steps that every
 * item of the scale shares.
 */
export interface RatingScaleParameters {
  /** Discrimination: the slope. */
  readonly a: number;
  /** Difficulty: the location, the mean of the thresholds. */
  readonly b: number;
  /** The steps τ_1 to τ_(K-1), summing to 0: the thresholds are b + τ_k. */
  readonly steps: readonly number[];
}

/**
 * The rating scale model (RSM, Andrich).
 *
 * @remarks
 * The partial credit model with thresholds b + τ_k, where the slope `a` and the steps τ are shared by all items of the
 * model in a calibration and `b` is estimated per item. P(k) ∝ exp(Σ_(j ≤ k) a (θ - b - τ_j)). As for the 1PL, `a` is
 * estimated so that theta keeps the N(0, 1) scale. The steps sum to 0; the vector holds τ_1 to τ_(K-2), and τ_(K-1) is
 * minus their sum.
 */
export const ratingScale: IrtModel<RatingScaleParameters> = {
  name: "RSM",
  maxCategories: Number.POSITIVE_INFINITY,
  categoryCount: ({ steps }) => steps.length + 1,
  parameters: (categoryCount) => [
    { name: "a", shared: true },
    { name: "b" },
    ...Array.from({ length: categoryCount - 2 }, (_, j) => ({ name: `step${j + 1}`, shared: true })),
  ],
  startingValues(proportions) {
    const thresholds = stepStarts(proportions);
    const b = thresholds.reduce((sum, threshold) => sum + threshold, 0) / thresholds.length;
    return { a: 1, b, steps: thresholds.map((threshold) => threshold - b) };
  },
  toVector(parameters) {
    const { a, b, steps }: Partial<RatingScaleParameters> = isRecord(parameters) ? parameters : {};
    const location = typeof a === "number" && typeof b === "number" && Number.isFinite(a) && Number.isFinite(b);
    if (!(location && Array.isArray(steps) && steps.length > 0)) {
      throw new DataError(
        `An RSM item needs finite a and b and at least one step, found a = ${describe(a)}, b = ${describe(b)}, steps = ${formatList(steps)}.`,
      );
    }
    if (!steps.every(Number.isFinite)) {
      throw new DataError(`An RSM item needs finite steps, found ${formatList(steps)}.`);
    }
    // Steps that do not sum to 0 move their mean into the location; the thresholds stay the same.
    const center = steps.reduce((sum, step) => sum + step, 0) / steps.length;
    return Float64Array.from([a, b + center, ...steps.slice(0, -1).map((step) => step - center)]);
  },
  fromVector(vector) {
    const steps = Array.from(vector.subarray(2));
    let last = 0;
    for (const step of steps) last -= step;
    steps.push(last);
    return { a: vector[0] ?? 0, b: vector[1] ?? 0, steps };
  },
  probabilities: ({ a, b, steps }, theta, out) => partialCreditProbabilities(a, b, steps, theta, out),
  thetaDerivatives: ({ a, b, steps }, theta, first, second) =>
    partialCreditThetaDerivatives(a, b, steps, theta, first, second),
  parameterDerivatives({ a, b, steps }, theta, out) {
    const size = steps.length + 1;
    partialCreditDerivatives(a, b, steps, theta, out);
    // From the derivatives with respect to the thresholds δ_j = b + τ_j, with τ_(K-1) = -(τ_1 + ... + τ_(K-2)), to
    // those with respect to b and τ_1 to τ_(K-2), in place: entry l + 1 takes δ_l's minus δ_(K-1)'s.
    for (let k = 0; k < size; k++) {
      const row = k * size;
      let sum = 0;
      for (let j = 1; j < size; j++) sum += out[row + j] ?? 0;
      const last = out[row + size - 1] ?? 0;
      for (let l = size - 2; l >= 1; l--) out[row + l + 1] = (out[row + l] ?? 0) - last;
      out[row + 1] = sum;
    }
  },
  information: ({ a, b, steps }, theta) => partialCreditInformation(a, b, steps, theta),
};

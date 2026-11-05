import { checkNonNegative, checkOptions, checkPositiveInteger, type FieldNames } from "@measurekit/core";
import type { ItemSelectionRule } from "./selection.js";

/** Options of {@link progressive} and {@link proportional}. */
export interface ProgressiveOptions {
  /**
   * The acceleration t of the weight: how fast selection turns from chance to information, catR's `AP`.
   *
   * @defaultValue `1`
   */
  readonly acceleration?: number;
}

const PROGRESSIVE_OPTIONS: FieldNames<ProgressiveOptions> = { acceleration: true };

/**
 * Returns the weight of each selection of a test: 0 at the first item, Σ_(i ≤ k) i^t / Σ_(i < Q) i^t after k items, and
 * 1 from the last planned item on.
 *
 * @throws `DataError` when `length` is not a positive integer or `acceleration` is not a non-negative number.
 */
function progressiveWeight(length: number, options: ProgressiveOptions): (administered: number) => number {
  checkOptions(options, PROGRESSIVE_OPTIONS);
  const { acceleration = 1 } = options;
  checkPositiveInteger(length, "The planned length");
  checkNonNegative(acceleration, "The option acceleration");
  // Each power is taken of i over the last planned i, so that the largest is 1 and none overflows at a high acceleration.
  const last = length - 1;
  let planned = 0;
  for (let i = 1; i < length; i++) planned += (i / last) ** acceleration;
  return (administered) => {
    if (administered === 0) return 0;
    if (administered >= last) return 1;
    let sum = 0;
    for (let i = 1; i <= administered; i++) sum += (i / last) ** acceleration;
    return sum / planned;
  };
}

/** Returns the rule that scores by `score` at the weight of each selection, as `progressiveWeight` gives it. */
function weighted(
  length: number,
  options: ProgressiveOptions,
  score: (weight: number, ...context: Parameters<ItemSelectionRule["scores"]>) => void,
): ItemSelectionRule {
  const weightAfter = progressiveWeight(length, options);
  return {
    scores(context, out) {
      score(weightAfter(context.administered.length), context, out);
    },
  };
}

/**
 * Returns the progressive method, catR's `progressive` with `rule = "length"`: each item scored by w I(θ̂) + (1 - w) U
 * max I, a weighted sum of its information and a uniform draw up to the largest information of the eligible items.
 *
 * @remarks
 * The weight w is 0 at the first item, so that it is chosen at random, and grows to 1 at the last planned item, Σ_(i ≤
 * k) i^t / Σ_(i < Q) i^t after k items, with t the acceleration and Q the planned length; after the planned length it
 * stays 1. Early items spread over the pool, which keeps their exposure low, and late ones are the most informative.
 * The rule draws one uniform number for each eligible item, in pool order, from `context.random`.
 *
 * @param length - The planned length of the test, Q, catR's `thr` with `rule = "length"`.
 * @param options - The acceleration.
 * @returns The selection rule.
 * @throws `DataError` when `length` is not a positive integer or `acceleration` is not a non-negative number.
 * @see {@link https://doi.org/10.1111/j.1745-3984.1998.tb00541.x | Revuelta and Ponsoda (1998)}
 * @see {@link https://doi.org/10.1348/000711007X230937 | Barrada, Olea, Ponsoda and Abad (2008)}
 */
export function progressive(length: number, options: ProgressiveOptions = {}): ItemSelectionRule {
  return weighted(length, options, (weight, { pool, estimate, eligible, random }, out) => {
    let largest = 0;
    pool.forEach(({ model, parameters }, index) => {
      if (eligible[index] !== 1) return;
      const information = model.information(parameters, estimate.theta);
      out[index] = information;
      largest = Math.max(largest, information);
    });
    pool.forEach((_, index) => {
      if (eligible[index] === 1) {
        out[index] = weight * (out[index] ?? 0) + (1 - weight) * random.uniform() * largest;
      }
    });
  });
}

/**
 * Returns the proportional method, catR's `proportional` with `rule = "length"`: an item drawn with probability
 * proportional to I(θ̂)^(Q w), its information to a power that grows over the test.
 *
 * @remarks
 * The weight w is that of {@link progressive}, so the power is 0 at the first item, where every eligible item is as
 * likely, and Q at the last planned item and after. The rule draws by an exponential race: each eligible item, in pool
 * order, takes one uniform number U from `context.random` and the score Q w ln I(θ̂) - ln(-ln(1 - U)), and the highest
 * score wins with the probability above. An item without information at the estimate is never drawn once the power is
 * above 0.
 *
 * @param length - The planned length of the test, Q, catR's `thr` with `rule = "length"`.
 * @param options - The acceleration.
 * @returns The selection rule.
 * @throws `DataError` when `length` is not a positive integer or `acceleration` is not a non-negative number.
 * @see {@link https://doi.org/10.1348/000711007X230937 | Barrada, Olea, Ponsoda and Abad (2008)}
 */
export function proportional(length: number, options: ProgressiveOptions = {}): ItemSelectionRule {
  return weighted(length, options, (weight, { pool, estimate, eligible, random }, out) => {
    const power = length * weight;
    pool.forEach(({ model, parameters }, index) => {
      if (eligible[index] !== 1) return;
      const logWeight = power === 0 ? 0 : power * Math.log(model.information(parameters, estimate.theta));
      out[index] = logWeight - Math.log(-Math.log(1 - random.uniform()));
    });
  });
}

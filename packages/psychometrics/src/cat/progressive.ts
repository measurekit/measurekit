import { checkNonNegative, checkOptions, checkPositiveInteger } from "@measurekit/core";
import type { ItemSelectionRule } from "./selection.js";

/** Options of {@link progressive} and {@link proportional}. */
export interface ProgressiveOptions {
  /** The planned length of the test, Q, catR's `thr` with `rule = "length"`. Required. */
  readonly length: number;
  /**
   * The acceleration t of the weight: how fast selection turns from chance to information, catR's `AP`.
   *
   * @defaultValue `1`
   */
  readonly acceleration?: number;
}

/**
 * Returns the weight of each selection of a test: 0 at the first item, Σ_(i ≤ k) i^t / Σ_(i < Q) i^t after k items, and
 * 1 from the last planned item on.
 *
 * @throws `DataError` when the options are not an object, `length` is not a positive integer or `acceleration` is not a
 * non-negative number.
 */
function progressiveWeight(options: ProgressiveOptions): (administered: number) => number {
  checkOptions(options);
  const { length, acceleration = 1 } = options;
  checkPositiveInteger(length, "The option length");
  checkNonNegative(acceleration, "The option acceleration");
  let planned = 0;
  for (let i = 1; i < length; i++) planned += i ** acceleration;
  return (administered) => {
    if (administered === 0) return 0;
    if (administered >= length - 1) return 1;
    let sum = 0;
    for (let i = 1; i <= administered; i++) sum += i ** acceleration;
    return sum / planned;
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
 * @param options - The planned length and the acceleration.
 * @returns The selection rule.
 * @throws `DataError` when the options are not an object, `length` is not a positive integer or `acceleration` is not a
 * non-negative number.
 * @see {@link https://doi.org/10.1111/j.1745-3984.1998.tb00541.x | Revuelta and Ponsoda (1998)}
 * @see {@link https://doi.org/10.1348/000711007X230937 | Barrada, Olea, Ponsoda and Abad (2008)}
 */
export function progressive(options: ProgressiveOptions): ItemSelectionRule {
  const weightAfter = progressiveWeight(options);
  return {
    scores({ pool, administered, estimate, eligible, random }, out) {
      const weight = weightAfter(administered.length);
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
    },
  };
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
 * @param options - The planned length and the acceleration.
 * @returns The selection rule.
 * @throws `DataError` when the options are not an object, `length` is not a positive integer or `acceleration` is not a
 * non-negative number.
 * @see {@link https://doi.org/10.1348/000711007X230937 | Barrada, Olea, Ponsoda and Abad (2008)}
 */
export function proportional(options: ProgressiveOptions): ItemSelectionRule {
  const weightAfter = progressiveWeight(options);
  return {
    scores({ pool, administered, estimate, eligible, random }, out) {
      const power = options.length * weightAfter(administered.length);
      pool.forEach(({ model, parameters }, index) => {
        if (eligible[index] !== 1) return;
        const logWeight = power === 0 ? 0 : power * Math.log(model.information(parameters, estimate.theta));
        out[index] = logWeight - Math.log(-Math.log(1 - random.uniform()));
      });
    },
  };
}

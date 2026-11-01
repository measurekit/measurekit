import { DataError, describe, isRecord, logisticPair, logisticProbabilities, validAsymptotes } from "@measurekit/core";
import type { ModelParameter } from "../irt/internal.js";
import type { MultidimensionalModel } from "./model.js";

/**
 * Parameters of a compensatory multidimensional logistic item in the slope-intercept form of mirt with D = 1, and the
 * asymptotes of the IRT form: the probability of a correct response is P(θ) = c + (d - c) / (1 + exp(-(a · θ +
 * intercept))).
 */
export interface MultidimensionalLogisticParameters {
  /** The slopes: one per dimension. */
  readonly a: readonly number[];
  /** The intercept, mirt's `d`. */
  readonly intercept: number;
  /** Lower asymptote (guessing); 0 in the M2PL. */
  readonly c: number;
  /** Upper asymptote; 1 in the M2PL and M3PL. */
  readonly d: number;
}

const LOWER_ASYMPTOTE: ModelParameter = { name: "c", lower: 0, upper: 1 };

/** The two logistic values of `logisticPair` that a model reads at once: the calls are synchronous. */
const tails = new Float64Array(2);

/** The multidimensional two-parameter logistic model (M2PL): slopes and an intercept per item, `c = 0`, `d = 1`. */
export const multidimensionalTwoParameterLogistic: MultidimensionalModel<MultidimensionalLogisticParameters> =
  logisticModel("M2PL", false);

/** The multidimensional three-parameter logistic model (M3PL): slopes, an intercept and `c` per item, `d = 1`. */
export const multidimensionalThreeParameterLogistic: MultidimensionalModel<MultidimensionalLogisticParameters> =
  logisticModel("M3PL", true);

/**
 * Checks parameters of the most general item: finite slopes, at least one, a finite intercept, and 0 ≤ c < d ≤ 1.
 *
 * @throws `DataError` when they are not.
 */
function checkLogisticParameters(name: string, parameters: MultidimensionalLogisticParameters): void {
  const { a, intercept, c, d }: Partial<MultidimensionalLogisticParameters> = isRecord(parameters) ? parameters : {};
  const slopes = Array.isArray(a) && a.length > 0 && a.every(Number.isFinite);
  if (!(slopes && Number.isFinite(intercept) && validAsymptotes(c, d))) {
    throw new DataError(
      `An ${name} item needs finite slopes a, at least one, a finite intercept and 0 ≤ c < d ≤ 1, found a = ${describe(a)}, intercept = ${describe(intercept)}, c = ${describe(c)}, d = ${describe(d)}.`,
    );
  }
}

/** Builds a logistic model whose parameter vector is the slopes, the intercept and, with `guessing`, c; d is 1. */
function logisticModel(name: string, guessing: boolean): MultidimensionalModel<MultidimensionalLogisticParameters> {
  return {
    name,
    maxCategories: 2,
    categoryCount: () => 2,
    dimensionCount: ({ a }) => a.length,
    parameters: (_, dimensions) => [
      ...Array.from({ length: dimensions }, (__, d) => ({ name: `a${d + 1}` })),
      { name: "intercept" },
      ...(guessing ? [LOWER_ASYMPTOTE] : []),
    ],
    startingValues(proportions, slopes) {
      // The asymptote starts between 0 and the share correct p, and the intercept where P is about p at theta 0.
      const p = proportions[1] ?? 0.5;
      const c = guessing ? Math.min(0.1, p / 2) : 0;
      const share = (p - c) / (1 - c);
      return { a: Array.from(slopes), intercept: Math.log(share / (1 - share)), c, d: 1 };
    },
    toVector(item) {
      checkLogisticParameters(name, item);
      if (!guessing && item.c !== 0) throw new DataError(`An ${name} item has c = 0, found c = ${item.c}.`);
      if (item.d !== 1) throw new DataError(`An ${name} item has d = 1, found d = ${item.d}.`);
      return Float64Array.of(...item.a, item.intercept, ...(guessing ? [item.c] : []));
    },
    fromVector(vector, dimensions) {
      return {
        a: Array.from(vector.subarray(0, dimensions)),
        intercept: vector[dimensions] ?? 0,
        c: guessing ? (vector[dimensions + 1] ?? 0) : 0,
        d: 1,
      };
    },
    probabilities: ({ a, intercept, c, d }, theta, out) =>
      logisticProbabilities(linearPredictor(a, theta) + intercept, c, d, out),
    thetaGradients({ a, intercept, c, d }, theta, out) {
      const slope = logisticSlope(a, theta, intercept, c, d);
      a.forEach((aj, j) => {
        out[j] = -slope * aj;
        out[a.length + j] = slope * aj;
      });
    },
    parameterDerivatives({ a, intercept, c, d }, theta, out) {
      logisticPair(linearPredictor(a, theta) + intercept, tails);
      const upper = tails[0] ?? 0;
      const lower = tails[1] ?? 0;
      const slope = (d - c) * upper * lower;
      const size = a.length + 1 + (guessing ? 1 : 0);
      // Category 0 is the complement of category 1, so its derivatives are the negatives.
      a.forEach((_, j) => {
        out[j] = -slope * (theta[j] ?? 0);
        out[size + j] = slope * (theta[j] ?? 0);
      });
      out[a.length] = -slope;
      out[size + a.length] = slope;
      if (guessing) {
        out[a.length + 1] = -lower;
        out[size + a.length + 1] = lower;
      }
    },
    information({ a, intercept, c, d }, theta, out) {
      logisticPair(linearPredictor(a, theta) + intercept, tails);
      const upper = tails[0] ?? 0;
      const lower = tails[1] ?? 0;
      const slope = (d - c) * upper * lower;
      // P (1 - P), which is 0 only where the probabilities underflow and the information vanishes.
      const variance = (1 - d + (d - c) * lower) * (c + (d - c) * upper);
      const factor = variance > 0 ? (slope * slope) / variance : 0;
      writeInformation(a, factor, out);
    },
  };
}

/** Writes the information of an item, `factor` times the outer product of the slopes `a`, by rows, into `out`. */
export function writeInformation(a: readonly number[], factor: number, out: Float64Array): void {
  a.forEach((ai, i) => {
    a.forEach((aj, j) => {
      out[i * a.length + j] = factor * (ai * aj);
    });
  });
}

/** Returns a · θ. */
export function linearPredictor(a: readonly number[], theta: Float64Array): number {
  let sum = 0;
  for (let j = 0; j < a.length; j++) sum += (a[j] ?? 0) * (theta[j] ?? 0);
  return sum;
}

/** Returns dP/dz of the logistic item at theta, with z the linear predictor, so that ∇P = dP/dz a. */
function logisticSlope(a: readonly number[], theta: Float64Array, intercept: number, c: number, d: number): number {
  logisticPair(linearPredictor(a, theta) + intercept, tails);
  return (d - c) * (tails[0] ?? 0) * (tails[1] ?? 0);
}

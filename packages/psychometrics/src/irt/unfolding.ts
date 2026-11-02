import { DataError, describe, isRecord } from "@measurekit/core";
import type { IrtModel, ModelParameter } from "./model.js";
import { writeThresholdDerivatives } from "./partial-credit.js";
import { formatList } from "./thresholds.js";
import { singleItemStart } from "./unfolding-start.js";

// The GGUM folds a partial credit item of M + 1 = 2C + 2 subjective categories y = 0 to M: category y has the logit
// a (y (θ - b) - T_(f(y))), f(y) = min(y, M - y), T_z = t_1 + ... + t_z and T_0 = 0, and the observed category z joins
// the subjective categories z and M - z, which a person below and above the item reaches. With π_y the subjective
// probabilities, P_z = π_z + π_(M-z), Y_z = z π_z + (M - z) π_(M-z) and ȳ = Σ Y_z, every derivative is a difference of
// these sums: dP_z/dθ = a (Y_z - ȳ P_z) = -dP_z/db, dP_z/dt_k = a P_z (S_k - [k ≤ z]) with S_k = P(Z ≥ k), and
// dP_z/da = (θ - b) (Y_z - ȳ P_z) - P_z (T_z - T̄) with T̄ = Σ P_z T_z.

/** Parameters of a generalized graded unfolding item with C + 1 categories. */
export interface UnfoldingParameters {
  /** Discrimination, above 0. */
  readonly a: number;
  /** Location: the theta at which a person most agrees with the item. */
  readonly b: number;
  /** The thresholds t_1 to t_C, relative to the location. */
  readonly t: readonly number[];
}

/** The bounds of the discrimination in estimation, the GGUM package's. */
const DISCRIMINATION: ModelParameter = { name: "a", lower: 0.1, upper: 10 };
/** The bounds of the location in estimation, the GGUM package's. */
const LOCATION: ModelParameter = { name: "b", lower: -10, upper: 10 };

/**
 * The generalized graded unfolding model (GGUM, Roberts, Donoghue and Laughlin), for agree-disagree items whose
 * persons agree most near the item's location.
 *
 * @remarks
 * An item with C + 1 categories has P(z | θ) ∝ exp(a (z (θ - b) - T_z)) + exp(a ((M - z) (θ - b) - T_z)), M = 2C + 1,
 * T_z = t_1 + ... + t_z and T_0 = 0. The probability of agreeing falls on both sides of `b`, so a person far below and
 * a person far above an item both disagree with it. Reflecting theta and every b, θ → -θ and b → -b, leaves every
 * probability unchanged. a, b and the thresholds are the GGUM package's α, δ and τ_1 to τ_C; mirt's `ggum` item type
 * writes the thresholds with the opposite sign. With C = 1 the item is dichotomous. The vector holds a, b, then t1 to
 * tC.
 *
 * @see {@link https://doi.org/10.1177/01466216000241001 | Roberts, Donoghue and Laughlin (2000)}
 */
export const generalizedGradedUnfolding: IrtModel<UnfoldingParameters> = unfoldingModel("GGUM", true, false);

/**
 * Returns the graded unfolding model with a = 1, which calibration's first stages fit: with `shared`, every item of the
 * model has the same thresholds, as in Roberts and Laughlin's graded unfolding model, so each number of categories
 * needs its own model; without it, each item has its own.
 */
export function unitUnfoldingModel(shared: boolean): IrtModel<UnfoldingParameters> {
  return unfoldingModel(shared ? "GUM" : "GGUM with a = 1", false, shared);
}

/** Builds an unfolding model whose vector holds a when `slope` is set, then b and the thresholds. */
function unfoldingModel(name: string, slope: boolean, shared: boolean): IrtModel<UnfoldingParameters> {
  const offset = slope ? 1 : 0;
  return {
    name,
    maxCategories: Number.POSITIVE_INFINITY,
    categoryCount: ({ t }) => t.length + 1,
    parameters: (categoryCount) => [
      ...(slope ? [DISCRIMINATION] : []),
      LOCATION,
      ...Array.from({ length: categoryCount - 1 }, (_, k) =>
        shared ? { name: `t${k + 1}`, shared } : { name: `t${k + 1}` },
      ),
    ],
    startingValues: (proportions) => singleItemStart(proportions),
    toVector(parameters) {
      const { a, b, t } = checkUnfoldingParameters(name, parameters);
      return Float64Array.from([...(slope ? [a] : []), b, ...t]);
    },
    fromVector: (vector) => ({
      a: slope ? (vector[0] ?? 0) : 1,
      b: vector[offset] ?? 0,
      t: Array.from(vector.subarray(offset + 1)),
    }),
    probabilities: ({ a, b, t }, theta, out) => unfoldingProbabilities(a, b, t, theta, out),
    thetaDerivatives: ({ a, b, t }, theta, first, second) => unfoldingThetaDerivatives(a, b, t, theta, first, second),
    parameterDerivatives: ({ a, b, t }, theta, out) => unfoldingDerivatives(a, b, t, theta, slope, out),
    information: ({ a, b, t }, theta) => unfoldingInformation(a, b, t, theta),
  };
}

/**
 * Checks parameters of an unfolding item: a finite a above 0, a finite b and at least one finite threshold.
 *
 * @throws `DataError` when they are not.
 */
function checkUnfoldingParameters(model: string, parameters: UnfoldingParameters): UnfoldingParameters {
  const { a, b, t }: Partial<UnfoldingParameters> = isRecord(parameters) ? parameters : {};
  const valid =
    typeof a === "number" &&
    Number.isFinite(a) &&
    a > 0 &&
    typeof b === "number" &&
    Number.isFinite(b) &&
    Array.isArray(t) &&
    t.length > 0 &&
    t.every(Number.isFinite);
  if (!valid) {
    throw new DataError(
      `A ${model} item needs a finite a above 0, a finite b and at least one finite threshold t, found a = ${describe(a)}, b = ${describe(b)}, t = ${formatList(t)}.`,
    );
  }
  return { a, b, t };
}

/**
 * The subjective probabilities of the last {@link weigh}, three entries per observed category z: π_z, π_(M-z) and T_z.
 * It grows to the largest item weighed, so that no call allocates.
 */
let subjective = new Float64Array(15);
/** The sums of the last {@link weigh}: ȳ, the variance of the subjective category, and T̄. */
const sums = new Float64Array(3);
const MEAN = 0;
const VARIANCE = 1;
const MEAN_THRESHOLD = 2;

/**
 * Writes the subjective probabilities of an unfolding item at θ into {@link subjective} and their sums into
 * {@link sums}, every weight taken from the largest logit so that none overflows.
 */
function weigh(a: number, b: number, t: ArrayLike<number>, theta: number): void {
  const count = t.length + 1;
  const top = 2 * count - 1;
  const largest = writeLogits(a, b, t, theta);
  let total = 0;
  for (let k = 0; k < 3 * count; k += 3) {
    subjective[k] = Math.exp((subjective[k] ?? 0) - largest);
    subjective[k + 1] = Math.exp((subjective[k + 1] ?? 0) - largest);
    total += (subjective[k] ?? 0) + (subjective[k + 1] ?? 0);
  }
  let first = 0;
  let second = 0;
  let thresholds = 0;
  for (let z = 0; z < count; z++) {
    const lower = (subjective[3 * z] ?? 0) / total;
    const upper = (subjective[3 * z + 1] ?? 0) / total;
    subjective[3 * z] = lower;
    subjective[3 * z + 1] = upper;
    first += z * lower + (top - z) * upper;
    second += z * z * lower + (top - z) * (top - z) * upper;
    thresholds += (lower + upper) * (subjective[3 * z + 2] ?? 0);
  }
  sums[MEAN] = first;
  sums[VARIANCE] = Math.max(0, second - first * first);
  sums[MEAN_THRESHOLD] = thresholds;
}

/**
 * Writes the logits of an unfolding item's subjective categories z and M - z and T_z into {@link subjective}, growing
 * it to the item, and returns the largest logit.
 */
function writeLogits(a: number, b: number, t: ArrayLike<number>, theta: number): number {
  const count = t.length + 1;
  if (subjective.length < 3 * count) subjective = new Float64Array(3 * count);
  const distance = theta - b;
  const top = 2 * count - 1;
  let cumulative = 0;
  let largest = Number.NEGATIVE_INFINITY;
  for (let z = 0; z < count; z++) {
    if (z > 0) cumulative += t[z - 1] ?? 0;
    const lower = a * (z * distance - cumulative);
    const upper = a * ((top - z) * distance - cumulative);
    subjective[3 * z] = lower;
    subjective[3 * z + 1] = upper;
    subjective[3 * z + 2] = cumulative;
    largest = Math.max(largest, lower, upper);
  }
  return largest;
}

/** Writes the probability of each category of an unfolding item at θ into `out`. */
function unfoldingProbabilities(a: number, b: number, t: ArrayLike<number>, theta: number, out: Float64Array): void {
  weigh(a, b, t, theta);
  for (let z = 0; z <= t.length; z++) out[z] = (subjective[3 * z] ?? 0) + (subjective[3 * z + 1] ?? 0);
}

/** Writes the first and second theta derivatives of an unfolding item's category probabilities. */
function unfoldingThetaDerivatives(
  a: number,
  b: number,
  t: ArrayLike<number>,
  theta: number,
  first: Float64Array,
  second: Float64Array,
): void {
  weigh(a, b, t, theta);
  const top = 2 * t.length + 1;
  const mean = sums[MEAN] ?? 0;
  const variance = sums[VARIANCE] ?? 0;
  for (let z = 0; z <= t.length; z++) {
    const lower = subjective[3 * z] ?? 0;
    const upper = subjective[3 * z + 1] ?? 0;
    first[z] = a * spread(z, top);
    second[z] = a * a * (lower * (z - mean) ** 2 + upper * (top - z - mean) ** 2 - variance * (lower + upper));
  }
}

/**
 * Writes the derivatives of an unfolding item's category probabilities with respect to its vector: a when `slope` is
 * set, then b and t_1 to t_C, by category, then entry.
 */
function unfoldingDerivatives(
  a: number,
  b: number,
  t: ArrayLike<number>,
  theta: number,
  slope: boolean,
  out: Float64Array,
): void {
  weigh(a, b, t, theta);
  const count = t.length + 1;
  const size = count + (slope ? 1 : 0);
  const top = 2 * count - 1;
  const distance = theta - b;
  const meanThreshold = sums[MEAN_THRESHOLD] ?? 0;
  for (let z = 0; z < count; z++) out[z * size] = (subjective[3 * z] ?? 0) + (subjective[3 * z + 1] ?? 0);
  writeThresholdDerivatives(a, count, out, size, slope ? 2 : 1);
  // Column 0 held the probabilities for the thresholds' derivatives; the slope's or the location's replace them.
  for (let z = 0; z < count; z++) {
    const probability = out[z * size] ?? 0;
    const change = spread(z, top);
    if (slope) {
      out[z * size] = distance * change - probability * ((subjective[3 * z + 2] ?? 0) - meanThreshold);
      out[z * size + 1] = -a * change;
    } else {
      out[z * size] = -a * change;
    }
  }
}

/** Returns the information of an unfolding item: a² Σ_z (Y_z - ȳ P_z)² / P_z, over the categories with P_z > 0. */
function unfoldingInformation(a: number, b: number, t: ArrayLike<number>, theta: number): number {
  weigh(a, b, t, theta);
  const top = 2 * t.length + 1;
  let information = 0;
  for (let z = 0; z <= t.length; z++) {
    const probability = (subjective[3 * z] ?? 0) + (subjective[3 * z + 1] ?? 0);
    // A category whose probability underflows adds nothing, and would divide 0 by 0.
    if (probability === 0) continue;
    information += spread(z, top) ** 2 / probability;
  }
  return a * a * information;
}

/** Returns Y_z - ȳ P_z of the last {@link weigh}, dP_z/dθ over a, for the item's M = `top`. */
function spread(z: number, top: number): number {
  const lower = subjective[3 * z] ?? 0;
  const upper = subjective[3 * z + 1] ?? 0;
  return z * lower + (top - z) * upper - (sums[MEAN] ?? 0) * (lower + upper);
}

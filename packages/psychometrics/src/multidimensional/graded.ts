import { DataError, describe, isRecord, logisticPair } from "@measurekit/core";
import { linearPredictor } from "./logistic.js";
import type { MultidimensionalModel } from "./model.js";

/**
 * Parameters of a multidimensional graded response item: with K ordered categories, P(X ≥ k) = 1 / (1 + exp(-(a · θ +
 * intercepts[k - 1]))) for k = 1 to K - 1, so that P(k) = P(X ≥ k) - P(X ≥ k + 1). This is mirt's `graded` item type
 * in its slope-intercept form.
 */
export interface MultidimensionalGradedParameters {
  /** The slopes: one per dimension. */
  readonly a: readonly number[];
  /** The intercepts of the K - 1 boundaries between adjacent categories, strictly decreasing. */
  readonly intercepts: readonly number[];
}

/** The values of `logisticPair` at a boundary: P(X ≥ k) and P(X < k), read at once. */
const tails = new Float64Array(2);

/** The multidimensional graded response model (Samejima): slopes and one intercept per boundary between categories. */
export const multidimensionalGradedResponse: MultidimensionalModel<MultidimensionalGradedParameters> = {
  name: "MGRM",
  maxCategories: Number.POSITIVE_INFINITY,
  categoryCount: ({ intercepts }) => intercepts.length + 1,
  dimensionCount: ({ a }) => a.length,
  parameters: (categoryCount, dimensions) => [
    ...Array.from({ length: dimensions }, (_, d) => ({ name: `a${d + 1}` })),
    ...Array.from({ length: categoryCount - 1 }, (_, k) => ({ name: `intercept${k + 1}` })),
  ],
  startingValues(proportions, slopes) {
    // At theta 0, P(X ≥ k) is the share of persons in categories k and above.
    const intercepts: number[] = [];
    let above = 1;
    for (let k = 1; k < proportions.length; k++) {
      above -= proportions[k - 1] ?? 0;
      intercepts.push(Math.log(above / (1 - above)));
    }
    return { a: Array.from(slopes), intercepts };
  },
  toVector(parameters) {
    const { a, intercepts }: Partial<MultidimensionalGradedParameters> = isRecord(parameters) ? parameters : {};
    const valid =
      Array.isArray(a) &&
      a.length > 0 &&
      a.every(Number.isFinite) &&
      Array.isArray(intercepts) &&
      intercepts.length > 0 &&
      intercepts.every(Number.isFinite);
    if (!valid) {
      throw new DataError(
        `An MGRM item needs finite slopes a, at least one, and at least one finite intercept, found a = ${list(a)}, intercepts = ${list(intercepts)}.`,
      );
    }
    for (let k = 1; k < intercepts.length; k++) {
      if (!((intercepts[k - 1] ?? 0) > (intercepts[k] ?? 0))) {
        throw new DataError(`An MGRM item needs strictly decreasing intercepts, found ${list(intercepts)}.`);
      }
    }
    return Float64Array.of(...a, ...intercepts);
  },
  fromVector: (vector, dimensions) => ({
    a: Array.from(vector.subarray(0, dimensions)),
    intercepts: Array.from(vector.subarray(dimensions)),
  }),
  probabilities({ a, intercepts }, theta, out) {
    // P(k) = P(X ≥ k) P(X < k + 1) (1 - exp(-(c_k - c_(k+1)))), with P(X ≥ 0) = P(X < K) = 1, equals the difference of
    // the cumulative probabilities and keeps the precision of small probabilities in both tails.
    const z = linearPredictor(a, theta);
    let above = 1;
    for (let k = 0; k < intercepts.length; k++) {
      logisticPair(z + (intercepts[k] ?? 0), tails);
      const gap = k === 0 ? 1 : -Math.expm1((intercepts[k] ?? 0) - (intercepts[k - 1] ?? 0));
      out[k] = above * (tails[1] ?? 0) * gap;
      above = tails[0] ?? 0;
    }
    out[intercepts.length] = above;
  },
  thetaGradients({ a, intercepts }, theta, out) {
    const z = linearPredictor(a, theta);
    const size = a.length;
    // Category k gains the slope of P(X ≥ k) and loses that of P(X ≥ k + 1), both multiples of a.
    let slope = 0;
    for (let k = 0; k < intercepts.length; k++) {
      logisticPair(z + (intercepts[k] ?? 0), tails);
      const next = (tails[0] ?? 0) * (tails[1] ?? 0);
      for (let j = 0; j < size; j++) out[k * size + j] = (slope - next) * (a[j] ?? 0);
      slope = next;
    }
    for (let j = 0; j < size; j++) out[intercepts.length * size + j] = slope * (a[j] ?? 0);
  },
  parameterDerivatives({ a, intercepts }, theta, out) {
    const size = a.length + intercepts.length;
    const categories = intercepts.length + 1;
    const z = linearPredictor(a, theta);
    out.fill(0, 0, categories * size);
    // Boundary k moves only P(X ≥ k), which adds to category k and subtracts from category k - 1; the slopes move all.
    for (let k = 1; k < categories; k++) {
      logisticPair(z + (intercepts[k - 1] ?? 0), tails);
      const density = (tails[0] ?? 0) * (tails[1] ?? 0);
      for (let j = 0; j < a.length; j++) {
        const slope = density * (theta[j] ?? 0);
        out[(k - 1) * size + j] = (out[(k - 1) * size + j] ?? 0) - slope;
        out[k * size + j] = (out[k * size + j] ?? 0) + slope;
      }
      out[(k - 1) * size + a.length + k - 1] = -density;
      out[k * size + a.length + k - 1] = density;
    }
  },
  information({ a, intercepts }, theta, out) {
    const z = linearPredictor(a, theta);
    let sum = 0;
    let above = 1;
    let slope = 0;
    for (let k = 0; k < intercepts.length; k++) {
      logisticPair(z + (intercepts[k] ?? 0), tails);
      const upper = tails[0] ?? 0;
      const lower = tails[1] ?? 0;
      const gap = k === 0 ? 1 : -Math.expm1((intercepts[k] ?? 0) - (intercepts[k - 1] ?? 0));
      const probability = above * lower * gap;
      const next = upper * lower;
      const derivative = slope - next;
      if (probability > 0) sum += (derivative * derivative) / probability;
      above = upper;
      slope = next;
    }
    const factor = above > 0 ? sum + (slope * slope) / above : sum;
    a.forEach((ai, i) => {
      a.forEach((aj, j) => {
        out[i * a.length + j] = factor * (ai * aj);
      });
    });
  },
};

/** Formats a list of numbers for a message, or describes any other value. */
function list(values: unknown): string {
  return Array.isArray(values) ? `[${values.map(describe).join(", ")}]` : describe(values);
}

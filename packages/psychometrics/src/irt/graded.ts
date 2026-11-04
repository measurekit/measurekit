import { DataError, gradedProbabilities, hasOrderedThresholds, logistic, type ThresholdItem } from "@measurekit/core";
import type { IrtModel } from "./model.js";
import {
  formatList,
  sharesAbove,
  thresholdParameterList,
  thresholdsFromVector,
  thresholdVector,
} from "./thresholds.js";

/**
 * The graded response model (GRM, Samejima): an item with K ordered categories has a slope `a` and thresholds
 * b1 to b(K-1), and P(X ≥ k) = 1 / (1 + exp(-a (θ - b_k))), so that P(k) = P(X ≥ k) - P(X ≥ k + 1). The thresholds
 * increase when a > 0 and decrease when a < 0. This is mirt's `graded` item type with `IRTpars = TRUE`.
 */
export const gradedResponse: IrtModel<ThresholdItem> = {
  name: "GRM",
  maxCategories: Number.POSITIVE_INFINITY,
  categoryCount: ({ b }) => b.length + 1,
  parameters: (categoryCount) => thresholdParameterList({ name: "a" }, categoryCount),
  startingValues(proportions) {
    // With a = 1, P(X ≥ k) at θ = 0 is the share of persons in categories k and above.
    return { a: 1, b: sharesAbove(proportions).map((above) => Math.log((1 - above) / above)) };
  },
  toVector(parameters) {
    const vector = thresholdVector("GRM", parameters);
    const { a, b } = parameters;
    if (!hasOrderedThresholds(a, b)) {
      throw new DataError(
        `A GRM item with more than two categories needs a nonzero a and thresholds b strictly increasing when a > 0 and strictly decreasing when a < 0, found a = ${a}, b = ${formatList(b)}.`,
      );
    }
    return vector;
  },
  fromVector: thresholdsFromVector,
  probabilities: ({ a, b }, theta, out) => gradedProbabilities(a, b, theta, out),
  thetaDerivatives({ a, b }, theta, first, second) {
    // Category k gains the slope and curvature of P(X ≥ k) and loses those of P(X ≥ k + 1).
    let slope = 0;
    let curvature = 0;
    for (let k = 0; k < b.length; k++) {
      const z = a * (theta - (b[k] ?? 0));
      const upper = logistic(z);
      const lower = logistic(-z);
      const nextSlope = a * upper * lower;
      const nextCurvature = nextSlope * a * (lower - upper);
      first[k] = slope - nextSlope;
      second[k] = curvature - nextCurvature;
      slope = nextSlope;
      curvature = nextCurvature;
    }
    first[b.length] = slope;
    second[b.length] = curvature;
  },
  parameterDerivatives({ a, b }, theta, out) {
    const size = b.length + 1;
    out.fill(0, 0, size * size);
    // Threshold k moves only P(X ≥ k), which adds to category k and subtracts from category k - 1.
    for (let k = 1; k < size; k++) {
      const distance = theta - (b[k - 1] ?? 0);
      const z = a * distance;
      const density = logistic(z) * logistic(-z);
      const slope = distance * density;
      const location = -a * density;
      out[(k - 1) * size] = (out[(k - 1) * size] ?? 0) - slope;
      out[k * size] = (out[k * size] ?? 0) + slope;
      out[(k - 1) * size + k] = -location;
      out[k * size + k] = location;
    }
  },
  information({ a, b }, theta) {
    let sum = 0;
    let above = 1;
    let slope = 0;
    for (let k = 0; k < b.length; k++) {
      const z = a * (theta - (b[k] ?? 0));
      const upper = logistic(z);
      const lower = logistic(-z);
      const gap = k === 0 ? 1 : -Math.expm1(-a * ((b[k] ?? 0) - (b[k - 1] ?? 0)));
      const probability = above * lower * gap;
      const nextSlope = a * upper * lower;
      const derivative = slope - nextSlope;
      if (probability > 0) sum += (derivative * derivative) / probability;
      above = upper;
      slope = nextSlope;
    }
    return above > 0 ? sum + (slope * slope) / above : sum;
  },
};

import { DataError, describe, isRecord, logistic } from "@measurekit/core";
import type { IrtModel, ModelParameter } from "./model.js";

/**
 * Parameters of a dichotomous logistic item in the IRT parameterization with D = 1: the probability of a correct
 * response is P(θ) = c + (d - c) / (1 + exp(-a (θ - b))).
 */
export interface LogisticParameters {
  /** Discrimination: the slope. */
  readonly a: number;
  /** Difficulty: the location. */
  readonly b: number;
  /** Lower asymptote (guessing); 0 in the 1PL and 2PL. */
  readonly c: number;
  /** Upper asymptote; 1 in the 1PL, 2PL and 3PL. */
  readonly d: number;
}

const DISCRIMINATION: ModelParameter = { name: "a" };
const DIFFICULTY: ModelParameter = { name: "b" };
const LOWER_ASYMPTOTE: ModelParameter = { name: "c", lower: 0, upper: 1 };
const UPPER_ASYMPTOTE: ModelParameter = { name: "d", lower: 0, upper: 1 };

/**
 * The one-parameter logistic model (1PL): every item has the same discrimination `a`, estimated, so theta keeps the
 * N(0, 1) scale. It is the Rasch model up to scale: the Rasch parameters are `a * b` and the Rasch ability `a * θ`.
 */
export const oneParameterLogistic: IrtModel<LogisticParameters> = logisticModel("1PL", [
  { ...DISCRIMINATION, shared: true },
  DIFFICULTY,
]);

/** The two-parameter logistic model (2PL): `a` and `b` per item, `c = 0`, `d = 1`. */
export const twoParameterLogistic: IrtModel<LogisticParameters> = logisticModel("2PL", [DISCRIMINATION, DIFFICULTY]);

/** The three-parameter logistic model (3PL): `a`, `b` and `c` per item, `d = 1`. */
export const threeParameterLogistic: IrtModel<LogisticParameters> = logisticModel("3PL", [
  DISCRIMINATION,
  DIFFICULTY,
  LOWER_ASYMPTOTE,
]);

/** The four-parameter logistic model (4PL): `a`, `b`, `c` and `d` per item. */
export const fourParameterLogistic: IrtModel<LogisticParameters> = logisticModel("4PL", [
  DISCRIMINATION,
  DIFFICULTY,
  LOWER_ASYMPTOTE,
  UPPER_ASYMPTOTE,
]);

const LOGISTIC_MODELS: readonly IrtModel<LogisticParameters>[] = [
  oneParameterLogistic,
  twoParameterLogistic,
  threeParameterLogistic,
  fourParameterLogistic,
];

/** Returns whether a model is one of the logistic models, the 1PL to the 4PL. */
export function isLogisticModel(model: IrtModel): model is IrtModel<LogisticParameters> {
  return LOGISTIC_MODELS.some((candidate) => candidate === model);
}

/**
 * Checks parameters of the most general logistic item: an object with finite `a` and `b`, and 0 ≤ c < d ≤ 1.
 *
 * @throws `DataError` when they are not.
 */
function checkLogisticParameters(parameters: LogisticParameters): void {
  const { a, b, c, d }: Partial<LogisticParameters> = isRecord(parameters) ? parameters : {};
  const asymptotes = typeof c === "number" && typeof d === "number" && c >= 0 && c < d && d <= 1;
  if (!(Number.isFinite(a) && Number.isFinite(b) && asymptotes)) {
    throw new DataError(
      `Logistic item parameters need finite a and b and 0 ≤ c < d ≤ 1, found a = ${describe(a)}, b = ${describe(b)}, c = ${describe(c)}, d = ${describe(d)}.`,
    );
  }
}

/**
 * Builds a logistic model whose parameter vector is the first `parameters.length` of a, b, c, d; the others are
 * fixed at c = 0 and d = 1.
 */
function logisticModel(name: string, parameters: readonly ModelParameter[]): IrtModel<LogisticParameters> {
  const size = parameters.length;
  return {
    name,
    maxCategories: 2,
    categoryCount: () => 2,
    parameters: () => parameters,
    startingValues(proportions) {
      // Asymptotes start between the extremes and the share correct p, so that b starts where P(b) is about p.
      const p = proportions[1] ?? 0.5;
      const c = size > 2 ? Math.min(0.1, p / 2) : 0;
      const d = size > 3 ? Math.max(0.9, (1 + p) / 2) : 1;
      const share = (p - c) / (d - c);
      return { a: 1, b: Math.log((1 - share) / share), c, d };
    },
    toVector(item) {
      checkLogisticParameters(item);
      if (size < 3 && item.c !== 0) throw new DataError(`A ${name} item has c = 0, found c = ${item.c}.`);
      if (size < 4 && item.d !== 1) throw new DataError(`A ${name} item has d = 1, found d = ${item.d}.`);
      return Float64Array.of(item.a, item.b, item.c, item.d).slice(0, size);
    },
    fromVector: (vector) => ({
      a: vector[0] ?? 0,
      b: vector[1] ?? 0,
      c: size > 2 ? (vector[2] ?? 0) : 0,
      d: size > 3 ? (vector[3] ?? 1) : 1,
    }),
    probabilities({ a, b, c, d }, theta, out) {
      const z = a * (theta - b);
      // 1 - P from the complementary logistic keeps the precision of small probabilities of a wrong response.
      out[0] = 1 - d + (d - c) * logistic(-z);
      out[1] = c + (d - c) * logistic(z);
    },
    thetaDerivatives({ a, b, c, d }, theta, first, second) {
      const z = a * (theta - b);
      const upper = logistic(z);
      const lower = logistic(-z);
      const slope = (d - c) * a * upper * lower;
      const curvature = slope * a * (lower - upper);
      first[0] = -slope;
      first[1] = slope;
      second[0] = -curvature;
      second[1] = curvature;
    },
    parameterDerivatives({ a, b, c, d }, theta, out) {
      const z = a * (theta - b);
      const upper = logistic(z);
      const lower = logistic(-z);
      const scale = (d - c) * upper * lower;
      const slope = scale * (theta - b);
      const location = -scale * a;
      // Category 0 is the complement of category 1, so its derivatives are the negatives.
      out[0] = -slope;
      out[1] = -location;
      out[size] = slope;
      out[size + 1] = location;
      if (size > 2) {
        out[2] = -lower;
        out[size + 2] = lower;
      }
      if (size > 3) {
        out[3] = -upper;
        out[size + 3] = upper;
      }
    },
    information({ a, b, c, d }, theta) {
      const z = a * (theta - b);
      const upper = logistic(z);
      const lower = logistic(-z);
      const slope = (d - c) * a * upper * lower;
      // P (1 - P), which is 0 only where the probabilities underflow and the information vanishes.
      const variance = (1 - d + (d - c) * lower) * (c + (d - c) * upper);
      return variance > 0 ? (slope * slope) / variance : 0;
    },
  };
}

import { describe, expect, it } from "vitest";
import { cholesky, dot, logistic, logLogistic, maxAbs, mean, normalizeLogs, normalQuantile } from "../src/index.js";

/** Relative tolerance of the sums that the normalization of logs rounds. */
const NORMALIZATION_TOLERANCE = 1e-12;

describe("normalQuantile", () => {
  it("returns infinite quantiles at 0 and 1 and NaN outside [0, 1]", () => {
    expect(normalQuantile(0)).toBe(Number.NEGATIVE_INFINITY);
    expect(normalQuantile(1)).toBe(Number.POSITIVE_INFINITY);
    expect(normalQuantile(-0.1)).toBeNaN();
    expect(normalQuantile(Number.NaN)).toBeNaN();
  });

  it("is 0 at one half and odd around it", () => {
    expect(normalQuantile(0.5)).toBe(0);
    for (const p of [0.001, 0.025, 0.1, 0.3]) expect(normalQuantile(1 - p)).toBeCloseTo(-normalQuantile(p), 12);
  });
});

describe("logistic", () => {
  it("is the probability of a logit", () => {
    expect(logistic(0)).toBe(0.5);
    expect(logistic(Math.log(3))).toBeCloseTo(0.75, 15);
    expect(logistic(-Infinity)).toBe(0);
    expect(logistic(Infinity)).toBe(1);
  });

  it("gives its logarithm finite in both tails in logLogistic", () => {
    for (const x of [-3, -0.5, 0, 0.5, 3]) expect(logLogistic(x)).toBeCloseTo(Math.log(logistic(x)), 14);
    expect(logLogistic(-1000)).toBe(-1000);
    expect(logLogistic(40)).toBe(-Math.exp(-40));
  });
});

describe("cholesky", () => {
  it("factors a positive definite matrix into L Lᵀ", () => {
    const lower = cholesky(Float64Array.of(4, 2, 2, 5), 2);
    expect(Array.from(lower ?? [])).toEqual([2, 0, 1, 2]);
  });

  it("returns null for a matrix that is singular or indefinite to working precision", () => {
    expect(cholesky(Float64Array.of(1, 1, 1, 1), 2)).toBeNull();
    expect(cholesky(Float64Array.of(1, 1, 1, 1 + 1e-17), 2)).toBeNull();
    expect(cholesky(Float64Array.of(1, 2, 2, 1), 2)).toBeNull();
    expect(cholesky(Float64Array.of(0), 1)).toBeNull();
  });
});

describe("normalizeLogs", () => {
  it("turns log weights into probabilities and returns the log of their sum", () => {
    const values = Float64Array.of(Math.log(1), Math.log(3));
    expect(normalizeLogs(values)).toBeCloseTo(Math.log(4), 14);
    expect(values[0]).toBeCloseTo(0.25, 14);
    expect(values[1]).toBeCloseTo(0.75, 14);
  });

  it("keeps weights that exp alone would underflow", () => {
    const values = Float64Array.of(-1000, -1000, -1000, -1000);
    expect(Math.abs(normalizeLogs(values) - (-1000 + Math.log(4)))).toBeLessThan(NORMALIZATION_TOLERANCE);
    expect(values[3]).toBeCloseTo(0.25, 14);
  });

  it("returns -Infinity and leaves the values when every weight is 0", () => {
    const values = Float64Array.of(Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY);
    expect(normalizeLogs(values)).toBe(Number.NEGATIVE_INFINITY);
    expect(Array.from(values)).toEqual([Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY]);
  });
});

describe("vectors", () => {
  it("multiplies and adds in dot", () => {
    expect(dot(Float64Array.of(1, 2, 3), Float64Array.of(4, -5, 6))).toBe(12);
    expect(dot(new Float64Array(0), new Float64Array(0))).toBe(0);
  });

  it("takes the largest absolute value in maxAbs, 0 for no values", () => {
    expect(maxAbs(Float64Array.of(1, -7, 3))).toBe(7);
    expect(maxAbs(new Float64Array(0))).toBe(0);
  });

  it("averages arrays and typed arrays alike in mean", () => {
    expect(mean([1, 2, 6])).toBe(3);
    expect(mean(Float64Array.of(0.5))).toBe(0.5);
  });
});

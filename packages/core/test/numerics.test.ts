import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  ARMIJO,
  backtrack,
  cholesky,
  dot,
  logistic,
  logLogistic,
  maxAbs,
  mean,
  normalizeLogs,
  normalQuantile,
} from "../src/index.js";

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

describe("backtrack", () => {
  /** The objective (x - 2)² of a point x = scale · length along a direction from x = 0. */
  const objective = (scale: number) => (length: number) => (scale * length - 2) ** 2;

  it("accepts the full step when it decreases the objective enough", () => {
    // From x = 0 along +1: the value is 4, the slope -4, and a length of 1 gives 1 <= 4 - 1e-4 · 4.
    expect(backtrack(objective(1), 4, -4)).toBe(1);
  });

  it("halves the step until the decrease is sufficient", () => {
    // Along +10 the objective is (10 t - 2)²: 64 at t = 1, 9 at 1/2, 0.25 at 1/4, which is below 4 - 1e-4 · (1/4) · 40.
    expect(backtrack(objective(10), 4, -40)).toBe(0.25);
  });

  it("requires a decrease of c1 times the length times the slope, and returns null when no length gives it", () => {
    expect(backtrack(() => 5, 4, -1)).toBeNull();
    // A trial of 3.6 misses the bound 4 - 0.5 · 1 · 1 = 3.5 at length 1, and meets 4 - 0.5 · (1/2) · 1 = 3.75 at 1/2.
    expect(backtrack(() => 3.6, 4, -1, { decrease: 0.5 })).toBe(0.5);
    expect(backtrack(() => 3.6, 4, -1, { decrease: 0.5, maxHalvings: 0 })).toBeNull();
  });

  it("starts at the length it is given and halves from there", () => {
    const tried: number[] = [];
    backtrack(
      (length) => {
        tried.push(length);
        return Number.POSITIVE_INFINITY;
      },
      0,
      -1,
      { start: 0.5, maxHalvings: 3 },
    );
    expect(tried).toEqual([0.5, 0.25, 0.125, 0.0625]);
  });

  it("tries the first length and up to maxHalvings more, 39 by default", () => {
    let calls = 0;
    const evaluate = (): number => {
      calls++;
      return Number.POSITIVE_INFINITY;
    };
    expect(backtrack(evaluate, 0, -1, { maxHalvings: 0 })).toBeNull();
    expect(calls).toBe(1);
    calls = 0;
    backtrack(evaluate, 0, -1);
    expect(calls).toBe(40);
  });

  it("accepts a rise within the slack, and a trial that does not rise when no decrease is asked", () => {
    expect(backtrack(() => 4 + 1e-13, 4, -1, { decrease: 0, slack: 1e-12 })).toBe(1);
    expect(backtrack(() => 4 + 1e-11, 4, -1, { decrease: 0, slack: 1e-12, maxHalvings: 2 })).toBeNull();
    expect(backtrack(() => 4, 4, 0, { decrease: 0 })).toBe(1);
  });

  it("leaves the last trial at the accepted length", () => {
    let at = Number.NaN;
    const length = backtrack(
      (t) => {
        at = t;
        return t <= 0.125 ? -1 : 1;
      },
      0,
      -1,
    );
    expect(length).toBe(0.125);
    expect(at).toBe(0.125);
  });

  it("rejects a trial that is NaN, as it satisfies no comparison", () => {
    expect(backtrack(() => Number.NaN, 4, -1, { maxHalvings: 2 })).toBeNull();
  });

  it("takes the Armijo constant 1e-4 by default", () => {
    expect(ARMIJO).toBe(1e-4);
    // At slope -1 and length 1 the accepted bound is the value less 1e-4.
    expect(backtrack(() => 4 - 0.9e-4, 4, -1, { maxHalvings: 0 })).toBeNull();
    expect(backtrack(() => 4 - 1.1e-4, 4, -1, { maxHalvings: 0 })).toBe(1);
  });

  it("finds a length that meets the condition along any descent direction of a quadratic", () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0.1, max: 4, noNaN: true }),
        fc.double({ min: -3, max: -0.01, noNaN: true }),
        (curvature, slope) => {
          const f = (t: number): number => 1 + slope * t + curvature * t * t;
          const length = backtrack(f, 1, slope);
          expect(length).not.toBeNull();
          expect(f(length ?? 0)).toBeLessThanOrEqual(1 + ARMIJO * (length ?? 0) * slope);
        },
      ),
    );
  });
});

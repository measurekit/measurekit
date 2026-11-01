import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  gradedProbabilities,
  logistic,
  logisticPair,
  logisticProbabilities,
  orderedThresholds,
  partialCreditProbabilities,
  softmax,
  validAsymptotes,
} from "../src/index.js";

/** Tolerance of probabilities that are sums and products of a few terms, each rounded once. */
const ROUNDING = 1e-14;

/** Returns the probabilities of a logistic item. */
function logisticOf(z: number, c = 0, d = 1): number[] {
  const out = new Float64Array(2);
  logisticProbabilities(z, c, d, out);
  return Array.from(out);
}

/** Returns the probabilities of a graded item. */
function gradedOf(a: number, b: readonly number[], theta: number): number[] {
  const out = new Float64Array(b.length + 1);
  gradedProbabilities(a, b, theta, out);
  return Array.from(out);
}

/** Returns the probabilities of a partial credit item. */
function partialCreditOf(a: number, location: number, steps: readonly number[], theta: number): number[] {
  const out = new Float64Array(steps.length + 1);
  partialCreditProbabilities(a, location, steps, theta, out);
  return Array.from(out);
}

const sum = (values: readonly number[]): number => values.reduce((total, value) => total + value, 0);

describe("logisticPair", () => {
  it("writes the logistic of x and of -x", () => {
    const out = new Float64Array(2);
    logisticPair(Math.log(3), out);
    expect(out[0]).toBeCloseTo(0.75, 15);
    expect(out[1]).toBeCloseTo(0.25, 15);
    logisticPair(0, out);
    expect(Array.from(out)).toEqual([0.5, 0.5]);
  });

  it("keeps each tail accurate where the other rounds to 1", () => {
    const out = new Float64Array(2);
    logisticPair(40, out);
    expect(out[0]).toBe(1);
    expect(out[1]).toBeCloseTo(Math.exp(-40), 30);
    logisticPair(-740, out);
    expect(out[0]).toBeGreaterThan(0);
    expect(out[1]).toBe(1);
  });
});

describe("logisticProbabilities", () => {
  it("gives the probabilities of the 2PL from the logit", () => {
    expect(logisticOf(0)).toEqual([0.5, 0.5]);
    const [wrong, correct] = logisticOf(Math.log(3));
    expect(correct).toBeCloseTo(0.75, 15);
    expect(wrong).toBeCloseTo(0.25, 15);
  });

  it("puts the asymptotes around the logistic, as c + (d - c) logistic(z)", () => {
    const [wrong, correct] = logisticOf(Math.log(3), 0.2, 0.9);
    expect(correct).toBeCloseTo(0.2 + 0.7 * 0.75, 15);
    expect(wrong).toBeCloseTo(0.1 + 0.7 * 0.25, 15);
  });

  it("keeps the probability of a wrong answer where 1 - P(correct) would be 0", () => {
    const [wrong, correct] = logisticOf(40);
    expect(correct).toBe(1);
    expect(wrong).toBeCloseTo(Math.exp(-40), 30);
  });

  it("stays finite and sums to 1 at extreme logits", () => {
    for (const z of [-1e6, -800, 800, 1e6]) {
      const probabilities = logisticOf(z, 0.1, 0.95);
      expect(probabilities.every(Number.isFinite)).toBe(true);
      expect(sum(probabilities)).toBeCloseTo(1, 15);
    }
  });
});

describe("gradedProbabilities", () => {
  it("is the logistic with two categories, for either sign of the slope", () => {
    expect(gradedOf(1.3, [0.4], 0.9)[1]).toBeCloseTo(logistic(1.3 * 0.5), 15);
    expect(gradedOf(-1.3, [0.4], 0.9)[1]).toBeCloseTo(logistic(-1.3 * 0.5), 15);
  });

  it("gives each category the difference of the cumulative probabilities", () => {
    const a = 1.2;
    const b = [-1, 0.2, 1.4];
    const theta = 0.3;
    const cumulative = [1, ...b.map((threshold) => logistic(a * (theta - threshold))), 0];
    const probabilities = gradedOf(a, b, theta);
    probabilities.forEach((probability, k) => {
      expect(probability).toBeCloseTo((cumulative[k] ?? 0) - (cumulative[k + 1] ?? 0), 14);
    });
  });

  it("keeps the precision of the tails, where the cumulative probabilities round to 1 and 0", () => {
    const probabilities = gradedOf(1, [-1, 1], 40);
    expect(probabilities[2]).toBe(1);
    expect(probabilities[1]).toBeCloseTo(Math.exp(-39) - Math.exp(-41), 30);
    expect(probabilities[0]).toBeGreaterThan(0);
  });

  it("stays in 0 to 1 and sums to 1 at extreme thetas", () => {
    for (const theta of [-1e4, 1e4]) {
      const probabilities = gradedOf(2, [-2, 0, 2], theta);
      expect(probabilities.every((p) => p >= 0 && p <= 1)).toBe(true);
      expect(sum(probabilities)).toBeCloseTo(1, 15);
    }
  });
});

describe("partialCreditProbabilities", () => {
  it("is the logistic of a (θ - b) with two categories", () => {
    expect(partialCreditOf(1.1, 0, [0.4], 0.9)[1]).toBeCloseTo(logistic(1.1 * 0.5), 15);
  });

  it("is uniform for steps at theta, and ordered by the logits otherwise", () => {
    for (const probability of partialCreditOf(1, 0, [0.7, 0.7, 0.7], 0.7)) expect(probability).toBeCloseTo(0.25, 15);
    const [first, second, third] = partialCreditOf(1, 0, [0, 1], 2);
    expect(first).toBeLessThan(defined(second));
    expect(second).toBeLessThan(defined(third));
  });

  it("adds the location to each step", () => {
    const shifted = partialCreditOf(1.4, 0.5, [0.1, -0.4, 0.8], 0.2);
    const written = partialCreditOf(1.4, 0, [0.6, 0.1, 1.3], 0.2);
    shifted.forEach((probability, k) => {
      expect(probability).toBeCloseTo(defined(written[k]), 15);
    });
  });

  it("does not overflow at large logits", () => {
    const probabilities = partialCreditOf(5, 0, [-100, -100], 100);
    expect(probabilities.every(Number.isFinite)).toBe(true);
    expect(probabilities[2]).toBeCloseTo(1, 15);
  });
});

describe("softmax", () => {
  it("turns logits into probabilities in place", () => {
    const values = Float64Array.of(0, Math.log(2), Math.log(5));
    softmax(values, 3);
    expect(values[0]).toBeCloseTo(1 / 8, 15);
    expect(values[1]).toBeCloseTo(2 / 8, 15);
    expect(values[2]).toBeCloseTo(5 / 8, 15);
  });

  it("reads and writes only the entries at the stride", () => {
    const values = Float64Array.of(0, 99, 99, Math.log(3), 99, 99);
    softmax(values, 2, 3);
    expect(values[0]).toBeCloseTo(0.25, 15);
    expect(values[3]).toBeCloseTo(0.75, 15);
    expect([values[1], values[2], values[4], values[5]]).toEqual([99, 99, 99, 99]);
  });

  it("does not overflow at large logits", () => {
    const values = Float64Array.of(1000, 1001, 999);
    softmax(values, 3);
    expect(sum(Array.from(values))).toBeCloseTo(1, 15);
    expect(values.every(Number.isFinite)).toBe(true);
  });
});

describe("validAsymptotes", () => {
  it("is 0 ≤ c < d ≤ 1", () => {
    expect(validAsymptotes(0, 1)).toBe(true);
    expect(validAsymptotes(0.2, 0.9)).toBe(true);
    expect(validAsymptotes(0.5, 0.5)).toBe(false);
    expect(validAsymptotes(0.6, 0.5)).toBe(false);
    expect(validAsymptotes(-0.1, 1)).toBe(false);
    expect(validAsymptotes(0, 1.1)).toBe(false);
  });

  it("is false for values that are not numbers", () => {
    expect(validAsymptotes(JSON.parse('"0"'), 1)).toBe(false);
    expect(validAsymptotes(0, JSON.parse("null"))).toBe(false);
    expect(validAsymptotes(Number.NaN, 1)).toBe(false);
    expect(validAsymptotes(undefined, undefined)).toBe(false);
  });
});

describe("orderedThresholds", () => {
  it("is strictly increasing for a positive slope and strictly decreasing for a negative one", () => {
    expect(orderedThresholds(1, [-1, 0, 2])).toBe(true);
    expect(orderedThresholds(-1, [2, 0, -1])).toBe(true);
    expect(orderedThresholds(1, [-1, 2, 0])).toBe(false);
    expect(orderedThresholds(-1, [-1, 0, 2])).toBe(false);
    expect(orderedThresholds(1, [0, 0])).toBe(false);
  });

  it("accepts one threshold for any slope, and rejects several for a slope of 0", () => {
    expect(orderedThresholds(0, [0.4])).toBe(true);
    expect(orderedThresholds(0, [0, 1])).toBe(false);
  });
});

describe("properties", () => {
  const slopes = fc.double({ min: -4, max: 4, noNaN: true });
  const locations = fc.double({ min: -6, max: 6, noNaN: true });
  const thresholds = fc.array(locations, { minLength: 1, maxLength: 6 });

  it("the probabilities of a logistic item are from 0 to 1 and sum to 1", () => {
    fc.assert(
      fc.property(
        fc.double({ min: -1e3, max: 1e3, noNaN: true }),
        fc.double({ min: 0, max: 0.45, noNaN: true }),
        fc.double({ min: 0.55, max: 1, noNaN: true }),
        (z, c, d) => {
          const probabilities = logisticOf(z, c, d);
          for (const p of probabilities) expect(p >= 0 && p <= 1).toBe(true);
          expect(Math.abs(sum(probabilities) - 1)).toBeLessThan(ROUNDING);
        },
      ),
    );
  });

  it("the probabilities of a graded item are from 0 to 1 and sum to 1", () => {
    fc.assert(
      fc.property(slopes, thresholds, fc.double({ min: -1e3, max: 1e3, noNaN: true }), (a, b, theta) => {
        const ordered = [...b].sort((x, y) => x - y).filter((x, i, all) => i === 0 || x > (all[i - 1] ?? 0));
        const probabilities = gradedOf(Math.abs(a) + 0.1, ordered, theta);
        for (const p of probabilities) expect(p >= 0 && p <= 1).toBe(true);
        expect(Math.abs(sum(probabilities) - 1)).toBeLessThan(ROUNDING * 10);
      }),
    );
  });

  it("the probabilities of a partial credit item are from 0 to 1 and sum to 1, with unordered steps", () => {
    fc.assert(
      fc.property(
        slopes,
        locations,
        thresholds,
        fc.double({ min: -1e3, max: 1e3, noNaN: true }),
        (a, location, steps, theta) => {
          const probabilities = partialCreditOf(a, location, steps, theta);
          for (const p of probabilities) expect(p >= 0 && p <= 1).toBe(true);
          expect(Math.abs(sum(probabilities) - 1)).toBeLessThan(ROUNDING * 10);
        },
      ),
    );
  });

  it("reflecting theta and the thresholds reverses the categories of a graded item", () => {
    fc.assert(
      fc.property(fc.double({ min: 0.1, max: 3, noNaN: true }), thresholds, locations, (a, b, theta) => {
        const increasing = [...new Set(b)].sort((x, y) => x - y);
        const mirrored = increasing.map((x) => -x).reverse();
        const forward = gradedOf(a, increasing, theta);
        const backward = gradedOf(a, mirrored, -theta);
        forward.forEach((probability, k) => {
          expect(Math.abs(probability - defined(backward[forward.length - 1 - k]))).toBeLessThan(ROUNDING * 10);
        });
      }),
    );
  });
});

/** Returns the value, failing the test when it is `undefined`. */
function defined(value: number | undefined): number {
  if (value === undefined) throw new Error("Expected a value.");
  return value;
}

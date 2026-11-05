import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { continuedFraction } from "../src/continued-fraction.js";
import { chiSquareUpperTail, logGamma, studentLowerTail } from "../src/index.js";

/** Relative tolerance of tails that have a closed form, at the precision of the continued fractions. */
const CLOSED_FORM_TOLERANCE = 1e-13;

describe("logGamma", () => {
  it("is the logarithm of the factorial at integers, and of the square root of π at one half", () => {
    expect(logGamma(1)).toBeCloseTo(0, 14);
    expect(logGamma(2)).toBeCloseTo(0, 14);
    expect(logGamma(5)).toBeCloseTo(Math.log(24), 13);
    expect(logGamma(11)).toBeCloseTo(Math.log(3_628_800), 12);
    expect(logGamma(0.5)).toBeCloseTo(Math.log(Math.PI) / 2, 14);
  });

  it("obeys the recurrence Γ(x + 1) = x Γ(x)", () => {
    fc.assert(
      fc.property(fc.double({ min: 0.5, max: 50, noNaN: true }), (x) => {
        expect(logGamma(x + 1) - logGamma(x)).toBeCloseTo(Math.log(x), 12);
      }),
    );
  });
});

describe("chiSquareUpperTail", () => {
  it("is 1 at 0", () => {
    for (const df of [1, 2, 7, 100]) expect(chiSquareUpperTail(0, df)).toBe(1);
  });

  it("is 0 at an infinite statistic", () => {
    for (const df of [1, 2, 7, 100]) expect(chiSquareUpperTail(Number.POSITIVE_INFINITY, df)).toBe(0);
  });

  it("is exp(-x / 2) with 2 degrees of freedom, below and above the switch to the continued fraction", () => {
    for (const x of [0.1, 1, 3.9, 4, 4.1, 10, 50, 300]) {
      expect(chiSquareUpperTail(x, 2) / Math.exp(-x / 2)).toBeCloseTo(1, 12);
    }
  });

  it("is 2 (1 - Φ(√x)) with 1 degree of freedom, as 1 - erf(√(x / 2))", () => {
    // 1 - erf(1 / √2) is the probability of a standard normal beyond one standard deviation.
    expect(chiSquareUpperTail(1, 1)).toBeCloseTo(0.31731050786291415, 14);
  });

  it("meets the continued fraction at the switch with many degrees of freedom", () => {
    // A statistic of df ± 2 is about 0.002 standard deviations from the center, where the tail is close to 1/2.
    for (const df of [1e5, 6e5, 5e6]) {
      const below = chiSquareUpperTail(df - 2, df);
      const above = chiSquareUpperTail(df + 2, df);
      expect(below).toBeGreaterThan(above);
      expect(below - above).toBeLessThan(1e-2);
    }
  });

  it("falls from 1 to 0 as the statistic grows, and rises with the degrees of freedom", () => {
    fc.assert(
      fc.property(
        fc.double({ min: 0, max: 200, noNaN: true }),
        fc.double({ min: 0, max: 50, noNaN: true }),
        fc.integer({ min: 1, max: 60 }),
        (x, step, df) => {
          const tail = chiSquareUpperTail(x, df);
          expect(tail).toBeGreaterThanOrEqual(0);
          expect(tail).toBeLessThanOrEqual(1);
          expect(chiSquareUpperTail(x + step, df)).toBeLessThanOrEqual(tail * (1 + 1e-12));
          expect(chiSquareUpperTail(x, df + 1)).toBeGreaterThanOrEqual(tail * (1 - 1e-12));
        },
      ),
    );
  });
});

describe("studentLowerTail", () => {
  it("is one half at 0", () => {
    for (const df of [1, 2, 9, 1000]) expect(studentLowerTail(0, df)).toBe(0.5);
  });

  it("is the Cauchy distribution function 1 / 2 + atan(t) / π with 1 degree of freedom", () => {
    for (const t of [-50, -3, -0.5, 0.2, 1, 7, 100]) {
      const expected = 0.5 + Math.atan(t) / Math.PI;
      expect(Math.abs(studentLowerTail(t, 1) - expected)).toBeLessThanOrEqual(CLOSED_FORM_TOLERANCE);
    }
  });

  it("is 1 / 2 + t / (2 √(2 + t²)) with 2 degrees of freedom", () => {
    for (const t of [-20, -1.5, 0.3, 2, 9]) {
      const expected = 0.5 + t / (2 * Math.sqrt(2 + t * t));
      expect(Math.abs(studentLowerTail(t, 2) - expected)).toBeLessThanOrEqual(CLOSED_FORM_TOLERANCE);
    }
  });

  it("approaches the normal distribution as the degrees of freedom grow", () => {
    // Φ(1.959964) = 0.975.
    expect(studentLowerTail(1.959964, 1e6)).toBeCloseTo(0.975, 5);
  });

  it("is symmetric, increasing in t, and within 0 and 1", () => {
    fc.assert(
      fc.property(
        fc.double({ min: -30, max: 30, noNaN: true }),
        fc.double({ min: 0, max: 5, noNaN: true }),
        fc.integer({ min: 1, max: 200 }),
        (t, step, df) => {
          const lower = studentLowerTail(t, df);
          expect(lower).toBeGreaterThanOrEqual(0);
          expect(lower).toBeLessThanOrEqual(1);
          expect(lower + studentLowerTail(-t, df)).toBeCloseTo(1, 12);
          expect(studentLowerTail(t + step, df)).toBeGreaterThanOrEqual(lower - 1e-12);
        },
      ),
    );
  });
});

describe("continuedFraction", () => {
  /** Evaluates 1 / (b₀ + a₁ / (b₁ + ...)) from the term `depth` upward, which meets no division by 0 where the tail is positive. */
  function fromTheTail(a: (n: number) => number, b: (n: number) => number, depth: number): number {
    let tail = b(depth);
    for (let n = depth - 1; n >= 0; n--) tail = b(n) + a(n + 1) / tail;
    return 1 / tail;
  }

  it("is 1 / √2 for 1 / (1 + 1 / (2 + 1 / (2 + ...))), as 1 + 1 / (2 + ...) is √2", () => {
    const result = continuedFraction(
      () => 1,
      (n) => (n === 0 ? 1 : 2),
    );
    expect(Math.abs(result - Math.SQRT1_2)).toBeLessThanOrEqual(CLOSED_FORM_TOLERANCE);
  });

  it("goes through a convergent whose denominator is 0, which is infinite, to the limit of the fraction", () => {
    // b₀ + a₁ / b₁ = 1 - 1 / 1 = 0, so the first convergent is infinite; the tail 2 + 1 / (2 + ...) is 1 + √2, which makes
    // the fraction 1 / (1 - 1 / √2) = 2 + √2.
    const a = (n: number): number => (n === 1 ? -1 : 1);
    const b = (n: number): number => (n <= 1 ? 1 : 2);
    const result = continuedFraction(a, b);
    expect(Number.isFinite(result)).toBe(true);
    expect(Math.abs(result - (2 + Math.SQRT2)) / (2 + Math.SQRT2)).toBeLessThanOrEqual(CLOSED_FORM_TOLERANCE);
    expect(Math.abs(result - fromTheTail(a, b, 200)) / result).toBeLessThanOrEqual(CLOSED_FORM_TOLERANCE);
  });

  it("goes through a convergent whose numerator is 0 to the limit of the fraction", () => {
    // b₁ + a₂ / b₂ = 1 - 2 / 2 = 0 makes the second convergent 0, which the ratio of its numerators to the first's
    // takes to 0 and the next to infinity.
    const a = (n: number): number => (n === 1 ? 3 : n === 2 ? -2 : 1);
    const b = (n: number): number => (n <= 1 ? 1 : 2);
    const result = continuedFraction(a, b);
    expect(Number.isFinite(result)).toBe(true);
    expect(Math.abs(result - fromTheTail(a, b, 200)) / result).toBeLessThanOrEqual(CLOSED_FORM_TOLERANCE);
  });
});

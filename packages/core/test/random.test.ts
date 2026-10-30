import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { createRandom, DataError } from "../src/index.js";

describe("createRandom", () => {
  it("reproduces the same sequence for the same seed", () => {
    fc.assert(
      fc.property(fc.maxSafeInteger(), (seed) => {
        const a = createRandom(seed);
        const b = createRandom(seed);
        const draws = (random: typeof a): number[] =>
          Array.from({ length: 20 }, (_, i) => (i % 2 ? random.normal() : random.uniform()));
        expect(draws(a)).toEqual(draws(b));
      }),
    );
  });

  it("gives different sequences for different seeds", () => {
    const first = (seed: number): number => createRandom(seed).uniform();
    expect(new Set([0, 1, 2, -1, 2 ** 32, 2 ** 40, Number.MAX_SAFE_INTEGER].map(first)).size).toBe(7);
  });

  it("draws uniform values in [0, 1) and normal values with mean 0 and variance 1", () => {
    const random = createRandom(42);
    const count = 100_000;
    const uniform = Array.from({ length: count }, () => random.uniform());
    const normal = Array.from({ length: count }, () => random.normal());
    expect(uniform.every((value) => value >= 0 && value < 1)).toBe(true);
    const mean = (values: number[]): number => values.reduce((sum, value) => sum + value, 0) / values.length;
    expect(mean(uniform)).toBeCloseTo(0.5, 2);
    expect(mean(normal)).toBeCloseTo(0, 2);
    expect(mean(normal.map((value) => value * value))).toBeCloseTo(1, 1);
    expect(normal.every(Number.isFinite)).toBe(true);
  });

  it("rejects a seed that is not a safe integer", () => {
    expect(() => createRandom(1.5)).toThrow(DataError);
    expect(() => createRandom(2 ** 53)).toThrow(DataError);
    expect(() => createRandom(JSON.parse('"5"'))).toThrow('A seed must be a safe integer, found "5".');
  });

  it("reproduces a sequence from the same seed and differs between neighboring seeds", () => {
    fc.assert(
      fc.property(fc.integer(), (seed) => {
        const a = createRandom(seed);
        const b = createRandom(seed);
        const c = createRandom(seed + 1);
        const draws = Array.from({ length: 8 }, () => [a.uniform(), a.normal()]);
        expect(Array.from({ length: 8 }, () => [b.uniform(), b.normal()])).toEqual(draws);
        expect(Array.from({ length: 8 }, () => [c.uniform(), c.normal()])).not.toEqual(draws);
      }),
    );
  });

  it("draws normals whose mean and variance lie within four standard errors of 0 and 1", () => {
    const random = createRandom(20261029);
    const count = 100_000;
    const uniforms = Array.from({ length: count }, () => random.uniform());
    expect(uniforms.every((u) => u >= 0 && u < 1)).toBe(true);
    const normals = Array.from({ length: count }, () => random.normal());
    const mean = normals.reduce((sum, value) => sum + value, 0) / count;
    const variance = normals.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (count - 1);
    // Four standard errors: of the mean 1 / √n, of the variance √(2 / n).
    expect(Math.abs(mean)).toBeLessThan(4 / Math.sqrt(count));
    expect(Math.abs(variance - 1)).toBeLessThan(4 * Math.sqrt(2 / count));
  });

  it("names the seed that it rejects", () => {
    expect(() => createRandom(2 ** 53)).toThrow("A seed must be a safe integer, found 9007199254740992.");
  });
});

import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { createRandom, DataError } from "../src/index.js";

describe("createRandom", () => {
  it("reproduces a sequence from the same seed and differs between neighboring seeds", () => {
    fc.assert(
      fc.property(fc.integer({ min: Number.MIN_SAFE_INTEGER, max: Number.MAX_SAFE_INTEGER - 1 }), (seed) => {
        const draws = (random: ReturnType<typeof createRandom>): number[][] =>
          Array.from({ length: 8 }, () => [random.uniform(), random.normal()]);
        const sequence = draws(createRandom(seed));
        expect(draws(createRandom(seed))).toEqual(sequence);
        expect(draws(createRandom(seed + 1))).not.toEqual(sequence);
      }),
    );
  });

  it("gives different sequences for seeds that differ only in their high bits", () => {
    const first = (seed: number): number => createRandom(seed).uniform();
    expect(new Set([0, 1, 2, -1, 2 ** 32, 2 ** 40, Number.MAX_SAFE_INTEGER].map(first)).size).toBe(7);
  });

  it("draws uniforms in [0, 1) and normals whose mean and variance lie within four standard errors of 0 and 1", () => {
    const random = createRandom(20261029);
    const count = 100_000;
    const uniforms = Array.from({ length: count }, () => random.uniform());
    expect(uniforms.every((u) => u >= 0 && u < 1)).toBe(true);
    // The standard error of the mean of uniforms is √(1 / 12n).
    expect(Math.abs(uniforms.reduce((sum, u) => sum + u, 0) / count - 0.5)).toBeLessThan(4 / Math.sqrt(12 * count));
    const normals = Array.from({ length: count }, () => random.normal());
    expect(normals.every(Number.isFinite)).toBe(true);
    const mean = normals.reduce((sum, value) => sum + value, 0) / count;
    const variance = normals.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (count - 1);
    // Four standard errors: of the mean 1 / √n, of the variance √(2 / n).
    expect(Math.abs(mean)).toBeLessThan(4 / Math.sqrt(count));
    expect(Math.abs(variance - 1)).toBeLessThan(4 * Math.sqrt(2 / count));
  });

  it("rejects a seed that is not a safe integer, naming it", () => {
    expect(() => createRandom(1.5)).toThrow(DataError);
    expect(() => createRandom(2 ** 53)).toThrow("A seed must be a safe integer, found 9007199254740992.");
    expect(() => createRandom(JSON.parse('"5"'))).toThrow('A seed must be a safe integer, found "5".');
  });
});

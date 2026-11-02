import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { type KappaWeights, kappaOfPairs } from "../src/index.js";

describe("kappaOfPairs", () => {
  it("gives (p_o - p_e) / (1 - p_e) for two categories", () => {
    // Seven of ten pairs agree; the margins are 6 and 4 against 5 and 5, so chance agreement is 0.6 * 0.5 + 0.4 * 0.5.
    const first = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0];
    const second = [1, 1, 1, 1, 0, 0, 0, 0, 0, 1];
    expect(kappaOfPairs(first, second)).toBeCloseTo(0.4, 14);
  });

  it("weights partial agreement by the distance of the categories over their range", () => {
    // Pairs (0, 0), (1, 2), (2, 2) over the range 2: the agreement of 1 and 2 is 1/2 linear and 3/4 quadratic, and the
    // chance agreements are 1/3, 1/2 and 7/12.
    const first = [0, 1, 2];
    const second = [0, 2, 2];
    expect(kappaOfPairs(first, second, "none")).toBeCloseTo(0.5, 14);
    expect(kappaOfPairs(first, second, "linear")).toBeCloseTo(2 / 3, 14);
    expect(kappaOfPairs(first, second, "quadratic")).toBeCloseTo(0.8, 14);
  });

  it("is 1 for equal lists, whatever the weights", () => {
    for (const weights of ["none", "linear", "quadratic"] as const) {
      expect(kappaOfPairs([0, 3, 1, 2, 2], [0, 3, 1, 2, 2], weights)).toBe(1);
    }
  });

  it("is 0 when no pair agrees and the lists use one category each", () => {
    expect(kappaOfPairs([0, 0, 0], [1, 1, 1])).toBe(0);
  });

  it("is null when chance agreement is 1, as when both lists use one category", () => {
    expect(kappaOfPairs([2, 2, 2], [2, 2, 2])).toBeNull();
    expect(kappaOfPairs([0], [0], "quadratic")).toBeNull();
  });

  it("takes the categories that occur, however far apart, without a table over their range", () => {
    const far = Number.MAX_SAFE_INTEGER;
    // p_o = 2/3 and p_e = 2/3 * 1/3 + 1/3 * 2/3 = 4/9, with no partial credit between categories a range apart.
    for (const weights of ["none", "linear", "quadratic"] as const) {
      expect(kappaOfPairs([0, far, 0], [0, far, far], weights)).toBeCloseTo(0.4, 14);
    }
  });

  it("does not depend on which list is first, on the order of the pairs, or on where the categories start", () => {
    const weights = fc.constantFrom<KappaWeights>("none", "linear", "quadratic");
    const pairs = fc.array(fc.tuple(fc.integer({ min: 0, max: 5 }), fc.integer({ min: 0, max: 5 })), {
      minLength: 2,
      maxLength: 40,
    });
    fc.assert(
      fc.property(pairs, weights, fc.integer({ min: 0, max: 7 }), (values, weight, shift) => {
        const first = values.map(([x]) => x);
        const second = values.map(([, y]) => y);
        const kappa = kappaOfPairs(first, second, weight);
        const swapped = kappaOfPairs(second, first, weight);
        const reversed = kappaOfPairs([...first].reverse(), [...second].reverse(), weight);
        const shifted = kappaOfPairs(
          first.map((x) => x + shift),
          second.map((y) => y + shift),
          weight,
        );
        if (kappa === null) {
          expect([swapped, reversed, shifted]).toEqual([null, null, null]);
          return;
        }
        expect(kappa).toBeLessThanOrEqual(1 + 1e-12);
        for (const other of [swapped, reversed, shifted]) expect(other).toBeCloseTo(kappa, 12);
      }),
    );
  });
});

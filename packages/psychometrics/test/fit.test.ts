import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  createRandom,
  DataError,
  type IrtItem,
  itemFit,
  oneParameterLogistic,
  personFit,
  type ResponseMatrix,
  responseMatrix,
  simulateItems,
  simulateResponses,
  simulateThetas,
  summedScoreFit,
  twoParameterLogistic,
  uniformDistribution,
  yenQ3,
} from "../src/index.js";
import { expectClose } from "./assertions.js";
import { fixtureItems } from "./fixture-items.js";
import rasch from "./fixtures/fit-rasch.json" with { type: "json" };
import residuals from "./fixtures/fit-residuals.json" with { type: "json" };
import summedScore from "./fixtures/fit-summed-score.json" with { type: "json" };

// Tolerances against R, explained in docs/validation.md.
/** Every fit statistic against eRm and mirt, from the same items and thetas. */
const FIT_TOLERANCE = 1e-12;

/** Expects each named statistic of `actual` to be within `tolerance` of the fixture's list of that statistic. */
function expectTable(
  actual: readonly object[],
  expected: Readonly<Record<string, readonly (number | null)[]>>,
  tolerance: number,
  label: string,
): void {
  const rows = actual.map((row) => new Map<string, unknown>(Object.entries(row)));
  for (const [name, values] of Object.entries(expected)) {
    values.forEach((value, i) => {
      const found = rows[i]?.get(name);
      expectClose(typeof found === "number" ? found : null, value, tolerance, `${label} ${name} ${i + 1}`);
    });
  }
}

/** Two 1PL items with b = 0, where a person at θ = 0 has P = 0.5, W = 0.25 and C / W² = 1. */
const HALVES: IrtItem[] = [0, 1].map(() => ({ model: oneParameterLogistic, parameters: { a: 1, b: 0, c: 0, d: 1 } }));

/** A seeded 2PL simulation: items with slopes from 0.5 to 2 and difficulties from -2 to 2, thetas N(0, 1). */
function simulated(seed: number, persons: number, itemCount: number): { matrix: ResponseMatrix; items: IrtItem[] } {
  const random = createRandom(seed);
  const items = simulateItems(twoParameterLogistic, itemCount, random, {
    distributions: { a: uniformDistribution(0.5, 2), b: uniformDistribution(-2, 2) },
  });
  return { matrix: simulateResponses(items, simulateThetas(persons, random), random), items };
}

/** Returns the numbers and nulls of a result, depth first. */
function numbers(value: unknown): (number | null)[] {
  if (typeof value === "number" || value === null) return [value];
  if (typeof value !== "object") return [];
  return Object.values(value).flatMap(numbers);
}

/** Expects two results to hold the same numbers, up to rounding, and the same nulls. */
function expectSameNumbers(actual: unknown, expected: unknown): void {
  const found = numbers(actual);
  const wanted = numbers(expected);
  expect(found).toHaveLength(wanted.length);
  wanted.forEach((value, i) => {
    expectClose(found[i] ?? null, value, 1e-12 * Math.max(1, Math.abs(value ?? 0)));
  });
}

/** Returns the numbers of a result that are NaN or infinite. */
function nonFinite(result: readonly object[]): unknown[] {
  return result.flatMap((row) =>
    Object.values(row).filter((value) => typeof value === "number" && !Number.isFinite(value)),
  );
}

describe("itemFit", () => {
  for (const fixture of rasch.cases) {
    it(`matches eRm's itemfit for the ${fixture.name} model`, () => {
      const result = itemFit(responseMatrix(fixture.responses), fixtureItems(fixture.items), fixture.thetas);
      expectTable(result, fixture.itemFit, FIT_TOLERANCE, fixture.name);
    });
  }

  it("leaves out persons with extreme scores and skips missing responses", () => {
    const rows = [
      [1, 0, null],
      [0, 1, 1],
      [1, 1, 1],
      [0, 0, null],
    ];
    const pool = [...HALVES, ...HALVES.slice(0, 1)];
    const result = itemFit(responseMatrix(rows), pool, [0, 0, 2, -2]);
    expect(result.map((item) => item.outfitMeanSquare)).toEqual([1, 1, 1]);
    expect(result.map((item) => item.infitMeanSquare)).toEqual([1, 1, 1]);
  });

  it("gives null where a statistic is undefined: no other responses, or a mean square without variance", () => {
    const result = itemFit(
      responseMatrix([
        [1, 0, null],
        [0, 1, null],
        [1, 1, 1],
      ]),
      [...HALVES, ...HALVES.slice(0, 1)],
      [0, 0, 1],
    );
    expect(result[0]).toEqual({
      id: "item1",
      infitMeanSquare: 1,
      outfitMeanSquare: 1,
      infitStandardized: null,
      outfitStandardized: null,
    });
    expect(result[2]).toMatchObject({ infitMeanSquare: null, outfitMeanSquare: null, infitStandardized: null });
  });

  it("rejects items that do not match the responses, and invalid thetas", () => {
    const matrix = responseMatrix([
      [1, 0],
      [0, 2],
    ]);
    expect(() => itemFit(matrix, HALVES, [0, 0])).toThrow(DataError);
    expect(() => itemFit(responseMatrix([[1, 0]]), HALVES.slice(0, 1), [0])).toThrow(DataError);
    expect(() => itemFit(responseMatrix([[1, 0]]), HALVES, [0, 1])).toThrow(DataError);
    expect(() => itemFit(responseMatrix([[1, 0]]), HALVES, [Number.NaN])).toThrow(DataError);
    expect(() => itemFit(responseMatrix([[1, 0]]), JSON.parse("null"), [0])).toThrow(
      "The items must be an array, found null.",
    );
    expect(() => personFit(responseMatrix([[1, 0]]), HALVES, JSON.parse("null"))).toThrow(
      "The thetas must be an array, found null.",
    );
    expect(() => summedScoreFit(responseMatrix([[1, 0]]), JSON.parse("null"))).toThrow(
      "The items must be an array, found null.",
    );
    expect(() => yenQ3(responseMatrix([[1, 0]]), JSON.parse("[null, null]"), [0])).toThrow(
      'Item "item1" must be an item with a model and parameters, found null.',
    );
  });
});

describe("yenQ3", () => {
  for (const fixture of residuals.cases) {
    it(`matches mirt's Q3 residuals for ${fixture.name}`, () => {
      const result = yenQ3(responseMatrix(fixture.responses), fixtureItems(fixture.items), fixture.thetas);
      fixture.q3.forEach((row, i) => {
        row.forEach((value, j) => {
          expectClose(result[i]?.[j] ?? null, value, FIT_TOLERANCE, `Q3 ${i + 1}, ${j + 1}`);
        });
      });
    });
  }

  it("uses the persons who answered both items, null with fewer than two or a constant residual", () => {
    const pool = [...HALVES, ...HALVES.slice(0, 1)];
    const result = yenQ3(
      responseMatrix([
        [1, 0, 1],
        [0, 1, null],
        [1, 1, null],
      ]),
      pool,
      [0, 0, 0],
    );
    expect(result).toEqual([
      [1, -0.5, null],
      [-0.5, 1, null],
      [null, null, 1],
    ]);
  });
});

describe("personFit", () => {
  for (const fixture of rasch.cases) {
    it(`matches eRm's personfit for the ${fixture.name} model`, () => {
      const result = personFit(responseMatrix(fixture.responses), fixtureItems(fixture.items), fixture.thetas);
      expectTable(result, fixture.personFit, FIT_TOLERANCE, fixture.name);
    });
  }

  for (const fixture of residuals.cases) {
    it(`matches mirt's Zh for ${fixture.name}`, () => {
      const result = personFit(responseMatrix(fixture.responses), fixtureItems(fixture.items), fixture.thetas);
      expectTable(
        result,
        { standardizedLogLikelihood: fixture.standardizedLogLikelihood },
        FIT_TOLERANCE,
        fixture.name,
      );
    });
  }

  it("gives lz but no mean squares at an extreme score, and nothing without responses", () => {
    const result = personFit(
      responseMatrix([
        [1, 1],
        [null, null],
      ]),
      HALVES,
      [1, 0],
    );
    // At θ = 1 each item has P = e / (1 + e): l0 - E(l0) = 2 (1 - P) and Var(l0) = 2 P (1 - P), lz = √(2 (1 - P) / P).
    const p = Math.E / (1 + Math.E);
    expect(result[0]).toMatchObject({ infitMeanSquare: null, outfitMeanSquare: null });
    expectClose(result[0]?.standardizedLogLikelihood ?? null, Math.sqrt((2 * (1 - p)) / p), 1e-15);
    expect(result[1]).toMatchObject({ infitMeanSquare: null, standardizedLogLikelihood: null });
  });
});

describe("summedScoreFit", () => {
  for (const fixture of summedScore.cases) {
    it(`matches mirt's S-X² for ${fixture.name}`, () => {
      const result = summedScoreFit(responseMatrix(fixture.responses), fixtureItems(fixture.items));
      expectTable(
        result,
        { chiSquare: fixture.chiSquare, significance: fixture.significance },
        FIT_TOLERANCE,
        fixture.name,
      );
      expect(result.map((item) => item.degreesOfFreedom)).toEqual(fixture.degreesOfFreedom);
    });
  }

  it("uses the persons who answered every item", () => {
    const { matrix, items: pool } = simulated(5, 300, 8);
    const withMissing = responseMatrix([...matrix.responses, [null, 1, 0, 1, 0, 1, 0, 1]]);
    expect(summedScoreFit(withMissing, pool)).toEqual(summedScoreFit(matrix, pool));
    expect(() => summedScoreFit(responseMatrix([[null, 1]]), HALVES)).toThrow(DataError);
  });

  it("gives null without degrees of freedom", () => {
    const result = summedScoreFit(
      responseMatrix([
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 0],
      ]),
      HALVES,
    );
    expect(result).toEqual([
      { id: "item1", chiSquare: null, degreesOfFreedom: 0, significance: null },
      { id: "item2", chiSquare: null, degreesOfFreedom: 0, significance: null },
    ]);
  });
});

describe("properties", () => {
  const seeds = fc.integer({ min: 0, max: 2 ** 31 });

  it("statistics never contain NaN or infinities", () => {
    fc.assert(
      fc.property(seeds, fc.integer({ min: 2, max: 30 }), fc.integer({ min: 1, max: 6 }), (seed, persons, count) => {
        const { matrix, items: pool } = simulated(seed, persons, count);
        const thetas = matrix.personIds.map((_, person) => (person % 7) - 3);
        expect(nonFinite(itemFit(matrix, pool, thetas))).toEqual([]);
        expect(nonFinite(personFit(matrix, pool, thetas))).toEqual([]);
        expect(nonFinite(summedScoreFit(matrix, pool))).toEqual([]);
        expect(
          yenQ3(matrix, pool, thetas)
            .flat()
            .every((value) => value === null || Number.isFinite(value)),
        ).toBe(true);
      }),
      { numRuns: 50 },
    );
  });

  it("item fit, S-X² and Q3 do not depend on the order of persons", () => {
    fc.assert(
      fc.property(seeds, (seed) => {
        const { matrix, items: pool } = simulated(seed, 60, 5);
        const thetas = matrix.personIds.map((_, person) => Math.sin(person));
        const reversed = responseMatrix([...matrix.responses].reverse());
        const reversedThetas = [...thetas].reverse();
        expectSameNumbers(itemFit(reversed, pool, reversedThetas), itemFit(matrix, pool, thetas));
        expectSameNumbers(summedScoreFit(reversed, pool), summedScoreFit(matrix, pool));
        expectSameNumbers(yenQ3(reversed, pool, reversedThetas), yenQ3(matrix, pool, thetas));
      }),
      { numRuns: 20 },
    );
  });
});

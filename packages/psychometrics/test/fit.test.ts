import { chiSquareUpperTail } from "@measurekit/core";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  type CalibratedItem,
  type Calibration,
  calibrate,
  createRandom,
  DataError,
  type IrtItem,
  type IrtModel,
  itemFit,
  type LimitedInformationFit,
  limitedInformationFit,
  nominalResponse,
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
import { defined, expectClose } from "./assertions.js";
import { definedEntries, type FixtureParameters, fixtureItems, MODELS } from "./fixture-items.js";
import limited from "./fixtures/fit-limited-information.json" with { type: "json" };
import rasch from "./fixtures/fit-rasch.json" with { type: "json" };
import residuals from "./fixtures/fit-residuals.json" with { type: "json" };
import summedScore from "./fixtures/fit-summed-score.json" with { type: "json" };

// Tolerances against R, explained in docs/validation.md.
/** Every fit statistic against eRm and mirt, from the same items and thetas. */
const FIT_TOLERANCE = 1e-12;
/** M2, RMSEA, its interval, SRMSR, TLI and CFI against mirt, each from its own calibration. */
const LIMITED_TOLERANCE = 1e-3;
/** The EM tolerance of those calibrations, as the script's. */
const LIMITED_EM_TOLERANCE = 1e-8;
/** M2 of the rejection sets, whose calibrations stop at the reference's 1e-6. */
const REJECTION_TOLERANCE = 2e-2;

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

/** Expects the fit to equal the reference's values. */
function expectLimitedFit(actual: LimitedInformationFit, expected: typeof limited.level.expected, label: string): void {
  expectClose(actual.statistic, expected.statistic, LIMITED_TOLERANCE, `${label} statistic`);
  expect(actual.degreesOfFreedom, label).toBe(expected.degreesOfFreedom);
  expectClose(actual.significance, expected.significance, LIMITED_TOLERANCE, `${label} significance`);
  expectClose(actual.rmsea, expected.rmsea, LIMITED_TOLERANCE, `${label} RMSEA`);
  expected.rmseaInterval.forEach((bound, i) => {
    expectClose(actual.rmseaInterval?.[i] ?? null, bound, LIMITED_TOLERANCE, `${label} RMSEA interval ${i}`);
  });
  expectClose(actual.srmsr, expected.srmsr, LIMITED_TOLERANCE, `${label} SRMSR`);
  expectClose(actual.tli, expected.tli, LIMITED_TOLERANCE, `${label} TLI`);
  expectClose(actual.cfi, expected.cfi, LIMITED_TOLERANCE, `${label} CFI`);
}

/** Returns the calibration of a fixture's responses under a model name, as tight as the reference's. */
function calibratedCase(
  matrix: ResponseMatrix,
  model: string,
  fixed?: Readonly<Record<string, FixtureParameters>>,
  latent?: string,
): Calibration<unknown> {
  return calibrate(matrix, {
    model: defined(MODELS[model]),
    tolerance: LIMITED_EM_TOLERANCE,
    maxIterations: 100_000,
    ...(fixed !== undefined && { fixed: definedEntries(fixed) }),
    ...(latent !== undefined && { latent: "normal" as const }),
  });
}

describe("limitedInformationFit", () => {
  const binary = responseMatrix(limited.binary.responses);
  const polytomous = responseMatrix(limited.polytomous.responses);
  const twoParameter = calibratedCase(binary, "2pl");

  for (const { name, model, fixed, latent, expected, mirtStatistic } of limited.binary.cases) {
    it(`matches mirt's M2 for ${name}`, () => {
      const calibration = calibratedCase(binary, model, fixed, latent);
      const anchors = fixed === undefined ? {} : { fixed: Object.keys(fixed) };
      const result = limitedInformationFit(binary, calibration, anchors);
      expectLimitedFit(result, expected, name);
      if (mirtStatistic !== undefined) expect(mirtStatistic).toBeGreaterThan(defined(result.statistic));
    });
  }

  for (const { name, model, statistic, expected } of limited.polytomous.cases) {
    it(`matches mirt for ${name}`, () => {
      const result = limitedInformationFit(polytomous, calibratedCase(polytomous, model), {
        statistic: statistic === "c2" ? "c2" : "m2-star",
      });
      expectLimitedFit(result, expected, name);
    });
  }

  it("gives 54 degrees of freedom to the 2PL and 65 to the 1PL on 12 items", () => {
    expect(limitedInformationFit(binary, twoParameter).degreesOfFreedom).toBe(54);
    const shared = limitedInformationFit(binary, calibratedCase(binary, "1pl"));
    expect(shared.degreesOfFreedom).toBe(65);
  });

  it("rejects a true model as often as mirt on the same sets", () => {
    const { sets, alpha, degreesOfFreedom } = limited.rejections;
    let rejected = 0;
    let rejectedByMirt = 0;
    for (const { responses, statistic } of sets) {
      const matrix = responseMatrix(responses.map((row) => Array.from(row, Number)));
      const result = limitedInformationFit(
        matrix,
        calibrate(matrix, { model: twoParameterLogistic, tolerance: 1e-6, maxIterations: 10_000 }),
      );
      expectClose(result.statistic, statistic, REJECTION_TOLERANCE, "M2 of a set");
      expect(result.degreesOfFreedom).toBe(degreesOfFreedom);
      if (defined(result.significance) < alpha) rejected++;
      if (chiSquareUpperTail(statistic, degreesOfFreedom) < alpha) rejectedByMirt++;
    }
    expect(Math.abs(rejected - rejectedByMirt) / sets.length).toBeLessThanOrEqual(0.02);
    expect(rejected / sets.length).toBeLessThan(0.2);
  });

  it("takes the level of the interval", () => {
    const calibration = calibratedCase(binary, "1pl");
    const result = limitedInformationFit(binary, calibration, { level: limited.level.level });
    expectLimitedFit(result, limited.level.expected, "the 0.95 interval");
  });

  it("gives M2 and C2 the value of M2* on dichotomous items", () => {
    const star = limitedInformationFit(binary, twoParameter);
    expectSameNumbers(limitedInformationFit(binary, twoParameter, { statistic: "m2" }), star);
    expectSameNumbers(limitedInformationFit(binary, twoParameter, { statistic: "c2" }), star);
  });

  it("uses the persons who answered every item", () => {
    const missing = responseMatrix([
      ...binary.responses,
      [null, ...defined(binary.responses[0]).slice(1)],
      [...defined(binary.responses[1]).slice(0, -1), null],
    ]);
    expectSameNumbers(limitedInformationFit(missing, twoParameter), limitedInformationFit(binary, twoParameter));
  });

  it("leaves the statistic null without degrees of freedom, with SRMSR", () => {
    const { matrix } = simulated(21, 300, 3);
    const result = limitedInformationFit(matrix, calibrate(matrix, { model: twoParameterLogistic }));
    expect(result).toMatchObject({
      statistic: null,
      degreesOfFreedom: 0,
      significance: null,
      rmsea: null,
      rmseaInterval: null,
      tli: null,
      cfi: null,
    });
    expect(result.srmsr).toBeGreaterThanOrEqual(0);
  });

  it("leaves SRMSR, TLI and CFI null for an anchor without variance", () => {
    const rows = simulated(22, 300, 5).matrix.responses.map((row) => [1, ...row]);
    const matrix = responseMatrix(rows);
    const anchor = { a: 1, b: -3, c: 0, d: 1 };
    const calibration = calibrate(matrix, {
      model: twoParameterLogistic,
      fixed: { [defined(matrix.itemIds[0])]: anchor },
    });
    const result = limitedInformationFit(matrix, calibration, { fixed: [defined(matrix.itemIds[0])] });
    expect(result.statistic).not.toBeNull();
    expect(result).toMatchObject({ srmsr: null, tli: null, cfi: null });
  });

  it("leaves TLI and CFI null when the independence model fits no worse", () => {
    // A full factorial: every pair of items has a correlation of exactly 0, so that N Σ r² is 0.
    const factorial = responseMatrix(
      Array.from({ length: 64 }, (_, person) => [0, 1, 2, 3, 4, 5].map((i) => (person >> i) & 1)),
    );
    const result = limitedInformationFit(factorial, calibrate(factorial, { model: oneParameterLogistic }));
    expect(defined(result.statistic)).toBeGreaterThan(0);
    expect(result).toMatchObject({ tli: null, cfi: null });
  });

  it("refuses more than 60 items", () => {
    const { matrix } = simulated(24, 50, 61);
    const calibration = calibrate(matrix, { model: twoParameterLogistic, maxIterations: 1 });
    expect(() => limitedInformationFit(matrix, calibration)).toThrow(/at most 60 items, found 61/);
  });

  it("refuses a calibration of several groups, covariates, testlets or a histogram", () => {
    const { matrix } = simulated(25, 200, 6);
    const groups = matrix.personIds.map((_, person) => (person % 2 === 0 ? "a" : "b"));
    const model = twoParameterLogistic;
    const grouped = calibrate(matrix, { model, groups, maxIterations: 2 });
    expect(() => limitedInformationFit(matrix, grouped)).toThrow(/one group, found 2/);
    const covariates = { x: matrix.personIds.map((_, person) => person % 3) };
    const regression = calibrate(matrix, { model, covariates, maxIterations: 2 });
    expect(() => limitedInformationFit(matrix, regression)).toThrow(/does not take covariates/);
    const testlets = { passage: matrix.itemIds.slice(0, 2) };
    const local = calibrate(matrix, { model, testlets, maxIterations: 2 });
    expect(() => limitedInformationFit(matrix, local)).toThrow(/does not take testlets/);
    const anchors = { [defined(matrix.itemIds[0])]: { a: 1, b: 0, c: 0, d: 1 } };
    const histogram = calibrate(matrix, { model, fixed: anchors, latent: "empirical-histogram", maxIterations: 2 });
    expect(() => limitedInformationFit(matrix, histogram)).toThrow(/empirical histogram/);
  });

  it("asks for the anchors when the parameters differ from the calibration's AIC", () => {
    const calibration = calibratedCase(binary, "2pl", limited.binary.cases[3]?.fixed);
    expect(() => limitedInformationFit(binary, calibration)).toThrow(/give the anchors' ids in the option fixed/);
    expect(() => limitedInformationFit(binary, calibration, { fixed: ["item1"] })).toThrow(
      /estimated 16 parameters, but these items have 22/,
    );
  });

  it("refuses M2 for polytomous items and a nominal item for every statistic", () => {
    const matrix = responseMatrix([
      [0, 1, 2],
      [1, 2, 0],
      [2, 0, 1],
      [0, 0, 0],
      [1, 1, 1],
      [2, 2, 2],
    ]);
    const graded = calibrate(matrix, { model: defined(MODELS.grm), maxIterations: 2 });
    expect(() => limitedInformationFit(matrix, graded, { statistic: "m2" })).toThrow(/"m2" needs dichotomous/);
    const nominal = calibrate(matrix, { model: nominalResponse, maxIterations: 2 });
    expect(() => limitedInformationFit(matrix, nominal)).toThrow(/ordered categories/);
    expect(() => limitedInformationFit(matrix, nominal, { statistic: "c2" })).toThrow(/ordered categories/);
  });

  it("needs two persons who answered every item", () => {
    const { matrix } = simulated(26, 50, 5);
    const calibration = calibrate(matrix, { model: twoParameterLogistic, maxIterations: 2 });
    const few = responseMatrix(matrix.responses.map((row, person) => (person === 0 ? row : [null, ...row.slice(1)])));
    expect(() => limitedInformationFit(few, calibration)).toThrow(
      /at least two persons who answered every item, found 1/,
    );
  });

  it("refuses a calibration of other items", () => {
    const { matrix } = simulated(27, 50, 5);
    const calibration = calibrate(matrix, { model: twoParameterLogistic, maxIterations: 2 });
    const other = { ...matrix, itemIds: [...matrix.itemIds].reverse() };
    expect(() => limitedInformationFit(other, calibration)).toThrow(/but the calibration has/);
  });

  it("refuses arguments of other types", () => {
    const { matrix } = simulated(28, 50, 5);
    const calibration = calibrate(matrix, { model: twoParameterLogistic, maxIterations: 2 });
    expect(() => limitedInformationFit(JSON.parse("null"), calibration)).toThrow(DataError);
    expect(() => limitedInformationFit(matrix, JSON.parse("null"))).toThrow(/must be the result of calibrate/);
    expect(() => limitedInformationFit(matrix, JSON.parse("[]"))).toThrow(DataError);
    expect(() => limitedInformationFit(matrix, calibration, JSON.parse("null"))).toThrow(DataError);
    const options = (value: string) => () => limitedInformationFit(matrix, calibration, JSON.parse(value));
    expect(options('{ "statistic": "m3" }')).toThrow(/The option statistic/);
    expect(options('{ "statistic": null }')).toThrow(/The option statistic/);
    expect(options('{ "level": 1 }')).toThrow(/The option level/);
    expect(options('{ "level": "0.9" }')).toThrow(/The option level/);
    expect(options('{ "fixed": "item1" }')).toThrow(/The option fixed/);
    expect(options('{ "fixed": ["nobody"] }')).toThrow(/not an item id/);
    expect(options('{ "fixed": [1] }')).toThrow(/not an item id/);
    const calibrationWith = (fields: string) => ({ ...calibration, ...JSON.parse(fields) });
    expect(() => limitedInformationFit(matrix, calibrationWith('{ "aic": null }'))).toThrow(
      /The AIC of the calibration/,
    );
    expect(() => limitedInformationFit(matrix, calibrationWith('{ "logLikelihood": "1" }'))).toThrow(/log-likelihood/);
    const group = defined(calibration.groups[0]);
    const normal = { ...group, distribution: "normal" as const, sd: 0 };
    expect(() => limitedInformationFit(matrix, { ...calibration, groups: [normal] })).toThrow(/standard deviation/);
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

  it("the limited-information fit is at least 0 and does not depend on the order of persons or items", () => {
    fc.assert(
      fc.property(seeds, fc.constantFrom<"m2" | "m2-star" | "c2">("m2", "m2-star", "c2"), (seed, statistic) => {
        const { matrix } = simulated(seed, 150, 5);
        const model: IrtModel = oneParameterLogistic;
        const calibration = calibrate(matrix, { model, maxIterations: 50 });
        const fit = limitedInformationFit(matrix, calibration, { statistic });
        expect(defined(fit.statistic)).toBeGreaterThanOrEqual(0);
        const persons = responseMatrix([...matrix.responses].reverse(), { itemIds: matrix.itemIds });
        expectSameNumbers(limitedInformationFit(persons, calibration, { statistic }), fit);
        const order = matrix.itemIds.map((_, i) => matrix.itemIds.length - 1 - i);
        const items = responseMatrix(
          matrix.responses.map((row) => order.map((i) => row[i] ?? null)),
          { itemIds: order.map((i) => defined(matrix.itemIds[i])) },
        );
        const reversed: Calibration<unknown> = {
          ...calibration,
          items: order.map((i): CalibratedItem<unknown> => defined(calibration.items[i])),
        };
        expectSameNumbers(limitedInformationFit(items, reversed, { statistic }), fit);
      }),
      { numRuns: 20 },
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

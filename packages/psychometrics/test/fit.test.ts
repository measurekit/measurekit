import { chiSquareUpperTail, normalQuantile } from "@measurekit/core";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  andersenTest,
  buildResponseMatrix,
  type CalibratedItem,
  type Calibration,
  type ConditionalModel,
  calibrate,
  correctedPersonFit,
  createRandom,
  DataError,
  generalizedPartialCredit,
  gradedResponse,
  type IrtItem,
  type IrtModel,
  type ItemSplit,
  itemFit,
  kernelItemCurves,
  type LimitedInformationFit,
  limitedInformationFit,
  localDependenceChiSquare,
  martinLofTest,
  maximumLikelihoodEstimate,
  nominalResponse,
  oneParameterLogistic,
  personFit,
  type RaschSplit,
  type ResponseMatrix,
  raschWaldTest,
  rootMeanSquareDeviation,
  simulateItems,
  simulateResponses,
  simulateThetas,
  summedScoreFit,
  threeParameterLogistic,
  twoParameterLogistic,
  uniformDistribution,
  yenQ3,
} from "../src/index.js";
import { defined, expectClose } from "./assertions.js";
import { definedEntries, type FixtureParameters, fixtureItems, MODELS } from "./fixture-items.js";
import {
  type Cat3plFixture,
  type FitResidualsFixture,
  type IrtFixedParametersFixture,
  readFixture,
} from "./fixtures.js";

const cat3pl = readFixture<Cat3plFixture>("cat-3pl.json");

interface FitCorrectedPersonFitFixture {
  threePl: { responses: (number | null)[][] };
  cases: {
    name: string;
    estimator: string;
    prior: { mean: number; sd: number };
    thetas: number[];
    standardized: (number | null)[];
  }[];
}
const correctedFit = readFixture<FitCorrectedPersonFitFixture>("fit-corrected-person-fit.json");

interface FitKernelCurvesFixture {
  cases: {
    name: string;
    data: string;
    options: { points?: number; bandwidth?: number };
    bandwidth: number;
    nodes: number[];
    probabilities: number[][][];
    standardErrors: number[][][];
  }[];
}
const kernelCurves = readFixture<FitKernelCurvesFixture>("fit-kernel-curves.json");

interface FitLimitedInformationFixture {
  binary: {
    responses: number[][];
    cases: {
      name: string;
      model: string;
      expected: {
        statistic: number;
        degreesOfFreedom: number;
        significance: number;
        rmsea: number;
        rmseaInterval: number[];
        srmsr: number;
        tli: number;
        cfi: number;
      };
      mirtStatistic?: number;
      fixed?: {
        item1: { a: number; b: number; c: number; d: number };
        item2: { a: number; b: number; c: number; d: number };
        item3: { a: number; b: number; c: number; d: number };
        item4: { a: number; b: number; c: number; d: number };
      };
      latent?: string;
    }[];
  };
  polytomous: {
    responses: number[][];
    cases: {
      name: string;
      model: string;
      statistic: string;
      expected: {
        statistic: number;
        degreesOfFreedom: number;
        significance: number;
        rmsea: number;
        rmseaInterval: number[];
        srmsr: number;
        tli: number;
        cfi: number;
      };
    }[];
  };
  level: {
    level: number;
    expected: {
      statistic: number;
      degreesOfFreedom: number;
      significance: number;
      rmsea: number;
      rmseaInterval: number[];
      srmsr: number;
      tli: number;
      cfi: number;
    };
  };
  rejections: { sets: { responses: string[]; statistic: number }[]; alpha: number; degreesOfFreedom: number };
  held: {
    responses: number[][];
    models: string[];
    fixedParameters: {
      item1: { c: number };
      item2: { c: number };
      item3: { c: number };
      item4: { c: number };
      item5: { c: number };
      item12: { a: number };
    };
    mirtStatistic: number;
    expected: {
      statistic: number;
      degreesOfFreedom: number;
      significance: number;
      rmsea: number;
      rmseaInterval: number[];
      srmsr: number;
      tli: number;
      cfi: number;
    };
  };
}
const limited = readFixture<FitLimitedInformationFixture>("fit-limited-information.json");

// biome-ignore-start lint/style/useNamingConvention: the keys are those of the fixture file.
interface FitLocalDependenceFixture {
  cases: {
    name: string;
    responses: (number | null)[][];
    items: { model: string; parameters: { a: number; b: number | number[]; c?: number; d?: number } }[];
    chiSquare: number[];
    cramersV: number[];
    degreesOfFreedom: number[];
    significance: number[];
    population?: { nodes: number[]; weights: number[] };
  }[];
}
// biome-ignore-end lint/style/useNamingConvention: the keys are those of the fixture file.
const localDependence = readFixture<FitLocalDependenceFixture>("fit-local-dependence.json");

interface FitRaschFixture {
  cases: {
    name: string;
    responses: number[][];
    items: { model: string; parameters: { a: number; b: number | number[]; c?: number; d?: number } }[];
    thetas: number[];
    itemFit: {
      infitMeanSquare: number[];
      outfitMeanSquare: number[];
      infitStandardized: number[];
      outfitStandardized: number[];
    };
    personFit: {
      infitMeanSquare: (number | null)[];
      outfitMeanSquare: (number | null)[];
      infitStandardized: (number | null)[];
      outfitStandardized: (number | null)[];
    };
  }[];
}
const rasch = readFixture<FitRaschFixture>("fit-rasch.json");

interface FitRaschTestsFixture {
  andersen: {
    name: string;
    model: string;
    responses: (number | null)[][];
    split: string | string[];
    statistic: number;
    degreesOfFreedom: number;
    significance: number;
    excluded: string[];
    persons: number;
    erm?: number;
  }[];
  wald: {
    name: string;
    model: string;
    responses: (number | null)[][];
    split: string | string[];
    z: (number | null)[];
    significance: (number | null)[];
    excluded: string[];
    persons: number;
  }[];
  martinLof: {
    name: string;
    model: string;
    responses: (number | null)[][];
    split: string | string[][];
    statistic: number;
    degreesOfFreedom: number;
    significance: number;
    ermDegreesOfFreedom: number;
    persons: number;
  }[];
}
const raschTests = readFixture<FitRaschTestsFixture>("fit-rasch-tests.json");

const residuals = readFixture<FitResidualsFixture>("fit-residuals.json");

interface FitRmsdFixture {
  cases: {
    name: string;
    responses: (number | null)[][];
    items: { model: string; parameters: { a: number; b: number | number[]; c?: number; d?: number } }[];
    population: { nodes: number[]; weights: number[] };
    rmsd: number[];
    meanDeviation: number[];
    tamMeanDeviation: number[];
  }[];
}
const rmsd = readFixture<FitRmsdFixture>("fit-rmsd.json");

interface FitSummedScoreFixture {
  cases: {
    name: string;
    responses: number[][];
    items: { model: string; parameters: { a: number; b: number | number[]; c?: number; d?: number } }[];
    chiSquare: number[];
    degreesOfFreedom: number[];
    significance: number[];
  }[];
}
const summedScore = readFixture<FitSummedScoreFixture>("fit-summed-score.json");

const fixedParameters = readFixture<IrtFixedParametersFixture>("irt-fixed-parameters.json");

// Tolerances against R, explained in docs/validation.md.
/** Every fit statistic against eRm and mirt, from the same items and thetas. */
const FIT_TOLERANCE = 1e-12;
/** M2, RMSEA, its interval, SRMSR, TLI and CFI against mirt, each from its own calibration. */
const LIMITED_TOLERANCE = 1e-3;
/** The EM tolerance of those calibrations, as the script's. */
const LIMITED_EM_TOLERANCE = 1e-8;
/** M2 of the rejection sets, whose calibrations stop at the reference's 1e-6. */
const REJECTION_TOLERANCE = 2e-2;
/** The RMSD and mean deviation against TAM, whose calibrations stop at 1e-10. */
const RMSD_TOLERANCE = 1e-8;
/** The likelihood-ratio statistics of the Rasch tests against eRm, whose nlm stops short of the maximum. */
const RASCH_STATISTIC_TOLERANCE = 1e-6;
/** The Wald z of an item against eRm, first-order in the estimates where nlm stops. */
const RASCH_WALD_TOLERANCE = 1e-3;
/** Andersen's statistic against base R's conditional likelihood, maximized by optim to 1e-15. */
const RASCH_BASE_TOLERANCE = 1e-8;
/** lz* against the base R write-up of PerFit's lzstar, unrounded: the same sums in another order. */
const CORRECTED_TOLERANCE = 1e-12;
/** Kernel curves, their nodes and standard errors against KernSmoothIRT and base R: the same sums. */
const KERNEL_TOLERANCE = 1e-12;

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
function simulated(
  seed: number,
  persons: number,
  itemCount: number,
): { matrix: ResponseMatrix; items: readonly IrtItem[] } {
  const random = createRandom(seed);
  const items = simulateItems(twoParameterLogistic, itemCount, random, {
    distributions: { a: uniformDistribution(0.5, 2), b: uniformDistribution(-2, 2) },
  });
  return { matrix: simulateResponses(items, simulateThetas(persons, random), random), items };
}

/** The responses of the calibration with held parameters, and its items as mirt estimated them. */
const heldMatrix = buildResponseMatrix(fixedParameters.responses, { itemIds: fixedParameters.itemIds });
const heldItems: IrtItem[] = fixedParameters.models.map((name, i) => ({
  model: defined(MODELS[name]),
  parameters: defined(fixedParameters.calibrations[0]?.items[i]).parameters,
}));

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
      const result = itemFit(buildResponseMatrix(fixture.responses), fixtureItems(fixture.items), fixture.thetas);
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
    const result = itemFit(buildResponseMatrix(rows), pool, [0, 0, 2, -2]);
    expect(result.map((item) => item.outfitMeanSquare)).toEqual([1, 1, 1]);
    expect(result.map((item) => item.infitMeanSquare)).toEqual([1, 1, 1]);
  });

  it("gives null where a statistic is undefined: no other responses, or a mean square without variance", () => {
    const result = itemFit(
      buildResponseMatrix([
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
    const matrix = buildResponseMatrix([
      [1, 0],
      [0, 2],
    ]);
    expect(() => itemFit(matrix, HALVES, [0, 0])).toThrow(DataError);
    expect(() => itemFit(buildResponseMatrix([[1, 0]]), HALVES.slice(0, 1), [0])).toThrow(DataError);
    expect(() => itemFit(buildResponseMatrix([[1, 0]]), HALVES, [0, 1])).toThrow(DataError);
    expect(() => itemFit(buildResponseMatrix([[1, 0]]), HALVES, [Number.NaN])).toThrow(DataError);
    expect(() => itemFit(buildResponseMatrix([[1, 0]]), JSON.parse("null"), [0])).toThrow(
      "The items must be an array, found null.",
    );
    expect(() => personFit(buildResponseMatrix([[1, 0]]), HALVES, JSON.parse("null"))).toThrow(
      "The thetas must be an array, found null.",
    );
    expect(() => summedScoreFit(buildResponseMatrix([[1, 0]]), JSON.parse("null"))).toThrow(
      "The items must be an array, found null.",
    );
    expect(() => yenQ3(buildResponseMatrix([[1, 0]]), JSON.parse("[null, null]"), [0])).toThrow(
      'Item "item1" must be an item with a model and parameters, found null.',
    );
  });
});

describe("yenQ3", () => {
  for (const fixture of residuals.cases) {
    it(`matches mirt's Q3 residuals for ${fixture.name}`, () => {
      const result = yenQ3(buildResponseMatrix(fixture.responses), fixtureItems(fixture.items), fixture.thetas);
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
      buildResponseMatrix([
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

describe("localDependenceChiSquare", () => {
  for (const fixture of localDependence.cases) {
    it(`matches mirt's LD residuals for ${fixture.name}`, () => {
      const options = "population" in fixture ? { population: fixture.population } : {};
      const result = localDependenceChiSquare(
        buildResponseMatrix(fixture.responses),
        fixtureItems(fixture.items),
        options,
      );
      expect(result.map((pair) => pair.degreesOfFreedom)).toEqual(fixture.degreesOfFreedom);
      result.forEach((pair, p) => {
        const label = pair.items.join(", ");
        expectClose(pair.chiSquare, fixture.chiSquare[p] ?? null, FIT_TOLERANCE, `χ² ${label}`);
        expectClose(pair.significance, fixture.significance[p] ?? null, FIT_TOLERANCE, `p ${label}`);
        expectClose(pair.cramersV, fixture.cramersV[p] ?? null, FIT_TOLERANCE, `V ${label}`);
      });
    });
  }

  it("lists each pair once, the first item with each later one, then the second", () => {
    const { matrix, items: pool } = simulated(41, 100, 4);
    expect(localDependenceChiSquare(matrix, pool).map((pair) => pair.items.join())).toEqual([
      "item1,item2",
      "item1,item3",
      "item1,item4",
      "item2,item3",
      "item2,item4",
      "item3,item4",
    ]);
  });

  it("flags a planted dependent pair with a positive V", () => {
    // Thresholds set before the first run: the planted pair has p below 1e-6 and the largest V.
    const random = createRandom(43);
    const { matrix, items: pool } = simulated(42, 2000, 8);
    const planted = buildResponseMatrix(
      matrix.responses.map((row) =>
        row.map((response, i) => (i === 1 && random.uniform() < 0.5 ? (row[0] ?? 0) : response)),
      ),
    );
    const [first, ...others] = localDependenceChiSquare(planted, pool);
    expect(defined(first?.significance)).toBeLessThan(1e-6);
    expect(others.every((pair) => defined(pair.cramersV) < defined(first?.cramersV))).toBe(true);
  });

  it("uses the persons who answered both items, and gives null where a table has no persons or no order", () => {
    const pool = [...HALVES, ...HALVES.slice(0, 1)];
    const [both, none, constant] = localDependenceChiSquare(
      buildResponseMatrix([
        [1, null, 1],
        [0, null, 1],
        [null, 1, 1],
      ]),
      pool,
    );
    expect(both).toEqual({
      items: ["item1", "item2"],
      chiSquare: null,
      degreesOfFreedom: 1,
      significance: null,
      cramersV: null,
    });
    // Item 3 is 1 for both persons of the pair: the observed gamma is undefined, the chi-square is not.
    expect(none?.items).toEqual(["item1", "item3"]);
    expect([none?.chiSquare === null, none?.cramersV]).toEqual([false, null]);
    expect(constant?.chiSquare).not.toBeNull();
  });

  it("takes at most 1,000 items, as it and Yen's Q3 hold a statistic of every pair of items", () => {
    // Two persons who answer the items alike or opposite: the items, not the persons, would cost the square.
    const rows = (items: number) => [
      Array.from({ length: items }, (_, j) => j % 2),
      Array.from({ length: items }, (_, j) => (j + 1) % 2),
    ];
    const pool = (items: number) => Array.from({ length: items }, () => defined(HALVES[0]));
    expect(localDependenceChiSquare(buildResponseMatrix(rows(1000)), pool(1000))).toHaveLength(499_500);
    expect(yenQ3(buildResponseMatrix(rows(1000)), pool(1000), [0, 0])).toHaveLength(1000);
    const message = (analysis: string) =>
      `${analysis} holds a matrix of every pair of items, so it takes at most 1000 items, found 1001.`;
    expect(() => localDependenceChiSquare(buildResponseMatrix(rows(1001)), pool(1001))).toThrow(
      message("Local dependence"),
    );
    expect(() => yenQ3(buildResponseMatrix(rows(1001)), pool(1001), [0, 0])).toThrow(message("Yen's Q3"));
  });

  it("refuses tables of the categories of the pairs of more than 2^24 cells in all, so that items of many do not cost their square", () => {
    // Two graded items of 4,097 categories: a pair's table of 16,785,409 cells, from 66 KB of thresholds.
    const graded = (categories: number): IrtItem => ({
      model: gradedResponse,
      parameters: { a: 1, b: Array.from({ length: categories - 1 }, (_, k) => -3 + (6 * k) / (categories - 2)) },
    });
    const matrix = buildResponseMatrix([
      [0, 1],
      [1, 0],
    ]);
    expect(localDependenceChiSquare(matrix, [graded(100), graded(100)])[0]?.degreesOfFreedom).toBe(99 * 99);
    expect(() => localDependenceChiSquare(matrix, [graded(4097), graded(4097)])).toThrow(
      "Local dependence holds a table of the categories of every pair of items, 16785409 cells, and takes at most 16777216; test fewer items, or items of fewer categories.",
    );
  });

  it("refuses arguments of other types", () => {
    const { matrix, items: pool } = simulated(44, 20, 3);
    expect(() => localDependenceChiSquare(JSON.parse("null"), pool)).toThrow(DataError);
    expect(() => localDependenceChiSquare(matrix, JSON.parse("{}"))).toThrow(
      "The items must be an array, found an object.",
    );
    expect(() => localDependenceChiSquare(matrix, pool, JSON.parse("null"))).toThrow(DataError);
    expect(() => localDependenceChiSquare(matrix, pool, JSON.parse('{ "population": null }'))).toThrow(
      /The population must be an object/,
    );
  });
});

describe("personFit", () => {
  for (const fixture of rasch.cases) {
    it(`matches eRm's personfit for the ${fixture.name} model`, () => {
      const result = personFit(buildResponseMatrix(fixture.responses), fixtureItems(fixture.items), fixture.thetas);
      expectTable(result, fixture.personFit, FIT_TOLERANCE, fixture.name);
    });
  }

  for (const fixture of residuals.cases) {
    it(`matches mirt's Zh for ${fixture.name}`, () => {
      const result = personFit(buildResponseMatrix(fixture.responses), fixtureItems(fixture.items), fixture.thetas);
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
      buildResponseMatrix([
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

describe("correctedPersonFit", () => {
  const sat12 = defined(residuals.cases.find((candidate) => candidate.name === "SAT12 2PL"));
  const threePl: IrtItem[] = cat3pl.items.slice(0, 25).map(({ id: _, ...parameters }) => ({
    model: threeParameterLogistic,
    parameters,
  }));

  for (const fixture of correctedFit.cases) {
    it(`matches PerFit's lzstar, unrounded, for ${fixture.name}`, () => {
      const three = fixture.name.startsWith("3PL");
      const matrix = buildResponseMatrix(three ? correctedFit.threePl.responses : sat12.responses);
      const estimator = defined(
        (["maximum-likelihood", "weighted-likelihood", "posterior-mode"] as const).find(
          (name) => name === fixture.estimator,
        ),
      );
      const result = correctedPersonFit(matrix, three ? threePl : fixtureItems(sat12.items), fixture.thetas, {
        estimator,
        prior: fixture.prior,
      });
      result.forEach((person, i) => {
        expectClose(person.standardized, fixture.standardized[i] ?? null, CORRECTED_TOLERANCE, `person ${i + 1}`);
      });
    });
  }

  it("is about N(0, 1) at estimated thetas where the items fit", () => {
    // From the design, set before the first run: a standard deviation within 0.05 of 1 on 5,000 persons. The mean is
    // 0.098 on 30 items, 0.073 on 60 and 0.045 on 100, as Snijders' correction is of the first order: held below 0.15.
    const { matrix, items: pool } = simulated(20261103, 5000, 30);
    const thetas = matrix.responses.map((row) => maximumLikelihoodEstimate(pool, row).theta);
    const values = correctedPersonFit(matrix, pool, thetas).flatMap(({ standardized }) =>
      standardized === null ? [] : [standardized],
    );
    const average = values.reduce((sum, value) => sum + value, 0) / values.length;
    const variance = values.reduce((sum, value) => sum + (value - average) ** 2, 0) / (values.length - 1);
    expect(Math.abs(Math.sqrt(variance) - 1)).toBeLessThan(0.05);
    expect(Math.abs(average)).toBeLessThan(0.15);
  });

  it("skips missing responses, as if the item had not been given", () => {
    const { matrix, items: pool } = simulated(49, 40, 6);
    const thetas = matrix.personIds.map((_, person) => Math.sin(person));
    const sparse = buildResponseMatrix(matrix.responses.map((row) => [null, ...row.slice(1)]));
    const shorter = buildResponseMatrix(matrix.responses.map((row) => row.slice(1)));
    expectSameNumbers(
      correctedPersonFit(sparse, pool, thetas, { estimator: "weighted-likelihood" }),
      correctedPersonFit(shorter, pool.slice(1), thetas, { estimator: "weighted-likelihood" }),
    );
  });

  it("gives nothing to an extreme score, a person without responses, or a pattern without variance", () => {
    const result = correctedPersonFit(
      buildResponseMatrix([
        [1, 1],
        [0, null],
        [null, null],
        [1, 0],
      ]),
      HALVES,
      [1, -1, 0, 0],
    );
    // At θ = 0 both items have P = 1/2, so w = ln(P / Q) = 0 and the variance is 0.
    expect(result.map(({ standardized }) => standardized)).toEqual([null, null, null, null]);
  });

  it("corrects by the prior of the posterior mode", () => {
    const { matrix, items: pool } = simulated(50, 30, 8);
    const thetas = matrix.personIds.map((_, person) => Math.cos(person));
    const standard = correctedPersonFit(matrix, pool, thetas, { estimator: "posterior-mode" });
    const given = correctedPersonFit(matrix, pool, thetas, { estimator: "posterior-mode", prior: { mean: 0, sd: 1 } });
    expectSameNumbers(given, standard);
    const shifted = correctedPersonFit(matrix, pool, thetas, {
      estimator: "posterior-mode",
      prior: { mean: 1, sd: 2 },
    });
    expect(shifted.some((person, i) => person.standardized !== standard[i]?.standardized)).toBe(true);
  });

  it("rejects polytomous items, unknown estimators and invalid priors, and arguments of other types", () => {
    const { matrix, items: pool } = simulated(51, 10, 2);
    const thetas = matrix.personIds.map(() => 0);
    const polytomous: IrtItem = { model: generalizedPartialCredit, parameters: { a: 1, b: [0, 1] } };
    expect(() => correctedPersonFit(matrix, [defined(pool[0]), polytomous], thetas)).toThrow(
      'lz* needs dichotomous items, but item "item2" has 3 categories.',
    );
    expect(() => correctedPersonFit(matrix, pool, thetas, JSON.parse('{ "estimator": "eap" }'))).toThrow(
      'The option estimator must be "maximum-likelihood", "weighted-likelihood" or "posterior-mode", found "eap".',
    );
    expect(() => correctedPersonFit(matrix, pool, thetas, { prior: { mean: 0, sd: 0 } })).toThrow(
      "The sd of the prior must be a positive number, found 0.",
    );
    expect(() => correctedPersonFit(matrix, pool, thetas, JSON.parse('{ "prior": null }'))).toThrow(
      "The option prior must be an object { mean, sd }, found null.",
    );
    expect(() => correctedPersonFit(matrix, pool, thetas, JSON.parse("null"))).toThrow(DataError);
    expect(() => correctedPersonFit(matrix, pool, JSON.parse('"0"'))).toThrow("The thetas must be an array");
    expect(() => correctedPersonFit(JSON.parse("{}"), pool, thetas)).toThrow(DataError);
  });
});

describe("summedScoreFit", () => {
  for (const fixture of summedScore.cases) {
    it(`matches mirt's S-X² for ${fixture.name}`, () => {
      const result = summedScoreFit(buildResponseMatrix(fixture.responses), fixtureItems(fixture.items));
      expectTable(
        result,
        { chiSquare: fixture.chiSquare, significance: fixture.significance },
        FIT_TOLERANCE,
        fixture.name,
      );
      expect(result.map((item) => item.degreesOfFreedom)).toEqual(fixture.degreesOfFreedom);
    });
  }

  it("matches mirt's S-X² of items with held parameters, counting the estimated ones", () => {
    const result = summedScoreFit(heldMatrix, heldItems, { fixedParameters: fixedParameters.fixedParameters });
    expectTable(result, fixedParameters.summedScore, FIT_TOLERANCE, "held parameters");
    expect(result.map((item) => item.degreesOfFreedom)).toEqual(fixedParameters.summedScore.degreesOfFreedom);
  });

  it("counts every parameter of an item by default and none of an anchor", () => {
    const counted = summedScoreFit(heldMatrix, heldItems);
    const held = summedScoreFit(heldMatrix, heldItems, { fixedParameters: fixedParameters.fixedParameters });
    const anchored = summedScoreFit(heldMatrix, heldItems, { fixed: ["item1", "item12"] });
    expect(held.map((item, i) => item.degreesOfFreedom - defined(counted[i]).degreesOfFreedom)).toEqual([
      1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1,
    ]);
    expect(anchored.map((item, i) => item.degreesOfFreedom - defined(counted[i]).degreesOfFreedom)).toEqual([
      3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2,
    ]);
  });

  it("rejects anchors and held parameters that are not the items'", () => {
    expect(() => summedScoreFit(heldMatrix, heldItems, { fixed: ["q1"] })).toThrow(
      'The option fixed names "q1", which is not an item id.',
    );
    expect(() => summedScoreFit(heldMatrix, heldItems, { fixedParameters: { item9: { c: 0 } } })).toThrow(
      'The option fixedParameters holds "c" of item "item9", whose 2PL model has a, b.',
    );
    expect(() =>
      summedScoreFit(heldMatrix, heldItems, { fixed: ["item1"], fixedParameters: { item1: { c: 0.25 } } }),
    ).toThrow('The option fixedParameters holds parameters of item "item1", which the option fixed holds whole.');
    expect(() => summedScoreFit(heldMatrix, heldItems, JSON.parse("null"))).toThrow("The options must be an object");
    expect(() => summedScoreFit(heldMatrix, heldItems, JSON.parse('{ "fixed": "item1" }'))).toThrow(
      "The option fixed must be an array of item ids",
    );
  });

  it("uses the persons who answered every item", () => {
    const { matrix, items: pool } = simulated(5, 300, 8);
    const withMissing = buildResponseMatrix([...matrix.responses, [null, 1, 0, 1, 0, 1, 0, 1]]);
    expect(summedScoreFit(withMissing, pool)).toEqual(summedScoreFit(matrix, pool));
    expect(() => summedScoreFit(buildResponseMatrix([[null, 1]]), HALVES)).toThrow(DataError);
  });

  it("takes items of a highest total score of at most 300, whose sums cost the square of the score for each item", () => {
    // Two graded items of 151 and of 151 or 152 categories: a highest total score of 300 or 301.
    const graded = (categories: number): IrtItem => ({
      model: gradedResponse,
      parameters: { a: 1, b: Array.from({ length: categories - 1 }, (_, k) => -3 + (6 * k) / (categories - 2)) },
    });
    const matrix = buildResponseMatrix([
      [0, 1],
      [1, 0],
    ]);
    expect(summedScoreFit(matrix, [graded(151), graded(151)])).toHaveLength(2);
    expect(() => summedScoreFit(matrix, [graded(151), graded(152)])).toThrow(
      "S-X² sums the scores of the other items for each item, so it takes items of a highest total score of at most 300, found 301; test a part of the form.",
    );
  });

  it("gives null without degrees of freedom", () => {
    const result = summedScoreFit(
      buildResponseMatrix([
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

describe("rootMeanSquareDeviation", () => {
  for (const fixture of rmsd.cases) {
    it(`matches TAM's IRT.RMSD for ${fixture.name}`, () => {
      const result = rootMeanSquareDeviation(buildResponseMatrix(fixture.responses), fixtureItems(fixture.items), {
        population: fixture.population,
      });
      result.forEach((item, i) => {
        expectClose(item.rmsd, fixture.rmsd[i] ?? null, RMSD_TOLERANCE, `${item.id} RMSD`);
        expectClose(item.meanDeviation, fixture.meanDeviation[i] ?? null, RMSD_TOLERANCE, `${item.id} MD`);
      });
    });
  }

  it("divides each item's mean deviation by its own categories, where IRT.RMSD takes another item's", () => {
    const fixture = defined(rmsd.cases.find((candidate) => candidate.name === "partial credit"));
    const differences = fixture.meanDeviation.map((value, i) => Math.abs(value - (fixture.tamMeanDeviation[i] ?? 0)));
    expect(Math.max(...differences)).toBeGreaterThan(1e-3);
  });

  it("finds a shifted item, and items that fit near 0", () => {
    // Thresholds set before the first run: above 0.1 for an item 1 logit harder than its responses, below 0.05 otherwise.
    const { matrix, items: pool } = simulated(45, 2000, 10);
    const shifted = pool.map((item, i) =>
      i === 0
        ? {
            ...item,
            parameters: { ...(item.parameters as { b: number }), b: (item.parameters as { b: number }).b + 1 },
          }
        : item,
    );
    const [first, ...others] = rootMeanSquareDeviation(matrix, shifted);
    expect(defined(first?.rmsd)).toBeGreaterThan(0.1);
    expect(defined(first?.meanDeviation)).toBeGreaterThan(0);
    expect(others.every((item) => defined(item.rmsd) < 0.05)).toBe(true);
  });

  it("integrates over N(0, 1) on 61 points by default, and leaves out persons without responses", () => {
    const { matrix, items: pool } = simulated(46, 200, 4);
    const nodes = Array.from({ length: 61 }, (_, q) => -6 + q / 5);
    const densities = nodes.map((node) => Math.exp(-(node ** 2) / 2));
    const total = densities.reduce((sum, density) => sum + density, 0);
    const population = { nodes, weights: densities.map((density) => density / total) };
    expectSameNumbers(rootMeanSquareDeviation(matrix, pool), rootMeanSquareDeviation(matrix, pool, { population }));
    const empty = buildResponseMatrix([...matrix.responses, [null, null, null, null]]);
    expectSameNumbers(rootMeanSquareDeviation(empty, pool), rootMeanSquareDeviation(matrix, pool));
    const unanswered = buildResponseMatrix(matrix.responses.map((row) => [null, ...row.slice(1)]));
    expect(rootMeanSquareDeviation(unanswered, pool)[0]).toEqual({ id: "item1", rmsd: null, meanDeviation: null });
  });

  it("refuses arguments of other types", () => {
    const { matrix, items: pool } = simulated(47, 20, 3);
    expect(() => rootMeanSquareDeviation(JSON.parse("[]"), pool)).toThrow(DataError);
    expect(() => rootMeanSquareDeviation(matrix, pool.slice(1))).toThrow("Found 2 items for 3 columns of responses.");
    expect(() => rootMeanSquareDeviation(matrix, pool, JSON.parse("null"))).toThrow(DataError);
    const population = (value: string) => () =>
      rootMeanSquareDeviation(matrix, pool, { population: JSON.parse(value) });
    expect(population('{ "nodes": "0", "weights": [1] }')).toThrow(/The nodes of the population must be an array/);
    expect(population('{ "nodes": [0, 1], "weights": [0.5, 0.6] }')).toThrow(/must sum to 1, found 1.1/);
  });
});

describe("limitedInformationFit", () => {
  const binary = buildResponseMatrix(limited.binary.responses);
  const polytomous = buildResponseMatrix(limited.polytomous.responses);
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

  it("matches the M2* written out for a calibration with held parameters, which add no columns to Δ", () => {
    const { held } = limited;
    const matrix = buildResponseMatrix(held.responses);
    const calibration = calibrate(matrix, {
      model: held.models.map((name) => defined(MODELS[name])),
      fixedParameters: held.fixedParameters,
      tolerance: LIMITED_EM_TOLERANCE,
      maxIterations: 100_000,
    });
    const result = limitedInformationFit(matrix, calibration, { fixedParameters: held.fixedParameters });
    expectLimitedFit(result, held.expected, "held parameters");
    expect(held.mirtStatistic).toBeLessThan(defined(result.statistic));
    expect(() => limitedInformationFit(matrix, calibration)).toThrow(
      "The calibration estimated 23 parameters, but these items have 32; give the anchors' ids in the option fixed and the held parameters in the option fixedParameters.",
    );
  });

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
      const matrix = buildResponseMatrix(responses.map((row) => Array.from(row, Number)));
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
    const missing = buildResponseMatrix([
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
    const matrix = buildResponseMatrix(rows);
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
    const factorial = buildResponseMatrix(
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

  it("refuses C2 of more than 1,830 moments, those of 60 dichotomous items, so that categories do not grow its matrix", () => {
    // 60 graded items of three categories: 120 moments of categories and 1,770 of pairs, where M2* has 60 and 1,770.
    const rows = Array.from({ length: 50 }, (_, person) =>
      Array.from({ length: 60 }, (_, item) => (person + item) % 3),
    );
    const matrix = buildResponseMatrix(rows);
    const calibration = calibrate(matrix, { model: gradedResponse, maxIterations: 1 });
    expect(limitedInformationFit(matrix, calibration, { statistic: "m2-star" }).degreesOfFreedom).toBeGreaterThan(0);
    expect(() => limitedInformationFit(matrix, calibration, { statistic: "c2" })).toThrow(
      'The statistic "c2" has 1890 moments, a category above the lowest of each item and each pair of items, and takes at most 1830, as many as 60 dichotomous items have; test fewer items, or items of fewer categories.',
    );
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
    const matrix = buildResponseMatrix([
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
    const few = buildResponseMatrix(
      matrix.responses.map((row, person) => (person === 0 ? row : [null, ...row.slice(1)])),
    );
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

describe("Rasch tests", () => {
  /** The model and split of a fixture case, read from its JSON. */
  function modelOf(name: string): ConditionalModel {
    if (name === "rasch" || name === "partial-credit" || name === "rating-scale") return name;
    throw new Error(`Unknown model ${name}.`);
  }
  function personSplit(split: unknown): RaschSplit {
    if (split === "median" || split === "mean") return split;
    if (Array.isArray(split) && split.every((label) => typeof label === "string")) return split;
    throw new Error("The case's split is not one of persons.");
  }
  function itemSplit(split: unknown): ItemSplit {
    if (split === "median" || split === "mean") return split;
    if (Array.isArray(split) && split.every((set) => Array.isArray(set) && set.every((id) => typeof id === "string"))) {
      return split;
    }
    throw new Error("The case's split is not one of items.");
  }
  const caseNamed = <T extends { readonly name: string }>(cases: readonly T[], name: string): T =>
    defined(cases.find((candidate) => candidate.name === name));

  /** Two groups, a and b, where leaving out item1 makes person 3 extreme in b, and item2 then constant there. */
  const Cascade = buildResponseMatrix([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [0, 0, 1, 0],
    [1, 1, 0, 1],
    [1, 0, 0, 0],
    [1, 1, 1, 0],
    [1, 1, 0, 1],
  ]);
  const cascadeSplit = ["a", "a", "a", "a", "b", "b", "b"];

  describe("andersenTest", () => {
    for (const expected of raschTests.andersen) {
      const reference = "erm" in expected ? "base R's conditional likelihood" : "eRm's LRtest";
      it(`matches ${reference} on ${expected.name}`, () => {
        const result = andersenTest(buildResponseMatrix(expected.responses), {
          model: modelOf(expected.model),
          split: personSplit(expected.split),
        });
        const tolerance = "erm" in expected ? RASCH_BASE_TOLERANCE : RASCH_STATISTIC_TOLERANCE;
        expectClose(result.statistic, expected.statistic, tolerance, "statistic");
        expectClose(result.significance, expected.significance, tolerance, "significance");
        expect([result.degreesOfFreedom, result.persons, result.converged]).toEqual([
          expected.degreesOfFreedom,
          expected.persons,
          true,
        ]);
        expect(result.excluded).toEqual(expected.excluded);
      });
    }

    it("maximizes each conditional likelihood on missing responses of polytomous items, where eRm's LRtest does not", () => {
      const expected = caseNamed(raschTests.andersen, "small, sparse, median");
      const { statistic } = andersenTest(buildResponseMatrix(expected.responses));
      expect(Math.abs(statistic - defined("erm" in expected ? expected.erm : null))).toBeGreaterThan(10);
    });

    it("splits persons with missing responses at the median of those who answered the same items", () => {
      const matrix = buildResponseMatrix([
        [1, 1, 0, null],
        [0, 1, 0, null],
        [1, 1, 1, null],
        [1, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 1, 1, 0],
        [0, 1, 1, 0],
      ]);
      // Medians: 2 among the first three persons, 2.5 among the others.
      const labels = ["lower", "lower", "upper", "upper", "lower", "upper", "lower"];
      expectSameNumbers(andersenTest(matrix, { split: labels }), andersenTest(matrix));
    });

    it("leaves out an item a group answers alike, and counts the persons again without it", () => {
      const result = andersenTest(Cascade, { split: cascadeSplit });
      // Without item1, person 5 has a total of 0, and item2 is 1 for the other two persons of group b.
      expect(result.excluded).toEqual(["item1", "item2"]);
      expect([result.degreesOfFreedom, result.persons]).toEqual([1, 6]);
      expectClose(result.statistic, 0, 1e-9);
    });

    it("leaves out an item whose highest score is beyond the persons of a group, before its categories take room", () => {
      // A fifth item that one person scores 2³¹ - 1: a table of its categories would take 2 GB and overflow 32 bits.
      const far = buildResponseMatrix(Cascade.responses.map((row, person) => [...row, person === 0 ? 2 ** 31 - 1 : 0]));
      const result = andersenTest(far, { split: cascadeSplit });
      expect(result).toEqual({
        ...andersenTest(Cascade, { split: cascadeSplit }),
        excluded: ["item1", "item2", "item5"],
      });
      expect(raschWaldTest(far, { split: cascadeSplit }).excluded).toEqual(["item1", "item2", "item5"]);
    });

    it("takes the model and the stopping rules of conditional maximum likelihood", () => {
      const expected = caseNamed(raschTests.andersen, "pcmdat2, median");
      const matrix = buildResponseMatrix(expected.responses);
      expect(andersenTest(matrix, { maxIterations: 1 }).converged).toBe(false);
      const rating = andersenTest(matrix, { model: "rating-scale" });
      expect(rating.degreesOfFreedom).toBe(4);
      expect(() => andersenTest(matrix, { model: "rasch" })).toThrow(/allows responses from 0 to 1/);
    });

    it("refuses a split into one group, and too few items to test", () => {
      const matrix = buildResponseMatrix(caseNamed(raschTests.andersen, "raschdat1, median").responses);
      expect(() => andersenTest(matrix, { split: matrix.personIds.map(() => "all") })).toThrow(
        "The split puts every person in one group; the test needs at least two groups of persons.",
      );
      const constant = buildResponseMatrix([
        [1, 1, 0],
        [1, 0, 1],
        [1, 0, 0],
        [1, 1, 1],
      ]);
      expect(() => andersenTest(constant, { split: ["a", "b", "a", "b"] })).toThrow(
        /at least two items with a response in each of their categories in every group, found 0/,
      );
    });

    it("refuses arguments of other types", () => {
      const matrix = buildResponseMatrix(caseNamed(raschTests.andersen, "raschdat1, median").responses);
      expect(() => andersenTest(JSON.parse("null"))).toThrow(DataError);
      expect(() => andersenTest(matrix, JSON.parse("null"))).toThrow("The options must be an object, found null.");
      expect(() => andersenTest(matrix, JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
      const options = (value: string) => () => andersenTest(matrix, JSON.parse(value));
      expect(options('{ "split": null }')).toThrow(/The option split must be "median", "mean" or an array of labels/);
      expect(options('{ "split": "upper" }')).toThrow(/The option split must be/);
      expect(options('{ "split": ["a", "b"] }')).toThrow(/one label for each of the 100 persons, found 2/);
      expect(() =>
        andersenTest(matrix, { split: matrix.personIds.map((_, i) => JSON.parse(i < 50 ? "1" : "2")) }),
      ).toThrow('The label of person "person1" must be a string, found 1.');
      expect(options('{ "model": "RM" }')).toThrow(/The option model/);
      expect(options('{ "tolerance": "small" }')).toThrow(DataError);
    });
  });

  describe("raschWaldTest", () => {
    for (const expected of raschTests.wald) {
      const reference = expected.model === "partial-credit" ? "base R's locations of eRm's PCM" : "eRm's Waldtest";
      it(`matches ${reference} on ${expected.name}`, () => {
        const result = raschWaldTest(buildResponseMatrix(expected.responses), {
          model: modelOf(expected.model),
          split: personSplit(expected.split),
        });
        result.items.forEach((item, i) => {
          expectClose(item.z, expected.z[i] ?? null, RASCH_WALD_TOLERANCE, `${item.id} z`);
          expectClose(item.significance, expected.significance[i] ?? null, RASCH_WALD_TOLERANCE, `${item.id} p`);
        });
        expect([result.excluded, result.persons, result.converged]).toEqual([
          expected.excluded,
          expected.persons,
          true,
        ]);
      });
    }

    it("gives a positive z to an item harder for the second group", () => {
      const random = createRandom(31);
      const items = [-1, -0.5, 0, 0.5, 1].map((b) => ({
        model: oneParameterLogistic,
        parameters: { a: 1, b, c: 0, d: 1 },
      }));
      const shifted = items.map((item, i) => (i === 0 ? { ...item, parameters: { ...item.parameters, b: 0 } } : item));
      const reference = simulateResponses(items, simulateThetas(1000, random), random);
      const focal = simulateResponses(shifted, simulateThetas(1000, random), random);
      const matrix = buildResponseMatrix([...reference.responses, ...focal.responses]);
      const split = matrix.personIds.map((_, person) => (person < 1000 ? "1 reference" : "2 focal"));
      const [first, ...others] = raschWaldTest(matrix, { split }).items;
      expect(defined(first?.z)).toBeGreaterThan(5);
      expect(others.every((item) => defined(item.z) < 0)).toBe(true);
    });

    it("gives null to the items it leaves out", () => {
      const result = raschWaldTest(Cascade, { split: cascadeSplit });
      expect(result.items.map((item) => item.z === null)).toEqual([true, true, false, false]);
      expect(result.excluded).toEqual(["item1", "item2"]);
      expect(result.persons).toBe(6);
    });

    it("refuses a split into more than two groups, and arguments of other types", () => {
      const matrix = buildResponseMatrix(caseNamed(raschTests.andersen, "raschdat1, three labels").responses);
      const split = personSplit(caseNamed(raschTests.andersen, "raschdat1, three labels").split);
      expect(() => raschWaldTest(matrix, { split })).toThrow("The Wald test compares two groups of persons, found 3.");
      expect(() => raschWaldTest(JSON.parse("[]"))).toThrow(DataError);
      expect(() => raschWaldTest(matrix, JSON.parse("null"))).toThrow(DataError);
      expect(() => raschWaldTest(matrix, JSON.parse('{ "split": {} }'))).toThrow(/The option split must be/);
    });
  });

  describe("martinLofTest", () => {
    for (const expected of raschTests.martinLof) {
      const reference =
        expected.model === "rating-scale" ? "eRm's MLoef, with base R's degrees of freedom," : "eRm's MLoef";
      it(`matches ${reference} on ${expected.name}`, () => {
        const result = martinLofTest(buildResponseMatrix(expected.responses), {
          model: modelOf(expected.model),
          split: itemSplit(expected.split),
        });
        expectClose(result.statistic, expected.statistic, RASCH_STATISTIC_TOLERANCE, "statistic");
        expectClose(result.significance, expected.significance, RASCH_STATISTIC_TOLERANCE, "significance");
        expect([result.degreesOfFreedom, result.persons, result.converged]).toEqual([
          expected.degreesOfFreedom,
          expected.persons,
          true,
        ]);
      });
    }

    it("counts the steps of each set of a rating scale model, which eRm's degrees of freedom leave out", () => {
      const expected = caseNamed(raschTests.martinLof, "rating scale, median");
      // Two sets of items with 3 steps each: (2 - 1) (3 - 1) more parameters than eRm counts.
      expect(expected.degreesOfFreedom - expected.ermDegreesOfFreedom).toBe(2);
    });

    it("uses the persons who answered every item", () => {
      const expected = caseNamed(raschTests.martinLof, "raschdat1, median");
      const matrix = buildResponseMatrix(expected.responses);
      const missing = buildResponseMatrix([...expected.responses, [null, ...defined(expected.responses[0]).slice(1)]]);
      expectSameNumbers(martinLofTest(missing), martinLofTest(matrix));
      const none = buildResponseMatrix(expected.responses.map((row) => [null, ...row.slice(1)]));
      expect(() => martinLofTest(none)).toThrow(
        "The Martin-Löf test needs persons who answered every item, found none.",
      );
    });

    it("refuses sets that do not hold every item once, a set of one item, and arguments of other types", () => {
      const matrix = buildResponseMatrix([
        [1, 0, 1, 1],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
        [1, 1, 1, 0],
      ]);
      const sets = (split: ItemSplit) => () => martinLofTest(matrix, { split });
      expect(sets([["item1", "item2", "item3", "item4"]])).toThrow(
        "The option split needs at least two sets of item ids, found 1.",
      );
      expect(sets([["item1", "item2", "item3"], ["item4"]])).toThrow(/needs at least two items, found "item4"/);
      expect(
        sets([
          ["item1", "item2"],
          ["item2", "item3", "item4"],
        ]),
      ).toThrow(/names "item2", which is not an item id or is in an earlier set/);
      expect(
        sets([
          ["item1", "item2"],
          ["item3", "item5"],
        ]),
      ).toThrow(/names "item5"/);
      expect(sets([["item1", "item2"], ["item3"]])).toThrow(
        'The option split must place every item in a set; item "item4" is in none.',
      );
      expect(
        sets([
          ["item1", "item2"],
          ["item3", "item4"],
        ]),
      ).not.toThrow();
      const equal = buildResponseMatrix([
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1],
      ]);
      expect(() => martinLofTest(equal)).toThrow(
        "Each set of items of the Martin-Löf test needs at least two items, found no items.",
      );
      expect(() => martinLofTest(JSON.parse("null"))).toThrow(DataError);
      expect(() => martinLofTest(matrix, JSON.parse("null"))).toThrow(DataError);
      expect(() => martinLofTest(matrix, JSON.parse('{ "split": "upper" }'))).toThrow(
        /The option split must be "median", "mean" or an array of sets of item ids/,
      );
      expect(() => martinLofTest(matrix, JSON.parse('{ "split": [["item1", "item2"], "item3"] }'))).toThrow(
        /Set 2 of the option split must be an array of item ids/,
      );
    });
  });

  it("keeps the Rasch model at about its level on Rasch responses, and rejects it on 2PL responses", () => {
    // Thresholds set before the first run: at 0.05 over 100 sets of 500 persons and 10 items, each test rejects at most
    // 10% of the sets and the Wald test flags at most 10% of the items; Andersen's p is below 0.001 on each of 5 2PL sets
    // with slopes from 0.4 to 2.
    const random = createRandom(2027);
    const difficulties = Array.from({ length: 10 }, (_, i) => -1.5 + (3 * i) / 9);
    const items = difficulties.map((b) => ({ model: oneParameterLogistic, parameters: { a: 1, b, c: 0, d: 1 } }));
    let andersen = 0;
    let martinLof = 0;
    let flagged = 0;
    let tested = 0;
    for (let set = 0; set < 100; set++) {
      const matrix = simulateResponses(items, simulateThetas(500, random), random);
      if (andersenTest(matrix).significance < 0.05) andersen++;
      if (martinLofTest(matrix).significance < 0.05) martinLof++;
      for (const item of raschWaldTest(matrix).items) {
        tested++;
        if (defined(item.significance) < 0.05) flagged++;
      }
    }
    expect([andersen <= 10, martinLof <= 10, flagged / tested <= 0.1]).toEqual([true, true, true]);
    const slopes = items.map((item, i) => ({
      model: twoParameterLogistic,
      parameters: { ...item.parameters, a: 0.4 + (1.6 * i) / 9 },
    }));
    for (let set = 0; set < 5; set++) {
      const matrix = simulateResponses(slopes, simulateThetas(500, random), random);
      expect(andersenTest(matrix).significance).toBeLessThan(0.001);
    }
  });
});

describe("kernelItemCurves", () => {
  for (const fixture of kernelCurves.cases) {
    it(`matches KernSmoothIRT's ksIRT and Ramsay's errors in base R for ${fixture.name}`, () => {
      const data = defined(rmsd.cases.find((candidate) => candidate.name === fixture.data));
      const result = kernelItemCurves(buildResponseMatrix(data.responses), fixture.options);
      expect(result).toHaveLength(fixture.probabilities.length);
      result.forEach((curve, i) => {
        expect(curve.id).toBe(`item${i + 1}`);
        curve.nodes.forEach((node, q) => {
          expectClose(node, fixture.nodes[q] ?? null, KERNEL_TOLERANCE, `node ${q}`);
        });
        const probabilities = defined(fixture.probabilities[i]);
        const errors = defined(fixture.standardErrors[i]);
        expect(curve.probabilities).toHaveLength(probabilities.length);
        curve.probabilities.forEach((row, k) => {
          row.forEach((p, q) => {
            expectClose(p, probabilities[k]?.[q] ?? null, KERNEL_TOLERANCE, `${curve.id} P${k} ${q}`);
            expectClose(curve.standardErrors[k]?.[q] ?? null, errors[k]?.[q] ?? null, KERNEL_TOLERANCE, `SE${k} ${q}`);
          });
        });
      });
    });
  }

  it("recovers the curves of simulated 2PL items", () => {
    // Threshold from the design, set before the first run: an RMSE below 0.1 against the true curves at the nodes.
    const { matrix, items: pool } = simulated(20261102, 2000, 20);
    const probabilities = new Float64Array(2);
    let squares = 0;
    let count = 0;
    kernelItemCurves(matrix).forEach((curve, i) => {
      const item = defined(pool[i]);
      curve.nodes.forEach((node, q) => {
        item.model.probabilities(item.parameters, node, probabilities);
        squares += ((curve.probabilities[1]?.[q] ?? 0) - (probabilities[1] ?? 0)) ** 2;
        count++;
      });
    });
    expect(Math.sqrt(squares / count)).toBeLessThan(0.1);
  });

  it("ranks ties in the order of the persons and smooths with Silverman's bandwidth", () => {
    // Totals 1, 1, 0, 2: the tied first two persons take ranks 2 and 3 in their order.
    const [first] = kernelItemCurves(
      buildResponseMatrix([
        [1, 0],
        [0, 1],
        [0, 0],
        [1, 1],
      ]),
      { points: 3 },
    );
    const curve = defined(first);
    const scores = [2, 3, 1, 4].map((rank) => normalQuantile(rank / 5));
    const h = 1.06 * 4 ** -0.2;
    const [from, to] = [normalQuantile(1 / 5), normalQuantile(4 / 5)];
    const nodes = [from, from + (to - from) / 2, to];
    expect(curve.nodes).toEqual(nodes);
    nodes.forEach((node, q) => {
      const weights = scores.map((score) => Math.exp(-(((node - score) / h) ** 2) / 2));
      const total = weights.reduce((sum, w) => sum + w, 0);
      const squares = weights.reduce((sum, w) => sum + w * w, 0);
      const p = (defined(weights[0]) + defined(weights[3])) / total;
      expectClose(curve.probabilities[1]?.[q] ?? null, p, 1e-15, `P1 ${q}`);
      expectClose(curve.probabilities[0]?.[q] ?? null, 1 - p, 1e-15, `P0 ${q}`);
      expectClose(curve.standardErrors[1]?.[q] ?? null, Math.sqrt(p * (1 - p) * squares) / total, 1e-15, `SE ${q}`);
    });
  });

  it("leaves out a missing response and ranks no person without any", () => {
    const rows = [
      [1, 0, 2],
      [0, null, 1],
      [1, 1, 0],
      [0, 1, 2],
      [1, 1, 1],
    ];
    const curves = kernelItemCurves(buildResponseMatrix(rows));
    const empty = kernelItemCurves(buildResponseMatrix([...rows, [null, null, null]]));
    expectSameNumbers(empty, curves);
    // The second item without the person who skipped it: the same weights as the others' at each node.
    const scores = [3, 1, 2, 4, 5].map((rank) => normalQuantile(rank / 6));
    const h = 1.06 * 5 ** -0.2;
    const item = defined(curves[1]);
    item.nodes.forEach((node, q) => {
      const weights = scores.map((score) => Math.exp(-(((node - score) / h) ** 2) / 2));
      const answered = defined(weights[0]) + defined(weights[2]) + defined(weights[3]) + defined(weights[4]);
      const p = (answered - defined(weights[0])) / answered;
      expectClose(item.probabilities[1]?.[q] ?? null, p, 1e-14, `P1 ${q}`);
    });
  });

  it("smooths 200,000 items, more than a function call takes arguments", () => {
    // Two persons who score 0 on every item: one category of each item, at two nodes, so that the curves stay small.
    const items = 200_000;
    const curves = kernelItemCurves(buildResponseMatrix([Array(items).fill(0), Array(items).fill(0)]), { points: 2 });
    expect(curves).toHaveLength(items);
    expect(curves[items - 1]?.probabilities).toEqual([[1, 1]]);
  });

  it("takes the number of nodes and the bandwidth", () => {
    const { matrix } = simulated(48, 300, 4);
    const [curve] = kernelItemCurves(matrix, { points: 7, bandwidth: 0.5 });
    expect(defined(curve).nodes).toHaveLength(7);
    expect(defined(curve).probabilities[0]).toHaveLength(7);
    const [wide] = kernelItemCurves(matrix, { points: 7, bandwidth: 1e6 });
    // A bandwidth far wider than the scores weighs every person alike: each curve is the item's mean.
    const shares = matrix.responses.filter((row) => row[0] === 1).length / matrix.responses.length;
    for (const p of defined(wide).probabilities[1] ?? []) expectClose(p, shares, 1e-12);
  });

  it("keeps its curves defined at a bandwidth that underflows the weights, from the nearest persons", () => {
    // At h = 1e-3 the weights of persons 0.1 apart underflow; each node takes its nearest person's response.
    const rows = [
      [0, null],
      [0, 0],
      [1, null],
      [1, 1],
      [1, 1],
    ];
    const [first, second] = kernelItemCurves(buildResponseMatrix(rows), { points: 5, bandwidth: 1e-3 });
    expect(defined(first).probabilities[1]).toEqual([0, 0, 1, 1, 1]);
    // The second item's persons are ranked 2, 4 and 5: at the lowest node its nearest person is the second.
    expect(defined(second).probabilities[1]?.[0]).toBe(0);
    expect(defined(second).probabilities[1]?.[4]).toBe(1);
    expect(defined(second).standardErrors.flat().every(Number.isFinite)).toBe(true);
    // Below 1e-154 the square of the bandwidth underflows to 0: each node still takes its nearest person's response.
    const [narrowest] = kernelItemCurves(buildResponseMatrix(rows), { points: 5, bandwidth: 1e-200 });
    expect(defined(narrowest).probabilities[1]).toEqual([0, 0, 1, 1, 1]);
    expect(defined(narrowest).standardErrors.flat().every(Number.isFinite)).toBe(true);
  });

  it("gives a constant item one curve of 1", () => {
    const [constant] = kernelItemCurves(
      buildResponseMatrix([
        [0, 1],
        [0, 0],
        [0, 1],
      ]),
    );
    expect(defined(constant).probabilities).toHaveLength(1);
    expect(new Set(defined(constant).probabilities[0])).toEqual(new Set([1]));
    expect(new Set(defined(constant).standardErrors[0])).toEqual(new Set([0]));
  });

  it("rejects an item without responses, an empty category below the highest, and fewer than two persons", () => {
    expect(() =>
      kernelItemCurves(
        buildResponseMatrix([
          [1, null],
          [0, null],
        ]),
      ),
    ).toThrow('Item "item2" has no responses; kernel item curves need at least one.');
    expect(() => kernelItemCurves(buildResponseMatrix([[0], [3], [1]]))).toThrow(
      'Item "item1" has no response in category 2; kernel item curves need responses in every category up to the highest.',
    );
    expect(() => kernelItemCurves(buildResponseMatrix([[0], [null]]))).toThrow(
      "Kernel item curves need at least two persons with a response, found 1.",
    );
  });

  it("refuses arguments of other types", () => {
    const matrix = buildResponseMatrix([[0], [1], [1]]);
    expect(() => kernelItemCurves(JSON.parse("{}"))).toThrow(DataError);
    expect(() => kernelItemCurves(matrix, JSON.parse("null"))).toThrow(DataError);
    expect(() => kernelItemCurves(matrix, { points: 1 })).toThrow(
      "The option points must be an integer from 2, found 1.",
    );
    expect(() => kernelItemCurves(matrix, JSON.parse('{ "points": "51" }'))).toThrow(
      'The option points must be an integer from 2, found "51".',
    );
    expect(() => kernelItemCurves(matrix, { bandwidth: 0 })).toThrow(
      "The option bandwidth must be a positive number, found 0.",
    );
    expect(() => kernelItemCurves(matrix, JSON.parse('{ "bandwidth": null }'))).toThrow(
      "The option bandwidth must be a positive number, found null.",
    );
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
        const persons = buildResponseMatrix([...matrix.responses].reverse(), { itemIds: matrix.itemIds });
        expectSameNumbers(limitedInformationFit(persons, calibration, { statistic }), fit);
        const order = matrix.itemIds.map((_, i) => matrix.itemIds.length - 1 - i);
        const items = buildResponseMatrix(
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

  it("the local-dependence statistics of a pair do not depend on the order of the items", () => {
    fc.assert(
      fc.property(seeds, (seed) => {
        const { matrix, items: pool } = simulated(seed, 80, 4);
        const reversed = buildResponseMatrix(
          matrix.responses.map((row) => [...row].reverse()),
          { itemIds: [...matrix.itemIds].reverse() },
        );
        const forward = localDependenceChiSquare(matrix, pool);
        const backward = localDependenceChiSquare(reversed, [...pool].reverse());
        for (const pair of forward) {
          const twin = defined(backward.find((other) => other.items.join() === [...pair.items].reverse().join()));
          expectSameNumbers({ ...twin, items: [] }, { ...pair, items: [] });
        }
      }),
      { numRuns: 20 },
    );
  });

  it("the Rasch tests are at least 0, the same for a split given as labels, and free of the order of persons", () => {
    fc.assert(
      fc.property(seeds, (seed) => {
        const random = createRandom(seed);
        const items = [-1, -0.5, 0, 0.5, 1].map((b) => ({
          model: oneParameterLogistic,
          parameters: { a: 1, b, c: 0, d: 1 },
        }));
        const matrix = simulateResponses(items, simulateThetas(300, random), random);
        const totals = matrix.responses.map((row) => row.reduce<number>((sum, response) => sum + (response ?? 0), 0));
        const median =
          [...totals]
            .sort((a, b) => a - b)
            .slice(149, 151)
            .reduce((sum, total) => sum + total, 0) / 2;
        const split = totals.map((total) => (total <= median ? "lower" : "upper"));
        const andersen = andersenTest(matrix);
        expect(andersen.statistic).toBeGreaterThanOrEqual(0);
        expect(martinLofTest(matrix).statistic).toBeGreaterThanOrEqual(0);
        expectSameNumbers(andersenTest(matrix, { split }), andersen);
        expectSameNumbers(raschWaldTest(matrix, { split }), raschWaldTest(matrix));
        const reversed = buildResponseMatrix([...matrix.responses].reverse());
        expectSameNumbers(andersenTest(reversed), andersen);
        expectSameNumbers(martinLofTest(reversed), martinLofTest(matrix));
      }),
      { numRuns: 20 },
    );
  });

  it("kernel curves are probabilities from 0 to 1 that sum to 1 at each node, with errors from 0", () => {
    fc.assert(
      fc.property(
        seeds,
        fc.integer({ min: 2, max: 60 }),
        fc.integer({ min: 1, max: 5 }),
        fc.double({ min: 0.01, max: 5, noNaN: true }),
        (seed, persons, count, bandwidth) => {
          const { matrix } = simulated(seed, persons, count);
          // A person with every item wrong and one with every item right give each item both categories.
          const rows = [...matrix.responses, matrix.itemIds.map(() => 0), matrix.itemIds.map(() => 1)];
          for (const curve of kernelItemCurves(buildResponseMatrix(rows), { bandwidth })) {
            curve.nodes.forEach((_, q) => {
              const shares = curve.probabilities.map((row) => defined(row[q]));
              expect(shares.every((p) => p >= 0 && p <= 1)).toBe(true);
              expect(Math.abs(shares.reduce((sum, p) => sum + p, 0) - 1)).toBeLessThan(1e-12);
              expect(curve.standardErrors.every((row) => defined(row[q]) >= 0)).toBe(true);
            });
          }
        },
      ),
      { numRuns: 50 },
    );
  });

  it("item fit, S-X² and Q3 do not depend on the order of persons", () => {
    fc.assert(
      fc.property(seeds, (seed) => {
        const { matrix, items: pool } = simulated(seed, 60, 5);
        const thetas = matrix.personIds.map((_, person) => Math.sin(person));
        const reversed = buildResponseMatrix([...matrix.responses].reverse());
        const reversedThetas = [...thetas].reverse();
        expectSameNumbers(itemFit(reversed, pool, reversedThetas), itemFit(matrix, pool, thetas));
        expectSameNumbers(summedScoreFit(reversed, pool), summedScoreFit(matrix, pool));
        expectSameNumbers(yenQ3(reversed, pool, reversedThetas), yenQ3(matrix, pool, thetas));
      }),
      { numRuns: 20 },
    );
  });
});

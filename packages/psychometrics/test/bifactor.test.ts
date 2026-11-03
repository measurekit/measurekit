import { mean } from "@measurekit/core";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  type BifactorOptions,
  calibrateBifactor,
  calibrateMultidimensional,
  createRandom,
  DataError,
  type LoadingPattern,
  type MultidimensionalCalibration,
  type MultidimensionalGradedParameters,
  type MultidimensionalLogisticParameters,
  type MultidimensionalModel,
  multidimensionalEstimates,
  multidimensionalGradedResponse,
  multidimensionalPosteriorModeEstimate,
  multidimensionalThreeParameterLogistic,
  multidimensionalTwoParameterLogistic,
  normalPrior,
  type ParameterPrior,
  type ResponseMatrix,
  responseMatrix,
  responseRecords,
  simulateMultidimensionalResponses,
} from "../src/index.js";
import { compressResponses } from "../src/irt/internal.js";
import { createWalk, patternsOf, structureOf } from "../src/multidimensional/internal.js";
import { defined, expectClose } from "./assertions.js";
import graded from "./fixtures/bifactor-graded.json" with { type: "json" };
import incomplete from "./fixtures/bifactor-incomplete.json" with { type: "json" };
import long from "./fixtures/bifactor-long.json" with { type: "json" };
import m2pl from "./fixtures/bifactor-m2pl.json" with { type: "json" };
import m3pl from "./fixtures/bifactor-m3pl.json" with { type: "json" };
import recovery from "./fixtures/bifactor-recovery.json" with { type: "json" };
import twoTier from "./fixtures/bifactor-two-tier.json" with { type: "json" };

type Parameters = MultidimensionalLogisticParameters | MultidimensionalGradedParameters;

/** Item parameters in a fixture: the logistic and the graded names, whichever the model has. */
interface FixtureParameters {
  readonly a: readonly number[];
  readonly intercept?: number;
  readonly intercepts?: readonly number[];
  readonly c?: number;
}

interface FixtureItem {
  readonly id: string;
  readonly parameters: FixtureParameters;
  readonly standardErrors?: Readonly<Record<string, number | undefined>>;
}

/** The truth a fixture was simulated from: slopes by item and dimension. */
interface Truth {
  readonly slopes: readonly (readonly number[])[];
  readonly correlation?: number;
}

/** A bifactor calibration of mirt on simulated responses, with the structure measurekit is given. */
interface Case {
  readonly model: string;
  readonly points: number;
  readonly dimensions: readonly string[];
  readonly itemIds: readonly string[];
  readonly responses: readonly (readonly (number | null)[])[];
  readonly specific: Readonly<Record<string, string | null>>;
  readonly primary: LoadingPattern;
  readonly items: readonly FixtureItem[];
  readonly correlations: readonly (readonly number[])[];
  readonly correlationStandardErrors?: readonly (readonly number[])[];
  readonly standardized: {
    readonly loadings: readonly (readonly number[])[];
    readonly communalities: readonly number[];
  };
  readonly logLikelihood: number;
  readonly aic: number;
  readonly bic: number;
  /** The cycles mirt ran and whether it stopped before its limit. */
  readonly mirt: { readonly iterations: number; readonly converged: boolean };
  readonly truth?: Truth;
  /** The normal prior on logit c that mirt and measurekit apply, where the case has one. */
  readonly priors?: { readonly c: { readonly mean: number; readonly sd: number } };
  readonly crossProduct?: { readonly items: readonly FixtureItem[] };
}

/** The tolerances of the comparisons with mirt, in docs/psychometrics/validation.md. */
const TOLERANCES = { parameters: 5e-4, errors: 1e-3, likelihood: 1e-4 } as const;
/** The tolerance at which EM runs for a comparison with mirt's optimum, which mirt reaches at 1e-10. */
const TIGHT = 1e-8;
/** The time that a calibration at a tight tolerance may take on a shared machine, in milliseconds. */
const SLOW = 120_000;

/**
 * A seeded bifactor 2PL: one general and two specific dimensions of four items each, with slopes and intercepts that
 * identify the model at a few hundred persons.
 */
function simulatedCase(seed: number, persons: number): { matrix: ResponseMatrix; specific: Record<string, string> } {
  const random = createRandom(seed);
  const items = Array.from({ length: 8 }, (_, i) => {
    const slopes = [0.9 + 0.2 * (i % 4), i < 4 ? 0.7 + 0.15 * (i % 4) : 0, i < 4 ? 0 : 0.7 + 0.15 * (i % 4)];
    return {
      model: multidimensionalTwoParameterLogistic,
      parameters: { a: slopes, intercept: 0.6 * ((i % 4) - 1.5), c: 0, d: 1 },
    };
  });
  const thetas = Array.from({ length: persons }, () => [random.normal(), random.normal(), random.normal()]);
  const matrix = simulateMultidimensionalResponses(items, thetas, random);
  return {
    matrix,
    specific: Object.fromEntries(matrix.itemIds.map((id, i) => [id, i < 4 ? "s1" : "s2"])),
  };
}

const MODELS = {
  m2pl: multidimensionalTwoParameterLogistic,
  m3pl: multidimensionalThreeParameterLogistic,
  mgrm: multidimensionalGradedResponse,
} as const;

function modelOf(name: string): MultidimensionalModel<Parameters> {
  return defined(MODELS[name as keyof typeof MODELS]);
}

function matrixOf(fixture: Pick<Case, "responses" | "itemIds">): ResponseMatrix {
  return responseMatrix(fixture.responses, { itemIds: [...fixture.itemIds] });
}

/** A normal prior on logit(x), as mirt's `norm` on g. Implemented here as user code. */
function logitNormalPrior(mean: number, sd: number): ParameterPrior {
  const logit = (x: number): number => Math.log(x / (1 - x));
  const variance = sd * sd;
  return {
    logDensity: (x) => (-0.5 * (logit(x) - mean) ** 2) / variance,
    logDensityDerivative: (x) => -(logit(x) - mean) / (variance * x * (1 - x)),
    logDensitySecondDerivative: (x) => -(1 - (logit(x) - mean) * (1 - 2 * x)) / (variance * (x * (1 - x)) ** 2),
  };
}

function calibrateCase(
  fixture: Case,
  extra: Partial<BifactorOptions<Parameters>> = {},
): MultidimensionalCalibration<Parameters> {
  return calibrateBifactor(matrixOf(fixture), {
    specific: fixture.specific,
    primary: fixture.primary,
    model: modelOf(fixture.model),
    ...(fixture.priors === undefined
      ? {}
      : { priors: { c: logitNormalPrior(fixture.priors.c.mean, fixture.priors.c.sd) } }),
    points: fixture.points,
    tolerance: TIGHT,
    maxIterations: 5000,
    ...extra,
  });
}

/** Compares a calibration with mirt's: the parameters, correlations and fit statistics, and the standard errors. */
function expectCalibration(
  fit: MultidimensionalCalibration<Parameters>,
  fixture: Case,
  items: readonly FixtureItem[],
  withErrors: boolean,
): void {
  expect(fixture.mirt.converged, "mirt stopped before its limit of cycles").toBe(true);
  expect(fit.dimensions).toEqual(fixture.dimensions);
  items.forEach((expected, i) => {
    const found = defined(fit.items[i]);
    const target = expected.parameters;
    expect(found.id).toBe(expected.id);
    target.a.forEach((slope, d) => {
      expectClose(defined(found.parameters.a[d]), slope, TOLERANCES.parameters, `${expected.id} a${d + 1}`);
    });
    if ("intercepts" in found.parameters) {
      (target.intercepts ?? []).forEach((value, k) => {
        const intercepts = "intercepts" in found.parameters ? found.parameters.intercepts : [];
        expectClose(defined(intercepts[k]), value, TOLERANCES.parameters, `${expected.id} intercept${k + 1}`);
      });
    } else {
      expectClose(found.parameters.intercept, defined(target.intercept), TOLERANCES.parameters, `${expected.id} d`);
      expectClose(found.parameters.c, defined(target.c), TOLERANCES.parameters, `${expected.id} c`);
    }
    for (const [name, error] of Object.entries(withErrors ? (expected.standardErrors ?? {}) : {})) {
      const actual = found.standardErrors?.[name] ?? Number.NaN;
      expectClose(actual / defined(error), 1, TOLERANCES.errors, `${expected.id} standard error of ${name}`);
    }
  });
  fixture.correlations.forEach((row, i) => {
    row.forEach((value, j) => {
      expectClose(fit.correlations[i]?.[j] ?? Number.NaN, value, TOLERANCES.parameters, `correlation ${i},${j}`);
    });
  });
  fixture.standardized.loadings.forEach((row, i) => {
    row.forEach((value, d) => {
      expectClose(defined(fit.loadings[i]?.[d]), value, TOLERANCES.parameters, `loading ${i + 1},${d + 1}`);
    });
    expectClose(defined(fit.communalities[i]), defined(fixture.standardized.communalities[i]), TOLERANCES.parameters);
  });
  expectClose(fit.logLikelihood, fixture.logLikelihood, TOLERANCES.likelihood, "log-likelihood");
  expectClose(fit.aic, fixture.aic, TOLERANCES.likelihood, "AIC");
  expectClose(fit.bic, fixture.bic, TOLERANCES.likelihood, "BIC");
}

/** The root mean squared error of the slopes on the dimensions `from` to `to`, over the items that load on them. */
function slopeError(
  fit: MultidimensionalCalibration<Parameters>,
  truth: readonly (readonly number[])[],
  from: number,
  to: number,
): number {
  let sum = 0;
  let count = 0;
  fit.items.forEach((item, i) => {
    for (let d = from; d < to; d++) {
      const expected = truth[i]?.[d] ?? 0;
      if (expected === 0) continue;
      sum += (defined(item.parameters.a[d]) - expected) ** 2;
      count++;
    }
  });
  return Math.sqrt(sum / count);
}

describe("structure", () => {
  const ids = ["q1", "q2", "q3", "q4", "q5"];

  it("lists the primary dimensions and then the specific ones, each in order of first appearance", () => {
    const structure = structureOf(
      {
        specific: { q1: "b", q2: "b", q3: "a", q4: "a", q5: null },
        primary: { q1: ["g"], q2: ["g"], q3: ["g"], q4: ["g"], q5: ["g"] },
      },
      ids,
    );
    expect(structure.design.dimensionNames).toEqual(["g", "b", "a"]);
    expect(structure.primaryCount).toBe(1);
    expect(structure.design.correlated).toBe(false);
    expect(structure.groups).toEqual([
      { dimension: 1, items: [0, 1] },
      { dimension: 2, items: [2, 3] },
      { dimension: -1, items: [4] },
    ]);
    expect(structure.design.loads.map((row) => row.map(Number))).toEqual([
      [1, 1, 0],
      [1, 1, 0],
      [1, 0, 1],
      [1, 0, 1],
      [1, 0, 0],
    ]);
  });

  it("makes the dimensions correlated when there are two primary dimensions", () => {
    const structure = structureOf(
      {
        specific: { q1: "s", q2: "s", q3: null, q4: null, q5: null },
        primary: { q1: ["x"], q2: ["x", "y"], q3: ["y"], q4: ["y"], q5: ["x"] },
      },
      ids,
    );
    expect(structure.design.dimensionNames).toEqual(["x", "y", "s"]);
    expect(structure.design.correlated).toBe(true);
    expect(structure.primaryCount).toBe(2);
  });

  it("refuses a specification that is not valid", () => {
    const specific = { q1: "s", q2: "s", q3: null, q4: null, q5: null };
    expect(() => structureOf({ specific: { ...specific, q9: "s" } }, ids)).toThrow(DataError);
    expect(() => structureOf({ specific: { q1: "s", q2: "s" } }, ids)).toThrow(/specific dimension of item "q3"/);
    expect(() =>
      structureOf({ specific: JSON.parse('{"q1":["s","t"],"q2":"s","q3":null,"q4":null,"q5":null}') }, ids),
    ).toThrow(/at most one specific dimension/);
    expect(() => structureOf({ specific: { ...specific, q3: "t" } }, ids)).toThrow(/"t" has one item/);
    expect(() =>
      structureOf({ specific, primary: { q1: ["a"], q2: ["b"], q3: ["c"], q4: ["a"], q5: ["a"] } }, ids),
    ).toThrow(/found 3: a, b, c/);
    expect(() =>
      structureOf({ specific, primary: { q1: ["s"], q2: ["s"], q3: ["s"], q4: ["s"], q5: ["s"] } }, ids),
    ).toThrow(/"s" names a primary and a specific dimension/);
    expect(() => structureOf({ specific, primary: { q1: ["g", "g"] } }, ids)).toThrow(DataError);
    expect(() => structureOf({ specific, primary: { q1: ["g"] } }, ids)).toThrow(/primary dimensions of item "q2"/);
  });
});

describe("patterns", () => {
  it("collects the distinct patterns and the distinct sub-patterns of each group in order", () => {
    const matrix = responseMatrix(
      [
        [1, 0, 1, 1],
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [null, null, 0, 1],
        [null, null, null, null],
      ],
      { itemIds: ["a", "b", "c", "d"] },
    );
    const table = patternsOf(compressResponses(matrix), [
      { dimension: 1, items: [0, 1] },
      { dimension: 2, items: [2, 3] },
    ]);
    expect(table.count).toBe(3);
    expect(Array.from(table.frequency).sort()).toEqual([1, 1, 2]);
    const [first, second] = table.groups;
    expect(Array.from(first?.codes ?? [])).toEqual([-1, -1, 0, 1, 1, 0]);
    expect(Array.from(first?.answered ?? [])).toEqual([0, 1, 1]);
    expect(Array.from(second?.codes ?? [])).toEqual([0, 1, 1, 1]);
  });
});

describe("walk", () => {
  /** The weights times the product of the probabilities of the responses, directly. */
  function direct(
    tables: Float64Array[],
    codes: readonly number[],
    nodes: number,
    weights: Float64Array,
  ): Float64Array {
    return Float64Array.from({ length: nodes }, (_, m) => {
      return codes.reduce(
        (product, code, d) => (code < 0 ? product : product * (tables[d]?.[code * nodes + m] ?? 0)),
        weights[m % weights.length] ?? 0,
      );
    });
  }

  it("multiplies the probabilities of the responses of each sub-pattern, sharing the rows of the items before", () => {
    const width = 5;
    const primaryNodes = 3;
    const weights = Float64Array.of(0.25, 0.5, 0.25);
    const nodes = primaryNodes * weights.length;
    const tables = Array.from({ length: width }, (_, d) =>
      Float64Array.from({ length: 2 * nodes }, (__, i) => 0.1 + ((i * 7 + d * 3) % 17) / 20),
    );
    const rows = [
      [-1, 0, 1, 0, 1],
      [0, 0, 1, -1, 1],
      [0, 1, 1, 0, 0],
      [1, 0, -1, 0, 1],
      [1, 1, 1, 1, 1],
    ];
    const codes = Int32Array.from(rows.flat());
    const patterns = {
      group: { dimension: 1, items: [0, 1, 2, 3, 4] },
      width,
      count: rows.length,
      codes,
      answered: Uint8Array.from(rows, () => 1),
    };
    const visited: number[] = [];
    createWalk({ patterns, tables, primaryNodes, weights })((u, row, shift) => {
      visited.push(u);
      const expected = direct(tables, rows[u] ?? [], nodes, weights);
      expected.forEach((value, m) => {
        expectClose((row[m] ?? 0) * Math.exp(shift[Math.floor(m / weights.length)] ?? 0), value, 1e-12);
      });
    });
    expect(visited).toEqual([0, 1, 2, 3, 4]);
  });

  it("keeps the product of many small probabilities from underflowing", () => {
    const width = 100;
    const weights = Float64Array.of(0.5, 0.5);
    const tables = Array.from({ length: width }, () => Float64Array.of(1e-5, 1e-5, 1e-5, 1e-5));
    const codes = Int32Array.from({ length: width }, () => 0);
    const patterns = {
      group: { dimension: 1, items: Array.from({ length: width }, (_, i) => i) },
      width,
      count: 1,
      codes,
      answered: Uint8Array.of(1),
    };
    createWalk({ patterns, tables, primaryNodes: 2, weights })((_, row, shift) => {
      // 100 probabilities of 1e-5 and a weight of 0.5 multiply to 1e-500, which a plain product loses: the row keeps a
      // positive part and the shift the rest.
      expect(row[0]).toBeGreaterThan(0);
      expectClose(Math.log(row[0] ?? 0) + (shift[0] ?? 0), Math.log(0.5) - 500 * Math.LN10, 1e-9);
    });
  });
});

describe("calibrateBifactor", () => {
  it("matches mirt's bfactor on 12 items in 3 specific dimensions, standard errors included", { timeout: SLOW }, () => {
    const fixture: Case = m2pl.items12;
    expectCalibration(calibrateCase(fixture, { standardErrors: "hessian" }), fixture, fixture.items, true);
  });

  it("matches mirt's cross-product standard errors", { timeout: SLOW }, () => {
    const fixture: Case = m2pl.items12;
    const fit = calibrateCase(fixture, { standardErrors: "cross-product" });
    expectCalibration(fit, fixture, defined(fixture.crossProduct).items, true);
  });

  it("matches mirt's bfactor on 40 items in 4 specific dimensions", { timeout: SLOW }, () => {
    const fixture: Case = m2pl.items40;
    const fit = calibrateCase(fixture);
    expectCalibration(fit, fixture, fixture.items, false);
    expect(fit.items.every((item) => item.standardErrors === null)).toBe(true);
  });

  it("matches mirt's bfactor on specific dimensions of 20 items, where long products are rescaled", {
    timeout: SLOW,
  }, () => {
    const fixture: Case = long.long;
    expectCalibration(calibrateCase(fixture), fixture, fixture.items, false);
  });

  it("matches mirt's two-tier model, with the correlation of its primary dimensions and its standard error", {
    timeout: SLOW,
  }, () => {
    const fixture: Case = twoTier.twoTier;
    const fit = calibrateCase(fixture, { standardErrors: "hessian" });
    expectCalibration(fit, fixture, fixture.items, true);
    const expected = defined(fixture.correlationStandardErrors?.[0]?.[1]);
    expectClose(defined(fit.correlationStandardErrors?.[1]?.[0]) / expected, 1, TOLERANCES.errors);
    expect(fit.correlations[2]?.[0]).toBe(0);
    expect(fit.correlations[2]?.[2]).toBe(1);
  });

  it("matches mirt's bfactor with the 3PL model", { timeout: SLOW }, () => {
    const fixture: Case = m3pl.m3pl;
    expectCalibration(calibrateCase(fixture, { standardErrors: "hessian" }), fixture, fixture.items, true);
  });

  it("matches mirt's bfactor with the graded model", { timeout: SLOW }, () => {
    const fixture: Case = graded.graded;
    expectCalibration(calibrateCase(fixture, { standardErrors: "hessian" }), fixture, fixture.items, true);
  });

  it("matches mirt's bfactor with items that have no specific dimension", { timeout: SLOW }, () => {
    const fixture: Case = incomplete.ungrouped;
    const fit = calibrateCase(fixture);
    expectCalibration(fit, fixture, fixture.items, false);
    expect(fit.items[4]?.parameters.a).toEqual([expect.any(Number), 0, 0, 0]);
    expect(fit.items[9]?.parameters.a).toEqual([expect.any(Number), 0, 0, 0]);
  });

  it("matches mirt's bfactor with missing responses and persons who skipped a specific dimension", {
    timeout: SLOW,
  }, () => {
    const fixture: Case = incomplete.missing;
    expectCalibration(calibrateCase(fixture), fixture, fixture.items, false);
  });

  it("takes the iterations of calibrateMultidimensional with uncorrelated dimensions", { timeout: SLOW }, () => {
    // The two EM runs are one sequence of estimates, so after the same number of iterations they agree to rounding. Two
    // specific dimensions and the general one are the three dimensions that EM integrates in full.
    const { matrix, specific } = simulatedCase(7, 600);
    const settings = { points: 11, tolerance: 0, maxIterations: 60, standardErrors: "hessian" } as const;
    const reduced = calibrateBifactor(matrix, { specific, ...settings });
    const full = calibrateMultidimensional(matrix, {
      model: multidimensionalTwoParameterLogistic,
      loadings: Object.fromEntries(matrix.itemIds.map((id) => [id, ["general", specific[id] ?? ""]])),
      correlated: false,
      ...settings,
    });
    expect(reduced.iterations).toBe(full.iterations);
    expectClose(reduced.logLikelihood, full.logLikelihood, 1e-9);
    reduced.items.forEach((item, i) => {
      const other = defined(full.items[i]);
      item.parameters.a.forEach((slope, d) => {
        expectClose(slope, defined(other.parameters.a[d]), 1e-9);
      });
      expectClose(item.parameters.intercept, other.parameters.intercept, 1e-9);
      for (const [name, error] of Object.entries(defined(other.standardErrors))) {
        expectClose(defined(item.standardErrors?.[name]) / error, 1, 1e-7, `${item.id} ${name}`);
      }
    });
  });

  it("applies priors as calibrateMultidimensional does, so that the estimates are posterior modes", {
    timeout: SLOW,
  }, () => {
    const { matrix, specific } = simulatedCase(7, 600);
    const priors = { a: normalPrior(1, 0.4), intercept: normalPrior(0, 1) };
    const settings = { points: 11, tolerance: 0, maxIterations: 60 } as const;
    const reduced = calibrateBifactor(matrix, { specific, priors, ...settings });
    const plain = calibrateBifactor(matrix, { specific, ...settings });
    const full = calibrateMultidimensional(matrix, {
      model: multidimensionalTwoParameterLogistic,
      loadings: Object.fromEntries(matrix.itemIds.map((id) => [id, ["general", specific[id] ?? ""]])),
      correlated: false,
      priors,
      ...settings,
    });
    reduced.items.forEach((item, i) => {
      item.parameters.a.forEach((slope, d) => {
        expectClose(slope, defined(full.items[i]?.parameters.a[d]), 1e-6);
      });
    });
    expectClose(reduced.logLikelihood, full.logLikelihood, 1e-6);
    const spread = (fit: MultidimensionalCalibration<Parameters>): number => {
      const slopes = fit.items.flatMap((item) => item.parameters.a);
      return Math.max(...slopes) - Math.min(...slopes);
    };
    expect(spread(reduced)).toBeLessThan(spread(plain));
  });

  describe("input", () => {
    const fixture: Case = m2pl.items12;
    const quick = { points: 9, maxIterations: 40 } as const;

    it("accepts response records as it accepts a matrix", () => {
      const records = fixture.responses.flatMap((row, person) =>
        row.flatMap((response, item) =>
          response === null ? [] : [{ person: `p${person}`, item: defined(fixture.itemIds[item]), response }],
        ),
      );
      const fromRecords = calibrateBifactor(responseRecords(records, { itemIds: [...fixture.itemIds] }), {
        specific: fixture.specific,
        ...quick,
      });
      const fromMatrix = calibrateBifactor(matrixOf(fixture), { specific: fixture.specific, ...quick });
      expect(fromRecords).toEqual(fromMatrix);
    });

    it("leaves a person without responses out of the calibration", () => {
      const silent = fixture.itemIds.map(() => null);
      const padded = responseMatrix([...fixture.responses, silent], { itemIds: [...fixture.itemIds] });
      const fit = calibrateBifactor(padded, { specific: fixture.specific, ...quick });
      const plain = calibrateBifactor(matrixOf(fixture), { specific: fixture.specific, ...quick });
      expect(fit).toEqual(plain);
    });

    it("names the dimensions of the default model: one general dimension and the specific ones", () => {
      const fit = calibrateBifactor(matrixOf(fixture), { specific: fixture.specific, ...quick });
      expect(fit.dimensions).toEqual(["general", "s1", "s2", "s3"]);
      expect(fit.correlationStandardErrors).toBeNull();
      expect(fit.items.every((item) => item.standardErrors === null)).toBe(true);
    });

    it("refuses input of other types, as JavaScript and JSON pass it", () => {
      const { specific } = fixture;
      const matrix = matrixOf(fixture);
      const refused = (data: unknown, options: unknown): void => {
        expect(() => calibrateBifactor(JSON.parse(JSON.stringify(data)), JSON.parse(JSON.stringify(options)))).toThrow(
          DataError,
        );
      };
      refused(null, { specific });
      refused("responses", { specific });
      refused(matrix, null);
      refused(matrix, {});
      refused(matrix, { specific: null });
      refused(matrix, { specific: [] });
      refused(matrix, { specific: "s1" });
      refused(matrix, { specific, primary: null });
      refused(matrix, { specific, primary: [] });
      refused(matrix, { specific, points: "9" });
      refused(matrix, { specific, points: 1 });
      refused(matrix, { specific, points: 2.5 });
      refused(matrix, { specific, standardErrors: "oakes" });
      refused(matrix, { specific, standardErrors: null });
      refused(matrix, { specific, tolerance: "1e-4" });
      refused(matrix, { specific, tolerance: -1 });
      refused(matrix, { specific, maxIterations: 0 });
      refused(matrix, { specific, maxIterations: null });
      refused(matrix, { specific, model: {} });
      refused(matrix, { specific, model: null });
    });

    it("refuses a calibration without responses", () => {
      const silent = responseMatrix(
        [
          [null, null],
          [null, null],
        ],
        { itemIds: ["a", "b"] },
      );
      expect(() => calibrateBifactor(silent, { specific: { a: "s", b: "s" } })).toThrow(/at least one response/);
    });

    it("refuses a grid of more nodes than a calibration allows", () => {
      expect(() => calibrateBifactor(matrixOf(fixture), { specific: fixture.specific, points: 1001 })).toThrow(
        /more than 1000000 nodes/,
      );
    });

    it("limits the probabilities of the items' categories that EM keeps at the nodes of their groups", () => {
      // 1,000 points make a group of 10⁶ nodes, which 17 dichotomous items fill with 3.4e7 probabilities, above 2 ** 24.
      const ids = Array.from({ length: 17 }, (_, j) => `q${j}`);
      const wide = responseMatrix(
        [Array.from({ length: 17 }, (_, j) => j % 2), Array.from({ length: 17 }, (_, j) => 1 - (j % 2))],
        { itemIds: ids },
      );
      const specific = Object.fromEntries(ids.map((id) => [id, "passage"]));
      expect(() => calibrateBifactor(wide, { specific, points: 1000, maxIterations: 1 })).toThrow(
        "The items of the calibration would keep 34000000 values at the nodes of their grids, more than 16777216; use fewer points.",
      );
    });

    it("counts toward the limit the tables of pairs of items that Louis' standard errors take, and only those", () => {
      // A group of 10⁶ nodes: 5 dichotomous items keep 1e7 probabilities, and their 10 pairs 4e7 more for Louis' formula.
      const ids = Array.from({ length: 5 }, (_, j) => `q${j}`);
      const wide = responseMatrix(
        [Array.from({ length: 5 }, (_, j) => j % 2), Array.from({ length: 5 }, (_, j) => 1 - (j % 2))],
        { itemIds: ids },
      );
      const options = {
        specific: Object.fromEntries(ids.map((id) => [id, "passage"])),
        points: 1000,
        maxIterations: 1,
      };
      expect(() => calibrateBifactor(wide, { ...options, standardErrors: "hessian" })).toThrow(
        "The items of the calibration would keep 50000000 values at the nodes of their grids, more than 16777216; use fewer points.",
      );
      expect(calibrateBifactor(wide, { ...options, standardErrors: "cross-product" }).items).toHaveLength(5);
    });

    it("scores persons from its items, as any multidimensional calibration", () => {
      const fit = calibrateBifactor(matrixOf(fixture), { specific: fixture.specific, ...quick });
      const prior = { mean: fit.dimensions.map(() => 0), covariance: fit.correlations };
      const scored = multidimensionalEstimates(matrixOf(fixture), fit.items, (items, responses) =>
        multidimensionalPosteriorModeEstimate(items, responses, { prior }),
      );
      expect(scored).toHaveLength(fixture.responses.length);
      const totals = fixture.responses.map((row) => row.reduce<number>((sum, response) => sum + (response ?? 0), 0));
      const general = scored.map((person) => defined(person.estimate?.theta[0]));
      const high = general.filter((_, person) => defined(totals[person]) >= 8);
      const low = general.filter((_, person) => defined(totals[person]) <= 4);
      expect(mean(high)).toBeGreaterThan(mean(low) + 1);
      expect(scored[0]?.estimate?.theta).toHaveLength(4);
    });
  });

  it("runs the sample of the guide", () => {
    const random = createRandom(5);
    const items = Array.from({ length: 12 }, (_, i) => {
      const a = [1.2, 0, 0, 0];
      a[1 + Math.floor(i / 4)] = 0.9;
      return {
        model: multidimensionalTwoParameterLogistic,
        parameters: { a, intercept: ((i % 4) - 1.5) * 0.6, c: 0, d: 1 },
      };
    });
    const thetas = Array.from({ length: 2000 }, () => [
      random.normal(),
      random.normal(),
      random.normal(),
      random.normal(),
    ]);
    const matrix = simulateMultidimensionalResponses(items, thetas, random);
    const specific = Object.fromEntries(matrix.itemIds.map((id, i) => [id, `passage${1 + Math.floor(i / 4)}`]));
    const fit = calibrateBifactor(matrix, { specific, standardErrors: "hessian" });
    expect(fit.dimensions).toEqual(["general", "passage1", "passage2", "passage3"]);
    const first = defined(fit.items[0]);
    expect(first.parameters.a.map((value) => Number(value.toFixed(2)))).toEqual([1.19, 1.07, 0, 0]);
    expect(Number(first.parameters.intercept.toFixed(2))).toBe(-0.96);
    expect(Object.values(defined(first.standardErrors)).map((value) => Number(value.toFixed(2)))).toEqual([
      0.11, 0.17, 0.08,
    ]);
    expect(defined(fit.items[5]).parameters.a.map((value) => Number(value.toFixed(2)))).toEqual([1.08, 0, 0.87, 0]);
    expect(fit.logLikelihood.toFixed(2)).toBe("-14468.29");
    expect(fit.converged).toBe(true);
    const prior = { mean: [0, 0, 0, 0], covariance: fit.correlations };
    const scores = multidimensionalEstimates(matrix, fit.items, (calibrated, responses) =>
      multidimensionalPosteriorModeEstimate(calibrated, responses, { prior }),
    );
    expect(defined(scores[0]?.estimate).theta.map((value) => Number(value.toFixed(2)))).toEqual([
      0.63, 0.14, 0.14, 0.22,
    ]);
  });

  describe("recovery", () => {
    it("recovers the general and the specific slopes of 12 items, as mirt does", { timeout: SLOW }, () => {
      const fixture: Case = m2pl.items12;
      const fit = calibrateCase(fixture, { tolerance: 1e-4 });
      const mirt = { a: slopeOfFixture(fixture, 0, 1), s: slopeOfFixture(fixture, 1, 4) };
      expect(slopeError(fit, defined(fixture.truth).slopes, 0, 1)).toBeLessThan(mirt.a + 0.005);
      expect(slopeError(fit, defined(fixture.truth).slopes, 1, 4)).toBeLessThan(mirt.s + 0.005);
    });

    it("recovers the general and the specific slopes of 40 items, as mirt does", { timeout: SLOW }, () => {
      const fixture: Case = m2pl.items40;
      const fit = calibrateCase(fixture, { tolerance: 1e-4 });
      const mirt = { a: slopeOfFixture(fixture, 0, 1), s: slopeOfFixture(fixture, 1, 5) };
      expect(slopeError(fit, defined(fixture.truth).slopes, 0, 1)).toBeLessThan(mirt.a + 0.005);
      expect(slopeError(fit, defined(fixture.truth).slopes, 1, 5)).toBeLessThan(mirt.s + 0.005);
    });

    it("recovers the slopes and the correlation of a two-tier model of 80 items on 21 nodes", { timeout: SLOW }, () => {
      const fit = calibrateBifactor(responseMatrix(recovery.responses, { itemIds: recovery.itemIds }), {
        specific: recovery.specific,
        primary: recovery.primary,
        tolerance: 1e-3,
      });
      expect(fit.converged).toBe(true);
      expect(slopeError(fit, recovery.truth.slopes, 0, 2)).toBeLessThan(RECOVERY.primary);
      expect(slopeError(fit, recovery.truth.slopes, 2, 18)).toBeLessThan(RECOVERY.specific);
      expect(Math.abs(defined(fit.correlations[1]?.[0]) - recovery.truth.correlation)).toBeLessThan(
        RECOVERY.correlation,
      );
    });
  });
});

/** The root mean squared error of mirt's slopes on dimensions `from` to `to` against the truth of the fixture. */
function slopeOfFixture(fixture: Case, from: number, to: number): number {
  let sum = 0;
  let count = 0;
  fixture.items.forEach((item, i) => {
    for (let d = from; d < to; d++) {
      const expected = fixture.truth?.slopes[i]?.[d] ?? 0;
      if (expected === 0) continue;
      sum += (defined(item.parameters.a[d]) - expected) ** 2;
      count++;
    }
  });
  return Math.sqrt(sum / count);
}

/** The largest error of the recovery of the two-tier case, fixed from its first run. */
const RECOVERY = { primary: 0.1, specific: 0.14, correlation: 0.04 } as const;

describe("properties", () => {
  const specific = { q1: "a", q2: "a", q3: "a", q4: "b", q5: "b", q6: "b" };
  const itemIds = Object.keys(specific);

  /** Responses in which every item has both categories, so that every item can be estimated. */
  const responses = fc
    .array(fc.array(fc.constantFrom(0, 1, null), { minLength: 6, maxLength: 6 }), { minLength: 30, maxLength: 30 })
    .filter((rows) =>
      itemIds.every((_, item) => [0, 1].every((category) => rows.some((row) => row[item] === category))),
    );

  const fitOf = (rows: readonly (readonly (number | null)[])[]): MultidimensionalCalibration<Parameters> =>
    calibrateBifactor(responseMatrix(rows, { itemIds }), { specific, points: 5, maxIterations: 25 });

  it("does not depend on the order of the persons", { timeout: SLOW }, () => {
    const { matrix, specific: groups } = simulatedCase(5, 1000);
    const settings = { specific: groups, points: 9, tolerance: 1e-8, maxIterations: 1000 };
    const first = calibrateBifactor(matrix, settings);
    expect(first.converged).toBe(true);
    const permutations = fc.shuffledSubarray(
      Array.from({ length: matrix.responses.length }, (_, person) => person),
      { minLength: matrix.responses.length, maxLength: matrix.responses.length },
    );
    fc.assert(
      fc.property(permutations, (permutation) => {
        const rows = permutation.map((person) => defined(matrix.responses[person]));
        const second = calibrateBifactor(responseMatrix(rows, { itemIds: matrix.itemIds }), settings);
        expectClose(first.logLikelihood, second.logLikelihood, 1e-6);
        first.items.forEach((item, i) => {
          const other = defined(second.items[i]);
          item.parameters.a.forEach((slope, d) => {
            expectClose(slope, defined(other.parameters.a[d]), 1e-5);
          });
          expectClose(item.parameters.intercept, other.parameters.intercept, 1e-5);
        });
      }),
      { numRuns: 8 },
    );
  });

  it("gives every item a slope of 0 on the specific dimensions it does not load on", () => {
    fc.assert(
      fc.property(responses, (rows) => {
        const fit = fitOf(rows);
        fit.items.forEach((item, i) => {
          const own = i < 3 ? 1 : 2;
          item.parameters.a.forEach((slope, d) => {
            if (d === 0) expect(slope).not.toBe(0);
            else if (d !== own) expect(slope).toBe(0);
          });
        });
      }),
      { numRuns: 15 },
    );
  });

  it("returns a negative log-likelihood and uncorrelated dimensions", () => {
    fc.assert(
      fc.property(responses, (rows) => {
        const fit = fitOf(rows);
        expect(fit.logLikelihood).toBeLessThan(0);
        expect(fit.correlations).toEqual([
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ]);
      }),
      { numRuns: 15 },
    );
  });
});

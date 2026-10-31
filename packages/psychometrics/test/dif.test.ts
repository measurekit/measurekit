import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  DataError,
  type DifGroup,
  type DifPurification,
  type LogisticRegressionDif,
  logisticRegressionDif,
  lordChiSquare,
  type MantelHaenszel,
  type MantelTest,
  mantelHaenszel,
  mantelTest,
  type OrdinalLogisticDif,
  oneParameterLogistic,
  ordinalLogisticRegressionDif,
  type PurificationOptions,
  responseMatrix,
  twoParameterLogistic,
} from "../src/index.js";
import fixture from "./fixtures/dif.json" with { type: "json" };
import polytomous from "./fixtures/dif-polytomous.json" with { type: "json" };
import { expectClose } from "./helpers.js";

// Tolerances against R, explained in docs/validation.md.
/** Mantel-Haenszel statistics against difR and base R. */
const MANTEL_HAENSZEL_TOLERANCE = 1e-12;
/** Logistic regression chi-squares and R² changes against difR's glm fits. */
const LOGISTIC_TOLERANCE = 1e-9;
/** Lord's chi-square, relative, against difR on converged ltm calibrations. */
const LORD_RELATIVE_TOLERANCE = 1e-5;
/** The mean-sigma transformation of the focal scale. */
const LORD_TRANSFORMATION_TOLERANCE = 1e-8;
/** Mantel's chi-square, the generalized chi-square and the standardized mean difference against base R. */
const MANTEL_TEST_TOLERANCE = 1e-12;
/** Ordinal logistic regression chi-squares, R² changes and slope changes against MASS::polr. */
const ORDINAL_TOLERANCE = 1e-8;
/** lordif's statistics, which it rounds to 4 decimals. */
const LORDIF_TOLERANCE = 5e-5 + 1e-9;

/** Item 1 everyone answered correctly; items 2 and 3 vary. */
const CONSTANT_ROWS = [
  [1, 0, 1],
  [1, 1, 0],
  [1, 0, 0],
  [1, 1, 1],
  [1, 1, 0],
  [1, 0, 1],
];
const CONSTANT_GROUPS: DifGroup[] = ["reference", "reference", "reference", "focal", "focal", "focal"];

/** Two persons in each of two score groups: 1 holds responses 0 and 1 to item 1, 5 holds 2 and 3. */
const DISJOINT_ROWS = [
  [0, 1],
  [1, 0],
  [2, 3],
  [3, 2],
];

/** Three items, of highest scores 2, 2 and 1, answered by twelve persons. */
const SCALING_ROWS = [
  [0, 0, 0],
  [1, 0, 1],
  [1, 1, 0],
  [2, 1, 1],
  [2, 2, 1],
  [0, 1, 0],
  [0, 1, 0],
  [1, 1, 1],
  [2, 1, 0],
  [2, 2, 1],
  [1, 2, 1],
  [2, 0, 1],
];
const SCALING_GROUPS: DifGroup[] = SCALING_ROWS.map((_, person) => (person < 6 ? "reference" : "focal"));

function groups(focal: readonly number[]): DifGroup[] {
  return focal.map((flag) => (flag === 1 ? "focal" : "reference"));
}

/** Logistic regression tests of a fixture, by test, then item. */
type LogisticFixture = Readonly<
  Record<
    "uniform" | "nonuniform" | "both",
    {
      readonly chiSquare: readonly (number | null)[];
      readonly significance: readonly (number | null)[];
      readonly rSquaredChange: readonly (number | null)[];
    }
  >
>;

/** Mantel-Haenszel statistics of a fixture, by statistic, then item. */
interface MantelHaenszelFixture {
  readonly chiSquare: readonly (number | null)[];
  readonly significance: readonly (number | null)[];
  readonly oddsRatio: readonly (number | null)[];
  readonly delta: readonly (number | null)[];
  readonly deltaStandardError: readonly (number | null)[];
  readonly etsCategory: readonly (string | null)[];
}

/** A purification of a fixture: R's anchors, 1-based, rounds and convergence. */
interface PurificationFixture {
  readonly anchors: readonly number[];
  readonly iterations: number;
  readonly converged: boolean;
}

function expectLogistic(
  result: readonly (LogisticRegressionDif | OrdinalLogisticDif)[],
  expected: LogisticFixture,
  tolerance = LOGISTIC_TOLERANCE,
): void {
  for (const type of ["uniform", "nonuniform", "both"] as const) {
    const tests = expected[type];
    result.forEach((item, i) => {
      const test = item[type];
      expectClose(test.chiSquare, tests.chiSquare[i] ?? null, tolerance, `${type} ${i + 1}`);
      expectClose(test.significance, tests.significance[i] ?? null, tolerance, `${type} p ${i + 1}`);
      expectClose(test.rSquaredChange, tests.rSquaredChange[i] ?? null, tolerance, `${type} R²`);
      expect(test.degreesOfFreedom).toBe(type === "both" ? 2 : 1);
    });
  }
}

function expectMantelHaenszel(result: readonly MantelHaenszel[], expected: MantelHaenszelFixture): void {
  result.forEach((item, i) => {
    for (const name of ["chiSquare", "significance", "oddsRatio", "delta", "deltaStandardError"] as const) {
      expectClose(item[name], expected[name][i] ?? null, MANTEL_HAENSZEL_TOLERANCE, `${name} ${i + 1}`);
    }
    expect(item.etsCategory).toBe(expected.etsCategory[i]);
  });
}

/** Mantel statistics of a fixture, by statistic, then item. */
interface MantelTestFixture {
  readonly chiSquare: readonly (number | null)[];
  readonly significance: readonly (number | null)[];
  readonly generalizedChiSquare: readonly (number | null)[];
  readonly generalizedDegreesOfFreedom: readonly number[];
  readonly generalizedSignificance: readonly (number | null)[];
  readonly standardizedMeanDifference: readonly (number | null)[];
  readonly effectSize: readonly (number | null)[];
  readonly etsCategory: readonly (string | null)[];
}

/** Ordinal logistic regression tests of a fixture: those of a logistic regression, and the slope change by item. */
type OrdinalFixture = LogisticFixture & { readonly slopeChange: readonly (number | null)[] };

/** lordif's rounded p-values and R² changes by test, then item, and its slope change by item. */
type LordifFixture = Readonly<
  Record<
    "uniform" | "nonuniform" | "both",
    { readonly significance: readonly number[]; readonly rSquaredChange: readonly number[] }
  >
> & { readonly slopeChange: readonly number[] };

function expectMantelTest(result: readonly MantelTest[], expected: MantelTestFixture): void {
  result.forEach((item, i) => {
    const label = `item ${i + 1}`;
    expectClose(item.chiSquare, expected.chiSquare[i] ?? null, MANTEL_TEST_TOLERANCE, `chi-square ${label}`);
    expectClose(item.significance, expected.significance[i] ?? null, MANTEL_TEST_TOLERANCE, `p ${label}`);
    const { generalized } = item;
    const chiSquare = expected.generalizedChiSquare[i] ?? null;
    expectClose(generalized.chiSquare, chiSquare, MANTEL_TEST_TOLERANCE, `generalized chi-square ${label}`);
    expectClose(generalized.significance, expected.generalizedSignificance[i] ?? null, MANTEL_TEST_TOLERANCE, label);
    expect(generalized.degreesOfFreedom, label).toBe(expected.generalizedDegreesOfFreedom[i]);
    const smd = expected.standardizedMeanDifference[i] ?? null;
    expectClose(item.standardizedMeanDifference, smd, MANTEL_TEST_TOLERANCE, `SMD ${label}`);
    expectClose(item.effectSize, expected.effectSize[i] ?? null, MANTEL_TEST_TOLERANCE, `effect size ${label}`);
    expect(item.etsCategory, label).toBe(expected.etsCategory[i]);
  });
}

/** Compares ordinal logistic regression with polr's, and with lordif's statistics, which it rounds to 4 decimals. */
function expectOrdinal(
  result: readonly OrdinalLogisticDif[],
  expected: OrdinalFixture,
  lordif: LordifFixture | null = null,
): void {
  expectLogistic(result, expected, ORDINAL_TOLERANCE);
  result.forEach((item, i) => {
    expectClose(item.slopeChange, expected.slopeChange[i] ?? null, ORDINAL_TOLERANCE, `slope change ${i + 1}`);
    if (lordif === null) return;
    for (const type of ["uniform", "nonuniform", "both"] as const) {
      const { significance, rSquaredChange } = lordif[type];
      expectClose(item[type].significance, significance[i] ?? null, LORDIF_TOLERANCE, `lordif ${type} p ${i + 1}`);
      expectClose(item[type].rSquaredChange, rSquaredChange[i] ?? null, LORDIF_TOLERANCE, `lordif ${type} R² ${i + 1}`);
    }
    expectClose(item.slopeChange, lordif.slopeChange[i] ?? null, LORDIF_TOLERANCE, `lordif slope change ${i + 1}`);
  });
}

function expectPurification(actual: DifPurification | null, expected: PurificationFixture): void {
  expect(actual).toEqual({
    anchors: expected.anchors.map((index) => `item${index}`),
    iterations: expected.iterations,
    converged: expected.converged,
  });
}

describe("logisticRegressionDif", () => {
  for (const dataset of fixture.cases) {
    it(`matches difR's difLogistic on ${dataset.name} data`, () => {
      const result = logisticRegressionDif(responseMatrix(dataset.responses), groups(dataset.focal));
      expect(result.purification).toBeNull();
      expectLogistic(result.items, dataset.logistic);
    });

    it(`matches difR's purified difLogistic on ${dataset.name} data`, () => {
      const expected = dataset.purifiedLogistic;
      const result = logisticRegressionDif(responseMatrix(dataset.responses), groups(dataset.focal), { purify: true });
      expectLogistic(result.items, expected);
      expectPurification(result.purification, expected);
    });

    it(`matches base R's rule for sparse data, purified and not, on ${dataset.name} data`, () => {
      const matrix = responseMatrix(dataset.sparse.responses);
      expectLogistic(logisticRegressionDif(matrix, groups(dataset.focal)).items, dataset.sparse.logistic);
      const purified = logisticRegressionDif(matrix, groups(dataset.focal), { purify: true });
      expectLogistic(purified.items, dataset.sparse.purifiedLogistic);
      expectPurification(purified.purification, dataset.sparse.purifiedLogistic);
    });
  }

  it("gives null tests for an item every person answered alike or with persons of one group", () => {
    const { items } = logisticRegressionDif(responseMatrix(CONSTANT_ROWS), CONSTANT_GROUPS);
    expect(items[0]?.both).toEqual({ chiSquare: null, degreesOfFreedom: 2, significance: null, rSquaredChange: null });
    expect(items[1]?.uniform.chiSquare).not.toBeNull();
    const rows = CONSTANT_ROWS.map((row, person) => [person < 3 ? (row[0] ?? null) : null, ...row.slice(1)]);
    expect(logisticRegressionDif(responseMatrix(rows), CONSTANT_GROUPS).items[0]?.uniform.chiSquare).toBeNull();
  });
});

describe("lordChiSquare", () => {
  for (const dataset of fixture.cases) {
    it(`matches difR's difLord on ${dataset.name} data`, () => {
      const result = lordChiSquare(responseMatrix(dataset.responses), groups(dataset.focal), {
        model: twoParameterLogistic,
        quadrature: { method: "gauss-hermite", points: 21 },
        tolerance: 1e-10,
        maxIterations: 10_000,
      });
      const { lord } = dataset;
      result.items.forEach((item, i) => {
        const expected = lord.chiSquare[i] ?? 0;
        expectClose(item.chiSquare, expected, LORD_RELATIVE_TOLERANCE * expected, `chi-square ${i + 1}`);
        expect(item.degreesOfFreedom).toBe(2);
      });
      expectClose(result.transformation.slope, lord.transformation.slope, LORD_TRANSFORMATION_TOLERANCE);
      expectClose(result.transformation.intercept, lord.transformation.intercept, LORD_TRANSFORMATION_TOLERANCE);
      expect(result.reference.converged && result.focal.converged).toBe(true);
    });
  }

  it("rejects models other than the 2PL, 3PL and 4PL, and a group without persons", () => {
    const matrix = responseMatrix(CONSTANT_ROWS);
    const custom = { ...twoParameterLogistic, name: "custom 2PL" };
    expect(() => lordChiSquare(matrix, CONSTANT_GROUPS, { model: oneParameterLogistic })).toThrow(DataError);
    expect(() => lordChiSquare(matrix, CONSTANT_GROUPS, { model: custom })).toThrow(DataError);
    const references: DifGroup[] = CONSTANT_GROUPS.map(() => "reference");
    expect(() => lordChiSquare(matrix, references, { model: twoParameterLogistic })).toThrow(DataError);
  });
});

describe("mantelHaenszel", () => {
  for (const dataset of fixture.cases) {
    for (const expected of dataset.mantelHaenszel) {
      const correction = expected.continuityCorrection ? "with" : "without";
      it(`matches difR's difMH ${correction} continuity correction on ${dataset.name} data`, () => {
        const result = mantelHaenszel(responseMatrix(dataset.responses), groups(dataset.focal), {
          continuityCorrection: expected.continuityCorrection,
        });
        expectMantelHaenszel(result.items, expected);
      });
    }

    it(`matches difR's purified difMH on ${dataset.name} data`, () => {
      const expected = dataset.purifiedMantelHaenszel;
      const result = mantelHaenszel(responseMatrix(dataset.responses), groups(dataset.focal), { purify: true });
      expectMantelHaenszel(result.items, expected);
      expectPurification(result.purification, expected);
    });

    it(`matches base R's rule for sparse data, purified and not, on ${dataset.name} data`, () => {
      const matrix = responseMatrix(dataset.sparse.responses);
      expectMantelHaenszel(mantelHaenszel(matrix, groups(dataset.focal)).items, dataset.sparse.mantelHaenszel);
      const purified = mantelHaenszel(matrix, groups(dataset.focal), { purify: true });
      expectMantelHaenszel(purified.items, dataset.sparse.purifiedMantelHaenszel);
      expectPurification(purified.purification, dataset.sparse.purifiedMantelHaenszel);
    });
  }

  it("gives null statistics for an item every person answered alike", () => {
    const result = mantelHaenszel(responseMatrix(CONSTANT_ROWS), CONSTANT_GROUPS).items;
    expect(result[0]).toEqual({
      id: "item1",
      chiSquare: null,
      significance: null,
      oddsRatio: null,
      delta: null,
      deltaStandardError: null,
      etsCategory: null,
    });
  });

  it("analyzes each item on the persons who answered it, matched on their scaled score", () => {
    const base = mantelHaenszel(responseMatrix(CONSTANT_ROWS), CONSTANT_GROUPS).items;
    // A person who did not answer item 1, or answered no other item, changes nothing for item 1.
    const rows = [...CONSTANT_ROWS, [null, 1, 0], [1, null, null]];
    const sparse = mantelHaenszel(responseMatrix(rows), [...CONSTANT_GROUPS, "focal", "reference"]).items;
    expect(sparse[0]).toEqual(base[0]);
    // [1, 1, null] answered 1 of 1 other item correct, which scales to 2 of 2: score 3, not 2. Score 2 then holds
    // [0, 1, 1] and [1, 1, 0] alone: A = 0, B = 1, C = 1, D = 0, so Σ (A - E(A)) = -0.5 and Var(A) = 0.25.
    const scaled = [
      [1, 1, null],
      [1, 1, 1],
      [0, 1, 1],
      [1, 1, 0],
    ];
    const result = mantelHaenszel(responseMatrix(scaled), ["reference", "focal", "reference", "focal"], {
      continuityCorrection: false,
    });
    expect(result.items[0]?.chiSquare).toBeCloseTo(1, 15);
  });

  it("purifies the matching criterion until the flags settle, and reports no rounds without flags", () => {
    const { purification } = mantelHaenszel(responseMatrix(CONSTANT_ROWS), CONSTANT_GROUPS, { purify: true });
    expect(purification).toEqual({ anchors: ["item1", "item2", "item3"], iterations: 0, converged: true });
  });

  it("rejects invalid groups, polytomous items and a group without responses", () => {
    const matrix = responseMatrix(CONSTANT_ROWS);
    expect(() => mantelHaenszel(matrix, CONSTANT_GROUPS.slice(1))).toThrow(DataError);
    // Groups from untyped data, as JavaScript callers may pass them.
    const invalid: DifGroup[] = JSON.parse(JSON.stringify(["other", ...CONSTANT_GROUPS.slice(1)]));
    expect(() => mantelHaenszel(matrix, invalid)).toThrow(DataError);
    expect(() =>
      mantelHaenszel(
        responseMatrix([
          [2, 1],
          [0, 1],
        ]),
        ["reference", "focal"],
      ),
    ).toThrow(DataError);
    expect(() =>
      mantelHaenszel(
        responseMatrix([
          [1, 1],
          [null, null],
        ]),
        ["reference", "focal"],
      ),
    ).toThrow("Mantel-Haenszel DIF needs persons with responses in both groups.");
  });
});

describe("mantelTest", () => {
  for (const dataset of polytomous.cases) {
    it(`matches base R's Mantel and generalized Mantel-Haenszel statistics on ${dataset.name} data`, () => {
      const result = mantelTest(responseMatrix(dataset.responses), groups(dataset.focal));
      expect(result.purification).toBeNull();
      expectMantelTest(result.items, dataset.complete.mantel);
    });

    it(`matches base R's purified statistics on ${dataset.name} data`, () => {
      const expected = dataset.complete.purifiedMantel;
      const result = mantelTest(responseMatrix(dataset.responses), groups(dataset.focal), { purify: true });
      expectMantelTest(result.items, expected);
      expectPurification(result.purification, expected);
    });

    it(`matches base R's rule for sparse data, purified and not, on ${dataset.name} data`, () => {
      const { sparse } = dataset;
      const matrix = responseMatrix(sparse.responses);
      expectMantelTest(mantelTest(matrix, groups(dataset.focal)).items, sparse.mantel);
      const purified = mantelTest(matrix, groups(dataset.focal), { purify: true });
      expectMantelTest(purified.items, sparse.purifiedMantel);
      expectPurification(purified.purification, sparse.purifiedMantel);
    });
  }

  it("gives the Mantel-Haenszel chi-square without the continuity correction for dichotomous items", () => {
    const { responses, focal } = fixture.cases[0] ?? { responses: [], focal: [] };
    const matrix = responseMatrix(responses);
    const mantel = mantelTest(matrix, groups(focal)).items;
    const reference = mantelHaenszel(matrix, groups(focal), { continuityCorrection: false }).items;
    reference.forEach((item, i) => {
      expectClose(mantel[i]?.chiSquare ?? null, item.chiSquare, MANTEL_TEST_TOLERANCE, `chi-square ${i + 1}`);
      expectClose(mantel[i]?.generalized.chiSquare ?? null, item.chiSquare, MANTEL_TEST_TOLERANCE, `general ${i + 1}`);
      expect(mantel[i]?.generalized.degreesOfFreedom).toBe(1);
    });
  });

  it("derives the statistics of two score groups by hand", () => {
    const { items } = mantelTest(responseMatrix(DISJOINT_ROWS), ["reference", "focal", "reference", "focal"]);
    // Score group 1 holds [0, 1] and [1, 0], group 5 holds [2, 3] and [3, 2]; in each, F = E(F) + 1/2 and Var(F) = 1/4,
    // so the chi-square is 1 / (1/2) = 2. The standardized mean difference is 1: the focal mean is 1 higher in both.
    // The responses 0, 1, 2 and 3 have a sample variance of 5/3.
    expect(items[0]?.chiSquare).toBeCloseTo(2, 14);
    expect(items[0]?.significance).toBeCloseTo(0.15729920705028516, 14);
    expect(items[0]?.standardizedMeanDifference).toBeCloseTo(1, 14);
    expect(items[0]?.effectSize).toBeCloseTo(Math.sqrt(3 / 5), 14);
    expect(items[0]?.etsCategory).toBe("AA");
    expect(items[1]?.standardizedMeanDifference).toBeCloseTo(-1, 14);
  });

  it("gives a null generalized chi-square where the score groups use disjoint categories", () => {
    const { items } = mantelTest(responseMatrix(DISJOINT_ROWS), ["reference", "focal", "reference", "focal"]);
    expect(items[0]?.generalized).toEqual({ chiSquare: null, degreesOfFreedom: 3, significance: null });
  });

  it("counts only the categories used in score groups of more than one person", () => {
    const rows = [...DISJOINT_ROWS, [4, 3]];
    const { items } = mantelTest(responseMatrix(rows), ["reference", "focal", "reference", "focal", "focal"]);
    expect(items[0]?.generalized.degreesOfFreedom).toBe(3);
    expect(items[0]?.chiSquare).toBeCloseTo(2, 14);
    expect(items[0]?.standardizedMeanDifference).toBeCloseTo(1, 14);
  });

  it("gives null statistics for an item every person answered alike", () => {
    const result = mantelTest(responseMatrix(CONSTANT_ROWS), CONSTANT_GROUPS).items[0];
    expect(result).toEqual({
      id: "item1",
      chiSquare: null,
      significance: null,
      generalized: { chiSquare: null, degreesOfFreedom: 0, significance: null },
      standardizedMeanDifference: 0,
      effectSize: null,
      etsCategory: null,
    });
  });

  it("scales the matching score of a person who skipped an item to the highest possible score of its items", () => {
    const answered = [...SCALING_ROWS, [2, 1, 1]];
    const skipped = [...SCALING_ROWS, [2, 1, null]];
    const scalingGroups = [...SCALING_GROUPS, "focal" as const];
    // [2, 1, null] answered 1 of 2 on item 2, which scales to 3 / 2 of the other items' 3 points, 2 with halves up: the
    // score of item 1 is 2 + 2, as that of [2, 1, 1].
    const full = mantelTest(responseMatrix(answered), scalingGroups).items[0];
    expect(full?.chiSquare).not.toBeNull();
    expect(mantelTest(responseMatrix(skipped), scalingGroups).items[0]).toEqual(full);
  });

  it("rejects groups without responses and invalid purification", () => {
    const missing = responseMatrix([
      [2, 1],
      [null, null],
    ]);
    expect(() => mantelTest(missing, ["reference", "focal"])).toThrow(
      "Mantel DIF needs persons with responses in both groups.",
    );
  });
});

describe("ordinalLogisticRegressionDif", () => {
  for (const dataset of polytomous.cases) {
    it(`matches MASS::polr and lordif's rundif on ${dataset.name} data`, () => {
      const result = ordinalLogisticRegressionDif(responseMatrix(dataset.responses), groups(dataset.focal));
      expect(result.purification).toBeNull();
      expectOrdinal(result.items, dataset.complete.ordinal, dataset.complete.lordif);
    });

    it(`matches polr and rundif on ${dataset.name} data matched on theta`, () => {
      const result = ordinalLogisticRegressionDif(responseMatrix(dataset.responses), groups(dataset.focal), {
        matching: dataset.theta,
      });
      expectOrdinal(result.items, dataset.complete.ordinalTheta, dataset.complete.lordifTheta);
    });

    it(`matches polr's purified statistics on ${dataset.name} data`, () => {
      const expected = dataset.complete.purifiedOrdinal;
      const result = ordinalLogisticRegressionDif(responseMatrix(dataset.responses), groups(dataset.focal), {
        purify: true,
      });
      expectOrdinal(result.items, expected);
      expectPurification(result.purification, expected);
    });

    it(`matches polr's rule for sparse data, purified, not and matched on theta, on ${dataset.name} data`, () => {
      const { sparse } = dataset;
      const matrix = responseMatrix(sparse.responses);
      expectOrdinal(ordinalLogisticRegressionDif(matrix, groups(dataset.focal)).items, sparse.ordinal);
      const options = { matching: sparse.theta };
      expectOrdinal(ordinalLogisticRegressionDif(matrix, groups(dataset.focal), options).items, sparse.ordinalTheta);
      const purified = ordinalLogisticRegressionDif(matrix, groups(dataset.focal), { purify: true });
      expectOrdinal(purified.items, sparse.purifiedOrdinal);
      expectPurification(purified.purification, sparse.purifiedOrdinal);
    });
  }

  it("equals logistic regression for dichotomous items", () => {
    for (const dataset of fixture.cases) {
      const matrix = responseMatrix(dataset.responses);
      const ordinal = ordinalLogisticRegressionDif(matrix, groups(dataset.focal)).items;
      const logistic = logisticRegressionDif(matrix, groups(dataset.focal)).items;
      logistic.forEach((item, i) => {
        for (const type of ["uniform", "nonuniform", "both"] as const) {
          for (const name of ["chiSquare", "significance", "rSquaredChange"] as const) {
            const actual = ordinal[i]?.[type][name] ?? null;
            expectClose(actual, item[type][name], ORDINAL_TOLERANCE, `${dataset.name} ${type} ${name} ${i + 1}`);
          }
        }
      });
    }
  });

  it("uses the categories the persons used, whichever integers the responses are", () => {
    const dataset = polytomous.cases[0];
    const matrix = responseMatrix(dataset?.responses ?? []);
    const options = { matching: dataset?.theta ?? [] };
    const spread = [0, 3, 4, 9, 20, 21];
    const relabeled = responseMatrix(matrix.responses.map((row) => row.map((response) => spread[response ?? 0] ?? 0)));
    const expected = ordinalLogisticRegressionDif(matrix, groups(dataset?.focal ?? []), options);
    expect(ordinalLogisticRegressionDif(relabeled, groups(dataset?.focal ?? []), options)).toEqual(expected);
  });

  it("scales the matching score of a person who skipped an item as the Mantel test does", () => {
    const answered = [...SCALING_ROWS, [2, 1, 1]];
    const skipped = [...SCALING_ROWS, [2, 1, null]];
    const scalingGroups = [...SCALING_GROUPS, "focal" as const];
    const full = ordinalLogisticRegressionDif(responseMatrix(answered), scalingGroups).items[0];
    expect(full?.both.chiSquare).not.toBeNull();
    expect(ordinalLogisticRegressionDif(responseMatrix(skipped), scalingGroups).items[0]).toEqual(full);
  });

  it("leaves out the persons without a matching score and does not depend on its scale", () => {
    const dataset = polytomous.cases[0];
    const focal = dataset?.focal.slice(0, 400) ?? [];
    const rows = dataset?.responses.slice(0, 400) ?? [];
    const theta = dataset?.theta.slice(0, 400) ?? [];
    const dropped = theta.map((score, person) => (person % 5 === 0 ? null : score));
    const kept = rows.map((_, person) => person % 5 !== 0);
    const withNulls = ordinalLogisticRegressionDif(responseMatrix(rows), groups(focal), { matching: dropped });
    const without = ordinalLogisticRegressionDif(
      responseMatrix(rows.filter((_, person) => kept[person])),
      groups(focal.filter((_, person) => kept[person])),
      { matching: theta.filter((_, person) => kept[person]) },
    );
    expect(withNulls.items).toEqual(without.items);
    const rescaled = ordinalLogisticRegressionDif(responseMatrix(rows), groups(focal), {
      matching: dropped.map((score) => (score === null ? null : 3 * score + 7)),
    });
    rescaled.items.forEach((item, i) => {
      const reference = without.items[i];
      expectClose(item.both.chiSquare, reference?.both.chiSquare ?? null, 1e-7, `both ${i + 1}`);
      expectClose(item.slopeChange, reference?.slopeChange ?? null, 1e-7, `slope change ${i + 1}`);
    });
  });

  it("gives null tests for an item every person answered alike, persons of one group or an equal matching score", () => {
    const alike = ordinalLogisticRegressionDif(responseMatrix(CONSTANT_ROWS), CONSTANT_GROUPS).items;
    expect(alike[0]?.both).toEqual({ chiSquare: null, degreesOfFreedom: 2, significance: null, rSquaredChange: null });
    expect(alike[0]?.slopeChange).toBeNull();
    expect(alike[1]?.uniform.chiSquare).not.toBeNull();
    const oneGroup = CONSTANT_ROWS.map((row, person) => [person < 3 ? (row[0] ?? null) : null, ...row.slice(1)]);
    const options = { matching: [1, 2, 3, 4, 5, 6] };
    expect(ordinalLogisticRegressionDif(responseMatrix(oneGroup), CONSTANT_GROUPS, options).items[0]?.uniform).toEqual({
      chiSquare: null,
      degreesOfFreedom: 1,
      significance: null,
      rSquaredChange: null,
    });
    const equal = ordinalLogisticRegressionDif(responseMatrix(CONSTANT_ROWS), CONSTANT_GROUPS, {
      matching: [4, 4, 4, 4, 4, 4],
    });
    expect(equal.items[1]?.both.chiSquare).toBeNull();
  });

  it("computes the tests where a group term diverges", () => {
    // Group 1 answers 0 or 1, group 2 only 2: the group term has no finite estimate.
    const rows = [[0], [1], [0], [1], [0], [1], [2], [2], [2], [2], [2], [2]];
    const separated = rows.map((_, person) => (person < 6 ? "reference" : "focal")) as DifGroup[];
    const matching = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6];
    const { uniform } = ordinalLogisticRegressionDif(responseMatrix(rows), separated, { matching }).items[0] ?? {};
    expect(uniform?.chiSquare).toBeGreaterThan(10);
    expect(uniform?.significance).toBeLessThan(0.01);
  });

  it("rejects a matching score that purification would have to change, and an invalid one", () => {
    const matrix = responseMatrix(CONSTANT_ROWS);
    expect(() =>
      ordinalLogisticRegressionDif(matrix, CONSTANT_GROUPS, { matching: [1, 2, 3, 4, 5, 6], purify: true }),
    ).toThrow("The option purify purifies the total score, so it cannot be used with the option matching.");
    expect(() => ordinalLogisticRegressionDif(matrix, CONSTANT_GROUPS, { matching: [1, 2, 3] })).toThrow(
      "Found 3 matching scores for 6 persons.",
    );
    const infinite: (number | null)[] = JSON.parse('[1, 2, 3, 4, 5, "high"]');
    expect(() => ordinalLogisticRegressionDif(matrix, CONSTANT_GROUPS, { matching: infinite })).toThrow(
      'The matching score of person "person6" must be a finite number or null, found "high".',
    );
    expect(() => ordinalLogisticRegressionDif(matrix, CONSTANT_GROUPS, { matching: JSON.parse("null") })).toThrow(
      "The option matching must be an array, found null.",
    );
    expect(() =>
      ordinalLogisticRegressionDif(matrix, CONSTANT_GROUPS, { matching: [1, 2, Number.NaN, 4, 5, 6] }),
    ).toThrow('The matching score of person "person3" must be a finite number or null, found NaN.');
  });

  it("rejects groups without responses", () => {
    const missing = responseMatrix([
      [2, 1],
      [null, null],
    ]);
    expect(() => ordinalLogisticRegressionDif(missing, ["reference", "focal"])).toThrow(
      "Ordinal logistic regression DIF needs persons with responses in both groups.",
    );
  });
});

describe("options and groups", () => {
  const matrix = responseMatrix(CONSTANT_ROWS);

  it("reject a purify or continuity correction option that is not a boolean", () => {
    const purify: boolean = JSON.parse('"yes"');
    expect(() => mantelHaenszel(matrix, CONSTANT_GROUPS, { purify })).toThrow(
      'The option purify must be a boolean, found "yes".',
    );
    expect(() => logisticRegressionDif(matrix, CONSTANT_GROUPS, { purify })).toThrow(
      'The option purify must be a boolean, found "yes".',
    );
    expect(() => mantelHaenszel(matrix, CONSTANT_GROUPS, { continuityCorrection: JSON.parse("0") })).toThrow(
      "The option continuityCorrection must be a boolean, found 0.",
    );
  });

  it("reject groups and options of other types in the polytomous statistics", () => {
    const purify: boolean = JSON.parse('"yes"');
    const analyzes: ((options: PurificationOptions) => unknown)[] = [
      (options) => mantelTest(matrix, CONSTANT_GROUPS, options),
      (options) => ordinalLogisticRegressionDif(matrix, CONSTANT_GROUPS, options),
    ];
    for (const analyze of analyzes) {
      expect(() => analyze({ purify })).toThrow('The option purify must be a boolean, found "yes".');
      expect(() => analyze(JSON.parse("null"))).toThrow("The options must be an object, found null.");
    }
    expect(() => mantelTest(matrix, JSON.parse("null"))).toThrow("The groups must be an array, found null.");
    expect(() => ordinalLogisticRegressionDif(matrix, JSON.parse('"focal"'))).toThrow(
      'The groups must be an array, found "focal".',
    );
    expect(() => mantelTest(matrix, CONSTANT_GROUPS.slice(1))).toThrow(DataError);
  });

  it("reject a group other than reference or focal, and groups, options and models of other types", () => {
    const groups: DifGroup[] = JSON.parse(JSON.stringify(CONSTANT_GROUPS.map((_, i) => (i === 0 ? "both" : "focal"))));
    expect(() => mantelHaenszel(matrix, groups)).toThrow(
      'The group of person "person1" must be "reference" or "focal", found "both".',
    );
    expect(() => lordChiSquare(matrix, CONSTANT_GROUPS, { model: JSON.parse("null") })).toThrow(
      "The option model must be the 2PL, 3PL or 4PL model, found null.",
    );
    expect(() => mantelHaenszel(matrix, JSON.parse("null"))).toThrow("The groups must be an array, found null.");
    expect(() => lordChiSquare(matrix, CONSTANT_GROUPS, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => logisticRegressionDif(matrix, CONSTANT_GROUPS, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => logisticRegressionDif(matrix, CONSTANT_GROUPS, { purify: JSON.parse("null") })).toThrow(
      "The option purify must be a boolean, found null.",
    );
  });
});

describe("properties", () => {
  /** Response matrices of 0 and 1: 2 to 30 persons, 1 to 6 items. */
  const binaryRows = fc
    .tuple(fc.integer({ min: 2, max: 30 }), fc.integer({ min: 1, max: 6 }))
    .chain(([persons, items]) =>
      fc.array(fc.array(fc.integer({ min: 0, max: 1 }), { minLength: items, maxLength: items }), {
        minLength: persons,
        maxLength: persons,
      }),
    );

  it("finds no DIF between identical groups", () => {
    fc.assert(
      fc.property(binaryRows, (rows) => {
        const matrix = responseMatrix([...rows, ...rows]);
        const groups: DifGroup[] = [...rows.map(() => "reference" as const), ...rows.map(() => "focal" as const)];
        for (const item of mantelHaenszel(matrix, groups).items) {
          if (item.chiSquare === null) continue;
          expect(item.chiSquare).toBe(0);
          expect(item.oddsRatio).toBe(1);
          expect(item.etsCategory).toBe("A");
        }
        for (const item of logisticRegressionDif(matrix, groups).items) {
          if (item.both.chiSquare === null) continue;
          expect(item.both.chiSquare).toBeLessThan(1e-9);
        }
      }),
    );
  });

  it("Mantel-Haenszel statistics do not depend on the order of persons", () => {
    fc.assert(
      fc.property(binaryRows, fc.array(fc.boolean(), { minLength: 30, maxLength: 30 }), (rows, focal) => {
        const groups: DifGroup[] = rows.map((_, i) => (focal[i] === true || i === 0 ? "focal" : "reference"));
        groups[1] = "reference";
        const forward = mantelHaenszel(responseMatrix(rows), groups).items;
        const backward = mantelHaenszel(responseMatrix([...rows].reverse()), [...groups].reverse()).items;
        forward.forEach((item, i) => {
          expectClose(backward[i]?.chiSquare ?? null, item.chiSquare, 1e-12 * Math.max(1, item.chiSquare ?? 0));
          expect(backward[i]?.etsCategory).toBe(item.etsCategory);
        });
      }),
    );
  });

  /** Response matrices of 0 to 3: 2 to 40 persons, 1 to 5 items. */
  const gradedRows = fc
    .tuple(fc.integer({ min: 2, max: 40 }), fc.integer({ min: 1, max: 5 }))
    .chain(([persons, items]) =>
      fc.array(fc.array(fc.integer({ min: 0, max: 3 }), { minLength: items, maxLength: items }), {
        minLength: persons,
        maxLength: persons,
      }),
    );

  it("finds no polytomous DIF between identical groups", () => {
    fc.assert(
      fc.property(gradedRows, (rows) => {
        const matrix = responseMatrix([...rows, ...rows]);
        const groups: DifGroup[] = [...rows.map(() => "reference" as const), ...rows.map(() => "focal" as const)];
        for (const item of mantelTest(matrix, groups).items) {
          if (item.chiSquare !== null) expect(item.chiSquare).toBe(0);
          if (item.generalized.chiSquare !== null) expect(item.generalized.chiSquare).toBeLessThan(1e-9);
          if (item.standardizedMeanDifference !== null) expect(item.standardizedMeanDifference).toBe(0);
          if (item.etsCategory !== null) expect(item.etsCategory).toBe("AA");
        }
        for (const item of ordinalLogisticRegressionDif(matrix, groups).items) {
          if (item.both.chiSquare !== null) expect(item.both.chiSquare).toBeLessThan(1e-6);
        }
      }),
    );
  });

  it("gives chi-squares from 0 and significances from 0 to 1 for any responses", () => {
    fc.assert(
      fc.property(gradedRows, (rows) => {
        const groups: DifGroup[] = rows.map((_, person) => (person % 2 === 0 ? "focal" : "reference"));
        const matrix = responseMatrix(rows);
        const probabilities: (number | null)[] = [];
        for (const item of mantelTest(matrix, groups).items) {
          expect(item.chiSquare ?? 0).toBeGreaterThanOrEqual(0);
          expect(item.generalized.chiSquare ?? 0).toBeGreaterThanOrEqual(0);
          probabilities.push(item.significance, item.generalized.significance);
        }
        for (const item of ordinalLogisticRegressionDif(matrix, groups).items) {
          for (const test of [item.uniform, item.nonuniform, item.both]) {
            expect(test.chiSquare ?? 0).toBeGreaterThanOrEqual(0);
            probabilities.push(test.significance);
          }
        }
        for (const probability of probabilities) {
          expect(probability ?? 0.5).toBeGreaterThanOrEqual(0);
          expect(probability ?? 0.5).toBeLessThanOrEqual(1);
        }
      }),
    );
  });

  it("Mantel statistics do not depend on the order of persons", () => {
    fc.assert(
      fc.property(gradedRows, fc.array(fc.boolean(), { minLength: 40, maxLength: 40 }), (rows, focal) => {
        const groups: DifGroup[] = rows.map((_, i) => (focal[i] === true || i === 0 ? "focal" : "reference"));
        groups[1] = "reference";
        const forward = mantelTest(responseMatrix(rows), groups).items;
        const backward = mantelTest(responseMatrix([...rows].reverse()), [...groups].reverse()).items;
        forward.forEach((item, i) => {
          const other = backward[i];
          expectClose(other?.chiSquare ?? null, item.chiSquare, 1e-12 * Math.max(1, item.chiSquare ?? 0));
          expectClose(
            other?.generalized.chiSquare ?? null,
            item.generalized.chiSquare,
            1e-9 * Math.max(1, item.generalized.chiSquare ?? 0),
          );
          expectClose(other?.standardizedMeanDifference ?? null, item.standardizedMeanDifference, 1e-12);
          expect(other?.etsCategory).toBe(item.etsCategory);
        });
      }),
    );
  });
});

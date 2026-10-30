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
  mantelHaenszel,
  oneParameterLogistic,
  responseMatrix,
  twoParameterLogistic,
} from "../src/index.js";
import fixture from "./fixtures/dif.json" with { type: "json" };
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

function expectLogistic(result: readonly LogisticRegressionDif[], expected: LogisticFixture): void {
  for (const type of ["uniform", "nonuniform", "both"] as const) {
    const tests = expected[type];
    result.forEach((item, i) => {
      const test = item[type];
      expectClose(test.chiSquare, tests.chiSquare[i] ?? null, LOGISTIC_TOLERANCE, `${type} ${i + 1}`);
      expectClose(test.significance, tests.significance[i] ?? null, LOGISTIC_TOLERANCE, `${type} p ${i + 1}`);
      expectClose(test.rSquaredChange, tests.rSquaredChange[i] ?? null, LOGISTIC_TOLERANCE, `${type} R²`);
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
});

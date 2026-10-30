import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  classifyDifficulty,
  classifyDiscrimination,
  cronbachAlpha,
  DataError,
  DIFFICULTY_THRESHOLDS,
  distractorAnalysis,
  itemStatistics,
  kr20,
  kr21,
  mcdonaldOmega,
  type ResponseMatrix,
  responseMatrix,
  type ScoredResponses,
  scoreResponses,
  splitHalf,
  standardErrorOfMeasurement,
} from "../src/index.js";
import bfi from "./fixtures/ctt-bfi-neuroticism.json" with { type: "json" };
import lsat7 from "./fixtures/ctt-lsat7.json" with { type: "json" };
import sat12 from "./fixtures/ctt-sat12.json" with { type: "json" };
import simulated from "./fixtures/ctt-simulated-2pl.json" with { type: "json" };
import { defined, expectClose } from "./helpers.js";

/** Absolute tolerance of CTT statistics against R. */
const TOLERANCE = 1e-10;
/** Minres loadings against base R's minimization, both to their rounding. */
const OMEGA_LOADING_TOLERANCE = 1e-10;

const ITEM_FIELDS = [
  "mean",
  "sd",
  "itemTotal",
  "itemRest",
  "biserial",
  "discriminationIndex",
  "alphaIfDeleted",
] as const;

const OPTION_FIELDS = ["proportion", "upper", "lower", "pointBiserial"] as const;

describe("classification", () => {
  it("classifies difficulty with bounds in the moderate class", () => {
    expect([0.29, 0.3, 0.7, 0.71].map((p) => classifyDifficulty(p))).toEqual([
      "difficult",
      "moderate",
      "moderate",
      "easy",
    ]);
    expect(classifyDifficulty(0.75, { ...DIFFICULTY_THRESHOLDS, easy: 0.8 })).toBe("moderate");
  });

  it("classifies discrimination by Ebel's bounds", () => {
    expect([0.4, 0.39, 0.3, 0.2, 0.19, -0.1].map((value) => classifyDiscrimination(value))).toEqual([
      "very-good",
      "good",
      "good",
      "marginal",
      "poor",
      "poor",
    ]);
    expect(classifyDiscrimination(0.35, { veryGood: 0.35, good: 0.25, marginal: 0.15 })).toBe("very-good");
  });

  it("rejects values and thresholds out of range", () => {
    expect(() => classifyDifficulty(1.2)).toThrow("A p-value must be a number from 0 to 1, found 1.2.");
    expect(() => classifyDifficulty(0.5, { difficult: 0.8, easy: 0.2 })).toThrow(DataError);
    expect(() => classifyDiscrimination(Number.NaN)).toThrow(DataError);
    expect(() => classifyDiscrimination(0.5, { veryGood: 0.2, good: 0.3, marginal: 0.1 })).toThrow(DataError);
    expect(() => classifyDiscrimination(Number.POSITIVE_INFINITY)).toThrow(
      "A discrimination value must be a finite number, found Infinity.",
    );
  });

  it("rejects values and thresholds of other types from JavaScript", () => {
    expect(() => classifyDifficulty(JSON.parse('"0.5"'))).toThrow(
      'A p-value must be a number from 0 to 1, found "0.5".',
    );
    expect(() => classifyDifficulty(0.5, JSON.parse("null"))).toThrow(
      "Difficulty thresholds must be an object { difficult, easy }, found null.",
    );
    expect(() =>
      classifyDiscrimination(0.5, JSON.parse('{ "veryGood": "0.4", "good": 0.3, "marginal": 0.2 }')),
    ).toThrow('The very good threshold must be a finite number, found "0.4".');
  });
});

describe("distractorAnalysis", () => {
  const scored = scoreResponses(sat12.choices, sat12.key, {
    itemIds: sat12.itemIds,
    optionLabels: sat12.optionLabels,
  });

  it("scores the SAT12 choices as mirt's key2binary does", () => {
    expect(scored.responses).toEqual(sat12.responses);
  });

  it("matches R's option statistics", () => {
    const analysis = distractorAnalysis(scored, { groupFraction: sat12.groupFraction });
    sat12.options.forEach((expected, i) => {
      const actual = defined(analysis[i]);
      expect(actual).toMatchObject({ id: expected.id, key: expected.key });
      expected.options.forEach((option, j) => {
        const statistics = defined(actual.options[j]);
        expect(statistics).toMatchObject({ label: option.label, count: option.count });
        for (const field of OPTION_FIELDS) {
          expectClose(statistics[field], option[field], TOLERANCE, `${expected.id} ${option.label} ${field}`);
        }
      });
    });
  });

  it("lists options nobody chose", () => {
    const [item] = distractorAnalysis(scoreResponses([["A"], ["B"]], ["A"], { optionLabels: ["A", "B", "C"] }));
    expect(defined(item).options[2]).toEqual({
      label: "C",
      count: 0,
      proportion: 0,
      upper: 0,
      lower: 0,
      pointBiserial: null,
    });
  });

  it("rejects scored responses from JSON that lack an array", () => {
    const scored = scoreResponses([["A"], ["B"]], ["A"]);
    for (const field of ["choices", "key", "optionLabels"]) {
      const parsed: ScoredResponses = JSON.parse(JSON.stringify({ ...scored, [field]: undefined }));
      expect(() => distractorAnalysis(parsed)).toThrow(
        "Scored responses need the arrays choices, key and optionLabels.",
      );
    }
    expect(() => distractorAnalysis(JSON.parse("null"))).toThrow(
      "Scored responses must be an object from scoreResponses, found null.",
    );
    expect(() => distractorAnalysis(scored, JSON.parse("null"))).toThrow("The options must be an object, found null.");
  });
});

describe("upper and lower groups", () => {
  // Totals 3, 2, 2, 0: with groups of two, the persons tied at 2 count half in each group.
  const matrix = responseMatrix([
    [1, 1, 1],
    [1, 1, 0],
    [0, 1, 1],
    [0, 0, 0],
  ]);

  it("share the places at a tied boundary equally", () => {
    const [first] = itemStatistics(matrix, { groupFraction: 0.5 });
    expect(defined(first).discriminationIndex).toBe(0.5);
  });

  it.each([0, 0.6, Number.NaN, "0.3"])("reject a group fraction of %s", (groupFraction) => {
    const options = JSON.parse(JSON.stringify({ groupFraction }));
    expect(() => itemStatistics(matrix, options)).toThrow(DataError);
    expect(() => distractorAnalysis(scoreResponses([["A"]], ["A"]), options)).toThrow(DataError);
  });
});

const FIXTURES = [lsat7, bfi, simulated, sat12].map((fixture) => ({
  fixture,
  matrix: responseMatrix(fixture.responses, { itemIds: fixture.itemIds }),
}));

describe("itemStatistics", () => {
  for (const { fixture, matrix } of FIXTURES) {
    it(`matches R's item statistics of ${fixture.source}`, () => {
      const items = itemStatistics(matrix, { groupFraction: fixture.groupFraction });
      fixture.items.forEach((expected, i) => {
        const actual = defined(items[i]);
        expect(actual.id).toBe(expected.id);
        expect(actual.count).toBe(expected.count);
        for (const field of ITEM_FIELDS)
          expectClose(actual[field], expected[field], TOLERANCE, `${expected.id} ${field}`);
      });
    });
  }
});

describe("reliability", () => {
  for (const { fixture, matrix } of FIXTURES) {
    it(`matches R's reliability coefficients and standard error of measurement of ${fixture.source}`, () => {
      expectClose(cronbachAlpha(matrix), fixture.alpha, TOLERANCE, "alpha");
      for (const [statistic, expected] of [
        [kr20, fixture.kr20],
        [kr21, fixture.kr21],
      ] as const) {
        if (expected === null) expect(() => statistic(matrix)).toThrow("needs dichotomous items");
        else expectClose(statistic(matrix), expected, TOLERANCE, statistic.name);
      }
      expectClose(standardErrorOfMeasurement(matrix), fixture.sem, TOLERANCE, "SEM");
    });

    it(`matches R's split-half coefficients of ${fixture.source}`, () => {
      for (const [split, expected] of [
        ["odd-even", fixture.splitHalf.oddEven],
        ["first-second", fixture.splitHalf.firstSecond],
      ] as const) {
        const actual = splitHalf(matrix, { split });
        expectClose(actual.correlation, expected.correlation, TOLERANCE, `${split} correlation`);
        expectClose(actual.spearmanBrown, expected.spearmanBrown, TOLERANCE, `${split} Spearman-Brown`);
        expectClose(actual.guttman, expected.guttman, TOLERANCE, `${split} Guttman`);
      }
    });
  }

  for (const { fixture, matrix } of FIXTURES) {
    it(`matches psych's omega, by base R's minres loadings, of ${fixture.source}`, () => {
      const result = mcdonaldOmega(matrix);
      expect(result.converged).toBe(true);
      expectClose(result.omega, fixture.omega.omega, TOLERANCE, "omega");
      result.loadings.forEach((loading, i) => {
        expectClose(loading, defined(fixture.omega.loadings[i]), OMEGA_LOADING_TOLERANCE, `loading ${i + 1}`);
      });
    });
  }

  it("gives omega 1 - Σ (1 - λ²) / Σ r for three items with equal correlations, whose loadings are √r", () => {
    // Every item has variance 0.24 and every pair covariance 0.04, so r = 1/6 and λ = √(1/6): Σ (1 - λ²) = 2.5 and
    // Σ r = 3 + 6 / 6 = 4, so omega = 1 - 2.5 / 4.
    const matrix = responseMatrix([
      [1, 1, 1],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [0, 0, 0],
      [1, 1, 1],
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
      [0, 0, 0],
    ]);
    const { omega, loadings } = mcdonaldOmega(matrix);
    for (const loading of loadings) expectClose(loading, Math.sqrt(1 / 6), 1e-12);
    expectClose(omega, 0.375, 1e-12);
  });

  it("rejects omega for fewer than three items or an item without variance", () => {
    expect(() =>
      mcdonaldOmega(
        responseMatrix([
          [1, 0],
          [0, 1],
          [1, 1],
        ]),
      ),
    ).toThrow("Omega needs at least three items, as one factor is not identified by two.");
    expect(() =>
      mcdonaldOmega(
        responseMatrix([
          [1, 0, 1],
          [1, 1, 0],
          [1, 1, 1],
        ]),
      ),
    ).toThrow('Omega needs items with variance; item "item1" has none.');
    // Two persons with opposite responses: every person's standardized scores sum to 0.
    expect(() =>
      mcdonaldOmega(
        responseMatrix([
          [0, 1, 0, 1],
          [1, 0, 1, 0],
        ]),
      ),
    ).toThrow("Omega is undefined when the item correlations sum to 0: the standardized total does not vary.");
  });

  const polytomous = responseMatrix([
    [2, 1],
    [0, 1],
    [1, 0],
  ]);

  it("rejects KR-20 and KR-21 for polytomous items", () => {
    expect(() => kr20(polytomous)).toThrow(
      'KR-20 needs dichotomous items, but person "person1" scored 2 on item "item1".',
    );
    expect(() => kr21(polytomous)).toThrow("KR-21 needs dichotomous items");
  });

  it("rejects matrices and options of other types from JavaScript", () => {
    const message = "A response matrix needs the arrays personIds, itemIds and responses.";
    for (const statistic of [cronbachAlpha, kr20, kr21, mcdonaldOmega, splitHalf, standardErrorOfMeasurement]) {
      expect(() => statistic(JSON.parse("null"))).toThrow(message);
    }
    expect(() => splitHalf(polytomous, JSON.parse("null"))).toThrow("The options must be an object, found null.");
  });

  it("uses a given reliability for the standard error of measurement and checks its range", () => {
    expect(standardErrorOfMeasurement(polytomous, 0.75)).toBeCloseTo(Math.sqrt(4 / 3) / 2, 15);
    expect(() => standardErrorOfMeasurement(polytomous, 1.2)).toThrow(
      "A reliability must be a number from 0 to 1, found 1.2.",
    );
    expect(() => standardErrorOfMeasurement(polytomous, JSON.parse("null"))).toThrow(
      "A reliability must be a number from 0 to 1, found null.",
    );
  });
});

describe("edge cases", () => {
  it("gives a constant item null correlations, D of 0, and counts it in alpha", () => {
    const rows = [
      [1, 0, 1, 1],
      [1, 1, 1, 0],
      [0, 0, 1, 0],
      [1, 1, 1, 1],
      [0, 1, 1, 0],
    ];
    const items = itemStatistics(responseMatrix(rows));
    expect(items[2]).toMatchObject({ mean: 1, sd: 0, itemTotal: null, itemRest: null, biserial: null });
    expect(defined(items[2]).discriminationIndex).toBe(0);
    const without = cronbachAlpha(responseMatrix(rows.map((row) => row.filter((_, item) => item !== 2))));
    // Alpha with k items, one of them constant, equals alpha without it times k (k - 2) / (k - 1)².
    expect(cronbachAlpha(responseMatrix(rows))).toBeCloseTo((without * 4 * 2) / 9, 14);
    expect(defined(items[0]).alphaIfDeleted).not.toBeNull();
  });

  it("rejects reliability when every item is constant, but still lists item statistics", () => {
    const matrix = responseMatrix([
      [1, 0],
      [1, 0],
    ]);
    expect(() => cronbachAlpha(matrix)).toThrow("Alpha needs total scores with variance.");
    expect(itemStatistics(matrix).map((item) => item.itemTotal)).toEqual([null, null]);
  });

  it("skips persons without responses", () => {
    const rows = [
      [1, 0, 1],
      [1, 1, 1],
      [0, 0, 1],
      [0, 1, 0],
    ];
    const withEmptyRow = responseMatrix([...rows, [null, null, null]]);
    expect(itemStatistics(withEmptyRow)).toEqual(itemStatistics(responseMatrix(rows)));
    expect(cronbachAlpha(withEmptyRow)).toBe(cronbachAlpha(responseMatrix(rows)));
  });

  it("gives an item without responses null statistics and rejects reliability", () => {
    const matrix = responseMatrix([
      [1, 0, null],
      [0, 1, null],
      [1, 1, null],
    ]);
    const items = itemStatistics(matrix);
    expect(items[2]).toMatchObject({ count: 0, mean: null, sd: null });
    // Nobody answered every item, so no statistic that uses the total score is defined.
    expect(items.flatMap((item) => [item.itemTotal, item.discriminationIndex, item.alphaIfDeleted])).toEqual(
      Array(9).fill(null),
    );
    for (const statistic of [cronbachAlpha, kr20, kr21, splitHalf, standardErrorOfMeasurement]) {
      expect(() => statistic(matrix)).toThrow("needs at least two persons who answered every item.");
    }
    const options = distractorAnalysis(
      scoreResponses(
        [
          ["A", null],
          ["B", null],
        ],
        ["A", "B"],
      ),
    )[1]?.options;
    expect(options?.map((option) => option.proportion)).toEqual([null, null]);
  });

  it("handles a single item", () => {
    const matrix = responseMatrix([[1], [0], [1]]);
    expect(itemStatistics(matrix)[0]).toMatchObject({ itemTotal: 1, itemRest: null, alphaIfDeleted: null });
    for (const statistic of [cronbachAlpha, kr20, kr21, splitHalf, standardErrorOfMeasurement]) {
      expect(() => statistic(matrix)).toThrow("needs at least two items.");
    }
  });

  it("handles two persons", () => {
    const matrix = responseMatrix([
      [1, 1, 1],
      [0, 1, 0],
    ]);
    expect(itemStatistics(matrix).map((item) => [item.itemTotal, item.discriminationIndex])).toEqual([
      [1, 1],
      [null, 0],
      [1, 1],
    ]);
    expect(cronbachAlpha(matrix)).toBe(0.75);
    expect(kr21(matrix)).toBeCloseTo(0.5, 15);
    expect(splitHalf(matrix, { split: "first-second" })).toEqual({ correlation: 1, spearmanBrown: 1, guttman: 1 });
    expect(() => splitHalf(matrix)).toThrow("Split-half reliability needs half scores with variance.");
    expect(() => splitHalf(matrix, { split: JSON.parse('"halves"') })).toThrow(
      'The option split must be "odd-even" or "first-second", found "halves".',
    );
    expect(standardErrorOfMeasurement(matrix)).toBeCloseTo(Math.SQRT1_2, 15);
    expect(() =>
      splitHalf(
        responseMatrix([
          [1, 0],
          [0, 1],
        ]),
      ),
    ).toThrow("correlate at -1");
  });
});

/** Response matrices of 2 to 25 persons and 1 to 8 items with up to 5 categories and about 10% missing responses. */
const arbitraryRows = fc
  .record({
    persons: fc.integer({ min: 2, max: 25 }),
    items: fc.integer({ min: 1, max: 8 }),
    maxScore: fc.integer({ min: 1, max: 4 }),
  })
  .chain(({ persons, items, maxScore }) =>
    fc.array(
      fc.array(fc.option(fc.integer({ min: 0, max: maxScore }), { freq: 9, nil: null }), {
        minLength: items,
        maxLength: items,
      }),
      { minLength: persons, maxLength: persons },
    ),
  );

/** Rows with the same rows in another order. */
const permutedRows = <T>(rows: fc.Arbitrary<T[][]>): fc.Arbitrary<[T[][], T[][]]> =>
  rows.chain((original) =>
    fc
      .shuffledSubarray(original, { minLength: original.length, maxLength: original.length })
      .map((shuffled): [T[][], T[][]] => [original, shuffled]),
  );

/** Returns the result of `compute`, or "DataError" when it throws one. */
function outcome(compute: () => unknown): unknown {
  try {
    return compute();
  } catch (error) {
    if (error instanceof DataError) return "DataError";
    throw error;
  }
}

/** Expects the same structure, with numbers equal up to rounding. */
function expectNear(actual: unknown, expected: unknown, path = "result"): void {
  if (typeof expected === "number" && typeof actual === "number") {
    expect(Math.abs(actual - expected), path).toBeLessThanOrEqual(1e-9 * Math.max(1, Math.abs(expected)));
  } else if (typeof expected === "object" && expected !== null && typeof actual === "object" && actual !== null) {
    expect(Object.keys(actual), path).toEqual(Object.keys(expected));
    for (const [key, value] of Object.entries(expected)) {
      expectNear(Reflect.get(actual, key), value, `${path}.${key}`);
    }
  } else {
    expect(actual, path).toEqual(expected);
  }
}

const statistics = (matrix: ResponseMatrix): unknown[] =>
  [
    () => itemStatistics(matrix),
    () => cronbachAlpha(matrix),
    () => kr20(matrix),
    () => kr21(matrix),
    () => splitHalf(matrix),
    () => splitHalf(matrix, { split: "first-second" }),
    () => standardErrorOfMeasurement(matrix),
  ].map(outcome);

describe("properties", () => {
  it("alpha is at most 1", () => {
    fc.assert(
      fc.property(arbitraryRows, (rows) => {
        const alpha = outcome(() => cronbachAlpha(responseMatrix(rows)));
        if (alpha !== "DataError") expect(alpha).toBeLessThanOrEqual(1 + 1e-12);
      }),
    );
  });

  it("results never contain NaN or infinities", () => {
    fc.assert(
      fc.property(arbitraryRows, (rows) => {
        const numbers = JSON.stringify(statistics(responseMatrix(rows)), (_, value) =>
          typeof value === "number" && !Number.isFinite(value) ? "not finite" : value,
        );
        expect(numbers).not.toContain("not finite");
      }),
    );
  });

  it("statistics do not depend on the order of persons", () => {
    fc.assert(
      fc.property(permutedRows(arbitraryRows), ([original, shuffled]) => {
        expectNear(statistics(responseMatrix(shuffled)), statistics(responseMatrix(original)));
      }),
    );
  });

  it("distractor statistics do not depend on the order of persons", () => {
    const choices = fc
      .record({ persons: fc.integer({ min: 2, max: 25 }), items: fc.integer({ min: 1, max: 6 }) })
      .chain(({ persons, items }) =>
        fc.array(
          fc.array(fc.option(fc.constantFrom("A", "B", "C", "D"), { freq: 9, nil: null }), {
            minLength: items,
            maxLength: items,
          }),
          { minLength: persons, maxLength: persons },
        ),
      );
    fc.assert(
      fc.property(permutedRows(choices), ([original, shuffled]) => {
        const key = (original[0] ?? []).map(() => "A");
        const analyze = (rows: (string | null)[][]): unknown =>
          outcome(() => distractorAnalysis(scoreResponses(rows, key, { optionLabels: ["A", "B", "C", "D"] })));
        expectNear(analyze(shuffled), analyze(original));
      }),
    );
  });
});

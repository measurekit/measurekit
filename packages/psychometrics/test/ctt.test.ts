import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  augmentedSubscores,
  classifyDifficulty,
  classifyDiscrimination,
  createRandom,
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
  subscoreValue,
} from "../src/index.js";
import { defined, expectClose } from "./assertions.js";
import augmented from "./fixtures/ctt-augmented-subscores.json" with { type: "json" };
import bfi from "./fixtures/ctt-bfi-neuroticism.json" with { type: "json" };
import lsat7 from "./fixtures/ctt-lsat7.json" with { type: "json" };
import sat12 from "./fixtures/ctt-sat12.json" with { type: "json" };
import simulated from "./fixtures/ctt-simulated-2pl.json" with { type: "json" };
import subscores from "./fixtures/ctt-subscores.json" with { type: "json" };

/** Absolute tolerance of CTT statistics against R. */
const TOLERANCE = 1e-10;
/** Minres loadings against base R's minimization, both to their rounding. */
const OMEGA_LOADING_TOLERANCE = 1e-10;
/** Subscore PRMSEs against the subscore package's CTTsub: closed forms of the same sums. */
const SUBSCORE_TOLERANCE = 1e-12;
/** Augmented subscores against subscore.Wainer: the same matrices, inverted by another method. */
const AUGMENTED_SCORE_TOLERANCE = 1e-10;
/** Augmented reliabilities and weights against subscore.Wainer and base R. */
const AUGMENTED_TOLERANCE = 1e-10;

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
    expect(classifyDifficulty(0.75, { thresholds: { ...DIFFICULTY_THRESHOLDS, easy: 0.8 } })).toBe("moderate");
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
    expect(classifyDiscrimination(0.35, { thresholds: { veryGood: 0.35, good: 0.25, marginal: 0.15 } })).toBe(
      "very-good",
    );
  });

  it("rejects values and thresholds out of range", () => {
    expect(() => classifyDifficulty(1.2)).toThrow("A p-value must be a number from 0 to 1, found 1.2.");
    expect(() => classifyDifficulty(0.5, { thresholds: { difficult: 0.8, easy: 0.2 } })).toThrow(DataError);
    expect(() => classifyDiscrimination(Number.NaN)).toThrow(DataError);
    expect(() => classifyDiscrimination(0.5, { thresholds: { veryGood: 0.2, good: 0.3, marginal: 0.1 } })).toThrow(
      DataError,
    );
    expect(() => classifyDiscrimination(Number.POSITIVE_INFINITY)).toThrow(
      "A discrimination value must be a finite number, found Infinity.",
    );
  });

  it("rejects values and thresholds of other types from JavaScript", () => {
    expect(() => classifyDifficulty(JSON.parse('"0.5"'))).toThrow(
      'A p-value must be a number from 0 to 1, found "0.5".',
    );
    expect(() => classifyDifficulty(0.5, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => classifyDifficulty(0.5, { thresholds: JSON.parse("null") })).toThrow(
      "Difficulty thresholds must be an object { difficult, easy }, found null.",
    );
    expect(() =>
      classifyDiscrimination(0.5, { thresholds: JSON.parse('{ "veryGood": "0.4", "good": 0.3, "marginal": 0.2 }') }),
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

  it("takes at most 1,000 items for omega, whose correlations hold every pair of items", () => {
    const rows = (items: number) => [
      Array.from({ length: items }, (_, j) => j % 2),
      Array.from({ length: items }, (_, j) => (j + 1) % 2),
      Array.from({ length: items }, (_, j) => (j % 3 === 0 ? 1 : 0)),
    ];
    expect(() => mcdonaldOmega(responseMatrix(rows(1001)))).toThrow(
      "Omega holds a matrix of every pair of items, so it takes at most 1000 items, found 1001.",
    );
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
    expect(standardErrorOfMeasurement(polytomous, { reliability: 0.75 })).toBeCloseTo(Math.sqrt(4 / 3) / 2, 15);
    expect(() => standardErrorOfMeasurement(polytomous, { reliability: 1.2 })).toThrow(
      "A reliability must be a number from 0 to 1, found 1.2.",
    );
    expect(() => standardErrorOfMeasurement(polytomous, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => standardErrorOfMeasurement(polytomous, { reliability: JSON.parse("null") })).toThrow(
      "A reliability must be a number from 0 to 1, found null.",
    );
  });
});

/** The loadings of the eight items of the recipe on one factor, and the cuts of their five categories. */
const FACTOR_LOADINGS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8];
const FACTOR_CUTS = [-1.2, -0.4, 0.4, 1.2];
/** The range of the share of seeded samples whose 95% interval holds the population value. */
const INTERVAL_COVERAGE = [0.92, 0.98] as const;
const INTERVAL_SAMPLES = 200;
const INTERVAL_RESAMPLES = 200;
const INTERVAL_PERSONS = 200;

describe("the recipe for intervals of alpha and omega", () => {
  /** Persons answering items of five categories that load on one factor, as in ctt.md. */
  function sample(persons: number, random: ReturnType<typeof createRandom>): ResponseMatrix {
    const rows = Array.from({ length: persons }, () => {
      const factor = random.normal();
      return FACTOR_LOADINGS.map((loading) => {
        const score = loading * factor + Math.sqrt(1 - loading ** 2) * random.normal();
        return FACTOR_CUTS.filter((cut) => score > cut).length;
      });
    });
    return responseMatrix(rows);
  }

  /** The recipe of ctt.md: the percentile interval of a statistic over bootstrap samples of the persons. */
  function percentileInterval(
    data: ResponseMatrix,
    statistic: (resample: ResponseMatrix) => number,
    random: ReturnType<typeof createRandom>,
    { resamples = 1000, level = 0.95 } = {},
  ): [number, number] {
    const persons = data.responses.length;
    const values = Array.from({ length: resamples }, () => {
      const drawn = Array.from({ length: persons }, () => data.responses[Math.floor(random.uniform() * persons)] ?? []);
      return statistic(responseMatrix(drawn, { itemIds: data.itemIds }));
    }).sort((a, b) => a - b);
    const tail = (1 - level) / 2;
    return [values[Math.floor(tail * resamples)] ?? 0, values[Math.ceil((1 - tail) * resamples) - 1] ?? 0];
  }

  const omegaOf = (resample: ResponseMatrix) => mcdonaldOmega(resample).omega;

  it("covers the population value of alpha and of omega in 92 to 98 percent of 200 seeded samples", () => {
    const population = sample(200_000, createRandom(1));
    const truth = { alpha: cronbachAlpha(population), omega: omegaOf(population) };
    const random = createRandom(2);
    const covered = { alpha: 0, omega: 0 };
    for (let set = 0; set < INTERVAL_SAMPLES; set++) {
      const data = sample(INTERVAL_PERSONS, random);
      const [alphaLow, alphaHigh] = percentileInterval(data, cronbachAlpha, random, { resamples: INTERVAL_RESAMPLES });
      const [omegaLow, omegaHigh] = percentileInterval(data, omegaOf, random, { resamples: INTERVAL_RESAMPLES });
      if (alphaLow <= truth.alpha && truth.alpha <= alphaHigh) covered.alpha++;
      if (omegaLow <= truth.omega && truth.omega <= omegaHigh) covered.omega++;
    }
    for (const name of ["alpha", "omega"] as const) {
      expect(covered[name] / INTERVAL_SAMPLES, name).toBeGreaterThanOrEqual(INTERVAL_COVERAGE[0]);
      expect(covered[name] / INTERVAL_SAMPLES, name).toBeLessThanOrEqual(INTERVAL_COVERAGE[1]);
    }
  });

  it("brackets the estimate, and a lower level gives a narrower interval", () => {
    const data = sample(INTERVAL_PERSONS, createRandom(3));
    const wide = percentileInterval(data, cronbachAlpha, createRandom(4));
    const narrow = percentileInterval(data, cronbachAlpha, createRandom(4), { level: 0.5 });
    expect(wide[0]).toBeLessThan(cronbachAlpha(data));
    expect(wide[1]).toBeGreaterThan(cronbachAlpha(data));
    expect(narrow[0]).toBeGreaterThan(wide[0]);
    expect(narrow[1]).toBeLessThan(wide[1]);
  });

  it("gives the same interval for the same seed", () => {
    const data = sample(INTERVAL_PERSONS, createRandom(3));
    expect(percentileInterval(data, omegaOf, createRandom(4), { resamples: 100 })).toStrictEqual(
      percentileInterval(data, omegaOf, createRandom(4), { resamples: 100 }),
    );
  });
});

describe("subscoreValue", () => {
  for (const reference of subscores.cases) {
    it(`matches the subscore package's CTTsub on ${reference.name}`, () => {
      const matrix = responseMatrix(reference.responses, { itemIds: reference.itemIds });
      const result = subscoreValue(matrix, sectionsOf(reference.sections));
      expect(result.map(({ section }) => section)).toEqual(reference.sectionNames);
      result.forEach((value, i) => {
        expectClose(
          value.subscorePrmse,
          defined(reference.subscorePrmse[i]),
          SUBSCORE_TOLERANCE,
          `${value.section} PRMSE`,
        );
        expectClose(
          value.totalPrmse,
          defined(reference.totalPrmse[i]),
          SUBSCORE_TOLERANCE,
          `${value.section} total PRMSE`,
        );
        expect(value.addedValue, value.section).toBe(reference.addedValue[i]);
      });
    });
  }

  it("finds added value for the sections of weakly correlated abilities and none where the total predicts better", () => {
    const added = (name: string): boolean[] =>
      subscores.cases.find((reference) => reference.name === name)?.addedValue ?? [];
    expect(added("bfi")).toEqual([true, true, true, true, true]);
    expect(added("scored.data")).toEqual([false, false, false]);
  });

  /** Two sections of two items; the first section's items are equal, so that its subscore is its true score. */
  const HandRows = [
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
  ];
  const HandSections = { first: ["item1", "item2"], second: ["item3", "item4"] };

  it("derives the PRMSEs of a small design by hand", () => {
    // Subscores 2a and b1 + b2, with variances 6/5 and 7/10, and covariance 1/5; so var(x) = 6/5 + 7/10 + 2/5 = 23/10.
    // First section: items equal, alpha = 1, true covariance with x = 6/5 + 1/5 = 7/5, PRMSE of x = (7/5)² / (6/5 × 23/10)
    // = 49/69. Second: alpha = 2 (1 - 3/5 / (7/10)) = 2/7, true covariance = 9/10 - (5/7)(7/10) = 2/5, PRMSE of x =
    // (2/5)² / (2/7 × 7/10 × 23/10) = 8/23.
    const [first, second] = subscoreValue(responseMatrix(HandRows), HandSections);
    expect(first).toEqual({
      section: "first",
      subscorePrmse: 1,
      totalPrmse: expect.closeTo(49 / 69, 14),
      addedValue: true,
    });
    expect(second?.subscorePrmse).toBeCloseTo(2 / 7, 14);
    expect(second?.totalPrmse).toBeCloseTo(8 / 23, 14);
    expect(second?.addedValue).toBe(false);
  });

  it("leaves out items in no section and the persons who missed an item of a section", () => {
    const base = subscoreValue(responseMatrix(HandRows), HandSections);
    const extra = HandRows.map((row, person) => [...row, person % 2]);
    expect(subscoreValue(responseMatrix(extra), HandSections)).toEqual(base);
    const missing = [...extra, [null, 1, 0, 1, 1], [0, 0, 1, null, 0]];
    expect(subscoreValue(responseMatrix(missing), HandSections)).toEqual(base);
    const outside = extra.map((row, person) => (person === 0 ? [...row.slice(0, 4), null] : row));
    expect(subscoreValue(responseMatrix(outside), HandSections)).toEqual(base);
  });

  it("gives a section whose subscore is constant or whose alpha is not above 0 null PRMSEs and no added value", () => {
    // The first section's items sum to 1 for everyone; the second's have covariance -0.1, so alpha = -1; the third's
    // have covariance 0.1, so alpha = 1/2.
    const rows = [
      [0, 1, 1, 0, 1, 1],
      [1, 0, 1, 0, 1, 0],
      [0, 1, 0, 1, 0, 0],
      [1, 0, 0, 1, 0, 1],
      [0, 1, 1, 1, 1, 1],
      [1, 0, 0, 0, 0, 0],
    ];
    const [constant, anticorrelated, ordinary] = subscoreValue(responseMatrix(rows), {
      constant: ["item1", "item2"],
      anticorrelated: ["item3", "item4"],
      ordinary: ["item5", "item6"],
    });
    const none = { subscorePrmse: null, totalPrmse: null, addedValue: false };
    expect(constant).toEqual({ section: "constant", ...none });
    expect(anticorrelated).toEqual({ section: "anticorrelated", ...none });
    expect(ordinary?.subscorePrmse).toBeCloseTo(0.5, 14);
    expect(ordinary?.totalPrmse).not.toBeNull();
  });

  it("rejects sections that are not an object of arrays of item ids, a single section, and a section of one item", () => {
    const matrix = responseMatrix(HandRows);
    expect(() => subscoreValue(matrix, JSON.parse("null"))).toThrow(
      "The sections must be an object of item ids by section name, found null.",
    );
    expect(() => subscoreValue(matrix, JSON.parse('[["item1", "item2"]]'))).toThrow(
      "The sections must be an object of item ids by section name, found an array.",
    );
    expect(() => subscoreValue(matrix, JSON.parse('{"first": "item1"}'))).toThrow(
      'The items of section "first" must be an array of item ids, found "item1".',
    );
    expect(() => subscoreValue(matrix, { first: ["item1", "item2"] })).toThrow(
      "Subscore value needs at least two sections, found 1.",
    );
    expect(() => subscoreValue(matrix, { first: ["item1", "item2"], second: ["item3"] })).toThrow(
      'Section "second" needs at least two items, found 1.',
    );
    expect(() => subscoreValue(matrix, {})).toThrow("Subscore value needs at least two sections, found 0.");
  });

  it("rejects an item that is not in the matrix or is in two sections, and a matrix that is not valid", () => {
    const matrix = responseMatrix(HandRows);
    expect(() => subscoreValue(matrix, { first: ["item1", "item2"], second: ["item3", "other"] })).toThrow(
      'Section "second" has the item "other", which is not an item of the matrix.',
    );
    expect(() => subscoreValue(matrix, { first: ["item1", "item2"], second: ["item2", "item3"] })).toThrow(
      'The item "item2" is in the sections "first" and "second"; a section\'s items are its own.',
    );
    expect(() => subscoreValue(matrix, { first: ["item1", "item1"], second: ["item3", "item4"] })).toThrow(
      'The item "item1" is in the sections "first" and "first"; a section\'s items are its own.',
    );
    expect(() => subscoreValue(JSON.parse("null"), HandSections)).toThrow(DataError);
    expect(() => subscoreValue(matrix, { first: JSON.parse("[1, 2]"), second: ["item3", "item4"] })).toThrow(
      'Section "first" has the item 1, which is not an item of the matrix.',
    );
  });

  it("rejects fewer than two persons who answered every item and total scores without variance", () => {
    expect(() =>
      subscoreValue(
        responseMatrix([
          [1, 0, 1, 1],
          [1, null, 0, 1],
        ]),
        HandSections,
      ),
    ).toThrow("Subscore value needs at least two persons who answered every item of the sections.");
    expect(() =>
      subscoreValue(
        responseMatrix([
          [1, 0, 1, 0],
          [0, 1, 0, 1],
          [1, 0, 0, 1],
          [0, 1, 1, 0],
        ]),
        HandSections,
      ),
    ).toThrow("Subscore value needs total scores with variance.");
  });
});

describe("augmentedSubscores", () => {
  for (const reference of augmented.cases) {
    it(`matches subscore.Wainer on ${reference.name}`, () => {
      const matrix = responseMatrix(reference.responses, { itemIds: reference.itemIds });
      const result = augmentedSubscores(matrix, sectionsOf(reference.sections));
      expect(result.sections.map(({ section }) => section)).toEqual(reference.sectionNames);
      expect(result.persons).toEqual(reference.persons);
      expect(result.scores).toHaveLength(reference.scores.length);
      result.scores.forEach((row, person) => {
        row.forEach((score, j) => {
          expectClose(score, defined(reference.scores[person]?.[j]), AUGMENTED_SCORE_TOLERANCE, `person ${person}`);
        });
      });
      result.sections.forEach((section, j) => {
        expectClose(section.reliability, defined(reference.reliability[j]), AUGMENTED_TOLERANCE, "alpha");
        expectClose(
          section.augmentedReliability,
          defined(reference.augmentedReliability[j]),
          AUGMENTED_TOLERANCE,
          `${section.section} augmented reliability`,
        );
        expectClose(section.mean, defined(reference.mean[j]), AUGMENTED_SCORE_TOLERANCE, `${section.section} mean`);
        expectClose(
          section.standardDeviation,
          defined(reference.standardDeviation[j]),
          AUGMENTED_SCORE_TOLERANCE,
          `${section.section} standard deviation`,
        );
        result.weights[j]?.forEach((weight, k) => {
          expectClose(weight, defined(reference.weights[j]?.[k]), AUGMENTED_TOLERANCE, `weight ${j}, ${k}`);
        });
      });
    });
  }

  it("is closer to the true subscores than the observed subscores are, as in the simulation", () => {
    const reference = defined(augmented.cases.find(({ name }) => name === "simulated"));
    const matrix = responseMatrix(reference.responses, { itemIds: reference.itemIds });
    const sections = sectionsOf(reference.sections);
    const result = augmentedSubscores(matrix, sections);
    const truth = defined(reference.truth);
    Object.values(sections).forEach((ids, j) => {
      const columns = ids.map((id) => reference.itemIds.indexOf(id));
      const observed = result.persons.map((person) =>
        columns.reduce((sum, column) => sum + defined(reference.responses[person]?.[column]), 0),
      );
      const trueScores = result.persons.map((_, i) => defined(truth[i]?.[j]));
      const augmentedScores = result.scores.map((row) => defined(row[j]));
      expectClose(correlationOf(augmentedScores, trueScores), defined(reference.augmentedCorrelation?.[j]), 1e-10);
      expectClose(correlationOf(observed, trueScores), defined(reference.observedCorrelation?.[j]), 1e-10);
      expect(correlationOf(augmentedScores, trueScores)).toBeGreaterThan(correlationOf(observed, trueScores) + 0.05);
    });
  });

  /** Two sections of two equal items, the four combinations of their scores once each. */
  const AgreeingRows = [
    [0, 0, 0, 0],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
  ];
  const AgreeingSections = { first: ["item1", "item2"], second: ["item3", "item4"] };

  it("leaves subscores that are perfectly reliable as they are", () => {
    // Alpha is 1 for both sections, so S_true is S, B is the identity, and the reliabilities stay 1.
    const result = augmentedSubscores(responseMatrix(AgreeingRows), AgreeingSections);
    expect(result.weights).toEqual([
      [1, 0],
      [0, 1],
    ]);
    expect(result.sections.map(({ augmentedReliability }) => augmentedReliability)).toEqual([1, 1]);
    expect(result.scores).toEqual([
      [0, 0],
      [0, 2],
      [2, 0],
      [2, 2],
    ]);
  });

  it("shrinks uncorrelated subscores toward their means by their reliabilities", () => {
    // Each section's items take the patterns 00, 11 and 10, each with every pattern of the other section, so the
    // subscores 0, 2 and 1 are uncorrelated. Their variance is 3/4 and each item's 1/4, so alpha = 2 (1 - (1/2) / (3/4))
    // = 2/3; then B is the diagonal of the alphas, and so are the augmented reliabilities.
    const patterns = [
      [0, 0],
      [1, 1],
      [1, 0],
    ];
    const rows = patterns.flatMap((first) => patterns.map((second) => [...first, ...second]));
    const result = augmentedSubscores(responseMatrix(rows), AgreeingSections);
    result.weights.forEach((row, j) => {
      row.forEach((weight, k) => {
        expect(weight).toBeCloseTo(j === k ? 2 / 3 : 0, 14);
      });
    });
    result.sections.forEach((section) => {
      expect(section.reliability).toBeCloseTo(2 / 3, 14);
      expect(section.augmentedReliability).toBeCloseTo(2 / 3, 14);
      expect(section.mean).toBeCloseTo(1, 14);
    });
    rows.forEach((row, person) => {
      const sums = [(row[0] ?? 0) + (row[1] ?? 0), (row[2] ?? 0) + (row[3] ?? 0)];
      sums.forEach((sum, j) => {
        expect(result.scores[person]?.[j]).toBeCloseTo(1 + (2 / 3) * (sum - 1), 14);
      });
    });
  });

  it("gives the augmented subscores as the means plus the weights times the deviations", () => {
    const reference = defined(augmented.cases.find(({ name }) => name === "TIMSS11G8M.data"));
    const matrix = responseMatrix(reference.responses, { itemIds: reference.itemIds });
    const sections = sectionsOf(reference.sections);
    const result = augmentedSubscores(matrix, sections);
    const sums = reference.responses.map((row) =>
      Object.values(sections).map((ids) =>
        ids.reduce((sum, id) => sum + defined(row[reference.itemIds.indexOf(id)]), 0),
      ),
    );
    const means = result.sections.map(({ mean }) => mean);
    result.scores.forEach((row, person) => {
      row.forEach((score, j) => {
        const deviations = defined(sums[defined(result.persons[person])]).map((sum, k) => sum - defined(means[k]));
        const expected =
          defined(means[j]) + deviations.reduce((total, d, k) => total + defined(result.weights[j]?.[k]) * d, 0);
        expect(score).toBeCloseTo(expected, 10);
      });
    });
  });

  it("scores the persons who answered every item of the sections and ignores other items", () => {
    const base = augmentedSubscores(responseMatrix(ThreeSectionRows), ThreeSections);
    const extra = ThreeSectionRows.map((row, person) => [...row, person % 2]);
    expect(augmentedSubscores(responseMatrix(extra), ThreeSections)).toEqual(base);
    const missing = [...extra, [null, 1, 0, 1, 1, 0, 1], [0, 0, 1, null, 0, 1, 0]];
    const withMissing = augmentedSubscores(responseMatrix(missing), ThreeSections);
    expect(withMissing).toEqual(base);
    expect(withMissing.persons).toHaveLength(ThreeSectionRows.length);
  });

  it("does not change its input or share it with the result", () => {
    const rows = ThreeSectionRows.map((row) => [...row]);
    const matrix = responseMatrix(rows);
    const result = augmentedSubscores(matrix, ThreeSections);
    expect(rows).toEqual(ThreeSectionRows);
    expect(result.scores).not.toBe(matrix.responses);
  });

  it("rejects sections that cannot be augmented", () => {
    const matrix = responseMatrix(ThreeSectionRows);
    expect(() => augmentedSubscores(matrix, { first: ["item1", "item2"] })).toThrow(
      "Augmented subscores needs at least two sections, found 1.",
    );
    expect(() => augmentedSubscores(matrix, { first: ["item1", "item2"], second: ["item3"] })).toThrow(
      'Section "second" needs at least two items, found 1.',
    );
    expect(() => augmentedSubscores(matrix, { first: ["item1", "item2"], second: ["item3", "other"] })).toThrow(
      'Section "second" has the item "other", which is not an item of the matrix.',
    );
    expect(() => augmentedSubscores(matrix, { first: ["item1", "item2"], second: ["item2", "item3"] })).toThrow(
      'The item "item2" is in the sections "first" and "second"; a section\'s items are its own.',
    );
  });

  it("rejects sections whose subscores are linearly dependent and fewer than two complete persons", () => {
    const copy = [
      [0, 1, 0, 1],
      [1, 1, 1, 1],
      [0, 0, 0, 0],
      [1, 0, 1, 0],
      [1, 1, 1, 1],
    ];
    expect(() => augmentedSubscores(responseMatrix(copy), AgreeingSections)).toThrow(
      "Augmented subscores need observed subscores that are not linearly dependent, with more persons than sections.",
    );
    expect(() =>
      augmentedSubscores(
        responseMatrix([
          [1, 0, 1, 1],
          [1, null, 0, 1],
        ]),
        AgreeingSections,
      ),
    ).toThrow("Augmented subscores needs at least two persons who answered every item of the sections.");
  });

  it("rejects input of other types, as JavaScript or JSON pass it", () => {
    const matrix = responseMatrix(ThreeSectionRows);
    expect(() => augmentedSubscores(JSON.parse("null"), ThreeSections)).toThrow(DataError);
    expect(() => augmentedSubscores(matrix, JSON.parse("null"))).toThrow(
      "The sections must be an object of item ids by section name, found null.",
    );
    expect(() => augmentedSubscores(matrix, JSON.parse('{"first": "item1", "second": ["item3", "item4"]}'))).toThrow(
      'The items of section "first" must be an array of item ids, found "item1".',
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

/** Returns the result of `compute`, or `null` when it throws a `DataError`. */
function attempt<T>(compute: () => T): T | null {
  try {
    return compute();
  } catch (error) {
    if (error instanceof DataError) return null;
    throw error;
  }
}

/** Persons answering three sections of two items each, and a seventh item that belongs to none. */
const ThreeSectionRows = [
  [0, 0, 0, 0, 0, 0],
  [0, 0, 0, 1, 0, 0],
  [1, 1, 0, 0, 0, 0],
  [0, 0, 1, 1, 0, 0],
  [1, 1, 1, 0, 1, 0],
  [1, 0, 1, 1, 0, 1],
  [0, 1, 0, 0, 1, 1],
  [1, 1, 1, 1, 1, 1],
  [1, 1, 0, 1, 1, 1],
  [1, 1, 1, 1, 0, 1],
  [0, 0, 1, 1, 1, 1],
  [1, 0, 0, 0, 0, 0],
  [0, 1, 1, 1, 0, 0],
  [0, 0, 0, 0, 1, 1],
  [1, 1, 0, 0, 1, 1],
];
const ThreeSections = { first: ["item1", "item2"], second: ["item3", "item4"], third: ["item5", "item6"] };

/** Returns the Pearson correlation of two lists of numbers. */
function correlationOf(x: readonly number[], y: readonly number[]): number {
  const meanX = x.reduce((sum, value) => sum + value, 0) / x.length;
  const meanY = y.reduce((sum, value) => sum + value, 0) / y.length;
  let products = 0;
  let squaresX = 0;
  let squaresY = 0;
  x.forEach((value, i) => {
    products += (value - meanX) * (defined(y[i]) - meanY);
    squaresX += (value - meanX) ** 2;
    squaresY += (defined(y[i]) - meanY) ** 2;
  });
  return products / Math.sqrt(squaresX * squaresY);
}

/** The sections of a fixture, without the entries that the other cases' sections give the union type. */
function sectionsOf(
  sections: Readonly<Record<string, readonly string[] | undefined>>,
): Record<string, readonly string[]> {
  return Object.fromEntries(
    Object.entries(sections).flatMap(([name, ids]) => (ids === undefined ? [] : [[name, ids]])),
  );
}

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

  it("the subscore PRMSE is alpha of the section's items, whatever the order of persons, items and sections", () => {
    fc.assert(
      fc.property(permutedRows(arbitraryRows), ([rows, shuffled]) => {
        const items = rows[0]?.length ?? 0;
        if (items < 4) return;
        const half = Math.floor(items / 2);
        const ids = Array.from({ length: items }, (_, i) => `item${i + 1}`);
        const sections = { first: ids.slice(0, half), second: ids.slice(half) };
        const reversed = { second: [...sections.second].reverse(), first: [...sections.first].reverse() };
        const forward = attempt(() => subscoreValue(responseMatrix(rows), sections));
        const other = attempt(() => subscoreValue(responseMatrix(shuffled), reversed));
        if (forward === null || other === null) return;
        forward.forEach((value) => {
          const same = defined(other.find(({ section }) => section === value.section));
          // Summing the same item variances in another order moves alpha by rounding, which at exactly 0 decides null.
          if (value.subscorePrmse === null || same.subscorePrmse === null) {
            expect(Math.abs(value.subscorePrmse ?? same.subscorePrmse ?? 0)).toBeLessThan(1e-9);
            return;
          }
          expectClose(same.subscorePrmse, value.subscorePrmse, 1e-9);
          expectClose(same.totalPrmse, value.totalPrmse, 1e-9);
          expect(value.addedValue).toBe(value.subscorePrmse > defined(value.totalPrmse));
          if (Math.abs(value.subscorePrmse - defined(value.totalPrmse)) > 1e-9) {
            expect(same.addedValue).toBe(value.addedValue);
          }
        });
      }),
    );
  });

  it("subscore PRMSEs equal alpha of the section's items and never contain NaN", () => {
    fc.assert(
      fc.property(arbitraryRows, (rows) => {
        const items = rows[0]?.length ?? 0;
        if (items < 4) return;
        const half = Math.floor(items / 2);
        const ids = Array.from({ length: items }, (_, i) => `item${i + 1}`);
        const matrix = responseMatrix(rows);
        const result = attempt(() => subscoreValue(matrix, { first: ids.slice(0, half), second: ids.slice(half) }));
        if (result === null) return;
        const complete = rows.filter((row) => !row.includes(null));
        result.forEach((value, i) => {
          for (const number of [value.subscorePrmse, value.totalPrmse])
            expect(number === null || Number.isFinite(number)).toBe(true);
          const columns = i === 0 ? [0, half] : [half, items];
          const section = responseMatrix(complete.map((row) => row.slice(columns[0], columns[1])));
          const alpha = attempt(() => cronbachAlpha(section));
          if (value.subscorePrmse !== null && alpha !== null) expectClose(value.subscorePrmse, alpha, 1e-9);
        });
      }),
    );
  });

  it("augmented reliabilities are finite and do not depend on the order of persons", () => {
    fc.assert(
      fc.property(permutedRows(arbitraryRows), ([rows, shuffled]) => {
        const items = rows[0]?.length ?? 0;
        if (items < 4) return;
        const half = Math.floor(items / 2);
        const ids = Array.from({ length: items }, (_, i) => `item${i + 1}`);
        const sections = { first: ids.slice(0, half), second: ids.slice(half) };
        const forward = attempt(() => augmentedSubscores(responseMatrix(rows), sections));
        const other = attempt(() => augmentedSubscores(responseMatrix(shuffled), sections));
        if (forward === null || other === null) return;
        for (const { augmentedReliability, mean } of forward.sections) {
          expect(Number.isFinite(augmentedReliability)).toBe(true);
          expect(Number.isFinite(mean)).toBe(true);
        }
        expectNear(other.sections, forward.sections);
        expectNear(other.weights, forward.weights);
      }),
    );
  });
});

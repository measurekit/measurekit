import { shuffle } from "@measurekit/core";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  angoffCutScore,
  beukCutScore,
  bookmarkCutScore,
  borderlineGroupCutScores,
  borderlineRegressionCutScores,
  classificationAccuracy,
  contrastingGroupsCutScores,
  createRandom,
  cronbachAlpha,
  DataError,
  fourParameterLogistic,
  generalizedPartialCredit,
  gradedResponse,
  hofsteeCutScore,
  livingstonLewisClassification,
  nominalResponse,
  oneParameterLogistic,
  orderedItemBooklet,
  type Random,
  responseMatrix,
  type TestItem,
  threeParameterLogistic,
} from "../src/index.js";
import { defined, expectClose } from "./assertions.js";
import { type FixtureParameters, item2pl, MODELS } from "./fixture-items.js";
import fixture from "./fixtures/standard-setting.json" with { type: "json" };
import borderline from "./fixtures/standard-setting-borderline.json" with { type: "json" };
import cuts from "./fixtures/standard-setting-cut-scores.json" with { type: "json" };
import livingstonLewis from "./fixtures/standard-setting-livingston-lewis.json" with { type: "json" };

/** Classification probabilities against R's pnorm: Cody's algorithm on both sides. */
const CLASSIFICATION_TOLERANCE = 1e-14;

/** Cut scores and Bookmark locations against R, explained in docs/validation.md: closed forms and uniroot to 1e-14. */
const TOLERANCE = 1e-12;

/** Borderline cut scores, standard errors and R² against lm and base R: closed forms of the same sums. */
const BORDERLINE_TOLERANCE = 1e-12;

/** Contrasting-groups cut scores against glm, which stops at a deviance change of 1e-14. */
const LOGISTIC_TOLERANCE = 1e-6;

/** Livingston and Lewis's effective length and beta against betafunctions: closed forms of the same moments. */
const PARAMETER_TOLERANCE = 1e-6;
/** Livingston and Lewis's indices against base R's integrate to a relative 1e-12 from the same parameters. */
const INDEX_TOLERANCE = 1e-6;
/** Livingston and Lewis's indices against betafunctions' own, whose integrate keeps R's default tolerance. */
const PACKAGE_TOLERANCE = 1e-4;

type FixtureItems = readonly { readonly id: string; readonly model: string; readonly parameters: FixtureParameters }[];

function items(fixtureItems: FixtureItems): TestItem[] {
  return fixtureItems.map(({ id, model, parameters }) => ({ id, model: defined(MODELS[model]), parameters }));
}

const logistic = (value: number): number => 1 / (1 + Math.exp(-value));

/** Returns a draw of a beta distribution as the ratio of two gamma draws. */
function betaDraw(random: Random, alpha: number, beta: number): number {
  const x = gammaDraw(random, alpha);
  return x / (x + gammaDraw(random, beta));
}

/**
 * Returns a draw of a gamma distribution of unit scale by Marsaglia and Tsang's method, which needs a shape of at least
 * 1; a smaller shape takes a draw of shape + 1 times U^(1 / shape).
 */
function gammaDraw(random: Random, shape: number): number {
  if (shape < 1) return gammaDraw(random, shape + 1) * random.uniform() ** (1 / shape);
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    const z = random.normal();
    const v = (1 + c * z) ** 3;
    if (v > 0 && Math.log(random.uniform()) < 0.5 * z * z + d - d * v + d * Math.log(v)) return d * v;
  }
}

describe("angoffCutScore", () => {
  it("matches the base R reference", () => {
    const result = angoffCutScore(fixture.angoff.ratings);
    expectClose(result.cutScore, fixture.angoff.cutScore, TOLERANCE, "cut score");
    result.panelistCutScores.forEach((cut, i) => {
      expectClose(cut, defined(fixture.angoff.panelistCutScores[i]), TOLERANCE, `panelist ${i + 1}`);
    });
    expectClose(result.sd, fixture.angoff.sd, TOLERANCE, "sd");
    expectClose(result.standardError, fixture.angoff.standardError, TOLERANCE, "standard error");
  });

  it("sums each panelist's ratings, of dichotomous and polytomous items, and averages the sums", () => {
    // Panelists' cut scores 3.4, 3.5 and 3.3: mean 3.4, sd √((0 + 0.01 + 0.01) / 2) = 0.1.
    const result = angoffCutScore([
      [0.6, 0.8, 2],
      [0.7, 0.9, 1.9],
      [0.5, 0.7, 2.1],
    ]);
    expectClose(result.cutScore, 3.4, 1e-14);
    result.panelistCutScores.forEach((cut, i) => {
      expectClose(cut, defined([3.4, 3.5, 3.3][i]), 1e-14);
    });
    expectClose(result.sd, 0.1, 1e-14);
    expectClose(result.standardError, 0.1 / Math.sqrt(3), 1e-14);
  });

  it("gives no spread for one panelist", () => {
    expect(angoffCutScore([[0.5, 0.75]])).toEqual({
      cutScore: 1.25,
      panelistCutScores: [1.25],
      sd: null,
      standardError: null,
    });
  });

  it("rejects missing, uneven and invalid ratings", () => {
    expect(() => angoffCutScore([])).toThrow("An Angoff cut score needs ratings of at least one item by one panelist.");
    expect(() => angoffCutScore([[0.5, 0.6], [0.5]])).toThrow("Panelist 2 has 1 ratings, panelist 1 has 2.");
    expect(() => angoffCutScore([[0.5, -0.1]])).toThrow(
      "The rating of item 2 by panelist 1 must be a non-negative number, found -0.1.",
    );
    expect(() => angoffCutScore([[Number.NaN]])).toThrow(DataError);
    expect(() => angoffCutScore(JSON.parse("null"))).toThrow(
      "Angoff ratings must be an array of rows, one per panelist, found null.",
    );
    expect(() => angoffCutScore(JSON.parse('[[0.5], "a"]'))).toThrow(
      'The ratings of panelist 2 must be an array, found "a".',
    );
  });
});

describe("beukCutScore", () => {
  const { distribution } = cuts.beuk;

  cuts.beuk.cases.forEach(({ judgments, expected }, index) => {
    it(`matches the line's crossing written out in base R, case ${index + 1}`, () => {
      const result = beukCutScore(distribution, judgments);
      expectClose(result.cutScore, expected.cutScore, TOLERANCE, "cut score");
      expectClose(result.failureRate, expected.failureRate, TOLERANCE, "failure rate");
    });
  });

  it("moves a cut less the surer the panel is of it than of the pass rate", () => {
    const judgments = { cutScore: 24, passRate: 0.55, cutScoreSd: 3, passRateSd: 0.12 };
    const surerOfCut = beukCutScore(distribution, { ...judgments, cutScoreSd: 0.5 });
    const surerOfRate = beukCutScore(distribution, { ...judgments, passRateSd: 0.01 });
    expect(Math.abs(surerOfCut.cutScore - 24)).toBeLessThan(Math.abs(surerOfRate.cutScore - 24));
  });

  it("rejects invalid judgments and a line that does not cross the failure rates", () => {
    const judgments = { cutScore: 24, passRate: 0.55, cutScoreSd: 3, passRateSd: 0.12 };
    expect(() => beukCutScore(distribution, { ...judgments, cutScoreSd: 0 })).toThrow(
      "The judgment cutScoreSd must be a positive number, found 0.",
    );
    expect(() => beukCutScore(distribution, { ...judgments, passRate: 1.2 })).toThrow("The judgment passRate");
    expect(() => beukCutScore(distribution, { ...judgments, cutScore: 41 })).toThrow(
      "The judgment cutScore must be a score from 0 to 40, found 41.",
    );
    expect(() => beukCutScore(distribution, { ...judgments, passRate: 0, cutScoreSd: 1e6 })).toThrow(
      "The Beuk line does not cross the observed failure rates between the scores 0 and 40.",
    );
    expect(() => beukCutScore(distribution, JSON.parse("null"))).toThrow("The Beuk judgments must be an object");
    expect(() => beukCutScore(JSON.parse("{}"), judgments)).toThrow(DataError);
  });
});

describe("orderedItemBooklet", () => {
  const bookletItems = items(fixture.bookmark.items);

  for (const reference of fixture.bookmark.booklets) {
    it(`matches the base R pages at RP ${reference.responseProbability}`, () => {
      const pages = orderedItemBooklet(bookletItems, { responseProbability: reference.responseProbability });
      expect(pages.map(({ id, category }) => [id, category])).toEqual(
        reference.pages.map(({ id, category }) => [id, category]),
      );
      pages.forEach((page, i) => {
        expectClose(page.location, defined(reference.pages[i]).location, TOLERANCE, `page ${i + 1}`);
      });
    });
  }

  it("locates a logistic item where P(θ) = RP: b + ln((RP - c) / (d - RP)) / a", () => {
    const item = { id: "q1", model: fourParameterLogistic, parameters: { a: 1.3, b: 0.4, c: 0.2, d: 0.95 } };
    const [page] = orderedItemBooklet([item]);
    expect(page).toMatchObject({ id: "q1", category: 1 });
    expectClose(defined(page).location, 0.4 + Math.log((0.67 - 0.2) / (0.95 - 0.67)) / 1.3, 1e-12);
    const [rasch] = orderedItemBooklet([item2pl("q2", 1, -0.5)], { responseProbability: 2 / 3 });
    // Huynh (2006): at RP 2/3, a Rasch item's location is b + ln 2.
    expectClose(defined(rasch).location, -0.5 + Math.LN2, 1e-12);
  });

  it("locates a 3PL item at RP (2 + c) / 3 where P(θ) I(θ) is largest", () => {
    // Huynh (1998): the information of a correct response, P(θ) I(θ), of a 3PL item peaks where P(θ) = (2 + c) / 3.
    const parameters = { a: 1.4, b: -0.3, c: 0.25, d: 1 };
    const [page] = orderedItemBooklet([{ id: "q", model: threeParameterLogistic, parameters }], {
      responseProbability: (2 + parameters.c) / 3,
    });
    const correctInformation = (theta: number): number => {
      const probabilities = new Float64Array(2);
      threeParameterLogistic.probabilities(parameters, theta, probabilities);
      return defined(probabilities[1]) * threeParameterLogistic.information(parameters, theta);
    };
    const location = defined(page).location;
    const slope = (correctInformation(location + 1e-5) - correctInformation(location - 1e-5)) / 2e-5;
    expect(Math.abs(slope)).toBeLessThan(1e-8);
  });

  it("gives a polytomous item a page per score point, at P(X ≥ k) = RP", () => {
    const pages = orderedItemBooklet([{ id: "essay", model: gradedResponse, parameters: { a: 0.8, b: [-1, 0.5, 2] } }]);
    expect(pages.map(({ category }) => category)).toEqual([1, 2, 3]);
    pages.forEach((page, k) => {
      expectClose(page.location, defined([-1, 0.5, 2][k]) + Math.log(0.67 / 0.33) / 0.8, 1e-12);
    });
  });

  it("keeps items with equal locations in item order", () => {
    const pages = orderedItemBooklet([item2pl("b", 1, 0), item2pl("a", 1, 0), item2pl("c", 1, -1)]);
    expect(pages.map(({ id }) => id)).toEqual(["c", "b", "a"]);
  });

  it("rejects items whose probabilities do not rise through the response probability", () => {
    expect(() => orderedItemBooklet([])).toThrow("An ordered item booklet needs at least one item.");
    expect(() => orderedItemBooklet([item2pl("q", 1, 0), item2pl("q", 1, 1)])).toThrow(
      'Item ids must be unique; "q" appears twice.',
    );
    expect(() => orderedItemBooklet([item2pl("q", -1, 0)])).toThrow(
      'Item "q" needs an expected score that increases with theta: a model other than the NRM, a > 0.',
    );
    const nominal = { id: "n", model: nominalResponse, parameters: { a: [0, 1], c: [0, 0] } };
    expect(() => orderedItemBooklet([nominal])).toThrow(DataError);
    const guessing = { id: "g", model: threeParameterLogistic, parameters: { a: 1, b: 0, c: 0.7, d: 1 } };
    expect(() => orderedItemBooklet([guessing])).toThrow(
      'Item "g" has probabilities of a correct response from 0.7 to 1, which never equal the response probability 0.67.',
    );
    expect(() => orderedItemBooklet([item2pl("q", 1, 0)], { responseProbability: 1 })).toThrow(
      "The option responseProbability must be a number above 0 and below 1, found 1.",
    );
  });
});

describe("bookmarkCutScore", () => {
  const bookletItems = items(fixture.bookmark.items);

  for (const reference of fixture.bookmark.booklets) {
    it(`matches mirt's expected test scores at every page at RP ${reference.responseProbability}`, () => {
      reference.pages.forEach((page, i) => {
        const result = bookmarkCutScore(bookletItems, i + 1, { responseProbability: reference.responseProbability });
        expectClose(result.theta, page.location, TOLERANCE, `theta ${i + 1}`);
        expectClose(result.cutScore, defined(reference.cutScores[i]), TOLERANCE, `cut score ${i + 1}`);
      });
    });
  }

  it("cuts at the last page mastered, with the expected total score there", () => {
    // At RP 0.5 the 1PL items lie at b = 0 and 1; after page 1, θ = 0 and the expected score is 0.5 + P(0 - 1).
    const pool = [0, 1].map((b, j) => ({
      id: `q${j + 1}`,
      model: oneParameterLogistic,
      parameters: { a: 1, b, c: 0, d: 1 },
    }));
    const result = bookmarkCutScore(pool, 1, { responseProbability: 0.5 });
    expectClose(result.theta, 0, 1e-12);
    expectClose(result.cutScore, 0.5 + logistic(-1), 1e-12);
  });

  it("rejects pages outside the booklet", () => {
    const pool = [item2pl("q1", 1, 0), item2pl("q2", 1, 1)];
    for (const page of [0, 3, 1.5]) {
      expect(() => bookmarkCutScore(pool, page)).toThrow(`A bookmark follows a page from 1 to 2, found ${page}.`);
    }
    expect(() => bookmarkCutScore(pool, 1, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => orderedItemBooklet(JSON.parse("null"))).toThrow("The items must be an array, found null.");
  });
});

describe("borderlineGroupCutScores", () => {
  for (const reference of Object.values(borderline.group)) {
    it(`matches base R's mean, median and sd for ${reference.name}`, () => {
      const result = borderlineGroupCutScores(reference.marks, reference.ratings, reference.borderline, {
        statistic: reference.statistic === "median" ? "median" : "mean",
      });
      expect(result.stations).toHaveLength(reference.stations.length);
      reference.stations.forEach((expected, s) => {
        const station = defined(result.stations[s]);
        expectClose(station.cutScore, expected.cutScore, BORDERLINE_TOLERANCE, `cut ${s + 1}`);
        expectClose(station.standardError, expected.standardError, BORDERLINE_TOLERANCE, `standard error ${s + 1}`);
        expect(station.persons).toBe(expected.persons);
      });
      expectClose(result.cutScore, reference.cutScore, BORDERLINE_TOLERANCE, "test");
    });
  }

  it("takes the mean of the persons at the borderline grade by default", () => {
    // Station 1's borderline persons have 10 and 14; station 2's have 6, 8 and 13.
    const marks = [
      [10, 6],
      [14, 8],
      [20, 13],
      [5, 2],
    ];
    const ratings = [
      [2, 2],
      [2, 2],
      [3, 2],
      [1, 1],
    ];
    const result = borderlineGroupCutScores(marks, ratings, 2);
    expect(result.stations.map((station) => station.cutScore)).toEqual([12, 9]);
    expect(result.stations.map((station) => station.persons)).toEqual([2, 3]);
    expect(result.cutScore).toBe(21);
    expect(borderlineGroupCutScores(marks, ratings, 2, { statistic: "median" }).cutScore).toBe(20);
    expect(borderlineGroupCutScores(marks, ratings, [1, 2]).stations[0]?.cutScore).toBeCloseTo(29 / 3, 12);
  });

  it("gives one borderline person a cut without a standard error", () => {
    const result = borderlineGroupCutScores([[7], [9]], [[2], [3]], 2);
    expect(result.stations).toEqual([{ cutScore: 7, standardError: null, persons: 1 }]);
    expect(result.cutScore).toBe(7);
  });

  it("rejects marks, ratings and grades that are not numbers, and too large marks", () => {
    expect(() => borderlineGroupCutScores([[1, 2]], [[2]], 2)).toThrow(
      "Person 1 needs 2 ratings, one per station, found 1.",
    );
    expect(() => borderlineGroupCutScores([[1]], [[2], [3]], 2)).toThrow(
      "The marks and the ratings need a row for each person, found 1 and 2.",
    );
    expect(() => borderlineGroupCutScores([], [], 2)).toThrow(
      "A borderline method needs the marks of at least one person.",
    );
    expect(() => borderlineGroupCutScores([[]], [[]], 2)).toThrow("A borderline method needs at least one station.");
    expect(() => borderlineGroupCutScores(JSON.parse('[["1"]]'), [[2]], 2)).toThrow(
      'Person 1 needs a finite mark at station 1, found "1".',
    );
    expect(() => borderlineGroupCutScores([[1]], JSON.parse("[[null]]"), 2)).toThrow(
      "Person 1 needs a finite rating at station 1, found null.",
    );
    expect(() => borderlineGroupCutScores(JSON.parse("{}"), [[2]], 2)).toThrow("The marks must be an array");
    expect(() => borderlineGroupCutScores([[1]], JSON.parse("[{}]"), 2)).toThrow(
      "The ratings of person 1 must be an array, one per station",
    );
    expect(() => borderlineGroupCutScores([[1]], [[2]], [])).toThrow("The borderline grades need at least one grade.");
    expect(() => borderlineGroupCutScores([[1]], [[2]], JSON.parse('"2"'))).toThrow(
      "The borderline grades must be a number or an array of numbers",
    );
    expect(() => borderlineGroupCutScores([[1]], [[2]], [2, Number.NaN])).toThrow(
      "Borderline grade 2 must be a finite number, found NaN.",
    );
    expect(() => borderlineGroupCutScores([[1]], [[2]], 2, JSON.parse('{ "statistic": "mode" }'))).toThrow(
      'The option statistic must be "mean" or "median", found "mode".',
    );
    expect(() => borderlineGroupCutScores([[1]], [[2]], 2, JSON.parse("null"))).toThrow(
      "The options must be an object",
    );
    expect(() => borderlineGroupCutScores([[1e308], [1e308]], [[2], [2]], 2)).toThrow(
      "The borderline group's cut of station 1 is not finite: the marks are too large; rescale them.",
    );
    expect(() => borderlineGroupCutScores([[1e308, 1e308]], [[2, 2]], 2)).toThrow(
      "The borderline group's cut of the test is not finite: the marks are too large; rescale them.",
    );
  });
});

describe("borderlineRegressionCutScores", () => {
  for (const reference of Object.values(borderline.regression)) {
    it(`matches lm, predict and the covariance of the stations' cuts for ${reference.name}`, () => {
      const result = borderlineRegressionCutScores(reference.marks, reference.ratings, reference.borderline);
      expect(result.stations).toHaveLength(reference.stations.length);
      reference.stations.forEach((expected, s) => {
        const station = defined(result.stations[s]);
        for (const name of ["cutScore", "standardError", "rSquared", "intercept", "slope"] as const) {
          expectClose(station[name], expected[name], BORDERLINE_TOLERANCE, `${name} ${s + 1}`);
        }
      });
      expectClose(result.cutScore, reference.cutScore, BORDERLINE_TOLERANCE, "cut");
      expectClose(result.standardError, reference.standardError, BORDERLINE_TOLERANCE, "standard error");
    });
  }

  it("reads the line at the borderline grade, and gives one station's test its standard error", () => {
    // Marks 10 + 5 x exactly: the line passes through every person, so the cut at 2 is 20 with no error.
    const exact = borderlineRegressionCutScores([[15], [20], [25], [30]], [[1], [2], [3], [4]], 2);
    expect(exact.stations[0]).toEqual({ cutScore: 20, standardError: 0, rSquared: 1, intercept: 10, slope: 5 });
    const { marks, ratings } = defined(borderline.regression["six stations, ratings 1 to 4"]);
    const first = borderlineRegressionCutScores(
      marks.map((row) => [defined(row[0])]),
      ratings.map((row) => [defined(row[0])]),
      2,
    );
    expectClose(first.standardError, defined(first.stations[0]?.standardError), 1e-15);
    const station = defined(first.stations[0]);
    expectClose(defined(station.intercept) + 2 * defined(station.slope), station.cutScore, 1e-12);
  });

  it("adds the stations' covariances to the test's variance", () => {
    // Two copies of a station have residuals that agree, so the test's cut has twice the station's standard error;
    // the station and its marks reflected about their mean have opposite residuals and opposite lines, so their sum
    // holds the mean mark of both at every grade and its standard error is 0.
    const marks = [12, 15, 14, 19, 22, 21];
    const ratings = [1, 2, 2, 3, 4, 4];
    const twice = borderlineRegressionCutScores(
      marks.map((mark) => [mark, mark]),
      ratings.map((rating) => [rating, rating]),
      2,
    );
    expectClose(twice.standardError, 2 * defined(twice.stations[0]?.standardError), 1e-12);
    const reflected = borderlineRegressionCutScores(
      marks.map((mark) => [mark, 40 - mark]),
      ratings.map((rating) => [rating, rating]),
      2,
    );
    expectClose(reflected.cutScore, 40, 1e-12);
    expectClose(reflected.standardError, 0, 1e-12);
  });

  it("refuses more than 2^28 persons × stations² before it sums them", () => {
    const row = new Array<number>(1024).fill(1);
    const rows = Array.from({ length: 257 }, () => row);
    expect(() => borderlineRegressionCutScores(rows, rows, 2)).toThrow(
      "Borderline regression sums over the persons for each pair of stations, so it takes at most 268435456 persons × stations², found 257 × 1024²; set the standard of fewer persons or stations at a time.",
    );
  });

  it("rejects input of other types and results that are not finite", () => {
    expect(() => borderlineRegressionCutScores([[1]], [[2]], JSON.parse('"2"'))).toThrow(
      'The borderline grade must be a finite number, found "2".',
    );
    expect(() => borderlineRegressionCutScores(JSON.parse("null"), [[2]], 2)).toThrow("The marks must be an array");
    expect(() => borderlineRegressionCutScores([[1], [2]], JSON.parse("[[1], 2]"), 2)).toThrow(
      "The ratings of person 2 must be an array, one per station",
    );
    expect(() => borderlineRegressionCutScores([[1e200], [-1e200], [0]], [[1], [2], [3]], 2)).toThrow(
      "The borderline regression of station 1 is not finite: the marks, the ratings or the borderline grade are too large; rescale them.",
    );
    // Each station's cut is 5e307, and four of them pass the largest double.
    const large = Array.from({ length: 3 }, () => new Array<number>(4).fill(5e307));
    const grades = [1, 2, 3].map((grade) => new Array<number>(4).fill(grade));
    expect(() => borderlineRegressionCutScores(large, grades, 2)).toThrow(
      "The borderline regression of the test is not finite: the marks, the ratings or the borderline grade are too large; rescale them.",
    );
  });
});

describe("classificationAccuracy", () => {
  const { estimates, cases } = fixture.classification;
  for (const expected of cases) {
    it(`matches cacIRT's class.Rud and base R's pnorm with cuts at ${expected.cuts.join(", ")}`, () => {
      const result = classificationAccuracy(estimates, expected.cuts);
      expectClose(result.accuracy, expected.accuracy, CLASSIFICATION_TOLERANCE, "accuracy");
      expectClose(result.consistency, expected.consistency, CLASSIFICATION_TOLERANCE, "consistency");
      result.persons.forEach((person, i) => {
        expect(person.category, `person ${i + 1}`).toBe(expected.category[i]);
        person.probabilities.forEach((probability, k) => {
          expectClose(
            probability,
            defined(expected.probabilities[i]?.[k]),
            CLASSIFICATION_TOLERANCE,
            `person ${i + 1}`,
          );
        });
      });
    });
  }

  it("puts a person at a cut above it, with even odds of passing", () => {
    const { persons, accuracy } = classificationAccuracy([{ theta: 0.5, standardError: 0.3 }], [0.5]);
    expect(persons[0]).toEqual({ category: 1, probabilities: [0.5, 0.5] });
    expect(accuracy).toBe(0.5);
  });

  it("rejects missing or unordered cuts and invalid estimates", () => {
    const estimate = { theta: 0, standardError: 1 };
    expect(() => classificationAccuracy([], [0])).toThrow("Classification accuracy needs at least one estimate.");
    expect(() => classificationAccuracy([estimate], [])).toThrow("Classification accuracy needs at least one cut.");
    expect(() => classificationAccuracy([estimate], [1, 1])).toThrow("The cuts must increase; cut 2 is 1.");
    // Zero has two signs in floating point, but it is one cut.
    expect(() => classificationAccuracy([estimate], [-0, 0])).toThrow("The cuts must increase; cut 2 is 0.");
    expect(() => classificationAccuracy([estimate], [0, -0])).toThrow("The cuts must increase; cut 2 is 0.");
    expect(() => classificationAccuracy([{ theta: 0, standardError: 0 }], [0])).toThrow(
      "The standard error of estimate 1 must be a positive number, found 0.",
    );
    expect(() => classificationAccuracy(JSON.parse("[null]"), [0])).toThrow(
      "Estimate 1 must be an object { theta, standardError }, found null.",
    );
    expect(() => classificationAccuracy([estimate], JSON.parse('["0"]'))).toThrow(
      'Cut 1 must be a finite number, found "0".',
    );
  });
});

describe("contrastingGroupsCutScores", () => {
  for (const fixtureCase of Object.values(cuts.contrasting)) {
    it(`matches glm per boundary for ${fixtureCase.name}`, () => {
      const result = contrastingGroupsCutScores(fixtureCase.scores, fixtureCase.categories, fixtureCase.options);
      expect(result.cutScores).toHaveLength(fixtureCase.cutScores.length);
      fixtureCase.cutScores.forEach((expected, k) => {
        expectClose(result.cutScores[k] ?? null, expected, LOGISTIC_TOLERANCE, `boundary ${k + 1}`);
      });
      expect(result.converged).toBe(fixtureCase.converged);
    });
  }

  it("recovers a known cut within 0.2 raw points from 6,000 persons", () => {
    // Scores of a 40-item test; P(master | x) = logistic(x - 22), so the cut at probability 0.5 is 22. The cut's standard
    // error is 0.061 by the expected information, so 0.2 is 3.3 of them; at 2,000 persons it would be 1.9.
    const random = createRandom(20261102);
    const scores = Array.from({ length: 6000 }, () => {
      let score = 0;
      const theta = random.normal();
      for (let i = 0; i < 40; i++) if (random.uniform() < logistic(theta - (i - 19.5) / 12)) score++;
      return score;
    });
    const masters = scores.map((score) => (random.uniform() < logistic(score - 22) ? 1 : 0));
    expectClose(defined(contrastingGroupsCutScores(scores, masters).cutScores[0]), 22, 0.2);
  });

  it("refuses a category that nobody holds, without a boundary for each number below a large one", () => {
    expect(() => contrastingGroupsCutScores([1, 2, 3, 4], [0, 0, 2, 2])).toThrow(
      "Cut scores need persons in every category from 0 to the highest, 2; category 1 has none.",
    );
    // A boundary for each number up to 2⁵³ - 1 would run for years; the check stops at the first empty category.
    expect(() => contrastingGroupsCutScores([1, 2, 3], [0, 1, Number.MAX_SAFE_INTEGER])).toThrow(
      `Cut scores need persons in every category from 0 to the highest, ${Number.MAX_SAFE_INTEGER}; category 2 has none.`,
    );
  });

  it("rejects invalid scores, categories and options", () => {
    expect(() => contrastingGroupsCutScores([1, 2, 3], [0, 1])).toThrow(
      "The scores and the categories need one entry for each person, found 3 and 2.",
    );
    expect(() => contrastingGroupsCutScores([1, 2], [0, 1.5])).toThrow(
      "Person 2 needs a category that is an integer from 0, found 1.5.",
    );
    expect(() => contrastingGroupsCutScores([1, JSON.parse('"2"')], [0, 1])).toThrow(
      'Person 2 needs a finite score, found "2".',
    );
    expect(() => contrastingGroupsCutScores([1, 2], [0, 0])).toThrow(
      "Cut scores need persons in at least two categories.",
    );
    expect(() => contrastingGroupsCutScores([1, 2, 3], [1, 2, 2])).toThrow(
      "Cut scores need persons in every category from 0 to the highest, 2; category 0 has none.",
    );
    expect(() => contrastingGroupsCutScores([1, 2, 3, 4], [0, 1, 0, 1], { probability: 1 })).toThrow(
      "The option probability must be a number between 0 and 1, exclusive, found 1.",
    );
    expect(() => contrastingGroupsCutScores(JSON.parse("{}"), [0, 1])).toThrow("The scores must be an array");
    expect(() => contrastingGroupsCutScores([1, 2], [0, 1], JSON.parse("null"))).toThrow(
      "The options must be an object",
    );
  });
});

describe("recipes", () => {
  it("gives Nedelsky's cut score as the Angoff cut score of 1 over the options left", () => {
    const { optionsLeft, cutScore } = cuts.recipes.nedelsky;
    const result = angoffCutScore(optionsLeft.map((row) => row.map((options) => 1 / options)));
    expectClose(result.cutScore, cutScore, TOLERANCE);
  });

  it("gives Ebel's cut score as the Angoff cut score of each item rated with its cell's proportion", () => {
    const { items: counts, proportions, cutScore } = cuts.recipes.ebel;
    const ratings = counts.flatMap((count, cell) => new Array<number>(count).fill(defined(proportions[cell])));
    expectClose(angoffCutScore([ratings]).cutScore, cutScore, TOLERANCE);
  });
});

describe("hofsteeCutScore", () => {
  for (const [i, reference] of fixture.hofstee.cases.entries()) {
    it(`matches the base R crossing of case ${i + 1}`, () => {
      const result = hofsteeCutScore(fixture.hofstee.distribution, reference.bounds);
      expectClose(result.cutScore, reference.cutScore, TOLERANCE, "cut score");
      expectClose(result.failureRate, reference.failureRate, TOLERANCE, "failure rate");
    });
  }

  const bounds = { minimumCutScore: 1, maximumCutScore: 3, minimumFailureRate: 0.25, maximumFailureRate: 0.75 };

  it("crosses the panel's line with the failure rates, linear between scores", () => {
    // One person at each score from 0 to 3: F(c) = c / 4. The line 0.75 - (c - 1) / 4 meets it at c = 2.
    expect(hofsteeCutScore([1, 1, 1, 1], bounds)).toEqual({ cutScore: 2, failureRate: 0.5 });
    // Two persons at 1: F rises from 0.2 at 1 to 0.6 at 2, and meets 0.75 - (c - 1) / 4 at c = 1 + 0.55 / 0.65.
    const result = hofsteeCutScore([1, 2, 1, 1], bounds);
    expectClose(result.cutScore, 1 + 0.55 / 0.65, 1e-15);
    expectClose(result.failureRate, 0.75 - 0.55 / 0.65 / 4, 1e-15);
  });

  it("keeps the cut score within the bounds where their difference rounds", () => {
    // 1.4663184973103873e-93 + (5.304227003211107e-93 - 1.4663184973103873e-93) rounds one ulp above the maximum.
    const bounds = {
      minimumCutScore: 1.4663184973103873e-93,
      maximumCutScore: 5.304227003211107e-93,
      minimumFailureRate: 0,
      maximumFailureRate: 5e-324,
    };
    expect(hofsteeCutScore([0, 1], bounds)).toEqual({ cutScore: bounds.maximumCutScore, failureRate: 0 });
  });

  it("rejects invalid bounds and a line that does not cross the failure rates", () => {
    expect(() => hofsteeCutScore([1, 1, 1, 1], { ...bounds, maximumCutScore: 4 })).toThrow(
      "The Hofstee cut score bounds need 0 ≤ k_min < k_max ≤ 3, found 1 and 4.",
    );
    expect(() => hofsteeCutScore([1, 1, 1, 1], { ...bounds, minimumFailureRate: 0.8 })).toThrow(
      "The Hofstee failure rate bounds need 0 ≤ f_min < f_max ≤ 1, found 0.8 and 0.75.",
    );
    // At k_min = 2, half the persons fail, more than f_max allows.
    const strict = { minimumCutScore: 2, maximumCutScore: 3, minimumFailureRate: 0.1, maximumFailureRate: 0.3 };
    expect(() => hofsteeCutScore([1, 1, 1, 1], strict)).toThrow(
      "The Hofstee line does not cross the observed failure rates between the minimum and maximum cut scores.",
    );
    expect(() => hofsteeCutScore([5], bounds)).toThrow(
      "The observed score distribution needs counts for at least the scores 0 and 1.",
    );
    // A bound given as a string from JavaScript would turn the cut score into a string.
    expect(() => hofsteeCutScore([1, 1, 1, 1], { ...bounds, minimumCutScore: JSON.parse('"1"') })).toThrow(
      'The bound minimumCutScore must be a finite number, found "1".',
    );
    expect(() => hofsteeCutScore([1, 1, 1, 1], JSON.parse("null"))).toThrow(
      "The Hofstee bounds must be an object of the cut scores and failure rates, found null.",
    );
  });
});

describe("livingstonLewisClassification", () => {
  for (const reference of Object.values(livingstonLewis.cases)) {
    it(`matches betafunctions' fit and base R's indices: ${reference.name}`, () => {
      const { scores, cuts: cutScores, reliability, minimum, maximum } = reference;
      const result = livingstonLewisClassification(scores, cutScores, { reliability, minimum, maximum });
      const { trueScore } = result;
      expectClose(trueScore.effectiveLength, reference.effectiveLength, PARAMETER_TOLERANCE, "effective length");
      expectClose(trueScore.alpha, reference.trueScore.alpha, PARAMETER_TOLERANCE, "alpha");
      expectClose(trueScore.beta, reference.trueScore.beta, PARAMETER_TOLERANCE, "beta");
      expectClose(trueScore.lower, reference.trueScore.lower, PARAMETER_TOLERANCE, "lower");
      expectClose(trueScore.upper, reference.trueScore.upper, PARAMETER_TOLERANCE, "upper");
      expectClose(result.accuracy, reference.accuracy, INDEX_TOLERANCE, "accuracy");
      expectClose(result.consistency, reference.consistency, INDEX_TOLERANCE, "consistency");
      expectClose(result.accuracy, reference.package.accuracy, PACKAGE_TOLERANCE, "betafunctions' accuracy");
      expectClose(result.consistency, reference.package.consistency, PACKAGE_TOLERANCE, "betafunctions' consistency");
      expect(result.cuts.map(({ cut }) => cut)).toEqual(cutScores);
      reference.perCut.forEach((expected, k) => {
        const found = defined(result.cuts[k]);
        expectClose(found.accuracy, expected.accuracy, INDEX_TOLERANCE, `cut ${k + 1} accuracy`);
        expectClose(found.consistency, expected.consistency, INDEX_TOLERANCE, `cut ${k + 1} consistency`);
        expectClose(found.accuracy, expected.package.accuracy, PACKAGE_TOLERANCE, `cut ${k + 1}, betafunctions`);
        expectClose(found.consistency, expected.package.consistency, PACKAGE_TOLERANCE, `cut ${k + 1}, betafunctions`);
      });
    });
  }

  it("recovers the accuracy and consistency of simulated classifications within 0.01 over 20 sets", () => {
    // True proportions from a beta on [0.05, 0.95], 40 items answered right with that probability each, and a second
    // test of 40 such items for consistency: the binomial error the method assumes. Cuts at 18 and 28.
    const random = createRandom(20261102);
    const differences = { accuracy: 0, consistency: 0 };
    const sets = 20;
    for (let set = 0; set < sets; set++) {
      const persons = 3000;
      const truth = Array.from({ length: persons }, () => 0.05 + 0.9 * betaDraw(random, 3, 2.5));
      const answer = (p: number): number[] => Array.from({ length: 40 }, () => (random.uniform() < p ? 1 : 0));
      const first = truth.map(answer);
      const second = truth.map((p) => answer(p).reduce((sum, x) => sum + x, 0));
      const scores = first.map((row) => row.reduce((sum, x) => sum + x, 0));
      const reliability = cronbachAlpha(responseMatrix(first));
      const result = livingstonLewisClassification(scores, [18, 28], { reliability, maximum: 40 });
      const category = (score: number): number => (score >= 28 ? 2 : score >= 18 ? 1 : 0);
      let accurate = 0;
      let consistent = 0;
      truth.forEach((p, i) => {
        const observed = category(defined(scores[i]));
        if (observed === category(40 * p)) accurate++;
        if (observed === category(defined(second[i]))) consistent++;
      });
      differences.accuracy += result.accuracy - accurate / persons;
      differences.consistency += result.consistency - consistent / persons;
    }
    expect(Math.abs(differences.accuracy / sets)).toBeLessThan(0.01);
    expect(Math.abs(differences.consistency / sets)).toBeLessThan(0.01);
  });

  it("classifies every person alike by a cut at or below the lowest score, and moves with the scale", () => {
    const { scores, reliability } = defined(livingstonLewis.cases["one cut"]);
    const below = livingstonLewisClassification(scores, [0], { reliability, maximum: 40 });
    expectClose(below.accuracy, 1, 1e-12, "accuracy");
    expectClose(below.consistency, 1, 1e-12, "consistency");
    const plain = livingstonLewisClassification(scores, [16, 24], { reliability, maximum: 40 });
    const shifted = livingstonLewisClassification(
      scores.map((score) => 2 * score + 10),
      [42, 58],
      { reliability, minimum: 10, maximum: 90 },
    );
    expectClose(shifted.accuracy, plain.accuracy, 1e-12, "accuracy on the doubled scale");
    expectClose(shifted.trueScore.alpha, plain.trueScore.alpha, 1e-9, "alpha on the doubled scale");
    expect(shifted.cuts.map(({ cut }) => cut)).toEqual([42, 58]);
  });

  it("keeps accuracy and consistency within [0, 1] where the quadrature's rounding would carry them past 1", () => {
    // With a cut beyond the highest score every person is below it, and the integrals of the beta came to 1 + 1.2e-13.
    const scores = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    const beyond = livingstonLewisClassification(scores, [50], { reliability: 0.8, maximum: 10 });
    for (const value of [
      beyond.accuracy,
      beyond.consistency,
      ...beyond.cuts.flatMap((cut) => [cut.accuracy, cut.consistency]),
    ]) {
      expect(value).toBeLessThanOrEqual(1);
      expect(value).toBeGreaterThanOrEqual(0);
    }
    expect(beyond.accuracy).toBe(1);
  });

  it("counts a cut a rounding below a whole number of trials as that number", () => {
    // N rounds to 46: a cut at 0.5 of the range scales to 23 trials exactly, so a score of 23 is above it.
    const { scores, reliability } = defined(livingstonLewis.cases["one cut"]);
    const exact = livingstonLewisClassification(scores, [20], { reliability, maximum: 40 });
    const nudged = livingstonLewisClassification(scores, [20 - 1e-12], { reliability, maximum: 40 });
    expectClose(nudged.accuracy, exact.accuracy, 1e-6, "a cut a rounding below");
  });

  it("rejects invalid scores, cuts and options", () => {
    const scores = [10, 14, 20, 25, 31];
    const options = { reliability: 0.8, maximum: 40 };
    expect(() => livingstonLewisClassification([10], [20], options)).toThrow(
      "Livingston and Lewis need at least two scores, found 1.",
    );
    expect(() => livingstonLewisClassification([10, 41], [20], options)).toThrow(
      "Score 2 must be a number from 0 to 40, found 41.",
    );
    expect(() => livingstonLewisClassification([10, Number.NaN], [20], options)).toThrow(DataError);
    expect(() => livingstonLewisClassification([12, 12, 12], [20], options)).toThrow(
      "Livingston and Lewis need scores that vary; every score is the same.",
    );
    expect(() => livingstonLewisClassification(scores, [], options)).toThrow(
      "Livingston and Lewis need at least one cut.",
    );
    expect(() => livingstonLewisClassification(scores, [20, 20], options)).toThrow(
      "The cuts must increase; cut 2 is 20.",
    );
    expect(() => livingstonLewisClassification(scores, [20], { ...options, reliability: 1 })).toThrow(
      "The option reliability must be a number between 0 and 1, exclusive, found 1.",
    );
    expect(() => livingstonLewisClassification(scores, [20], { reliability: 0.8, maximum: 0 })).toThrow(
      "The option maximum must be above the minimum 0, found 0.",
    );
    expect(() => livingstonLewisClassification(scores, [20], { ...options, minimum: Number.NaN })).toThrow(
      "The option minimum must be a finite number, found NaN.",
    );
  });

  it("rejects an effective length outside its limits and scores no beta fits", () => {
    // Scores only at the ends of the range have a variance near the largest, so N = 0.375 at a reliability of 0.6; a
    // reliability of 0.9999 makes N about 97,000.
    expect(() => livingstonLewisClassification([0, 40, 0, 40], [20], { reliability: 0.6, maximum: 40 })).toThrow(
      /Livingston and Lewis need an effective test length from 1 to 10000 after rounding/,
    );
    expect(() =>
      livingstonLewisClassification([12, 18, 20, 21, 29], [20], { reliability: 0.9999, maximum: 40 }),
    ).toThrow(/effective test length from 1 to 10000/);
    expect(() => livingstonLewisClassification([18, 19, 20, 21, 22], [20], { reliability: 0.01, maximum: 40 })).toThrow(
      /Livingston and Lewis find no true-score beta for these scores/,
    );
  });

  it("rejects scores, cuts and options of other types from JavaScript", () => {
    const options = { reliability: 0.8, maximum: 40 };
    expect(() => livingstonLewisClassification(JSON.parse("null"), [20], options)).toThrow(
      "Livingston and Lewis need an array of scores, found null.",
    );
    expect(() => livingstonLewisClassification([10, JSON.parse('"20"')], [20], options)).toThrow(
      'Score 2 must be a number from 0 to 40, found "20".',
    );
    expect(() => livingstonLewisClassification([10, 20], JSON.parse("{}"), options)).toThrow(
      "Livingston and Lewis need an array of cuts, found an object.",
    );
    expect(() => livingstonLewisClassification([10, 20], [JSON.parse('"15"')], options)).toThrow(
      'Cut 1 must be a finite number, found "15".',
    );
    expect(() => livingstonLewisClassification([10, 20], [15], JSON.parse("null"))).toThrow(
      "Livingston and Lewis need options { reliability, maximum }, found null.",
    );
    expect(() =>
      livingstonLewisClassification([10, 20], [15], JSON.parse('{"reliability":"0.8","maximum":40}')),
    ).toThrow('The option reliability must be a number between 0 and 1, exclusive, found "0.8".');
  });
});

describe("properties", () => {
  it("classification probabilities sum to 1, and accuracy and consistency lie in [0, 1]", () => {
    const estimate = fc.record({
      theta: fc.double({ min: -4, max: 4, noNaN: true }),
      standardError: fc.double({ min: 0.05, max: 2, noNaN: true }),
    });
    // SameValueZero takes -0 and 0 as one value, as the check that the cuts increase does; the default comparison draws them as two cuts.
    const cuts = fc.uniqueArray(fc.double({ min: -3, max: 3, noNaN: true }), {
      minLength: 1,
      maxLength: 4,
      comparator: "SameValueZero",
    });
    fc.assert(
      fc.property(fc.array(estimate, { minLength: 1, maxLength: 20 }), cuts, (estimates, values) => {
        const result = classificationAccuracy(
          estimates,
          [...values].sort((x, y) => x - y),
        );
        for (const person of result.persons) {
          expectClose(
            person.probabilities.reduce((sum, p) => sum + p, 0),
            1,
            1e-12,
          );
        }
        expect(result.consistency).toBeLessThanOrEqual(1 + 1e-12);
        expect(result.accuracy).toBeGreaterThanOrEqual(0);
      }),
    );
  });

  const arbitraryBookletItem = fc.oneof(
    fc
      .record({
        a: fc.double({ min: 0.3, max: 2.5, noNaN: true }),
        b: fc.double({ min: -3, max: 3, noNaN: true }),
        c: fc.double({ min: 0, max: 0.3, noNaN: true }),
      })
      .map((parameters) => ({ model: threeParameterLogistic, parameters: { ...parameters, d: 1 } })),
    fc
      .record({
        a: fc.double({ min: 0.3, max: 2.5, noNaN: true }),
        b: fc.array(fc.double({ min: -3, max: 3, noNaN: true }), { minLength: 1, maxLength: 3 }),
      })
      .map((parameters) => ({ model: generalizedPartialCredit, parameters })),
  );

  it("Bookmark cut scores rise with the page", () => {
    fc.assert(
      fc.property(fc.array(arbitraryBookletItem, { minLength: 1, maxLength: 6 }), (pool) => {
        const withIds = pool.map((item, j) => ({ ...item, id: `q${j + 1}` }));
        const pages = orderedItemBooklet(withIds);
        let previous = { theta: Number.NEGATIVE_INFINITY, cutScore: Number.NEGATIVE_INFINITY };
        for (let page = 1; page <= pages.length; page++) {
          const cut = bookmarkCutScore(withIds, page);
          expect(cut.theta).toBeGreaterThanOrEqual(previous.theta);
          expect(cut.cutScore).toBeGreaterThanOrEqual(previous.cutScore - 1e-12);
          previous = cut;
        }
      }),
      { numRuns: 50 },
    );
  });

  it("an Angoff cut score lies between the panelists' lowest and highest", () => {
    const ratings = fc.integer({ min: 1, max: 6 }).chain((items) =>
      fc.array(fc.array(fc.double({ min: 0, max: 1, noNaN: true }), { minLength: items, maxLength: items }), {
        minLength: 1,
        maxLength: 8,
      }),
    );
    fc.assert(
      fc.property(ratings, (rows) => {
        const { cutScore, panelistCutScores, sd } = angoffCutScore(rows);
        expect(cutScore).toBeGreaterThanOrEqual(Math.min(...panelistCutScores) - 1e-12);
        expect(cutScore).toBeLessThanOrEqual(Math.max(...panelistCutScores) + 1e-12);
        expect(sd === null || sd >= 0).toBe(true);
      }),
    );
  });

  it("a Hofstee cut score lies within the bounds, on the panel's line", () => {
    const cases = fc.record({
      distribution: fc.array(fc.integer({ min: 0, max: 20 }), { minLength: 2, maxLength: 12 }),
      cuts: fc.tuple(fc.double({ min: 0, max: 1, noNaN: true }), fc.double({ min: 0, max: 1, noNaN: true })),
      rates: fc.tuple(fc.double({ min: 0, max: 1, noNaN: true }), fc.double({ min: 0, max: 1, noNaN: true })),
    });
    fc.assert(
      fc.property(cases, ({ distribution, cuts, rates }) => {
        const highest = distribution.length - 1;
        const bounds = {
          minimumCutScore: Math.min(...cuts) * highest,
          maximumCutScore: Math.max(...cuts) * highest,
          minimumFailureRate: Math.min(...rates),
          maximumFailureRate: Math.max(...rates),
        };
        try {
          const { cutScore, failureRate } = hofsteeCutScore(distribution, bounds);
          expect(cutScore).toBeGreaterThanOrEqual(bounds.minimumCutScore);
          expect(cutScore).toBeLessThanOrEqual(bounds.maximumCutScore);
          expect(failureRate).toBeGreaterThanOrEqual(bounds.minimumFailureRate - 1e-12);
          expect(failureRate).toBeLessThanOrEqual(bounds.maximumFailureRate + 1e-12);
        } catch (error) {
          if (!(error instanceof DataError)) throw error;
        }
      }),
    );
  });
  it("moves each contrasting-groups cut by an increasing linear transformation of the scores", () => {
    const people = fc.array(fc.tuple(fc.integer({ min: 0, max: 30 }), fc.integer({ min: 0, max: 2 })), {
      minLength: 12,
      maxLength: 60,
    });
    fc.assert(
      fc.property(
        people,
        fc.double({ min: 0.2, max: 5, noNaN: true }),
        fc.double({ min: -50, max: 50, noNaN: true }),
        (pairs, slope, shift) => {
          const scores = pairs.map(([score]) => score);
          const categories = pairs.map(([, category]) => category);
          const highest = Math.max(...categories);
          fc.pre(highest > 0 && categories.includes(0) && new Set(categories).size === highest + 1);
          const before = contrastingGroupsCutScores(scores, categories);
          const after = contrastingGroupsCutScores(
            scores.map((score) => slope * score + shift),
            categories,
          );
          before.cutScores.forEach((cut, k) => {
            const moved = after.cutScores[k] ?? null;
            if (cut === null) expect(moved).toBeNull();
            else
              expectClose(moved, slope * cut + shift, 1e-6 * (1 + Math.abs(slope * cut + shift)), `boundary ${k + 1}`);
          });
        },
      ),
      { numRuns: 50 },
    );
  });
  /** Scores of 30 items answered with a person's beta-distributed probability, and a reliability and cuts. */
  const livingstonLewisCase = fc.record({
    seed: fc.integer(),
    persons: fc.integer({ min: 30, max: 300 }),
    shapes: fc.tuple(fc.double({ min: 0.3, max: 6, noNaN: true }), fc.double({ min: 0.3, max: 6, noNaN: true })),
    reliability: fc.double({ min: 0.5, max: 0.95, noNaN: true }),
    cuts: fc.uniqueArray(fc.integer({ min: 2, max: 28 }), { minLength: 1, maxLength: 3 }),
  });
  const simulatedScores = ({ seed, persons, shapes }: { seed: number; persons: number; shapes: [number, number] }) => {
    const random = createRandom(seed);
    return Array.from({ length: persons }, () => {
      const p = betaDraw(random, shapes[0], shapes[1]);
      let score = 0;
      for (let i = 0; i < 30; i++) if (random.uniform() < p) score++;
      return score;
    });
  };
  /** Livingston and Lewis's classification, or `null` where the scores and the reliability fit no beta. */
  const classifyOrNull = (scores: readonly number[], cuts: readonly number[], reliability: number) => {
    try {
      return livingstonLewisClassification(scores, cuts, { reliability, maximum: 30 });
    } catch (error) {
      if (!(error instanceof DataError)) throw error;
      return null;
    }
  };

  it("Livingston and Lewis's indices lie in [0, 1], and the overall ones at most each cut's", () => {
    fc.assert(
      fc.property(livingstonLewisCase, ({ cuts: cutScores, reliability, ...simulation }) => {
        const sorted = [...cutScores].sort((a, b) => a - b);
        const result = classifyOrNull(simulatedScores(simulation), sorted, reliability);
        if (result === null) return;
        for (const value of [result.accuracy, result.consistency]) {
          expect(value).toBeGreaterThanOrEqual(-1e-12);
          expect(value).toBeLessThanOrEqual(1 + 1e-12);
        }
        for (const cut of result.cuts) {
          expect(result.accuracy).toBeLessThanOrEqual(cut.accuracy + 1e-12);
          expect(result.consistency).toBeLessThanOrEqual(cut.consistency + 1e-12);
        }
      }),
      { numRuns: 60 },
    );
  });

  it("Livingston and Lewis classify every person alike below the lowest cut, whatever the order of the scores", () => {
    fc.assert(
      fc.property(livingstonLewisCase, fc.integer(), ({ cuts: cutScores, reliability, ...simulation }, order) => {
        const scores = simulatedScores(simulation);
        const below = classifyOrNull(scores, [-1], reliability);
        if (below === null) return;
        expectClose(below.accuracy, 1, 1e-12, "accuracy");
        expectClose(below.consistency, 1, 1e-12, "consistency");
        const sorted = [...cutScores].sort((a, b) => a - b);
        const permuted = Float64Array.from(scores);
        shuffle(permuted, createRandom(order));
        const first = defined(classifyOrNull(scores, sorted, reliability));
        const second = defined(classifyOrNull(Array.from(permuted), sorted, reliability));
        expectClose(second.accuracy, first.accuracy, 1e-12, "accuracy of the permuted scores");
        expectClose(second.consistency, first.consistency, 1e-12, "consistency of the permuted scores");
      }),
      { numRuns: 60 },
    );
  });
});

import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  angoffCutScore,
  bookmarkCutScore,
  classificationAccuracy,
  DataError,
  fourParameterLogistic,
  generalizedPartialCredit,
  gradedResponse,
  hofsteeCutScore,
  nominalResponse,
  oneParameterLogistic,
  orderedItemBooklet,
  type TestItem,
  threeParameterLogistic,
} from "../src/index.js";
import { defined, expectClose } from "./assertions.js";
import { type FixtureParameters, item2pl, MODELS } from "./fixture-items.js";
import fixture from "./fixtures/standard-setting.json" with { type: "json" };

/** Classification probabilities against R's pnorm: Cody's algorithm on both sides. */
const CLASSIFICATION_TOLERANCE = 1e-14;

/** Cut scores and Bookmark locations against R, explained in docs/validation.md: closed forms and uniroot to 1e-14. */
const TOLERANCE = 1e-12;

type FixtureItems = readonly { readonly id: string; readonly model: string; readonly parameters: FixtureParameters }[];

function items(fixtureItems: FixtureItems): TestItem[] {
  return fixtureItems.map(({ id, model, parameters }) => ({ id, model: defined(MODELS[model]), parameters }));
}

const logistic = (value: number): number => 1 / (1 + Math.exp(-value));

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
});

import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { cohenKappa, DataError, fleissKappa, intraclassCorrelation } from "../src/index.js";
import fixture from "./fixtures/rater.json" with { type: "json" };
import { expectClose } from "./helpers.js";

/** Kappa and intraclass correlations against irr: closed forms of the same counts and sums. */
const TOLERANCE = 1e-12;

const FIELDS = [
  "oneWay",
  "agreement",
  "consistency",
  "oneWayAverage",
  "agreementAverage",
  "consistencyAverage",
] as const;

/** Ratings of 2 to 20 subjects by `raters` raters in categories 0 to 4. */
function ratings(raters: number): fc.Arbitrary<number[][]> {
  return fc.array(fc.array(fc.integer({ min: 0, max: 4 }), { minLength: raters, maxLength: raters }), {
    minLength: 2,
    maxLength: 20,
  });
}

describe("intraclassCorrelation", () => {
  for (const dataset of fixture.cases) {
    it(`matches irr's icc and psych's ICC on ${dataset.name}`, () => {
      const result = intraclassCorrelation(dataset.ratings);
      for (const field of FIELDS) expectClose(result[field], dataset.icc[field], TOLERANCE, field);
    });
  }

  it("gives 1 for raters who agree exactly, and null coefficients without any variance", () => {
    const same = intraclassCorrelation([
      [1, 1],
      [2, 2],
      [4, 4],
    ]);
    for (const field of FIELDS) expectClose(same[field], 1, 1e-15, field);
    const constant = intraclassCorrelation([
      [3, 3],
      [3, 3],
    ]);
    for (const field of FIELDS) expect(constant[field]).toBeNull();
  });

  it("rejects one rater, non-finite ratings and fewer than two subjects rated by every rater", () => {
    expect(() => intraclassCorrelation([[1], [2]])).toThrow(
      "The intraclass correlation needs at least 2 raters, found 1.",
    );
    expect(() =>
      intraclassCorrelation([
        [1, Number.NaN],
        [2, 3],
      ]),
    ).toThrow("The rating of subject 1 by rater 2 must be a finite number or null, found NaN.");
    expect(() =>
      intraclassCorrelation([
        [1, null],
        [2, 3],
      ]),
    ).toThrow("The intraclass correlation needs at least 2 subjects rated by every rater, found 1.");
    expect(() => intraclassCorrelation([[1, 2], [3]])).toThrow("Subject 2 needs 2 ratings, one per rater.");
  });
});

describe("kappa", () => {
  for (const dataset of fixture.cases) {
    it(`matches irr's kappa2 and kappam.fleiss on ${dataset.name}`, () => {
      const pairs = dataset.ratings.map((row) => row.slice(0, 2));
      for (const weights of ["none", "linear", "quadratic"] as const) {
        expectClose(cohenKappa(pairs, { weights }), dataset.cohen[weights], TOLERANCE, weights);
      }
      expectClose(fleissKappa(dataset.ratings), dataset.fleiss, TOLERANCE, "Fleiss");
    });
  }

  it("gives 1 for perfect agreement, 0 for agreement at chance, and is undefined for one category", () => {
    expect(
      cohenKappa([
        [0, 0],
        [1, 1],
        [2, 2],
      ]),
    ).toBe(1);
    // Each rater rates half 0 and half 1, independently: p_o = p_e = 1/2.
    expect(
      cohenKappa([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
      ]),
    ).toBe(0);
    expect(
      fleissKappa([
        [0, 0, 0],
        [1, 1, 1],
      ]),
    ).toBe(1);
    expect(() =>
      cohenKappa([
        [2, 2],
        [2, 2],
      ]),
    ).toThrow("Cohen's kappa is undefined when the chance agreement is 1.");
    expect(() =>
      fleissKappa([
        [2, 2],
        [2, 2],
      ]),
    ).toThrow("Fleiss' kappa is undefined when the chance agreement is 1.");
  });

  it("rejects ratings that are not categories, other numbers of raters and unknown weights", () => {
    expect(() => cohenKappa([[0, 1, 2]])).toThrow("Cohen's kappa needs 2 raters, found 3.");
    expect(() => cohenKappa([[0, 1.5]])).toThrow(
      "The rating of subject 1 by rater 2 must be a category, an integer from 0, or null, found 1.5.",
    );
    expect(() => cohenKappa([[0, null]])).toThrow(
      "Cohen's kappa needs at least 1 subjects rated by every rater, found 0.",
    );
    expect(() => fleissKappa([[1], [0]])).toThrow(DataError);
    expect(() => cohenKappa([[0, 1]], { weights: JSON.parse('"cubic"') })).toThrow(
      'The option weights must be "none", "linear" or "quadratic", found "cubic".',
    );
    expect(() => cohenKappa([[0, 1]], JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => intraclassCorrelation(JSON.parse('[[1, "2"], [2, 3]]'))).toThrow(
      'The rating of subject 1 by rater 2 must be a finite number or null, found "2".',
    );
    expect(() => fleissKappa(JSON.parse("null"))).toThrow(
      "Fleiss' kappa needs an array of ratings, one row per subject.",
    );
  });

  it("rates more subjects than a spread of arguments holds", () => {
    const ratings = Array.from({ length: 200_000 }, (_, subject) => [subject % 4, subject % 4]);
    expect(cohenKappa(ratings, { weights: "quadratic" })).toBe(1);
  });
});

describe("properties", () => {
  it("kappas are at most 1 and do not depend on the order of subjects", () => {
    fc.assert(
      fc.property(ratings(3), (rows) => {
        const pairs = rows.map((row) => row.slice(0, 2));
        for (const statistic of [() => cohenKappa(pairs, { weights: "quadratic" }), () => fleissKappa(rows)]) {
          try {
            expect(statistic()).toBeLessThanOrEqual(1 + 1e-12);
          } catch (error) {
            if (!(error instanceof DataError)) throw error;
          }
        }
        try {
          expectClose(cohenKappa([...pairs].reverse()), cohenKappa(pairs), 1e-12);
        } catch (error) {
          if (!(error instanceof DataError)) throw error;
        }
      }),
    );
  });

  it("the intraclass correlation of the mean of raters is at least that of a single rater when positive", () => {
    fc.assert(
      fc.property(ratings(4), (rows) => {
        const result = intraclassCorrelation(rows);
        if (result.consistency !== null && result.consistency > 0 && result.consistencyAverage !== null) {
          expect(result.consistencyAverage).toBeGreaterThanOrEqual(result.consistency - 1e-12);
        }
      }),
    );
  });
});

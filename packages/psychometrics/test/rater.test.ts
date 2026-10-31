import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  adjacentAgreement,
  cohenKappa,
  createRandom,
  DataError,
  exactAgreement,
  expectedRating,
  type FacetedScore,
  fleissKappa,
  intraclassCorrelation,
  type ManyFacetRasch,
  manyFacetEstimates,
  manyFacetFit,
  manyFacetRasch,
  maximumLikelihoodEstimate,
  type PersonEstimate,
  standardizedMeanDifference,
} from "../src/index.js";
import { summarizeFacet } from "../src/rater/facet-summary.js";
import fixture from "./fixtures/rater.json" with { type: "json" };
import agreementFixture from "./fixtures/rater-agreement.json" with { type: "json" };
import manyFacetFixture from "./fixtures/rater-many-facet.json" with { type: "json" };
import { expectClose } from "./helpers.js";

/** Kappa and intraclass correlations against irr: closed forms of the same counts and sums. */
const TOLERANCE = 1e-12;
/** Agreement shares and standardized mean differences against irr and effsize: closed forms of the same counts and sums. */
const AGREEMENT_TOLERANCE = 1e-12;
/** Measures, thresholds, the mean and SD of theta against the base R maximum of the marginal likelihood. */
const ESTIMATE_TOLERANCE = 1e-8;
/** The marginal log-likelihood at the maximum. */
const LOG_LIKELIHOOD_TOLERANCE = 1e-8;
/** Standard errors from the Hessian, and the summaries of the facets that follow from them, relative, against numDeriv's. */
const STANDARD_ERROR_TOLERANCE = 1e-6;
/** Estimates, fit, fair averages and summaries computed from the same parameters. */
const DERIVED_TOLERANCE = 1e-10;

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

type ScoreColumns = Readonly<Record<string, readonly (string | number)[] | undefined>>;

/** The scores of a fixture, one record per row of its columns. */
function scoresOf(columns: ScoreColumns): FacetedScore[] {
  const facets = Object.keys(columns).filter((name) => name !== "score");
  return (columns.score ?? []).map((score, row) => ({
    levels: Object.fromEntries(facets.map((facet) => [facet, String(columns[facet]?.[row])])),
    score: Number(score),
  }));
}

/** The many-facet model that a fixture's base R maximum is, as a stored model. */
function modelOf(reference: (typeof manyFacetFixture.cases)[number]["models"][number]): ManyFacetRasch {
  return {
    person: "person",
    thresholdFacet: reference.thresholdFacet,
    facets: reference.summaries.map((summary) => ({
      facet: summary.facet,
      elements: summary.ids.map((id, e) => ({
        id,
        measure: summary.measures[e] ?? 0,
        standardError: summary.standardErrors[e] ?? null,
        count: summary.counts[e] ?? 0,
        observedAverage: summary.observedAverages[e] ?? 0,
        fairAverage: summary.fairAverages[e] ?? 0,
      })),
      separation: summary.separation,
      reliability: summary.reliability,
      homogeneity: summary.homogeneity,
    })),
    thresholds: reference.thresholds,
    population: { mean: reference.mean, sd: reference.sd },
    logLikelihood: reference.logLikelihood,
    aic: 0,
    bic: 0,
    converged: true,
    iterations: 0,
  };
}

/** Options that run EM to the precision of the base R maximum. */
const PRECISE = { tolerance: 1e-11, maxIterations: 5000 } as const;

/**
 * Scores of `persons` persons on two items by two of three raters, with categories 0 to 3, from the many-facet model
 * with item difficulties, rater severities and thresholds as given and theta ~ N(0.3, 1).
 */
function simulatedScores(seed: number, persons: number, severities = [-0.4, 0, 0.4]): FacetedScore[] {
  const random = createRandom(seed);
  const difficulties = [-0.5, 0.5];
  const thresholds = [-1, 0.2, 0.8];
  const scores: FacetedScore[] = [];
  for (let person = 0; person < persons; person++) {
    const theta = 0.3 + random.normal();
    for (const rater of [person % 3, (person + 1 + Math.floor(person / 3)) % 3]) {
      difficulties.forEach((difficulty, item) => {
        const weights = [1];
        let logit = 0;
        for (const tau of thresholds) {
          logit += theta - difficulty - (severities[rater] ?? 0) - tau;
          weights.push(Math.exp(logit));
        }
        let draw = random.uniform() * weights.reduce((sum, weight) => sum + weight, 0);
        let score = 0;
        while (score < 3 && draw > (weights[score] ?? 0)) draw -= weights[score++] ?? 0;
        scores.push({ levels: { person: `p${person}`, item: `i${item}`, rater: `r${rater}` }, score });
      });
    }
  }
  return scores;
}

describe("exactAgreement and adjacentAgreement", () => {
  for (const reference of agreementFixture.agreement) {
    it(`match irr's agree at every width on ${reference.name}`, () => {
      expectClose(exactAgreement(reference.ratings), reference.exact, AGREEMENT_TOLERANCE, "exact");
      expectClose(adjacentAgreement(reference.ratings), reference.adjacent, AGREEMENT_TOLERANCE, "adjacent");
      for (const { width, agreement } of reference.widths) {
        expectClose(adjacentAgreement(reference.ratings, { width }), agreement, AGREEMENT_TOLERANCE, `width ${width}`);
      }
    });
  }

  it("counts a subject when the ratings are within the width, derived by hand", () => {
    // Differences 0, 1, 2 and 0: exact 2 of 4, within 1 three of 4, within 2 all.
    const ratings = [
      [0, 0],
      [1, 2],
      [3, 1],
      [2, 2],
    ];
    expect(exactAgreement(ratings)).toBe(0.5);
    expect(adjacentAgreement(ratings)).toBe(0.75);
    expect(adjacentAgreement(ratings, { width: 0 })).toBe(0.5);
    expect(adjacentAgreement(ratings, { width: 2 })).toBe(1);
  });

  it("needs every rater of a subject within the width when there are more than two", () => {
    // The ratings 1, 2, 3 span 2 although neighbors differ by 1.
    const ratings = [
      [1, 2, 3],
      [2, 2, 2],
      [1, 2, 2],
    ];
    expect(exactAgreement(ratings)).toBeCloseTo(1 / 3, 15);
    expect(adjacentAgreement(ratings)).toBeCloseTo(2 / 3, 15);
    expect(adjacentAgreement(ratings, { width: 2 })).toBe(1);
  });

  it("uses the subjects every rater rated", () => {
    const rated = [
      [0, 0],
      [1, 3],
    ];
    expect(exactAgreement([...rated, [2, null], [null, 4], [null, null]])).toBe(exactAgreement(rated));
  });

  it("rejects a width that is not a non-negative integer, options of the wrong type and other invalid ratings", () => {
    const ratings = [[0, 1]];
    expect(() => adjacentAgreement(ratings, { width: -1 })).toThrow(
      "The option width must be a non-negative number, found -1.",
    );
    expect(() => adjacentAgreement(ratings, { width: 1.5 })).toThrow(
      "The option width must be a non-negative integer, found 1.5.",
    );
    expect(() => adjacentAgreement(ratings, { width: JSON.parse('"1"') })).toThrow(
      'The option width must be a non-negative number, found "1".',
    );
    expect(() => adjacentAgreement(ratings, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => exactAgreement([[0], [1]])).toThrow("Exact agreement needs at least 2 raters, found 1.");
    expect(() => adjacentAgreement([[0, 1.5]])).toThrow(
      "The rating of subject 1 by rater 2 must be a category, an integer from 0, or null, found 1.5.",
    );
    expect(() => exactAgreement([[0, null]])).toThrow(
      "Exact agreement needs at least 1 subjects rated by every rater, found 0.",
    );
    expect(() => adjacentAgreement(JSON.parse("null"))).toThrow(
      "Adjacent agreement needs an array of ratings, one row per subject.",
    );
  });
});

describe("standardizedMeanDifference", () => {
  for (const reference of agreementFixture.meanDifference) {
    it(`matches base R and effsize's cohen.d on ${reference.name}`, () => {
      expectClose(standardizedMeanDifference(reference.ratings), reference.value, AGREEMENT_TOLERANCE);
    });
  }

  it("divides the first rater's mean less the second's by the pooled standard deviation, derived by hand", () => {
    // Means 2 and 4 with variances 1 and 1: (2 - 4) / 1.
    expect(
      standardizedMeanDifference([
        [1, 3],
        [2, 4],
        [3, 5],
      ]),
    ).toBe(-2);
    // Means 2 and 1, variances 4 and 0: pooled √2, so 1 / √2.
    expect(
      standardizedMeanDifference([
        [0, 1],
        [2, 1],
        [4, 1],
      ]),
    ).toBeCloseTo(1 / Math.SQRT2, 15);
  });

  it("is 0 for equal means, changes sign with the order of the raters and ignores the scale and origin of the ratings", () => {
    const ratings = [
      [1, 2],
      [4, 3],
      [2, 2],
      [5, 6],
    ];
    // Means 3 and 3.25, variances 10/3 and 10.75/3: -0.25 / √(20.75 / 6).
    expect(standardizedMeanDifference(ratings)).toBeCloseTo(-0.25 / Math.sqrt(20.75 / 6), 14);
    const value = standardizedMeanDifference(ratings);
    expectClose(standardizedMeanDifference(ratings.map(([x = 0, y = 0]) => [y, x])), -value, 1e-15);
    expectClose(standardizedMeanDifference(ratings.map(([x = 0, y = 0]) => [3 * x + 7, 3 * y + 7])), value, 1e-12);
    expect(
      standardizedMeanDifference([
        [1, 3],
        [3, 1],
        [2, 2],
      ]),
    ).toBe(0);
  });

  it("uses the subjects both raters rated and accepts ratings that are not categories", () => {
    const rated = [
      [1.5, 2.25],
      [3, 2.5],
      [4.75, 4],
    ];
    expect(standardizedMeanDifference([...rated, [null, 9], [8, null]])).toBe(standardizedMeanDifference(rated));
  });

  it("rejects other numbers of raters, constant ratings, too few subjects and invalid ratings", () => {
    expect(() =>
      standardizedMeanDifference([
        [1, 2, 3],
        [2, 3, 4],
      ]),
    ).toThrow("The standardized mean difference needs 2 raters, found 3.");
    expect(() =>
      standardizedMeanDifference([
        [2, 2],
        [2, 2],
      ]),
    ).toThrow("The standardized mean difference is undefined when both raters' ratings are constant.");
    expect(() => standardizedMeanDifference([[1, 2]])).toThrow(
      "The standardized mean difference needs at least 2 subjects rated by every rater, found 1.",
    );
    expect(() =>
      standardizedMeanDifference([
        [1, Number.NaN],
        [2, 3],
      ]),
    ).toThrow("The rating of subject 1 by rater 2 must be a finite number or null, found NaN.");
    expect(() => standardizedMeanDifference(JSON.parse("null"))).toThrow(
      "The standardized mean difference needs an array of ratings, one row per subject.",
    );
  });
});

describe("manyFacetRasch", () => {
  for (const dataset of manyFacetFixture.cases) {
    for (const reference of dataset.models) {
      const label = `${reference.facets.join(" + ")}${reference.thresholdFacet === null ? "" : `, thresholds by ${reference.thresholdFacet}`}`;
      it(`matches the base R maximum of the marginal likelihood on ${dataset.name}: ${label}`, () => {
        const model = manyFacetRasch(scoresOf(dataset.scores), {
          facets: reference.facets,
          ...(reference.thresholdFacet === null ? {} : { thresholdFacet: reference.thresholdFacet }),
          ...PRECISE,
        });
        expect(model.converged).toBe(true);
        expect(model.thresholdFacet).toBe(reference.thresholdFacet);
        expectClose(model.logLikelihood, reference.logLikelihood, LOG_LIKELIHOOD_TOLERANCE, "log-likelihood");
        expectClose(model.population.mean, reference.mean, ESTIMATE_TOLERANCE, "mean");
        expectClose(model.population.sd, reference.sd, ESTIMATE_TOLERANCE, "sd");
        expectClose(model.aic, -2 * reference.logLikelihood + 2 * reference.parameters, 1e-6, "AIC");
        reference.summaries.forEach((expected, f) => {
          const facet = model.facets[f];
          expect(facet?.facet).toBe(expected.facet);
          expect(facet?.elements.map((element) => element.id)).toEqual(expected.ids);
          expected.ids.forEach((id, e) => {
            const element = facet?.elements[e];
            const at = `${expected.facet} ${id}`;
            expectClose(element?.measure ?? null, expected.measures[e] ?? null, ESTIMATE_TOLERANCE, at);
            const error = expected.standardErrors[e] ?? 0;
            expectClose(element?.standardError ?? null, error, STANDARD_ERROR_TOLERANCE * error, `${at} SE`);
            expect(element?.count, at).toBe(expected.counts[e]);
            expectClose(element?.observedAverage ?? null, expected.observedAverages[e] ?? null, 1e-12, `${at} average`);
            expectClose(
              element?.fairAverage ?? null,
              expected.fairAverages[e] ?? null,
              ESTIMATE_TOLERANCE,
              `${at} fair average`,
            );
          });
          const { separation, reliability, homogeneity } = expected;
          expectClose(facet?.separation ?? null, separation, STANDARD_ERROR_TOLERANCE * separation, "separation");
          expectClose(facet?.reliability ?? null, reliability, STANDARD_ERROR_TOLERANCE * reliability, "reliability");
          const chiSquare = facet?.homogeneity?.chiSquare ?? null;
          expectClose(chiSquare, homogeneity.chiSquare, STANDARD_ERROR_TOLERANCE * homogeneity.chiSquare, "chi-square");
        });
        reference.thresholds.forEach((expected, s) => {
          const set = model.thresholds[s];
          expect(set?.element).toBe(expected.element);
          expected.thresholds.forEach((tau, j) => {
            expectClose(set?.thresholds[j] ?? null, tau, ESTIMATE_TOLERANCE, `threshold ${j + 1} of set ${s + 1}`);
            const error = expected.standardErrors[j] ?? 0;
            expectClose(set?.standardErrors?.[j] ?? null, error, STANDARD_ERROR_TOLERANCE * error, "threshold SE");
          });
        });
      });
    }
  }

  it("leaves the standard errors, separation and homogeneity out with standardErrors none", () => {
    const model = manyFacetRasch(simulatedScores(3, 60), { facets: ["item", "rater"], standardErrors: "none" });
    for (const facet of model.facets) {
      expect(facet.separation).toBeNull();
      expect(facet.homogeneity).toBeNull();
      expect(facet.elements.every(({ standardError }) => standardError === null)).toBe(true);
    }
    expect(model.thresholds[0]?.standardErrors).toBeNull();
  });

  it("gives the one element of a facet the measure 0, and a two-category scale the threshold 0", () => {
    const dichotomous = simulatedScores(5, 80).map(({ levels, score }) => ({
      levels: { ...levels, task: "essay" },
      score: score > 1 ? 1 : 0,
    }));
    const model = manyFacetRasch(dichotomous, { facets: ["item", "rater", "task"] });
    expect(model.facets[2]?.elements).toMatchObject([{ id: "essay", measure: 0, standardError: 0 }]);
    expect(model.facets[2]?.separation).toBeNull();
    expect(model.thresholds[0]?.thresholds).toEqual([0]);
    expect(model.thresholds[0]?.standardErrors).toEqual([0]);
    expect(model.converged).toBe(true);
  });

  it("takes repeated scores of a person on one combination as independent observations", () => {
    const scores = simulatedScores(9, 60);
    const model = manyFacetRasch([...scores, ...scores], { facets: ["item", "rater"] });
    expect(model.converged).toBe(true);
    expect(model.facets[0]?.elements[0]?.count).toBe(2 * scores.filter(({ levels }) => levels.item === "i0").length);
  });

  it("accepts the Gauss-Hermite rule and a thresholdFacet of the person's raters", () => {
    const scores = simulatedScores(11, 90);
    const rectangular = manyFacetRasch(scores, { facets: ["item", "rater"], ...PRECISE });
    const hermite = manyFacetRasch(scores, {
      facets: ["item", "rater"],
      quadrature: { method: "gauss-hermite", points: 31 },
      ...PRECISE,
    });
    rectangular.facets.forEach((facet, f) => {
      facet.elements.forEach((element, e) => {
        expectClose(hermite.facets[f]?.elements[e]?.measure ?? null, element.measure, 0.02, element.id);
      });
    });
    const byRater = manyFacetRasch(scores, { facets: ["item", "rater"], thresholdFacet: "rater" });
    expect(byRater.thresholds.map(({ element }) => element)).toEqual(["r0", "r1", "r2"]);
  });

  it("names the subsets of a design that is not connected", () => {
    const scores: FacetedScore[] = [
      ...["a", "b"].flatMap((person, i) =>
        [0, 1].map((score) => ({ levels: { person, item: "i1", rater: "r1" }, score: score + i - i })),
      ),
      ...["c", "d"].flatMap((person) =>
        [0, 1].map((score) => ({ levels: { person, item: "i2", rater: "r2" }, score })),
      ),
    ];
    expect(() => manyFacetRasch(scores, { facets: ["item", "rater"] })).toThrow(
      /^The scores fall into 2 subsets that share no person, item or rater, so their measures are not on one scale \(subset 1: person "a", person "b", item "i1" and 1 more; subset 2: person "c", person "d", item "i2" and 1 more\)\.$/,
    );
  });

  it("rejects scores that are not integers from 0, lack a facet, or leave a category unused", () => {
    const base = simulatedScores(2, 20);
    const first = base[0];
    if (first === undefined) return;
    const options = { facets: ["item", "rater"] };
    expect(() => manyFacetRasch([...base.slice(0, 2), { ...first, score: 1.5 }], options)).toThrow(
      "Score 3 needs an integer score from 0, found 1.5.",
    );
    expect(() => manyFacetRasch([{ ...first, score: -1 }], options)).toThrow(
      "Score 1 needs an integer score from 0, found -1.",
    );
    expect(() => manyFacetRasch([first, { levels: { person: "p1", item: "i0" }, score: 1 }], options)).toThrow(
      'Score 2 needs a string element for the facet "rater", found undefined.',
    );
    expect(() => manyFacetRasch([first], options)).toThrow("The scores need at least two persons, found 1.");
    const skipped = base.map((entry) => ({ ...entry, score: entry.score === 2 ? 3 : entry.score }));
    expect(() => manyFacetRasch(skipped, options)).toThrow(
      "Category 2 has no scores in the rating scale, but category 3 has: its threshold cannot be estimated.",
    );
    const zeros = base.map((entry) => ({ ...entry, score: 0 }));
    expect(() => manyFacetRasch(zeros, options)).toThrow(
      "The scores of the rating scale need at least two categories, found only category 0.",
    );
    const gap = base.map((entry) =>
      entry.levels.rater === "r1" && entry.score === 1 ? { ...entry, score: 2 } : entry,
    );
    expect(() => manyFacetRasch(gap, { ...options, thresholdFacet: "rater" })).toThrow(
      'Category 1 has no scores in the thresholds of the rater "r1", but category 3 has: its threshold cannot be estimated.',
    );
  });

  it("gives each element of the threshold facet the categories its scores use", () => {
    const capped = simulatedScores(4, 90).map((entry) =>
      entry.levels.rater === "r0" ? { ...entry, score: Math.min(entry.score, 1) } : entry,
    );
    const model = manyFacetRasch(capped, { facets: ["item", "rater"], thresholdFacet: "rater" });
    expect(model.thresholds.map(({ thresholds }) => thresholds.length)).toEqual([1, 3, 3]);
    expect(model.thresholds[0]?.thresholds[0]).toBeCloseTo(0, 12);
    expect(() => manyFacetRasch(capped, { facets: ["item", "rater"] })).not.toThrow();
  });

  it("reports that EM has not converged when it runs out of iterations", () => {
    const model = manyFacetRasch(simulatedScores(6, 60), {
      facets: ["item", "rater"],
      tolerance: 1e-14,
      maxIterations: 3,
    });
    expect(model.converged).toBe(false);
    expect(model.iterations).toBe(3);
  });

  it("rejects invalid options", () => {
    const scores = simulatedScores(2, 20);
    const facets = ["item", "rater"];
    expect(() => manyFacetRasch(scores, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => manyFacetRasch(scores, { facets: JSON.parse('"item"') })).toThrow(
      'The option facets must be an array of facet names, such as ["item", "rater"], found "item".',
    );
    expect(() => manyFacetRasch(scores, { facets: [] })).toThrow("The option facets needs at least one facet.");
    expect(() => manyFacetRasch(scores, { facets: ["item", "item"] })).toThrow(
      'The person facet and the option facets need different names, found "item" twice.',
    );
    expect(() => manyFacetRasch(scores, { facets: ["person", "rater"] })).toThrow(
      'The person facet and the option facets need different names, found "person" twice.',
    );
    expect(() => manyFacetRasch(scores, { facets: JSON.parse('["item", 3]') })).toThrow(
      "Facet 2 of the option facets must be a non-empty string, found 3.",
    );
    expect(() => manyFacetRasch(scores, { facets, person: JSON.parse("3") })).toThrow(
      "The option person must be a non-empty string, found 3.",
    );
    expect(() => manyFacetRasch(scores, { facets, thresholdFacet: "task" })).toThrow(
      'The option thresholdFacet must be "item" or "rater", found "task".',
    );
    expect(() => manyFacetRasch(scores, { facets, standardErrors: JSON.parse('"exact"') })).toThrow(
      'The option standardErrors must be "none" or "hessian", found "exact".',
    );
    expect(() => manyFacetRasch(scores, { facets, tolerance: -1 })).toThrow(DataError);
    expect(() => manyFacetRasch(scores, { facets, maxIterations: JSON.parse('"many"') })).toThrow(DataError);
    expect(() => manyFacetRasch(scores, { facets, quadrature: JSON.parse("null") })).toThrow(DataError);
    expect(() => manyFacetRasch(scores, { facets, quadrature: { points: 1 } })).toThrow(
      "The quadrature points must be an integer from 2, found 1.",
    );
    expect(() => manyFacetRasch(JSON.parse("null"), { facets })).toThrow(DataError);
  });
});

describe("manyFacetEstimates", () => {
  for (const dataset of manyFacetFixture.cases) {
    for (const reference of dataset.models) {
      it(`matches the base R EAP estimates at the maximum on ${dataset.name}: ${reference.facets.join(" + ")}`, () => {
        const estimates = manyFacetEstimates(scoresOf(dataset.scores), modelOf(reference));
        expect(estimates.map(({ id }) => id)).toEqual(reference.persons.id);
        estimates.forEach((estimate, i) => {
          expectClose(estimate.theta, reference.persons.theta[i] ?? null, DERIVED_TOLERANCE, `theta ${i + 1}`);
          expectClose(estimate.standardError, reference.persons.standardError[i] ?? null, DERIVED_TOLERANCE, "SE");
        });
      });
    }
  }

  it("gives the estimator each person's items and responses, so that any estimator works", () => {
    const [dataset] = manyFacetFixture.cases;
    const [reference] = dataset?.models ?? [];
    if (dataset === undefined || reference === undefined) return;
    const scores = scoresOf(dataset.scores);
    const seen = new Map<string, number>();
    const estimates = manyFacetEstimates(scores, modelOf(reference), (items, responses) => {
      expect(items).toHaveLength(responses.length);
      seen.set(String(seen.size), items.length);
      return { theta: items.length, standardError: 1 };
    });
    const answered = new Map<string, number>();
    for (const { levels } of scores) answered.set(levels.person ?? "", (answered.get(levels.person ?? "") ?? 0) + 1);
    expect(estimates.map(({ theta }) => theta)).toEqual([...answered.values()]);
  });

  it("gives maximum likelihood estimates at which the expected scores add up to the raw score", () => {
    const [dataset] = manyFacetFixture.cases;
    const reference = dataset?.models[1];
    if (dataset === undefined || reference === undefined) return;
    const model = modelOf(reference);
    const scores = scoresOf(dataset.scores);
    const estimates = manyFacetEstimates(scores, model, (items, responses) =>
      maximumLikelihoodEstimate(items, responses),
    );
    estimates.forEach(({ id, theta }) => {
      const own = scores.filter(({ levels }) => levels.person === id);
      const raw = own.reduce((sum, { score }) => sum + score, 0);
      const maximum = own.length * (reference.thresholds[0]?.thresholds.length ?? 0);
      if (theta === null || raw === 0 || raw === maximum) return;
      const expected = own.reduce(
        (sum, { levels }) => sum + expectedRating(model, theta, { item: levels.item ?? "", rater: levels.rater ?? "" }),
        0,
      );
      expect(Math.abs(expected - raw), id).toBeLessThan(1e-4);
    });
  });

  it("rejects scores with an element the model lacks, a score beyond the categories and an estimator that is not a function", () => {
    const [dataset] = manyFacetFixture.cases;
    const reference = dataset?.models[0];
    if (dataset === undefined || reference === undefined) return;
    const model = modelOf(reference);
    const scores = scoresOf(dataset.scores).slice(0, 5);
    const unknown = scores.map((entry, i) =>
      i === 0 ? { ...entry, levels: { ...entry.levels, rater: "nobody" } } : entry,
    );
    expect(() => manyFacetEstimates(unknown, model)).toThrow('The facet "rater" of the model has no element "nobody".');
    const beyond = scores.map((entry, i) => (i === 1 ? { ...entry, score: 9 } : entry));
    expect(() => manyFacetEstimates(beyond, model)).toThrow("Score 2 is 9, but its thresholds give categories 0 to 3.");
    expect(() => manyFacetEstimates(scores, model, JSON.parse("null"))).toThrow(
      "The estimator must be a function of items and responses, such as posteriorMeanEstimate, found null.",
    );
    expect(() => manyFacetEstimates(JSON.parse("null"), model)).toThrow(DataError);
  });
});

describe("manyFacetFit", () => {
  for (const dataset of manyFacetFixture.cases) {
    for (const reference of dataset.models) {
      it(`matches base R's mean squares at the same person estimates on ${dataset.name}: ${reference.facets.join(" + ")}`, () => {
        const persons: PersonEstimate[] = reference.persons.id.map((id, i) => ({
          id,
          theta: reference.persons.theta[i] ?? null,
          standardError: reference.persons.standardError[i] ?? null,
        }));
        const fit = manyFacetFit(scoresOf(dataset.scores), modelOf(reference), persons);
        expect(fit).toHaveLength(reference.fit.length);
        fit.forEach((element, i) => {
          const expected = reference.fit[i];
          expect([element.facet, element.id]).toEqual([expected?.facet, expected?.id]);
          expectClose(element.infitMeanSquare, expected?.infitMeanSquare ?? null, DERIVED_TOLERANCE, "infit");
          expectClose(element.outfitMeanSquare, expected?.outfitMeanSquare ?? null, DERIVED_TOLERANCE, "outfit");
          expectClose(element.infitStandardized, expected?.infitStandardized ?? null, DERIVED_TOLERANCE, "infit z");
          expectClose(element.outfitStandardized, expected?.outfitStandardized ?? null, DERIVED_TOLERANCE, "outfit z");
        });
      });
    }
  }

  it("skips the scores of persons without an estimate and gives null statistics to an element without scores", () => {
    const [dataset] = manyFacetFixture.cases;
    const reference = dataset?.models[0];
    if (dataset === undefined || reference === undefined) return;
    const fit = manyFacetFit(scoresOf(dataset.scores), modelOf(reference), []);
    for (const element of fit) {
      expect(element.infitMeanSquare).toBeNull();
      expect(element.outfitStandardized).toBeNull();
    }
  });

  it("rejects estimates that are not objects with an id and a theta that is a number or null", () => {
    const [dataset] = manyFacetFixture.cases;
    const reference = dataset?.models[0];
    if (dataset === undefined || reference === undefined) return;
    const model = modelOf(reference);
    const scores = scoresOf(dataset.scores);
    expect(() => manyFacetFit(scores, model, JSON.parse("null"))).toThrow(
      "The persons must be an array of estimates { id, theta, standardError }, found null.",
    );
    expect(() => manyFacetFit(scores, model, JSON.parse('[{"id": "p1", "theta": "high"}]'))).toThrow(
      'Estimate 1 needs a string id and a theta that is a finite number or null, found "high".',
    );
  });
});

describe("expectedRating", () => {
  for (const dataset of manyFacetFixture.cases) {
    for (const reference of dataset.models) {
      it(`matches base R on ${dataset.name}: ${reference.facets.join(" + ")}`, () => {
        const model = modelOf(reference);
        for (const probe of reference.expectedRatings) {
          const levels: Record<string, string> = Object.fromEntries(Object.entries(probe.levels));
          expectClose(expectedRating(model, probe.theta, levels), probe.value, DERIVED_TOLERANCE, "levels");
          expectClose(expectedRating(model, probe.theta), probe.fair, DERIVED_TOLERANCE, "fair score");
        }
      });
    }
  }

  it("increases with theta from 0 to the highest category, and is higher from a milder rater", () => {
    const [dataset] = manyFacetFixture.cases;
    const reference = dataset?.models[0];
    if (dataset === undefined || reference === undefined) return;
    const model = modelOf(reference);
    const highest = reference.thresholds[0]?.thresholds.length ?? 0;
    expect(expectedRating(model, -40)).toBeLessThan(1e-6);
    expect(highest - expectedRating(model, 40)).toBeLessThan(1e-6);
    const raters = model.facets[1]?.elements ?? [];
    const harshest = raters.reduce((worst, rater) => (rater.measure > worst.measure ? rater : worst));
    const mildest = raters.reduce((best, rater) => (rater.measure < best.measure ? rater : best));
    expect(expectedRating(model, 0, { rater: mildest.id })).toBeGreaterThan(
      expectedRating(model, 0, { rater: harshest.id }),
    );
    let previous = -1;
    for (let theta = -4; theta <= 4; theta += 0.5) {
      const value = expectedRating(model, theta);
      expect(value).toBeGreaterThan(previous);
      previous = value;
    }
  });

  it("rejects a theta that is not finite, levels that name another facet or element, and a model that is not one", () => {
    const [dataset] = manyFacetFixture.cases;
    const reference = dataset?.models[0];
    if (dataset === undefined || reference === undefined) return;
    const model = modelOf(reference);
    expect(() => expectedRating(model, Number.NaN)).toThrow("Theta must be a finite number, found NaN.");
    expect(() => expectedRating(model, 0, { task: "t1" })).toThrow(
      'The levels name the facet "task", but the model has "item", "rater".',
    );
    expect(() => expectedRating(model, 0, { rater: "nobody" })).toThrow(
      'The facet "rater" of the model has no element "nobody".',
    );
    expect(() => expectedRating(model, 0, JSON.parse('{"rater": 3}'))).toThrow(
      'The level of the facet "rater" must be an element id, found 3.',
    );
    expect(() => expectedRating(model, 0, JSON.parse("null"))).toThrow(
      "The levels must be an object of elements by facet, found null.",
    );
    expect(() => expectedRating(JSON.parse("null"), 0)).toThrow(DataError);
    expect(() => expectedRating({ ...model, facets: JSON.parse('"item"') }, 0)).toThrow(
      'The model needs facets, an array of { facet, elements }, found "item".',
    );
    expect(() => expectedRating({ ...model, population: { mean: 0, sd: 0 } }, 0)).toThrow(
      "The model's population needs a finite mean and a positive finite sd.",
    );
    expect(() => expectedRating({ ...model, thresholds: [] }, 0)).toThrow(
      "The model needs one set of thresholds, with the element null, for the shared scale.",
    );
  });
});

describe("facet summaries", () => {
  for (const dataset of manyFacetFixture.cases) {
    for (const reference of dataset.models) {
      it(`give base R's separation, reliability and homogeneity of the same measures on ${dataset.name}: ${reference.facets.join(" + ")}`, () => {
        for (const expected of reference.summaries) {
          const summary = summarizeFacet(
            expected.facet,
            expected.ids.map((id, e) => ({
              id,
              measure: expected.measures[e] ?? 0,
              standardError: expected.standardErrors[e] ?? null,
              count: expected.counts[e] ?? 0,
              observedAverage: expected.observedAverages[e] ?? 0,
              fairAverage: expected.fairAverages[e] ?? 0,
            })),
          );
          expectClose(summary.separation, expected.separation, DERIVED_TOLERANCE, "separation");
          expectClose(summary.reliability, expected.reliability, DERIVED_TOLERANCE, "reliability");
          expectClose(summary.homogeneity?.chiSquare ?? null, expected.homogeneity.chiSquare, 1e-9, "chi-square");
          expect(summary.homogeneity?.degreesOfFreedom).toBe(expected.homogeneity.degreesOfFreedom);
          expectClose(
            summary.homogeneity?.significance ?? null,
            expected.homogeneity.significance,
            DERIVED_TOLERANCE,
            "p",
          );
        }
      });
    }
  }

  it("are null for one element, without standard errors, and give 0 separation when the errors explain the spread", () => {
    const element = (measure: number, standardError: number | null) => ({
      id: String(measure),
      measure,
      standardError,
      count: 1,
      observedAverage: 0,
      fairAverage: 0,
    });
    expect(summarizeFacet("rater", [element(0, 0)])).toMatchObject({
      separation: null,
      reliability: null,
      homogeneity: null,
    });
    expect(summarizeFacet("rater", [element(-1, null), element(1, null)])).toMatchObject({
      separation: null,
      homogeneity: null,
    });
    const noisy = summarizeFacet("rater", [element(-0.1, 1), element(0.1, 1)]);
    expect(noisy.separation).toBe(0);
    expect(noisy.reliability).toBe(0);
    expect(noisy.homogeneity?.chiSquare).toBeCloseTo(0.02, 12);
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

  /** Measures by facet and element. */
  function measuresOf(model: ManyFacetRasch): Map<string, number> {
    return new Map(
      model.facets.flatMap((facet) =>
        facet.elements.map((element) => [`${facet.facet}:${element.id}`, element.measure]),
      ),
    );
  }

  it("many-facet measures do not depend on the order of the scores", () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 1_000_000 }), (seed) => {
        const scores = simulatedScores(seed, 40);
        const options = { facets: ["item", "rater"], ...PRECISE };
        const forward = measuresOf(manyFacetRasch(scores, options));
        const backward = measuresOf(manyFacetRasch([...scores].reverse(), options));
        for (const [key, measure] of forward) expectClose(backward.get(key) ?? null, measure, 1e-7, key);
      }),
      { numRuns: 5 },
    );
  });

  it("reversing the scale reverses the many-facet measures and thresholds", () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 1_000_000 }), (seed) => {
        const scores = simulatedScores(seed, 40);
        const options = { facets: ["item", "rater"], ...PRECISE };
        const model = manyFacetRasch(scores, options);
        const reversed = manyFacetRasch(
          scores.map(({ levels, score }) => ({ levels, score: 3 - score })),
          options,
        );
        for (const [key, measure] of measuresOf(model))
          expectClose(measuresOf(reversed).get(key) ?? null, -measure, 1e-6, key);
        const thresholds = model.thresholds[0]?.thresholds ?? [];
        thresholds.forEach((tau, j) => {
          expectClose(reversed.thresholds[0]?.thresholds[2 - j] ?? null, -tau, 1e-6, `threshold ${j}`);
        });
        expectClose(reversed.population.mean, -model.population.mean, 1e-6, "mean");
        expectClose(reversed.logLikelihood, model.logLikelihood, 1e-6, "log-likelihood");
      }),
      { numRuns: 5 },
    );
  });

  it("many-facet measures sum to 0 within each facet, and fair averages lie between the categories", () => {
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 1_000_000 }), (seed) => {
        const model = manyFacetRasch(simulatedScores(seed, 40), { facets: ["item", "rater"], standardErrors: "none" });
        for (const facet of model.facets) {
          expectClose(
            facet.elements.reduce((sum, { measure }) => sum + measure, 0),
            0,
            1e-9,
            facet.facet,
          );
          for (const { fairAverage } of facet.elements) {
            expect(fairAverage).toBeGreaterThan(0);
            expect(fairAverage).toBeLessThan(3);
          }
        }
        expectClose(model.thresholds[0]?.thresholds.reduce((sum, tau) => sum + tau, 0) ?? null, 0, 1e-9, "thresholds");
      }),
      { numRuns: 5 },
    );
  });

  it("agreement grows with the width, lies from 0 to 1 and is 1 once the width spans the categories", () => {
    fc.assert(
      fc.property(ratings(3), (rows) => {
        let previous = exactAgreement(rows);
        expect(previous).toBeGreaterThanOrEqual(0);
        for (const width of [1, 2, 3, 4]) {
          const value = adjacentAgreement(rows, { width });
          expect(value).toBeGreaterThanOrEqual(previous);
          expect(value).toBeLessThanOrEqual(1);
          previous = value;
        }
        expect(previous).toBe(1);
      }),
    );
  });

  it("agreement and the standardized mean difference do not depend on the order of subjects", () => {
    fc.assert(
      fc.property(ratings(2), (rows) => {
        const reversed = [...rows].reverse();
        expect(adjacentAgreement(reversed)).toBe(adjacentAgreement(rows));
        try {
          expectClose(standardizedMeanDifference(reversed), standardizedMeanDifference(rows), 1e-12);
        } catch (error) {
          if (!(error instanceof DataError)) throw error;
        }
      }),
    );
  });
});

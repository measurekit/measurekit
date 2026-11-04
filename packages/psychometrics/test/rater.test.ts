import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  adjacentAgreement,
  cohenKappa,
  createRandom,
  DataError,
  exactAgreement,
  expectedRating,
  type FacetBias,
  type FacetedScore,
  fleissKappa,
  intraclassCorrelation,
  type ManyFacetRasch,
  manyFacetEstimates,
  manyFacetFit,
  manyFacetRasch,
  maximumLikelihoodEstimate,
  type PersonEstimate,
  ratingCorrelation,
  standardizedMeanDifference,
} from "../src/index.js";
import { standardDeviation } from "../src/math/internal.js";
import { summarizeFacet } from "../src/rater/internal.js";
import { defined, expectClose } from "./assertions.js";
import { type RaterAutomatedScoringFixture, readFixture } from "./fixtures.js";

interface RaterFixture {
  cases: {
    name: string;
    ratings: (number | null)[][];
    cohen: { none: number; linear: number; quadratic: number };
    fleiss: number;
    icc: {
      oneWay: number;
      agreement: number;
      consistency: number;
      oneWayAverage: number;
      agreementAverage: number;
      consistencyAverage: number;
    };
  }[];
}
const fixture = readFixture<RaterFixture>("rater.json");

interface RaterAgreementFixture {
  agreement: {
    name: string;
    ratings: (number | null)[][];
    subjects: number;
    exact: number;
    adjacent: number;
    widths: { width: number; agreement: number }[];
  }[];
  meanDifference: { name: string; ratings: (number | null)[][]; value: number }[];
}
const agreementFixture = readFixture<RaterAgreementFixture>("rater-agreement.json");

const scoringFixture = readFixture<RaterAutomatedScoringFixture>("rater-automated-scoring.json");

interface RaterBiasFixture {
  scores: { levels: { person: string; item: string; rater: string }; score: number }[];
  cells: { levels: string[]; bias: number; standardError: number; truth: number; tamBias: number }[];
  logLikelihood: number;
  tamLogLikelihood: number;
  mean: number;
  sd: number;
}
const biasFixture = readFixture<RaterBiasFixture>("rater-bias.json");

interface RaterCorrelationFixture {
  cases: { name: string; ratings: (number | null)[][]; value: number | null }[];
}
const correlationFixture = readFixture<RaterCorrelationFixture>("rater-correlation.json");

interface RaterManyFacetFixture {
  cases: {
    name: string;
    scores: { person: string[]; item: string[]; rater: string[]; score: number[] };
    models: {
      facets: string[];
      thresholdFacet: string | null;
      summaries: {
        facet: string;
        ids: string[];
        measures: number[];
        standardErrors: number[];
        counts: number[];
        observedAverages: number[];
        fairAverages: number[];
        separation: number;
        reliability: number;
        homogeneity: { chiSquare: number; degreesOfFreedom: number; significance: number };
      }[];
      thresholds: { element: string | null; thresholds: number[]; standardErrors: number[] }[];
      mean: number;
      sd: number;
      logLikelihood: number;
      parameters: number;
      persons: { id: string[]; theta: number[]; standardError: number[] };
      fit: {
        facet: string;
        id: string;
        infitMeanSquare: number;
        outfitMeanSquare: number;
        infitStandardized: number;
        outfitStandardized: number;
      }[];
      expectedRatings: { theta: number; levels: { item: string; rater?: string }; value: number; fair: number }[];
    }[];
  }[];
}
const manyFacetFixture = readFixture<RaterManyFacetFixture>("rater-many-facet.json");

/** Kappa and intraclass correlations against irr: closed forms of the same counts and sums. */
const TOLERANCE = 1e-12;
/** Agreement shares, standardized mean differences and correlations against irr, effsize and cor: closed forms of the same counts and sums. */
const AGREEMENT_TOLERANCE = 1e-12;
/** The evaluation of automated scores against RSMTool: closed forms of the same counts and sums. */
const SCORING_TOLERANCE = 1e-10;
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

/** RSMTool's metrics of one pair of scores, agreement in percent. */
interface ScoringMetrics {
  readonly kappa: number | null;
  readonly quadraticKappa: number | null;
  readonly exactPercent: number | null;
  readonly adjacentPercent: number | null;
  readonly smd: number | null;
  readonly correlation: number | null;
  readonly rSquared: number | null;
  readonly rmse: number | null;
  readonly machineSd: number | null;
  readonly humanSd: number | null;
}

/** The PERSUADE 2.0 case of the fixture: one human score, a machine score and a group per essay. */
const PERSUADE = scoringFixture.cases["PERSUADE 2.0"];
/** The ELLIPSE case of the fixture: two human scores and a machine score per essay. */
const ELLIPSE = scoringFixture.cases.ELLIPSE;

/** The recipe of rater.md: machine scores rounded and trimmed to the scale. */
function toScale(scores: readonly number[], [low = 0, high = 0]: readonly number[]): number[] {
  return scores.map((score) => Math.min(high, Math.max(low, Math.round(score))));
}

/** The recipe of rater.md: the metrics of a second rating against the first, the human, as rows [second, first]. */
function evaluate(human: readonly number[], machine: readonly number[]): ScoringMetrics {
  const pairs = machine.map((score, i) => [score, human[i] ?? null]);
  const squares = machine.reduce((sum, score, i) => sum + (score - (human[i] ?? 0)) ** 2, 0);
  const humanMean = human.reduce((sum, score) => sum + score, 0) / human.length;
  const spread = human.reduce((sum, score) => sum + (score - humanMean) ** 2, 0);
  return {
    kappa: cohenKappa(pairs),
    quadraticKappa: cohenKappa(pairs, { weights: "quadratic" }),
    exactPercent: 100 * exactAgreement(pairs),
    adjacentPercent: 100 * adjacentAgreement(pairs),
    smd: standardizedMeanDifference(pairs),
    correlation: ratingCorrelation(pairs),
    rSquared: 1 - squares / spread,
    rmse: Math.sqrt(squares / human.length),
    machineSd: standardDeviation(Float64Array.from(machine)),
    humanSd: standardDeviation(Float64Array.from(human)),
  };
}

/** Expects the recipe's metrics to be RSMTool's, agreement in percent at 100 times the tolerance. */
function expectMetrics(actual: ScoringMetrics, expected: ScoringMetrics, label: string): void {
  for (const [name, value] of Object.entries(expected)) {
    const found: unknown = Object.entries(actual).find(([key]) => key === name)?.[1];
    const scale = name.endsWith("Percent") ? 100 : 1;
    expectClose(typeof found === "number" ? found : null, value, scale * SCORING_TOLERANCE, `${label} ${name}`);
  }
}

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
    // The means of 0.1 and 0.7 repeated round apart from the ratings, which left mean squares of rounding: 1.6, 2.
    for (const rating of [0.1, 0.7]) {
      const fractional = intraclassCorrelation(Array.from({ length: 3 }, () => [rating, rating, rating]));
      for (const field of FIELDS) expect(fractional[field]).toBeNull();
    }
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
    expect(() => cohenKappa([[0, 1]], JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
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
    expect(() => adjacentAgreement(ratings, JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
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
    const value = defined(standardizedMeanDifference(ratings));
    expectClose(standardizedMeanDifference(ratings.map(([x = 0, y = 0]) => [y, x])), -value, 1e-15);
    expectClose(standardizedMeanDifference(ratings.map(([x = 0, y = 0]) => [3 * x + 7, 3 * y + 7])), value, 1e-12);
    expectClose(
      standardizedMeanDifference(ratings.map(([x = 0, y = 0]) => [x + 1e6 + 0.1, y + 1e6 + 0.1])),
      value,
      1e-9,
    );
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

  it("is null when both raters' ratings are constant, whole or not, and defined when only one is", () => {
    expect(
      standardizedMeanDifference([
        [2, 2],
        [2, 2],
      ]),
    ).toBeNull();
    // 0.7 repeated has a variance of -2.4e-16 by the sums, not 0.
    const decimals = Array.from({ length: 10 }, () => [0.7, 0.3]);
    expect(standardizedMeanDifference(decimals)).toBeNull();
    expectClose(
      standardizedMeanDifference([
        [1, 3],
        [2, 3],
        [3, 3],
      ]),
      -Math.SQRT2,
      1e-15,
    );
  });

  it("rejects other numbers of raters, too few subjects and invalid ratings", () => {
    expect(() =>
      standardizedMeanDifference([
        [1, 2, 3],
        [2, 3, 4],
      ]),
    ).toThrow("The standardized mean difference needs 2 raters, found 3.");
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

  it("matches RSMTool's Williamson SMD of each subgroup on the population's deviations", () => {
    const { human, machine: predicted, groups, scale, byGroup } = PERSUADE;
    const machine = toScale(predicted, scale);
    const population = {
      first: standardDeviation(Float64Array.from(machine)),
      second: standardDeviation(Float64Array.from(human)),
    };
    for (const { group, subjects, smd } of byGroup) {
      const rows = machine.flatMap((score, i) => (groups[i] === group ? [[score, human[i] ?? null]] : []));
      expect(rows, group).toHaveLength(subjects);
      expectClose(standardizedMeanDifference(rows, { population }), smd, SCORING_TOLERANCE, group);
    }
  });

  it("divides by the population's pooled deviation, derived by hand, and is null when it is 0", () => {
    const rows = [
      [3, 2],
      [4, 4],
    ];
    // Means 3.5 and 3 over √((1² + 3²) / 2) = √5.
    expectClose(standardizedMeanDifference(rows, { population: { first: 1, second: 3 } }), 0.5 / Math.sqrt(5), 1e-15);
    expect(standardizedMeanDifference(rows, { population: { first: 0, second: 0 } })).toBeNull();
    const constant = [
      [2, 2],
      [2, 2],
    ];
    expect(standardizedMeanDifference(constant, { population: { first: 1, second: 1 } })).toBe(0);
  });

  it("rejects population deviations that are not non-negative numbers, and options of other types", () => {
    const rows = [
      [1, 2],
      [2, 3],
    ];
    expect(() => standardizedMeanDifference(rows, { population: { first: -1, second: 1 } })).toThrow(
      "The population's standard deviation of the first rater must be a non-negative number, found -1.",
    );
    expect(() =>
      standardizedMeanDifference(rows, JSON.parse('{ "population": { "first": 1, "second": "1" } }')),
    ).toThrow('The population\'s standard deviation of the second rater must be a non-negative number, found "1".');
    expect(() => standardizedMeanDifference(rows, JSON.parse('{ "population": [1, 1] }'))).toThrow(
      "The option population must be an object { first, second }",
    );
    expect(() =>
      standardizedMeanDifference(rows, JSON.parse('{ "population": { "first": 1, "second": 1, "third": 1 } }')),
    ).toThrow('The option population may hold only first and second, found "third".');
    expect(() => standardizedMeanDifference(rows, JSON.parse('{ "population": null }'))).toThrow(DataError);
    expect(() => standardizedMeanDifference(rows, JSON.parse("null"))).toThrow(DataError);
  });
});

describe("the recipe for evaluating automated scores", () => {
  it("gives RSMTool's metrics of machine against human scores on PERSUADE 2.0", () => {
    expectMetrics(evaluate(PERSUADE.human, toScale(PERSUADE.machine, PERSUADE.scale)), PERSUADE.overall, "PERSUADE");
  });

  it("gives RSMTool's metrics, human-human agreement, degradation and disattenuated correlation on ELLIPSE", () => {
    const { first, second, machine, scale, humanSystem, humanHuman, degradation, disattenuated } = ELLIPSE;
    const system = evaluate(first, toScale(machine, scale));
    const humans = evaluate(first, second);
    expectMetrics(system, humanSystem, "human-system");
    expectMetrics(humans, humanHuman, "human-human");
    for (const [name, value] of Object.entries(degradation)) {
      const lost = (metrics: ScoringMetrics): number =>
        defined(Object.entries(metrics).find(([key]) => key === name)?.[1]);
      const scale = name.endsWith("Percent") ? 100 : 1;
      expectClose(lost(system) - lost(humans), value, scale * SCORING_TOLERANCE, `degradation of ${name}`);
    }
    const correlation = defined(system.correlation) / Math.sqrt(defined(humans.correlation));
    expectClose(correlation, disattenuated, SCORING_TOLERANCE, "disattenuated r");
  });
});

describe("ratingCorrelation", () => {
  for (const reference of correlationFixture.cases) {
    it(`matches base R's cor on ${reference.name}`, () => {
      expectClose(ratingCorrelation(reference.ratings), reference.value, AGREEMENT_TOLERANCE);
    });
  }

  it("is the covariance over the product of the standard deviations, derived by hand", () => {
    // Means 2 and 4, deviations -1, 0, 1 for both: perfectly correlated.
    expect(
      ratingCorrelation([
        [1, 3],
        [2, 4],
        [3, 5],
      ]),
    ).toBe(1);
    expect(
      ratingCorrelation([
        [1, 5],
        [2, 4],
        [3, 3],
      ]),
    ).toBe(-1);
    // Deviations -1, 0, 1 and 0, -1, 1: covariance 1 over (1)(1) is 1 / √2 of the variances 1 and 1: r = 1 / 2.
    expect(
      ratingCorrelation([
        [1, 2],
        [2, 1],
        [3, 3],
      ]),
    ).toBeCloseTo(0.5, 15);
  });

  it("does not change with the order of the raters, and keeps its sign and size under shifts and positive scales", () => {
    const ratings = [
      [1, 2],
      [4, 3],
      [2, 2.5],
      [5, 6],
    ];
    const value = defined(ratingCorrelation(ratings));
    expectClose(ratingCorrelation(ratings.map(([x = 0, y = 0]) => [y, x])), value, 1e-15);
    expectClose(ratingCorrelation(ratings.map(([x = 0, y = 0]) => [3 * x + 7, 0.5 * y - 2])), value, 1e-12);
    expectClose(ratingCorrelation(ratings.map(([x = 0, y = 0]) => [-x, y])), -value, 1e-15);
  });

  it("uses the subjects both raters rated and accepts ratings that are not categories", () => {
    const rated = [
      [1.5, 2.25],
      [3, 2.5],
      [4.75, 4],
    ];
    expect(ratingCorrelation([...rated, [null, 9], [8, null]])).toBe(ratingCorrelation(rated));
  });

  it("is null when either rater's ratings are constant, whole or not", () => {
    expect(
      ratingCorrelation([
        [1, 2],
        [2, 2],
        [3, 2],
      ]),
    ).toBeNull();
    expect(ratingCorrelation(Array.from({ length: 10 }, (_, k) => [0.7, k]))).toBeNull();
    expect(ratingCorrelation(Array.from({ length: 10 }, (_, k) => [k, 0.7]))).toBeNull();
    expect(ratingCorrelation(Array.from({ length: 10 }, () => [0.7, 0.7]))).toBeNull();
  });

  it("rejects other numbers of raters, too few subjects and invalid ratings", () => {
    expect(() =>
      ratingCorrelation([
        [1, 2, 3],
        [2, 3, 4],
      ]),
    ).toThrow("The rating correlation needs 2 raters, found 3.");
    expect(() => ratingCorrelation([[1, 2]])).toThrow(
      "The rating correlation needs at least 2 subjects rated by every rater, found 1.",
    );
    expect(() =>
      ratingCorrelation([
        [1, Number.NaN],
        [2, 3],
      ]),
    ).toThrow("The rating of subject 1 by rater 2 must be a finite number or null, found NaN.");
    expect(() => ratingCorrelation(JSON.parse("null"))).toThrow(
      "The rating correlation needs an array of ratings, one row per subject.",
    );
    expect(() => ratingCorrelation(JSON.parse('[[1, "2"], [2, 3]]'))).toThrow(DataError);
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

  it("rejects scores that are not integers from 0 to 2³¹ - 1, lack a facet, or leave a category unused", () => {
    const base = simulatedScores(2, 20);
    const first = base[0];
    if (first === undefined) return;
    const options = { facets: ["item", "rater"] };
    expect(() => manyFacetRasch([...base.slice(0, 2), { ...first, score: 1.5 }], options)).toThrow(
      "Score 3 needs an integer score from 0 to 2147483647, found 1.5.",
    );
    expect(() => manyFacetRasch([{ ...first, score: -1 }], options)).toThrow(
      "Score 1 needs an integer score from 0 to 2147483647, found -1.",
    );
    expect(() => manyFacetRasch([...base.slice(0, 2), { ...first, score: 2 ** 32 + 1 }], options)).toThrow(
      "Score 3 needs an integer score from 0 to 2147483647, found 4294967297.",
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

  it("refuses a model of more than 1,000 parameters or a scale of more than 255 categories", () => {
    const options = { facets: ["item", "rater"], standardErrors: "none", maxIterations: 1 } as const;
    // Person p is scored by raters p and p + 1, round 1,000 raters: 999 measures, 1 threshold, the mean and the SD.
    const raters = Array.from({ length: 1000 }, (_, person) =>
      [0, 1].map((step) => ({
        levels: { person: `p${person}`, item: "i1", rater: `r${(person + step) % 1000}` },
        score: (person + step) % 3,
      })),
    ).flat();
    expect(() => manyFacetRasch(raters, options)).toThrow(
      "A many-facet model takes at most 1000 parameters, as its estimation inverts a matrix of every pair of them; found 1002.",
    );
    const wide = Array.from({ length: 256 }, (_, s) => ({
      levels: { person: `p${s % 2}`, item: `i${Math.floor(s / 34)}`, rater: `r${s % 34}` },
      score: s,
    }));
    expect(() => manyFacetRasch(wide, options)).toThrow(
      "A many-facet model takes at most 255 categories in a rating scale, found 256.",
    );
  });

  it("refuses 2^24 expected counts at once, as the estimation that follows would not end in minutes", () => {
    // 92 items by 92 raters, every cell scored once, with 2 categories and 1,000 quadrature points: 8,464 virtual items
    // of 2,000 expected counts each, 16,928,000 in all. With no bound the estimation of 185 parameters ends in seconds.
    const cells = Array.from({ length: 92 * 92 }, (_, cell) => ({
      levels: { person: `p${cell % 2}`, item: `i${Math.floor(cell / 92)}`, rater: `r${cell % 92}` },
      score: cell % 2,
    }));
    const options = { facets: ["item", "rater"], standardErrors: "none", maxIterations: 1 } as const;
    expect(() => manyFacetRasch(cells, { ...options, quadrature: { points: 1000 } })).toThrow(
      "A many-facet model holds at most 16777216 expected counts, its items × quadrature points × categories, found 16928000.",
    );
  });

  it("refuses an interaction of more biases than a model takes, before it solves them", () => {
    // 40 items by 30 raters, every cell scored: 39 and 29 measures and 1,131 biases.
    const scores = Array.from({ length: 40 * 30 }, (_, cell) => ({
      levels: { person: `p${cell % 3}`, item: `i${Math.floor(cell / 30)}`, rater: `r${cell % 30}` },
      score: cell % 2,
    }));
    const options = { facets: ["item", "rater"], interactions: [["item", "rater"]], standardErrors: "none" } as const;
    expect(() => manyFacetRasch(scores, { ...options, maxIterations: 1 })).toThrow(
      "A many-facet model takes at most 1000 parameters, as its estimation inverts a matrix of every pair of them; found 1201.",
    );
  });

  it("rejects invalid options", () => {
    const scores = simulatedScores(2, 20);
    const facets = ["item", "rater"];
    expect(() => manyFacetRasch(scores, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => manyFacetRasch(scores, JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
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

/** The biases of a result by the ids of their elements, `item` first. */
function biasesByCell(model: ManyFacetRasch): Map<string, FacetBias> {
  return new Map(
    (model.interactions ?? []).map((cell) => {
      const [item, rater] = cell.facets[0] === "item" ? cell.levels : [cell.levels[1], cell.levels[0]];
      return [`${item}:${rater}`, cell];
    }),
  );
}

/** The item and rater of a cell of the bias fixture. */
function cellKey(levels: readonly string[]): string {
  return `${levels[0]}:${levels[1]}`;
}

/** Sums the biases of the cells that share one element of the facet at `position` of `levels`. */
function sumsByElement(cells: readonly FacetBias[], position: 0 | 1): Map<string, number> {
  const sums = new Map<string, number>();
  for (const { levels, bias } of cells) sums.set(levels[position], (sums.get(levels[position]) ?? 0) + bias);
  return sums;
}

/** A stored model of two items and two raters with the thresholds -0.5 and 0.5, the measures and the biases given. */
function storedModel(biases: readonly { levels: [string, string]; bias: number }[], itemMeasure = 0.5): ManyFacetRasch {
  const element = (id: string, measure: number) => ({
    id,
    measure,
    standardError: null,
    count: 1,
    observedAverage: 1,
    fairAverage: 1,
  });
  const summary = { separation: null, reliability: null, homogeneity: null };
  return {
    person: "person",
    thresholdFacet: null,
    facets: [
      { facet: "item", elements: [element("i1", itemMeasure), element("i2", -0.5)], ...summary },
      { facet: "rater", elements: [element("r1", 0.3), element("r2", -0.3)], ...summary },
    ],
    thresholds: [{ element: null, thresholds: [-0.5, 0.5], standardErrors: null }],
    interactions: biases.map(({ levels, bias }) => ({
      facets: ["item", "rater"],
      levels,
      bias,
      standardError: null,
      z: null,
    })),
    population: { mean: 0, sd: 1 },
    logLikelihood: 0,
    aic: 0,
    bic: 0,
    converged: true,
    iterations: 0,
  };
}

describe("manyFacetRasch with interactions", () => {
  const interactions = [["item", "rater"]] as const;
  const model = manyFacetRasch(biasFixture.scores, { facets: ["item", "rater"], interactions, ...PRECISE });

  it("matches the biases, their standard errors and the likelihood of the base R maximum", () => {
    expect(model.converged).toBe(true);
    expectClose(model.logLikelihood, biasFixture.logLikelihood, LOG_LIKELIHOOD_TOLERANCE, "log-likelihood");
    expectClose(model.population.mean, biasFixture.mean, ESTIMATE_TOLERANCE, "mean");
    expectClose(model.population.sd, biasFixture.sd, ESTIMATE_TOLERANCE, "sd");
    const cells = biasesByCell(model);
    expect(cells.size).toBe(biasFixture.cells.length);
    for (const expected of biasFixture.cells) {
      const cell = cells.get(cellKey(expected.levels));
      const at = cellKey(expected.levels);
      expectClose(cell?.bias ?? null, expected.bias, ESTIMATE_TOLERANCE, `${at} bias`);
      expectClose(
        cell?.standardError ?? null,
        expected.standardError,
        STANDARD_ERROR_TOLERANCE * expected.standardError,
        `${at} SE`,
      );
      const z = expected.bias / expected.standardError;
      expectClose(cell?.z ?? null, z, STANDARD_ERROR_TOLERANCE * Math.abs(z), `${at} z`);
    }
  });

  it("agrees with the biases and the likelihood of TAM within 1e-3", () => {
    const cells = biasesByCell(model);
    for (const expected of biasFixture.cells) {
      expectClose(cells.get(cellKey(expected.levels))?.bias ?? null, expected.tamBias, 1e-3, cellKey(expected.levels));
    }
    expectClose(model.logLikelihood, biasFixture.tamLogLikelihood, 1e-3, "log-likelihood");
  });

  it("recovers the planted biases and flags exactly the two cells that carry them", () => {
    const flagged = (model.interactions ?? []).filter(({ z }) => Math.abs(z ?? 0) > 3).map(({ levels }) => levels);
    expect(flagged).toEqual([
      ["i1", "r1"],
      ["i1", "r2"],
    ]);
    for (const expected of biasFixture.cells) {
      expectClose(biasesByCell(model).get(cellKey(expected.levels))?.bias ?? null, expected.truth, 0.2, "bias");
    }
  });

  it("gives the biases of each element a sum of 0 in both facets, with the main effects as without them", () => {
    const cells = model.interactions ?? [];
    for (const position of [0, 1] as const) {
      for (const [id, sum] of sumsByElement(cells, position)) expectClose(sum, 0, 1e-9, id);
    }
    const without = manyFacetRasch(biasFixture.scores, { facets: ["item", "rater"] });
    expect(without.interactions).toBeUndefined();
    expect(model.facets.map(({ facet }) => facet)).toEqual(without.facets.map(({ facet }) => facet));
    expect(model.logLikelihood).toBeGreaterThan(without.logLikelihood);
    const free = (8 - 1) * (3 - 1);
    expectClose(
      model.aic - without.aic,
      -2 * (model.logLikelihood - without.logLikelihood) + 2 * free,
      1e-9,
      "parameters added",
    );
  });

  it("gives the same biases whichever facet of the pair comes first", () => {
    const reversed = manyFacetRasch(biasFixture.scores, {
      facets: ["item", "rater"],
      interactions: [["rater", "item"]],
      ...PRECISE,
    });
    const cells = biasesByCell(reversed);
    expect(reversed.interactions?.[0]?.facets).toEqual(["rater", "item"]);
    for (const expected of biasFixture.cells) {
      expectClose(cells.get(cellKey(expected.levels))?.bias ?? null, expected.bias, 1e-7, cellKey(expected.levels));
    }
  });

  it("leaves the standard errors and the Wald statistics out with standardErrors none", () => {
    const cells = manyFacetRasch(biasFixture.scores, {
      facets: ["item", "rater"],
      interactions,
      standardErrors: "none",
    }).interactions;
    expect(cells).toHaveLength(biasFixture.cells.length);
    expect(cells?.every(({ standardError, z }) => standardError === null && z === null)).toBe(true);
  });

  it("estimates the bias of every pair of facets when there are several", () => {
    const scores = biasFixture.scores.map(({ levels, score }) => ({
      levels: { ...levels, task: Number(levels.person?.slice(1)) % 2 === 0 ? "t1" : "t2" },
      score,
    }));
    const result = manyFacetRasch(scores, {
      facets: ["item", "rater", "task"],
      interactions: [
        ["item", "rater"],
        ["rater", "task"],
      ],
      standardErrors: "none",
    });
    const pairs = new Set((result.interactions ?? []).map(({ facets }) => facets.join("*")));
    expect([...pairs]).toEqual(["item*rater", "rater*task"]);
    expect(result.interactions).toHaveLength(8 * 3 + 3 * 2);
    expect(result.converged).toBe(true);
  });

  it("estimates the cells that scores fill, with as many free biases as the cells have cycles", () => {
    const scores = simulatedScores(21, 90).filter(({ levels }) => !(levels.item === "i0" && levels.rater === "r2"));
    const result = manyFacetRasch(scores, { facets: ["item", "rater"], interactions, ...PRECISE });
    const cells = result.interactions ?? [];
    expect(cells).toHaveLength(5);
    for (const position of [0, 1] as const) {
      for (const [id, sum] of sumsByElement(cells, position)) expectClose(sum, 0, 1e-9, id);
    }
    const without = manyFacetRasch(scores, { facets: ["item", "rater"], ...PRECISE });
    expectClose(
      result.aic - without.aic,
      -2 * (result.logLikelihood - without.logLikelihood) + 2,
      1e-9,
      "one free bias",
    );
  });

  it("refuses an interaction that every score's cell fixes at 0", () => {
    const scores: FacetedScore[] = [];
    for (let person = 0; person < 6; person++) {
      scores.push({ levels: { person: `p${person}`, item: "i0", rater: "r0" }, score: person % 3 });
      scores.push({ levels: { person: `p${person}`, item: "i1", rater: "r1" }, score: (person + 1) % 3 });
    }
    expect(() => manyFacetRasch(scores, { facets: ["item", "rater"], interactions })).toThrow(
      'The interaction of the facets "item" and "rater" cannot be estimated: its 2 cells are fixed by the 2 item and 2 rater elements, so every bias would be 0.',
    );
  });

  it("rejects interactions that are not pairs of two different facets, or that repeat", () => {
    const scores = simulatedScores(2, 20);
    const facets = ["item", "rater"];
    expect(() => manyFacetRasch(scores, { facets, interactions: JSON.parse("null") })).toThrow(DataError);
    expect(() => manyFacetRasch(scores, { facets, interactions: JSON.parse('"item"') })).toThrow(
      'The option interactions must be an array of pairs of facet names, such as [["item", "rater"]], found "item".',
    );
    expect(() => manyFacetRasch(scores, { facets, interactions: JSON.parse('["item"]') })).toThrow(
      'Interaction 1 of the option interactions must be a pair of facet names, such as ["item", "rater"], found "item".',
    );
    expect(() => manyFacetRasch(scores, { facets, interactions: JSON.parse('[["item"]]') })).toThrow(
      "Interaction 1 of the option interactions needs two facet names, found 1.",
    );
    expect(() => manyFacetRasch(scores, { facets, interactions: [["item", "task"]] })).toThrow(
      'Interaction 1 of the option interactions must be "item" or "rater", found "task".',
    );
    expect(() => manyFacetRasch(scores, { facets, interactions: [["item", "item"]] })).toThrow(
      'Interaction 1 of the option interactions pairs the facet "item" with itself; it needs two different facets.',
    );
    expect(() =>
      manyFacetRasch(scores, {
        facets,
        interactions: [
          ["item", "rater"],
          ["rater", "item"],
        ],
      }),
    ).toThrow('Interaction 2 of the option interactions repeats the interaction of "rater" and "item".');
  });

  it("adds a bias to the location of the cell it belongs to, in the expected ratings, the estimates and the fit", () => {
    const biased = storedModel([{ levels: ["i1", "r1"], bias: 0.4 }]);
    const shifted = storedModel([], 0.9);
    const unbiased = storedModel([]);
    const cell = { item: "i1", rater: "r1" };
    expectClose(
      expectedRating(biased, 0.2, { levels: cell }),
      expectedRating(shifted, 0.2, { levels: cell }),
      1e-12,
      "the biased cell",
    );
    expectClose(
      expectedRating(biased, 0.2, { levels: { item: "i1", rater: "r2" } }),
      expectedRating(unbiased, 0.2, { levels: { item: "i1", rater: "r2" } }),
      1e-12,
      "another cell",
    );
    expect(expectedRating(biased, 0.2, { levels: cell })).toBeLessThan(expectedRating(unbiased, 0.2, { levels: cell }));
    expectClose(expectedRating(biased, 0.2), expectedRating(unbiased, 0.2), 1e-12, "the fair score fills no cell");
    const scores = [
      { levels: { person: "a", ...cell }, score: 2 },
      { levels: { person: "b", ...cell }, score: 0 },
    ];
    const [withBias] = manyFacetEstimates(scores.slice(0, 1), biased);
    const [withoutBias] = manyFacetEstimates(scores.slice(0, 1), unbiased);
    expect(withBias?.theta).toBeGreaterThan(withoutBias?.theta ?? 0);
    const thetas = [
      { id: "a", theta: 0.2, standardError: 0.5 },
      { id: "b", theta: 0.2, standardError: 0.5 },
    ];
    const fitOf = (stored: ManyFacetRasch) => manyFacetFit(scores, stored, thetas);
    expect(fitOf(biased)[0]?.outfitMeanSquare).not.toBe(fitOf(unbiased)[0]?.outfitMeanSquare);
  });

  it("rejects a stored model whose interactions are not cells of its facets", () => {
    const scores = [{ levels: { person: "a", item: "i1", rater: "r1" }, score: 1 }];
    const stored = storedModel([{ levels: ["i1", "r1"], bias: 0.4 }]);
    expect(() => manyFacetEstimates(scores, { ...stored, interactions: JSON.parse('"cells"') })).toThrow(
      'The model\'s interactions must be an array of { facets, levels, bias }, found "cells".',
    );
    expect(() => manyFacetEstimates(scores, { ...stored, interactions: JSON.parse("[null]") })).toThrow(DataError);
    expect(() => manyFacetEstimates(scores, { ...stored, interactions: JSON.parse('[{"facets":"item"}]') })).toThrow(
      'Interaction cell 1 of the model needs facets, an array of two facet names, found "item".',
    );
    expect(() =>
      manyFacetEstimates(scores, {
        ...stored,
        interactions: [{ facets: ["item", "task"], levels: ["i1", "t1"], bias: 0.1, standardError: null, z: null }],
      }),
    ).toThrow(
      "Interaction cell 1 of the model needs two facets of the model, two string element ids and a finite bias.",
    );
    expect(() =>
      manyFacetEstimates(scores, {
        ...stored,
        interactions: [
          { facets: ["item", "rater"], levels: ["i1", "r1"], bias: Number.NaN, standardError: null, z: null },
        ],
      }),
    ).toThrow(
      "Interaction cell 1 of the model needs two facets of the model, two string element ids and a finite bias.",
    );
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
    const estimates = manyFacetEstimates(scores, modelOf(reference), {
      estimator: (items, responses) => {
        expect(items).toHaveLength(responses.length);
        seen.set(String(seen.size), items.length);
        return { theta: items.length, standardError: 1 };
      },
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
    const estimates = manyFacetEstimates(scores, model, {
      estimator: (items, responses) => maximumLikelihoodEstimate(items, responses),
    });
    estimates.forEach(({ id, theta }) => {
      const own = scores.filter(({ levels }) => levels.person === id);
      const raw = own.reduce((sum, { score }) => sum + score, 0);
      const maximum = own.length * (reference.thresholds[0]?.thresholds.length ?? 0);
      if (theta === null || raw === 0 || raw === maximum) return;
      const expected = own.reduce(
        (sum, { levels }) =>
          sum + expectedRating(model, theta, { levels: { item: levels.item ?? "", rater: levels.rater ?? "" } }),
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
      "The options must be an object, found null.",
    );
    expect(() => manyFacetEstimates(scores, model, JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() => manyFacetEstimates(scores, model, { estimator: JSON.parse("null") })).toThrow(
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
          expectClose(expectedRating(model, probe.theta, { levels }), probe.value, DERIVED_TOLERANCE, "levels");
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
    expect(expectedRating(model, 0, { levels: { rater: mildest.id } })).toBeGreaterThan(
      expectedRating(model, 0, { levels: { rater: harshest.id } }),
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
    expect(() => expectedRating(model, 0, { levels: { task: "t1" } })).toThrow(
      'The levels name the facet "task", but the model has "item", "rater".',
    );
    expect(() => expectedRating(model, 0, { levels: { rater: "nobody" } })).toThrow(
      'The facet "rater" of the model has no element "nobody".',
    );
    expect(() => expectedRating(model, 0, { levels: JSON.parse('{"rater": 3}') })).toThrow(
      'The level of the facet "rater" must be an element id, found 3.',
    );
    expect(() => expectedRating(model, 0, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => expectedRating(model, 0, JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
    expect(() => expectedRating(model, 0, { levels: JSON.parse("null") })).toThrow(
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

  it("the SMD on the subjects' own deviations as the population's is the SMD, and swapping the raters negates it", () => {
    fc.assert(
      fc.property(ratings(2), (rows) => {
        const value = standardizedMeanDifference(rows);
        if (value === null) return;
        const deviation = (rater: number): number =>
          standardDeviation(Float64Array.from(rows, (row) => defined(row[rater])));
        const population = { first: deviation(0), second: deviation(1) };
        expectClose(standardizedMeanDifference(rows, { population }), value, 1e-12);
        const swapped = rows.map(([first, second]) => [defined(second), defined(first)]);
        expectClose(standardizedMeanDifference(swapped), -value, 1e-12);
        const reversed = { first: population.second, second: population.first };
        expectClose(standardizedMeanDifference(swapped, { population: reversed }), -value, 1e-12);
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
        expectClose(standardizedMeanDifference(reversed), standardizedMeanDifference(rows), 1e-12);
        expectClose(ratingCorrelation(reversed), ratingCorrelation(rows), 1e-12);
      }),
    );
  });

  it("the correlation of two raters lies from -1 to 1, is null exactly for a constant rater and is the same for either order of the raters", () => {
    const rating = fc.double({ min: -1000, max: 1000, noNaN: true });
    fc.assert(
      fc.property(fc.array(fc.tuple(rating, rating), { minLength: 2, maxLength: 30 }), (pairs) => {
        const rows = pairs.map(([x, y]) => [x, y]);
        const value = ratingCorrelation(rows);
        const constant = (k: number) => rows.every((row) => row[k] === rows[0]?.[k]);
        expect(value === null).toBe(constant(0) || constant(1));
        if (value !== null) {
          expect(value).toBeGreaterThanOrEqual(-1);
          expect(value).toBeLessThanOrEqual(1);
          expectClose(ratingCorrelation(rows.map(([x = 0, y = 0]) => [y, x])), value, 1e-12);
        }
      }),
    );
  });
});

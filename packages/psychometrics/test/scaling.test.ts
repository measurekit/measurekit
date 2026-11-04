import { chiSquareUpperTail, mean } from "@measurekit/core";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  buildResponseMatrix,
  type ConversionTableOptions,
  calibrate,
  conditionalScaleError,
  conversionTable,
  createRandom,
  DataError,
  drawJointThetas,
  type EstimateInterval,
  generalizedPartialCredit,
  type IrtItem,
  nominalResponse,
  normStatistics,
  percentileRanks,
  predictFormScore,
  predictThetaScore,
  type ScorePrediction,
  type ScoreScale,
  scaleConversion,
  scaleGrowth,
  scaleScore,
  summedScoreTable,
  type ThetaDistribution,
  thetaPosterior,
  twoParameterLogistic,
} from "../src/index.js";
import { covariance, normalCdf, variance } from "../src/math/internal.js";
import { correlationFactor, marginOf, quantile, thetasOf } from "../src/scaling/internal.js";
import { defined, expectClose } from "./assertions.js";
import { type FixtureItems, fixtureItems } from "./fixture-items.js";
import { readFixture } from "./fixtures.js";

interface ScalingConversionFixture {
  source: string;
  forms: {
    name: string;
    items: { model: string; parameters: { a: number; b: number | number[]; c?: number; d?: number } }[];
  }[];
  tables: {
    form: string;
    method: string;
    prior: { mean: number; sd: number };
    scale: { slope: number; intercept: number; lowest?: number; highest?: number; step?: number };
    rows: {
      score: number;
      theta: number;
      standardError: number | null;
      scaleScore: number;
      scaleStandardError: number;
    }[];
  }[];
  scaleConversion: {
    form: string;
    scale: { slope: number; intercept: number; lowest?: number; highest?: number; step?: number };
    conversion: number[];
    equivalents: number[];
    scaleScores: number[];
  }[];
  conditionalErrors: { form: string; conversion: number[]; thetas: number[]; errors: number[] }[];
}
const conversions = readFixture<ScalingConversionFixture>("scaling-conversion.json");

interface ScalingGrowthFixture {
  source: string;
  groups: { group: string; mean: number; sd: number }[];
  growth: { from: string; to: string; meanChange: number; sdRatio: number; effectSize: number }[];
}
const growth = readFixture<ScalingGrowthFixture>("scaling-growth.json");

// biome-ignore-start lint/style/useNamingConvention: the keys are those of the fixture file.
interface ScalingJointDrawsFixture {
  source: string;
  margins: {
    fine: { nodes: number[]; weights: number[] };
    default: { nodes: number[]; weights: number[] };
    uneven: { nodes: number[]; weights: number[] };
    empty_cells: { nodes: number[]; weights: number[] };
    rounded_sum: { nodes: number[]; weights: number[] };
    one_node: { nodes: number[]; weights: number[] };
  };
  inverse: { name: string; nodes: number[]; weights: number[]; u: number[]; theta: number[] }[];
  transform: { margins: string[]; correlation: number[][]; scores: number[][]; thetas: number[][] };
  corners: {
    margins: string[];
    correlation: number[][];
    corners: { corner: number[]; probability: number; error: number }[];
  }[];
  spearman: { correlation: number; expected: number; standardError: number; draws: number }[];
}
// biome-ignore-end lint/style/useNamingConvention: the keys are those of the fixture file.
const joint = readFixture<ScalingJointDrawsFixture>("scaling-joint-draws.json");

// biome-ignore-start lint/style/useNamingConvention: the keys are those of the fixture file.
interface ScalingNormStatisticsFixture {
  source: string;
  cases: {
    name: string;
    scores: number[];
    mokkenZErrors: number[];
    levels: {
      level: number;
      mean: { estimate: number; standardError: number; lower: number; upper: number };
      sd: { estimate: number; standardError: number; lower: number; upper: number };
      scores: {
        score: number;
        z: { estimate: number; standardError: number; lower: number; upper: number };
        percentile: { estimate: number; standardError: number; lower: number; upper: number };
      }[];
      stanines: { estimate: number; standardError: number; lower: number; upper: number }[];
    }[];
  }[];
}
// biome-ignore-end lint/style/useNamingConvention: the keys are those of the fixture file.
const normStatisticsFixture = readFixture<ScalingNormStatisticsFixture>("scaling-norm-statistics.json");

interface ScalingNormsFixture {
  source: string;
  cases: { name: string; sample: number[]; scores: number[]; ranks: number[] }[];
}
const norms = readFixture<ScalingNormsFixture>("scaling-norms.json");

interface ScalingPlausibleValuesFixture {
  source: string;
  itemIds: string[];
  responses: number[][];
  covariate: number[];
  draws: number;
  withoutCovariates: {
    mirt: { mean: number; variance: number; slope: number; eap: { variance: number; slope: number } };
    tam: { mean: number; variance: number; slope: number };
    exact: { mean: number; variance: number; slope: number };
  };
  withCovariates: {
    coefficient: number;
    mirt: { mean: number; variance: number; slope: number; eap: { variance: number; slope: number } };
    tam: { mean: number; variance: number; slope: number; coefficient: number };
    exact: { mean: number; variance: number; slope: number };
  };
  rubin: {
    plausibleValues: number[][];
    mean: {
      estimates: number[];
      variances: number[];
      pooled: { estimate: number; within: number; between: number; total: number; degreesOfFreedom: number };
    };
    slope: {
      estimates: number[];
      variances: number[];
      pooled: { estimate: number; within: number; between: number; total: number; degreesOfFreedom: number };
    };
  };
}
const plausible = readFixture<ScalingPlausibleValuesFixture>("scaling-plausible-values.json");

interface ScalingPredictionFixture {
  source: string;
  practice: { a: number; b: number }[];
  posteriors: {
    responses: (number | null)[];
    prior: { mean: number; sd: number };
    nodes: number[];
    weights: number[];
  }[];
  formPredictions: {
    posterior: number;
    form: string;
    conversion: number[];
    level: number;
    mean: number;
    lower: number;
    upper: number;
    scores: number[];
    probabilities: number[];
  }[];
  thetaPredictions: {
    posterior: number;
    scale: { slope: number; intercept: number; lowest: number; highest: number; step: number };
    standardError: number;
    level: number;
    mean: number;
    lower: number;
    upper: number;
    scores: number[];
    probabilities: number[];
  }[];
}
const predictions = readFixture<ScalingPredictionFixture>("scaling-prediction.json");

// Tolerances against R, explained in docs/validation.md.
/** Thetas and standard errors of the EAP of the summed score against mirt's EAPsum and the base R recursion. */
const EAP_SUM_TOLERANCE = 1e-12;
/** Thetas and standard errors of the true score against uniroot on mirt's expected.test and testinfo. */
const TRUE_SCORE_TOLERANCE = 1e-10;
/** Conditional errors of scale scores at the reference's own thetas. */
const CONDITIONAL_ERROR_TOLERANCE = 1e-12;
/** Scale scores and their conditional errors in a table, at thetas that differ from the reference's by rounding. */
const SCALE_TOLERANCE = 1e-10;
/** Probabilities of predicted scores, interpolated scale scores and percentile ranks against base R. */
const PREDICTION_TOLERANCE = 1e-12;
/** The means of predicted scores, which are in the hundreds. */
const MEAN_TOLERANCE = 1e-11;
/** Growth between grades against base R, written out. */
const GROWTH_TOLERANCE = 1e-12;
/** The inverse distribution function of a posterior and the transform of normal scores against base R. */
const JOINT_TRANSFORM_TOLERANCE = 1e-12;
/** The standard errors that a Monte Carlo estimate may differ from its reference by. */
const MONTE_CARLO_ERRORS = 4;
/** The draws that estimate a joint distribution function. */
const CORNER_DRAWS = 200_000;
/** The moments of 200 plausible values for each of 2,000 persons against mirt's and TAM's, which draw as many. */
const PLAUSIBLE_MOMENT_TOLERANCE = 0.02;
/** The same moments against their expectation computed from the posteriors, free of the Monte Carlo error of the draws. */
const PLAUSIBLE_EXACT_TOLERANCE = 0.01;
/** The posterior means and the regression coefficient of a calibration against mirt's, which integrates on 61 points. */
const PLAUSIBLE_ESTIMATE_TOLERANCE = 1e-3;
/** Rubin's rules against the formulas of mice::pool written out in base R. */
const RUBIN_TOLERANCE = 1e-12;

/** A scale of a fixture: the bounds and the step are absent where the scale has none. */
interface FixtureScale {
  readonly slope: number;
  readonly intercept: number;
  readonly lowest?: number | undefined;
  readonly highest?: number | undefined;
  readonly step?: number | undefined;
}

/** Returns a scale without the fields a fixture leaves out. */
function scaleOf({ slope, intercept, lowest, highest, step }: FixtureScale): ScoreScale {
  return {
    slope,
    intercept,
    ...(lowest !== undefined && { lowest }),
    ...(highest !== undefined && { highest }),
    ...(step !== undefined && { step }),
  };
}

/** Returns the items of the form a fixture names. */
function formItems(name: string): IrtItem[] {
  const form = defined(conversions.forms.find((entry) => entry.name === name));
  return fixtureItems(form.items as FixtureItems);
}

const item = (a: number, b: number, c = 0, d = 1): IrtItem => ({
  model: twoParameterLogistic,
  parameters: { a, b, c, d },
});

const SCALE: ScoreScale = { slope: 100, intercept: 500, lowest: 200, highest: 800, step: 10 };
const FORM: readonly IrtItem[] = [item(1.2, -0.8), item(0.9, 0), item(1.5, 0.9)];

/** Returns P(x | θ) for each total score x of the items, by summing over every response pattern. */
function enumeratedScores(items: readonly IrtItem[], theta: number): number[] {
  let distribution = [1];
  for (const { model, parameters } of items) {
    const probabilities = new Float64Array(model.categoryCount(parameters));
    model.probabilities(parameters, theta, probabilities);
    const next = new Array<number>(distribution.length + probabilities.length - 1).fill(0);
    distribution.forEach((weight, score) => {
      probabilities.forEach((probability, category) => {
        next[score + category] = defined(next[score + category]) + weight * probability;
      });
    });
    distribution = next;
  }
  return distribution;
}

/** Returns the standard deviation of the scale scores at theta by the definition, over every total score. */
function enumeratedDeviation(items: readonly IrtItem[], conversion: readonly number[], theta: number): number {
  const distribution = enumeratedScores(items, theta);
  const mean = distribution.reduce((sum, probability, score) => sum + probability * defined(conversion[score]), 0);
  return Math.sqrt(
    distribution.reduce((sum, probability, score) => sum + probability * (defined(conversion[score]) - mean) ** 2, 0),
  );
}

describe("scaleScore", () => {
  it("rounds to the nearest multiple of the step, halves up, and bounds", () => {
    expect(scaleScore(0.234, SCALE)).toBe(520);
    expect(scaleScore(0.25, SCALE)).toBe(530);
    expect(scaleScore(-0.25, SCALE)).toBe(480);
    expect(scaleScore(-4.2, SCALE)).toBe(200);
    expect(scaleScore(7, SCALE)).toBe(800);
    expect(scaleScore(0.234, { ...SCALE, step: 0 })).toBeCloseTo(523.4, 12);
    expect(scaleScore(0.234, { ...SCALE, step: 25 })).toBe(525);
  });

  it("rounds to whole scores and does not bound by default", () => {
    expect(scaleScore(0.5, { slope: 15, intercept: 100 })).toBe(108);
    expect(scaleScore(-3, { slope: 15, intercept: 100 })).toBe(55);
    expect(scaleScore(1e6, { slope: 15, intercept: 100 })).toBe(15_000_100);
  });

  it("reports 0 for a small negative score, never -0", () => {
    expect(scaleScore(-0.001, { slope: 1, intercept: 0 })).toBe(0);
    expect(scaleScore(-0.001, { slope: 1, intercept: 0, lowest: -5, highest: 5, step: 1 })).toBe(0);
  });

  it("reports a bound that is not a multiple of the step for the scores beyond it", () => {
    const scale = { slope: 100, intercept: 500, lowest: 205, highest: 795, step: 10 };
    expect(scaleScore(-3, scale)).toBe(205);
    expect(scaleScore(-2.9, scale)).toBe(210);
    expect(scaleScore(2.9, scale)).toBe(790);
    expect(scaleScore(5, scale)).toBe(795);
  });

  it("reports a multiple of a decimal step as the double nearest the decimal", () => {
    expect(scaleScore(0.523, { slope: 100, intercept: 0, step: 0.1 })).toBe(52.3);
    expect(scaleScore(0.3, { slope: 1, intercept: 0, step: 0.1 })).toBe(0.3);
    expect(scaleScore(1.2345, { slope: 1, intercept: 0, step: 0.05 })).toBe(1.25);
    expect(scaleConversion([1.5], [10.3, 20.7], { step: 0.1 })).toEqual([20.7]);
    const prediction = predictThetaScore(
      { nodes: [0.523], weights: [1] },
      { slope: 100, intercept: 0, lowest: 0, highest: 100, step: 0.1 },
    );
    expect(prediction.distribution.find(({ probability }) => probability > 0)?.score).toBe(52.3);
    expect(prediction.mean).toBe(52.3);
  });

  it("rejects an invalid theta or scale", () => {
    expect(() => scaleScore(Number.NaN, SCALE)).toThrow("Theta must be a finite number, found NaN.");
    expect(() => scaleScore(JSON.parse('"1"'), SCALE)).toThrow('Theta must be a finite number, found "1".');
    expect(() => scaleScore(0, JSON.parse("null"))).toThrow(
      "The score scale must be an object { slope, intercept }, found null.",
    );
    expect(() => scaleScore(0, JSON.parse('{"slope":1,"intercept":0,"round":1}'))).toThrow(
      'The score scale may hold only slope, intercept, lowest, highest and step, found "round".',
    );
    expect(() => scaleScore(0, { slope: 0, intercept: 0 })).toThrow(
      "The slope of the score scale must be a positive number, found 0.",
    );
    expect(() => scaleScore(0, { slope: -1, intercept: 0 })).toThrow(DataError);
    expect(() => scaleScore(0, { slope: 1, intercept: Number.NaN })).toThrow(
      "The intercept of the score scale must be a finite number, found NaN.",
    );
    expect(() => scaleScore(0, { slope: 1, intercept: 0, lowest: 5, highest: 5 })).toThrow(
      "The score scale needs a lowest score below the highest, found 5 and 5.",
    );
    expect(() => scaleScore(0, { slope: 1, intercept: 0, lowest: Number.NaN })).toThrow(
      "The score scale needs a number as lowest, found NaN.",
    );
    expect(() => scaleScore(0, { slope: 1, intercept: 0, highest: JSON.parse('"9"') })).toThrow(
      'The score scale needs a number as highest, found "9".',
    );
    expect(() => scaleScore(0, { slope: 1, intercept: 0, step: -1 })).toThrow(
      "The step of the score scale must be a non-negative number, found -1.",
    );
    expect(() => scaleScore(0, { slope: 1, intercept: 0, step: JSON.parse("null") })).toThrow(DataError);
  });
});

describe("conversionTable", () => {
  for (const reference of conversions.tables) {
    const { form, method, prior, scale } = reference;
    it(`matches the ${method} table of ${form} on the scale ${JSON.stringify(scale)}${prior.mean === 0 ? "" : ` under a prior of mean ${prior.mean}`}`, () => {
      const items = formItems(form);
      const options: ConversionTableOptions = method === "eap-sum" ? { method, prior } : { method: "true-score" };
      const rows = conversionTable(items, scaleOf(scale), options);
      expect(rows).toHaveLength(reference.rows.length);
      const tolerance = method === "eap-sum" ? EAP_SUM_TOLERANCE : TRUE_SCORE_TOLERANCE;
      reference.rows.forEach((expected, score) => {
        const actual = defined(rows[score]);
        const label = `score ${score}`;
        expect(actual.score).toBe(score);
        expectClose(actual.theta, expected.theta, tolerance, `${label} theta`);
        expectClose(actual.standardError, expected.standardError, tolerance, `${label} standard error`);
        expectClose(actual.scaleScore, expected.scaleScore, SCALE_TOLERANCE, `${label} scale score`);
        expectClose(actual.scaleStandardError, expected.scaleStandardError, SCALE_TOLERANCE, `${label} scale error`);
      });
    });
  }

  it("takes the EAP of the summed score by default, as summedScoreTable gives it", () => {
    const items = formItems("est3pl form 2, 3PL");
    const table = summedScoreTable(items);
    const rows = conversionTable(items, SCALE);
    expect(rows.map(({ theta }) => theta)).toEqual(table.map(({ theta }) => theta));
    expect(rows.map(({ standardError }) => standardError)).toEqual(table.map(({ standardError }) => standardError));
    expect(conversionTable(items, SCALE, { method: "eap-sum" })).toEqual(rows);
  });

  it("converts each theta to its scale score and gives the error of the table's own scores", () => {
    const rows = conversionTable(FORM, SCALE);
    expect(rows.map(({ score }) => score)).toEqual([0, 1, 2, 3]);
    for (const row of rows) {
      expect(row.scaleScore).toBe(scaleScore(row.theta, SCALE));
      const error = enumeratedDeviation(
        FORM,
        rows.map(({ scaleScore: score }) => score),
        row.theta,
      );
      expectClose(row.scaleStandardError, error, 1e-12, `score ${row.score}`);
    }
  });

  it("takes the prior and the quadrature of the EAP of the summed score", () => {
    const standard = conversionTable(FORM, SCALE);
    const shifted = conversionTable(FORM, SCALE, { prior: { mean: 1, sd: 0.5 } });
    expect(shifted.map(({ theta }) => theta)).toEqual(
      summedScoreTable(FORM, { prior: { mean: 1, sd: 0.5 } }).map(({ theta }) => theta),
    );
    expect(defined(shifted[0]).theta).toBeGreaterThan(defined(standard[0]).theta);
    const coarse = conversionTable(FORM, SCALE, { quadrature: { method: "gauss-hermite", points: 21 } });
    expect(coarse.map(({ theta }) => theta)).toEqual(
      summedScoreTable(FORM, { quadrature: { method: "gauss-hermite", points: 21 } }).map(({ theta }) => theta),
    );
  });

  it("puts a score whose probability underflows at the end of the range toward it, without a standard error", () => {
    const easy = Array.from({ length: 60 }, () => item(3, 0));
    const prior = { mean: -5, sd: 0.01 };
    const rows = conversionTable(easy, SCALE, { prior });
    const last = defined(rows.at(-1));
    expect(last).toMatchObject({ score: 60, theta: 6, standardError: null });
    expect(defined(rows[0]).standardError).not.toBeNull();
    const empty = rows.filter(({ standardError }) => standardError === null);
    expect(empty.length).toBeGreaterThanOrEqual(1);
    expect(empty.every(({ theta }) => theta === 6)).toBe(true);
    expect(conversionTable(easy, SCALE, { prior, range: [-4, 4] }).at(-1)?.theta).toBe(4);
    const hard = Array.from({ length: 60 }, () => item(3, 0));
    const lowEnd = conversionTable(hard, SCALE, { prior: { mean: 5, sd: 0.01 } });
    expect(lowEnd[0]).toMatchObject({ score: 0, theta: -6, standardError: null });
    expect(lowEnd.every(({ scaleStandardError }) => Number.isFinite(scaleStandardError))).toBe(true);
  });

  describe("by the true score", () => {
    const options: ConversionTableOptions = { method: "true-score" };

    it("finds the theta at which the expected total score equals each total score", () => {
      const items = formItems("LSAT7, 2PL");
      const rows = conversionTable(items, SCALE, options);
      rows.slice(1, -1).forEach(({ score, theta }) => {
        const expected = items.reduce((sum, { parameters }) => {
          const { a, b } = parameters as { a: number; b: number };
          return sum + 1 / (1 + Math.exp(-a * (theta - b)));
        }, 0);
        expectClose(expected, score, 1e-10, `score ${score}`);
      });
      const thetas = rows.slice(1, -1).map(({ theta }) => theta);
      expect(thetas).toEqual([...thetas].sort((x, y) => x - y));
    });

    it("gives the lower end of the range to scores at or below the lower asymptotes and the upper end at the top", () => {
      const items = formItems("est3pl form 1, 3PL");
      const lower = items.reduce((sum, { parameters }) => sum + (parameters as { c: number }).c, 0);
      const rows = conversionTable(items, SCALE, options);
      rows.forEach(({ score, theta, standardError }) => {
        if (score <= lower) expect({ theta, standardError }).toEqual({ theta: -6, standardError: null });
        else if (score === 20) expect({ theta, standardError }).toEqual({ theta: 6, standardError: null });
        else expect(standardError).toBeGreaterThan(0);
      });
      const wide = conversionTable(items, SCALE, { ...options, range: [-8, 9] });
      expect(defined(wide[0]).theta).toBe(-8);
      expect(defined(wide.at(-1)).theta).toBe(9);
    });

    it("applies to polytomous forms, whose lowest and highest total scores have no theta", () => {
      const rows = conversionTable(formItems("Science, GPCM"), SCALE, options);
      expect(rows).toHaveLength(13);
      expect(defined(rows[0])).toMatchObject({ theta: -6, standardError: null });
      expect(defined(rows[12])).toMatchObject({ theta: 6, standardError: null });
      expect(rows.slice(1, 12).every(({ standardError }) => (standardError ?? 0) > 0)).toBe(true);
    });

    it("needs a form whose test characteristic curve increases", () => {
      const nominal: IrtItem = { model: nominalResponse, parameters: { a: [0, 1, 2], c: [0, 0.5, 0] } };
      expect(() => conversionTable([...FORM, nominal], SCALE, options)).toThrow(
        "Item 4 needs an expected score that increases with theta: a model other than the NRM, a > 0.",
      );
      expect(() => conversionTable([item(-1, 0)], SCALE, options)).toThrow(
        "Item 1 needs an expected score that increases",
      );
      expect(conversionTable([...FORM, nominal], SCALE)).toHaveLength(6);
    });
  });

  it("leaves the scale scores unrounded with a step of 0", () => {
    const scale = { ...SCALE, step: 0 };
    for (const row of conversionTable(FORM, scale)) {
      expectClose(row.scaleScore, 100 * row.theta + 500, 1e-12);
    }
  });

  it("converts a one-item form", () => {
    const rows = conversionTable([item(1, 0)], SCALE);
    expect(rows).toHaveLength(2);
    expect(defined(rows[0]).theta).toBeLessThan(0);
    expect(defined(rows[1]).theta).toBeGreaterThan(0);
  });

  it("rejects invalid items, scales and options", () => {
    expect(() => conversionTable([], SCALE)).toThrow("A conversion table needs at least one item.");
    expect(() => conversionTable(JSON.parse("null"), SCALE)).toThrow("A conversion table needs an array of items");
    expect(() => conversionTable(FORM, { slope: 0, intercept: 0 })).toThrow(DataError);
    expect(() => conversionTable(FORM, SCALE, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => conversionTable(FORM, SCALE, JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
    expect(() => conversionTable(FORM, SCALE, { method: JSON.parse('"ratio"') })).toThrow(
      'The option method must be "eap-sum" or "true-score", found "ratio".',
    );
    expect(() => conversionTable(FORM, SCALE, { range: [2, -2] })).toThrow(
      "The range needs the lower bound below the upper, found [2, -2].",
    );
    expect(() => conversionTable(FORM, SCALE, { range: JSON.parse("5") })).toThrow(
      "The option range must be an array [lower, upper], found 5.",
    );
    expect(() => conversionTable(FORM, SCALE, { prior: { mean: 0, sd: 0 } })).toThrow(DataError);
    expect(() => conversionTable(FORM, SCALE, { quadrature: { points: 1 } })).toThrow(DataError);
    expect(() => conversionTable(FORM, SCALE, { prior: JSON.parse('"normal"') })).toThrow(
      'The option prior must be an object { mean, sd }, found "normal".',
    );
  });
});

describe("conditionalScaleError", () => {
  for (const reference of conversions.conditionalErrors) {
    it(`matches the base R error of the scale scores of ${reference.form}`, () => {
      const items = formItems(reference.form);
      reference.thetas.forEach((theta, j) => {
        const actual = conditionalScaleError(items, reference.conversion, theta);
        expectClose(actual, defined(reference.errors[j]), CONDITIONAL_ERROR_TOLERANCE, `theta ${theta}`);
      });
    });
  }

  it("is the standard deviation of the scale score over the total scores, as enumerating the patterns gives it", () => {
    const polytomous: IrtItem = { model: generalizedPartialCredit, parameters: { a: 1.1, b: [-0.5, 0.7] } };
    const items = [...FORM, polytomous];
    const conversion = [210, 230, 280, 400, 520, 610];
    for (const theta of [-2, -0.3, 0, 1.4, 3]) {
      expectClose(
        conditionalScaleError(items, conversion, theta),
        enumeratedDeviation(items, conversion, theta),
        1e-12,
        `theta ${theta}`,
      );
    }
  });

  it("is the slope times the standard deviation of the total score for a linear conversion", () => {
    for (const theta of [-1.5, 0, 0.8]) {
      const variance = FORM.reduce((sum, { parameters }) => {
        const { a, b } = parameters as { a: number; b: number };
        const probability = 1 / (1 + Math.exp(-a * (theta - b)));
        return sum + probability * (1 - probability);
      }, 0);
      expectClose(conditionalScaleError(FORM, [300, 350, 400, 450], theta), 50 * Math.sqrt(variance), 1e-12);
    }
  });

  it("is 0 for a conversion with one scale score", () => {
    expect(conditionalScaleError(FORM, [500, 500, 500, 500], 0.3)).toBe(0);
  });

  it("stays defined where the total scores' probabilities underflow", () => {
    const steep = Array.from({ length: 30 }, () => item(8, 0));
    const conversion = Array.from({ length: 31 }, (_, score) => 10 * score);
    for (const theta of [-60, 0, 60]) {
      expect(Number.isFinite(conditionalScaleError(steep, conversion, theta))).toBe(true);
    }
    expect(conditionalScaleError(steep, conversion, 60)).toBeLessThan(1e-6);
  });

  it("rejects invalid items, conversions and thetas", () => {
    expect(() => conditionalScaleError([], [], 0)).toThrow(
      "A conditional error of scale scores needs at least one item.",
    );
    expect(() => conditionalScaleError(FORM, [1, 2, 3], 0)).toThrow(
      "The conversion needs 4 scale scores, one for each total score from 0 to 3, found 3.",
    );
    expect(() => conditionalScaleError(FORM, [1, 2, Number.NaN, 4], 0)).toThrow(
      "The conversion needs finite scale scores, found NaN for the total score 2.",
    );
    expect(() => conditionalScaleError(FORM, JSON.parse("null"), 0)).toThrow(
      "The conversion must be an array of scale scores, found null.",
    );
    expect(() => conditionalScaleError(FORM, [1, 2, 3, JSON.parse('"4"')], 0)).toThrow(DataError);
    expect(() => conditionalScaleError(FORM, [1, 2, 3, 4], Number.POSITIVE_INFINITY)).toThrow(
      "Theta must be a finite number, found Infinity.",
    );
    expect(() => conditionalScaleError(FORM, [1, 2, 3, 4], JSON.parse('"0"'))).toThrow(DataError);
  });
});

describe("scaleConversion", () => {
  for (const reference of conversions.scaleConversion) {
    it(`matches the base R interpolation in the table of ${reference.form}`, () => {
      const { slope: _slope, intercept: _intercept, ...rounding } = scaleOf(reference.scale);
      const actual = scaleConversion(reference.equivalents, reference.conversion, rounding);
      expect(actual).toHaveLength(reference.scaleScores.length);
      reference.scaleScores.forEach((expected, i) => {
        expectClose(
          defined(actual[i]),
          expected,
          PREDICTION_TOLERANCE,
          `equivalent ${defined(reference.equivalents[i])}`,
        );
      });
    });
  }

  it("interpolates linearly between the neighboring scores and holds the ends", () => {
    const conversion = [300, 410, 520];
    expect(scaleConversion([0.5, 1.5, 0, 2], conversion, { step: 0 })).toEqual([355, 465, 300, 520]);
    expect(scaleConversion([-3, 9, 2.0001], conversion, { step: 0 })).toEqual([300, 520, 520]);
    expect(scaleConversion([0.25], conversion, { step: 0 })).toEqual([327.5]);
  });

  it("rounds and bounds the interpolated scores", () => {
    const conversion = [300, 410, 520];
    expect(scaleConversion([0.5, 1.5], conversion, { step: 10 })).toEqual([360, 470]);
    expect(scaleConversion([0.5, 1.5], conversion, { step: 10, lowest: 400, highest: 450 })).toEqual([400, 450]);
    expect(scaleConversion([0.5], conversion)).toEqual([355]);
  });

  it("returns the one scale score of a conversion of one score, and nothing for no equivalents", () => {
    expect(scaleConversion([-1, 0, 4], [250], { step: 0 })).toEqual([250, 250, 250]);
    expect(scaleConversion([], [250, 300])).toEqual([]);
  });

  it("gives the conversion's scores at the integers, however it is rounded", () => {
    const conversion = [10, 20, 40, 80];
    expect(scaleConversion([0, 1, 2, 3], conversion, { step: 0 })).toEqual(conversion);
  });

  it("rejects equivalents, conversions and options that are invalid", () => {
    expect(() => scaleConversion([0], [])).toThrow("The conversion needs at least one scale score.");
    expect(() => scaleConversion([Number.NaN], [1, 2])).toThrow("Equivalent 1 must be a finite number, found NaN.");
    expect(() => scaleConversion([0], [1, Number.POSITIVE_INFINITY])).toThrow(
      "The conversion needs finite scale scores, found Infinity for the total score 1.",
    );
    expect(() => scaleConversion(JSON.parse("null"), [1, 2])).toThrow(
      "The equivalents must be an array of total scores, found null.",
    );
    expect(() => scaleConversion([0], JSON.parse('"12"'))).toThrow(
      'The conversion must be an array of scale scores, found "12".',
    );
    expect(() => scaleConversion([0], [1, 2], JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => scaleConversion([0], [1, 2], JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
    expect(() => scaleConversion([0], [1, 2], { step: -5 })).toThrow(DataError);
    expect(() => scaleConversion([0], [1, 2], { lowest: 9, highest: 3 })).toThrow(
      "The options needs a lowest score below the highest, found 9 and 3.",
    );
  });
});

describe("scaleGrowth", () => {
  it("matches base R's growth between the grades of a concurrent calibration", () => {
    const rows = scaleGrowth(growth.groups);
    expect(rows).toHaveLength(growth.growth.length);
    for (const [k, row] of rows.entries()) {
      const expected = defined(growth.growth[k]);
      expect([row.from, row.to]).toEqual([expected.from, expected.to]);
      expectClose(row.meanChange, expected.meanChange, GROWTH_TOLERANCE, "mean change");
      expectClose(row.sdRatio, expected.sdRatio, GROWTH_TOLERANCE, "ratio of standard deviations");
      expectClose(row.effectSize, expected.effectSize, GROWTH_TOLERANCE, "effect size");
    }
  });

  it("gives one row for each pair of neighboring groups, in the order given", () => {
    const rows = scaleGrowth([
      { group: "a", mean: 0, sd: 1 },
      { group: "b", mean: 10, sd: 7 },
      { group: "c", mean: 4, sd: 7 },
    ]);
    expect(rows.map(({ from, to }) => [from, to])).toEqual([
      ["a", "b"],
      ["b", "c"],
    ]);
    expectClose(defined(rows[0]).effectSize, 2, GROWTH_TOLERANCE);
    expectClose(defined(rows[0]).sdRatio, 7, GROWTH_TOLERANCE);
    expect(rows[1]).toMatchObject({ from: "b", to: "c", meanChange: -6, sdRatio: 1 });
    expectClose(defined(rows[1]).effectSize, -6 / 7, GROWTH_TOLERANCE);
  });

  it("rejects fewer than two groups, labels that repeat and a scale without spread", () => {
    const group = { group: "a", mean: 0, sd: 1 };
    expect(() => scaleGrowth([group])).toThrow("Growth needs at least two groups on the shared scale, found 1.");
    expect(() => scaleGrowth([group, { ...group, mean: 1 }])).toThrow(
      'The groups need different labels, found "a" twice.',
    );
    expect(() => scaleGrowth([group, { group: "b", mean: 1, sd: 0 }])).toThrow(
      "The standard deviation of group 2 must be a positive number, found 0.",
    );
    expect(() => scaleGrowth([group, { group: "b", mean: Number.NaN, sd: 1 }])).toThrow(
      "The mean of group 2 must be a finite number, found NaN.",
    );
    expect(() => scaleGrowth([group, { group: "", mean: 1, sd: 1 }])).toThrow(
      'The group 2 needs a label that is a non-empty string, found "".',
    );
  });

  it("rejects input of other types", () => {
    expect(() => scaleGrowth(JSON.parse("null"))).toThrow("The groups must be an array, found null.");
    expect(() => scaleGrowth(JSON.parse('[{"group":"a","mean":0,"sd":1},null]'))).toThrow(
      "The group 2 must be an object { group, mean, sd }, found null.",
    );
    expect(() => scaleGrowth(JSON.parse('[{"group":"a","mean":0,"sd":1},{"group":"b","mean":"1","sd":1}]'))).toThrow(
      'The mean of group 2 must be a finite number, found "1".',
    );
    expect(() => scaleGrowth(JSON.parse('[{"group":"a","mean":0,"sd":1},{"group":2,"mean":1,"sd":1}]'))).toThrow(
      "The group 2 needs a label that is a non-empty string, found 2.",
    );
  });
});

describe("percentileRanks", () => {
  for (const reference of norms.cases) {
    it(`matches the base R percentile ranks of ${reference.name}`, () => {
      const actual = percentileRanks(reference.sample, reference.scores);
      reference.ranks.forEach((expected, i) => {
        expectClose(defined(actual[i]), expected, PREDICTION_TOLERANCE, `score ${defined(reference.scores[i])}`);
      });
    });
  }

  it("counts the scores below plus half of those at the score", () => {
    expect(percentileRanks([10, 20, 20, 30, 40], [20, 25, 50, 5, 10])).toEqual([40, 60, 100, 0, 10]);
    expect(percentileRanks([5], [4, 5, 6])).toEqual([0, 50, 100]);
  });

  it("does not depend on the order of the sample, and does not change it", () => {
    const sample = [30, 10, 40, 20, 20];
    const copy = [...sample];
    expect(percentileRanks(sample, [20])).toEqual(percentileRanks([10, 20, 20, 30, 40], [20]));
    expect(sample).toEqual(copy);
  });

  it("returns no ranks for no scores", () => {
    expect(percentileRanks([1, 2], [])).toEqual([]);
  });

  it("rejects a sample or scores that are invalid", () => {
    expect(() => percentileRanks([], [1])).toThrow("The norm sample needs at least one score.");
    expect(() => percentileRanks([1, Number.NaN], [1])).toThrow("The norm sample needs finite scores, found NaN at 1.");
    expect(() => percentileRanks([1, JSON.parse('"2"')], [1])).toThrow('found "2" at 1');
    expect(() => percentileRanks(JSON.parse("null"), [1])).toThrow(
      "The norm sample must be an array of scores, found null.",
    );
    expect(() => percentileRanks([1], JSON.parse("3"))).toThrow("The scores must be an array of numbers, found 3.");
    expect(() => percentileRanks([1], [0, Number.POSITIVE_INFINITY])).toThrow(
      "Score 2 must be a finite number, found Infinity.",
    );
  });
});

describe("normStatistics", () => {
  /** Estimates and standard errors against check.norms: delta-method sums of the same counts. */
  const NormTolerance = 1e-10;

  function expectInterval(actual: EstimateInterval | undefined, expected: EstimateInterval, label: string): void {
    const tolerance = NormTolerance * Math.max(1, Math.abs(expected.estimate));
    expectClose(actual?.estimate ?? null, expected.estimate, tolerance, `${label} estimate`);
    expectClose(actual?.standardError ?? null, expected.standardError, NormTolerance, `${label} standard error`);
    expectClose(actual?.lower ?? null, expected.lower, tolerance, `${label} lower`);
    expectClose(actual?.upper ?? null, expected.upper, tolerance, `${label} upper`);
  }

  for (const reference of normStatisticsFixture.cases) {
    it(`matches check.norms of the mokken package on ${reference.name}`, () => {
      for (const expected of reference.levels) {
        const result = normStatistics(reference.scores, { level: expected.level });
        const label = `${expected.level}`;
        expectInterval(result.mean, expected.mean, `${label} mean`);
        expectInterval(result.sd, expected.sd, `${label} sd`);
        expect(result.scores.map(({ score }) => score)).toEqual(expected.scores.map(({ score }) => score));
        expected.scores.forEach((norm, k) => {
          expectInterval(result.scores[k]?.z, norm.z, `${label} z of ${norm.score}`);
          expectInterval(result.scores[k]?.percentile, norm.percentile, `${label} percentile of ${norm.score}`);
        });
        expected.stanines.forEach((stanine, k) => {
          expectInterval(result.stanines[k], stanine, `${label} stanine boundary ${k + 1}`);
        });
      }
    });
  }

  it("gives z scores standard errors above check.norms', which subtracts its correction twice", () => {
    for (const reference of normStatisticsFixture.cases) {
      const result = normStatistics(reference.scores);
      reference.mokkenZErrors.forEach((mokken, k) => {
        const own = defined(result.scores[k]).z.standardError;
        expect(own).toBeGreaterThanOrEqual(mokken);
        expect(own - mokken).toBeLessThan(1e-5 * own);
      });
    }
  });

  it("derives the norms of two scores by hand", () => {
    // Scores 0 and 2: mean 1, SD √2, z of ∓1/√2, percentile ranks 25 and 75. The mean's standard error is the SD with
    // denominator N, 1, over √2; a rank is 50 + 50 p̂ for p̂ the share of 0s, whose variance is 1/2 · 1/2 / 2.
    const result = normStatistics([0, 2], { level: 0.5 });
    expectClose(result.mean.estimate, 1, 1e-15);
    expectClose(result.mean.standardError, 1 / Math.sqrt(2), 1e-15);
    expectClose(result.sd.estimate, Math.sqrt(2), 1e-15);
    expectClose(defined(result.scores[0]).z.estimate, -1 / Math.sqrt(2), 1e-15);
    expect(result.scores.map(({ percentile }) => percentile.estimate)).toEqual([25, 75]);
    expectClose(defined(result.scores[1]).percentile.standardError, 50 * Math.sqrt(1 / 8), 1e-12);
    expectClose(result.mean.upper - result.mean.estimate, 0.6744897501960817 / Math.sqrt(2), 1e-12);
  });

  it("gives the stanine boundaries at a quarter of an SD around the mean and every half SD out", () => {
    const { mean, sd, stanines } = normStatistics([1, 4, 4, 5, 7, 8, 9, 9, 12]);
    expect(stanines).toHaveLength(8);
    [-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75].forEach((multiple, k) => {
      expectClose(defined(stanines[k]).estimate, mean.estimate + multiple * sd.estimate, 1e-12, `${k + 1}`);
    });
  });

  it("takes 0.95 as the level by default", () => {
    expect(normStatistics([1, 2, 2, 3, 5])).toEqual(normStatistics([1, 2, 2, 3, 5], { level: 0.95 }));
  });

  it("rejects fewer than two scores, scores that do not vary and arguments of other types", () => {
    expect(() => normStatistics([3])).toThrow("Norm statistics need at least 2 scores, found 1.");
    expect(() => normStatistics([3, 3, 3])).toThrow("Norm statistics need scores that vary.");
    expect(() => normStatistics([1, Number.NaN])).toThrow("Score 2 must be a finite number, found NaN.");
    expect(() => normStatistics(JSON.parse('[1, "2"]'))).toThrow('Score 2 must be a finite number, found "2".');
    expect(() => normStatistics(JSON.parse("null"))).toThrow("The scores must be an array of numbers, found null.");
    expect(() => normStatistics([1, 2], { level: 1 })).toThrow("The option level");
    expect(() => normStatistics([1, 2], JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => normStatistics([1, 2], JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
  });
});

/** The posterior of a prediction fixture. */
function posteriorOf(index: number): ThetaDistribution {
  return defined(predictions.posteriors[index]);
}

/** Expects a prediction to match a reference. */
function expectPrediction(
  actual: ScorePrediction,
  expected: ScorePrediction & { scores: number[]; probabilities: number[] },
) {
  expectClose(actual.mean, expected.mean, MEAN_TOLERANCE, "mean");
  expect(actual.lower).toBe(expected.lower);
  expect(actual.upper).toBe(expected.upper);
  expect(actual.distribution.map(({ score }) => score)).toEqual(expected.scores);
  expected.probabilities.forEach((probability, i) => {
    expectClose(
      defined(actual.distribution[i]).probability,
      probability,
      PREDICTION_TOLERANCE,
      `P(${expected.scores[i]})`,
    );
  });
}

describe("predictFormScore", () => {
  predictions.formPredictions.forEach((reference, i) => {
    it(`matches the base R prediction ${i + 1} on ${reference.form} at the level ${reference.level}`, () => {
      const actual = predictFormScore(
        posteriorOf(reference.posterior),
        formItems(reference.form),
        reference.conversion,
        {
          level: reference.level,
        },
      );
      expectPrediction(actual, { ...actual, ...reference });
    });
  });

  it("gives the form's own distribution of the total score for a posterior at one node", () => {
    const polytomous: IrtItem = { model: generalizedPartialCredit, parameters: { a: 1.1, b: [-0.5, 0.7] } };
    const items = [...FORM, polytomous];
    const conversion = [210, 230, 280, 400, 520, 610];
    for (const theta of [-2, 0.4, 2.5]) {
      const actual = predictFormScore({ nodes: [theta], weights: [1] }, items, conversion);
      const expected = enumeratedScores(items, theta);
      expect(actual.distribution.map(({ score }) => score)).toEqual(conversion);
      expected.forEach((probability, score) => {
        expectClose(defined(actual.distribution[score]).probability, probability, 1e-12, `score ${score}`);
      });
      expectClose(
        actual.mean,
        expected.reduce((sum, probability, score) => sum + probability * defined(conversion[score]), 0),
        1e-10,
      );
    }
  });

  it("predicts a mixture of the nodes' distributions, with the posterior's weights", () => {
    const posterior = { nodes: [-1, 1], weights: [0.3, 0.7] };
    const low = enumeratedScores(FORM, -1);
    const high = enumeratedScores(FORM, 1);
    const { distribution } = predictFormScore(posterior, FORM, [0, 1, 2, 3]);
    distribution.forEach(({ probability }, score) => {
      expectClose(probability, 0.3 * defined(low[score]) + 0.7 * defined(high[score]), 1e-12);
    });
  });

  it("pools the total scores that convert to the same scale score, in ascending order of score", () => {
    const posterior = thetaPosterior(FORM, [1, null, 0]);
    const { distribution } = predictFormScore(posterior, FORM, [500, 500, 300, 700]);
    expect(distribution.map(({ score }) => score)).toEqual([300, 500, 700]);
    const separate = predictFormScore(posterior, FORM, [1, 2, 3, 4]).distribution;
    expectClose(
      defined(distribution[1]).probability,
      defined(separate[0]).probability + defined(separate[1]).probability,
      1e-12,
    );
    const reversed = predictFormScore(posterior, FORM, [700, 500, 300, 100]).distribution;
    expect(reversed.map(({ score }) => score)).toEqual([100, 300, 500, 700]);
    expectClose(defined(reversed[0]).probability, defined(separate[3]).probability, 1e-12);
  });

  it("takes its interval from the cumulative probability, wider for a higher level", () => {
    const posterior = thetaPosterior(FORM, [null, null, null]);
    const narrow = predictFormScore(posterior, FORM, [200, 400, 600, 800], { level: 0.2 });
    const wide = predictFormScore(posterior, FORM, [200, 400, 600, 800], { level: 0.99 });
    expect(narrow.lower).toBeGreaterThanOrEqual(wide.lower);
    expect(narrow.upper).toBeLessThanOrEqual(wide.upper);
    expect(wide.lower).toBe(200);
    expect(wide.upper).toBe(800);
    expect(predictFormScore(posterior, FORM, [200, 400, 600, 800]).mean).toBeGreaterThan(200);
  });

  it("scales weights that sum to 1 within the tolerance to a distribution that sums to 1", () => {
    const { distribution } = predictFormScore({ nodes: [0], weights: [0.9999995] }, FORM, [0, 1, 2, 3]);
    expectClose(
      distribution.reduce((sum, { probability }) => sum + probability, 0),
      1,
      1e-15,
    );
  });

  it("predicts a knowledge tracing learner's posterior, which has the same shape", () => {
    const state = { nodes: Array.from({ length: 61 }, (_, q) => -6 + q / 5), weights: Array(61).fill(1 / 61) };
    const { distribution } = predictFormScore(state, FORM, [0, 1, 2, 3]);
    expect(distribution).toHaveLength(4);
  });

  it("rejects an invalid posterior, form, conversion or level", () => {
    const posterior = { nodes: [0], weights: [1] };
    const conversion = [0, 1, 2, 3];
    expect(() => predictFormScore(posterior, [], conversion)).toThrow("A score prediction needs at least one item.");
    expect(() => predictFormScore(posterior, FORM, [0, 1])).toThrow(
      "The conversion needs 4 scale scores, one for each total score from 0 to 3, found 2.",
    );
    expect(() => predictFormScore({ nodes: [0], weights: [0.9] }, FORM, conversion)).toThrow(
      "The weights of the posterior must sum to 1, found 0.9.",
    );
    expect(() => predictFormScore({ nodes: [0, 1], weights: [1] }, FORM, conversion)).toThrow(
      "The posterior needs at least one node and as many weights as nodes, found 2 and 1.",
    );
    expect(() => predictFormScore({ nodes: [], weights: [] }, FORM, conversion)).toThrow(DataError);
    expect(() => predictFormScore({ nodes: [Number.NaN], weights: [1] }, FORM, conversion)).toThrow(
      "The nodes of the posterior must be finite numbers, found NaN at 0.",
    );
    expect(() => predictFormScore({ nodes: [0, 1], weights: [1.5, -0.5] }, FORM, conversion)).toThrow(
      "The weights of the posterior must be non-negative numbers, found -0.5 at 1.",
    );
    expect(() => predictFormScore(JSON.parse("null"), FORM, conversion)).toThrow(
      "The posterior must be an object { nodes, weights }, found null.",
    );
    expect(() => predictFormScore(posterior, FORM, conversion, { level: 0 })).toThrow(
      "The option level must be a number between 0 and 1, exclusive, found 0.",
    );
    expect(() => predictFormScore(posterior, FORM, conversion, { level: 1 })).toThrow(DataError);
    expect(() => predictFormScore(posterior, FORM, conversion, { level: JSON.parse('"0.8"') })).toThrow(
      'The option level must be a number between 0 and 1, exclusive, found "0.8".',
    );
    expect(() => predictFormScore(posterior, FORM, conversion, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => predictFormScore(posterior, FORM, conversion, JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
  });
});

describe("predictThetaScore", () => {
  predictions.thetaPredictions.forEach((reference, i) => {
    it(`matches the base R prediction ${i + 1} of the scale ${JSON.stringify(reference.scale)} at the standard error ${reference.standardError} and level ${reference.level}`, () => {
      const actual = predictThetaScore(posteriorOf(reference.posterior), scaleOf(reference.scale), {
        standardError: reference.standardError,
        level: reference.level,
      });
      expectPrediction(actual, { ...actual, ...reference });
    });
  });

  it("reports each node as its own score with no error", () => {
    const posterior = { nodes: [-1, 0, 0.234, 5], weights: [0.1, 0.2, 0.3, 0.4] };
    const { distribution, mean, lower, upper } = predictThetaScore(posterior, SCALE);
    const mass = distribution.filter(({ probability }) => probability > 0);
    expect(mass).toEqual([
      { score: 400, probability: 0.1 },
      { score: 500, probability: 0.2 },
      { score: 520, probability: 0.3 },
      { score: 800, probability: 0.4 },
    ]);
    expectClose(mean, 0.1 * 400 + 0.2 * 500 + 0.3 * 520 + 0.4 * 800, 1e-10);
    expect([lower, upper]).toEqual([400, 800]);
    expect(distribution).toHaveLength(61);
    expect(distribution.map(({ score }) => score)).toEqual(Array.from({ length: 61 }, (_, k) => 200 + 10 * k));
  });

  it("spreads a node over the scores its estimate rounds to, as the normal distribution gives it", () => {
    const posterior = { nodes: [0], weights: [1] };
    const { distribution, mean } = predictThetaScore(posterior, SCALE, { standardError: 0.1 });
    const at = (score: number) => defined(distribution.find((entry) => entry.score === score)).probability;
    expectClose(at(500), normalCdf(0.5) - normalCdf(-0.5), 1e-14);
    expectClose(at(510), normalCdf(1.5) - normalCdf(0.5), 1e-14);
    expectClose(at(490), at(510), 1e-14);
    expectClose(mean, 500, 1e-10);
  });

  it("lists the lowest and highest scores and the multiples of the step between them when the bounds are not multiples", () => {
    const scale = { slope: 100, intercept: 500, lowest: 205, highest: 795, step: 10 };
    const { distribution } = predictThetaScore({ nodes: [0], weights: [1] }, scale, { standardError: 0.5 });
    expect(distribution.map(({ score }) => score)).toEqual([
      205,
      ...Array.from({ length: 59 }, (_, k) => 210 + 10 * k),
      795,
    ]);
    expectClose(
      distribution.reduce((sum, { probability }) => sum + probability, 0),
      1,
      1e-14,
    );
  });

  it("agrees with the scores drawn from the posterior and the estimate's error", () => {
    const scale = { slope: 100, intercept: 500, lowest: 205, highest: 795, step: 10 };
    const posterior = { nodes: [-2.5, -0.4, 0.3, 1.8], weights: [0.2, 0.3, 0.4, 0.1] };
    const standardError = 0.45;
    const { distribution } = predictThetaScore(posterior, scale, { standardError });
    const random = createRandom(20261031);
    const draws = 100_000;
    const counts = new Map<number, number>();
    for (let i = 0; i < draws; i++) {
      let u = random.uniform();
      let q = 0;
      while (q < posterior.nodes.length - 1 && u >= defined(posterior.weights[q])) {
        u -= defined(posterior.weights[q]);
        q++;
      }
      const theta = defined(posterior.nodes[q]) + standardError * random.normal();
      const score = scaleScore(theta, scale);
      counts.set(score, (counts.get(score) ?? 0) + 1);
    }
    for (const { score, probability } of distribution) {
      const found = (counts.get(score) ?? 0) / draws;
      expect(Math.abs(found - probability), `score ${score}`).toBeLessThanOrEqual(
        4 * Math.sqrt((probability * (1 - probability)) / draws) + 1e-12,
      );
    }
  });

  it("approaches the prediction with no error as the error goes to 0", () => {
    const posterior = { nodes: [-1.03, 0.017, 1.21], weights: [0.25, 0.5, 0.25] };
    const exact = predictThetaScore(posterior, SCALE);
    const tiny = predictThetaScore(posterior, SCALE, { standardError: 1e-9 });
    exact.distribution.forEach(({ probability }, j) => {
      expectClose(defined(tiny.distribution[j]).probability, probability, 1e-9);
    });
  });

  it("moves up with the posterior and widens with the error and the level", () => {
    const base = { nodes: [-0.2, 0, 0.2], weights: [0.25, 0.5, 0.25] };
    const higher = { nodes: [0.8, 1, 1.2], weights: [0.25, 0.5, 0.25] };
    expect(predictThetaScore(higher, SCALE, { standardError: 0.3 }).mean).toBeGreaterThan(
      predictThetaScore(base, SCALE, { standardError: 0.3 }).mean,
    );
    const sharp = predictThetaScore(base, SCALE, { standardError: 0.1 });
    const vague = predictThetaScore(base, SCALE, { standardError: 0.6 });
    expect(vague.upper - vague.lower).toBeGreaterThan(sharp.upper - sharp.lower);
    const low = predictThetaScore(base, SCALE, { standardError: 0.3, level: 0.5 });
    const high = predictThetaScore(base, SCALE, { standardError: 0.3, level: 0.99 });
    expect(high.upper - high.lower).toBeGreaterThan(low.upper - low.lower);
  });

  it("reports the end scores for thetas beyond the bounds, without error and with it", () => {
    const posterior = { nodes: [-50, 50], weights: [0.5, 0.5] };
    for (const standardError of [0, 0.2]) {
      const { distribution } = predictThetaScore(posterior, SCALE, { standardError });
      expectClose(defined(distribution[0]).probability, 0.5, 1e-14);
      expectClose(defined(distribution.at(-1)).probability, 0.5, 1e-14);
    }
  });

  it("works for a scale with one multiple of the step between the bounds and with a step wider than the range", () => {
    const narrow = { slope: 100, intercept: 500, lowest: 495, highest: 505, step: 10 };
    expect(predictThetaScore({ nodes: [0], weights: [1] }, narrow, { standardError: 0.1 }).distribution).toHaveLength(
      3,
    );
    const wide = { slope: 1, intercept: 0, lowest: 0.2, highest: 0.9, step: 1 };
    const { distribution } = predictThetaScore({ nodes: [0.4, 0.6], weights: [0.5, 0.5] }, wide);
    expect(distribution).toEqual([
      { score: 0.2, probability: 0.5 },
      { score: 0.9, probability: 0.5 },
    ]);
  });

  it("needs finite bounds and a step above 0", () => {
    const posterior = { nodes: [0], weights: [1] };
    const message = "Predicting a theta score needs a scale with a finite lowest and highest score and a step above 0";
    expect(() => predictThetaScore(posterior, { slope: 100, intercept: 500 })).toThrow(message);
    expect(() => predictThetaScore(posterior, { ...SCALE, step: 0 })).toThrow(message);
    expect(() => predictThetaScore(posterior, { ...SCALE, lowest: Number.NEGATIVE_INFINITY })).toThrow(message);
  });

  it("reports at most 10,000 scores, whose boundaries each node takes, so that a scale of a few bytes asks no billions", () => {
    const posterior = { nodes: [0], weights: [1] };
    const widest = { slope: 1, intercept: 0, lowest: 0, highest: 9998, step: 1 };
    expect(predictThetaScore(posterior, widest, { standardError: 1 }).distribution).toHaveLength(9999);
    expect(predictThetaScore(posterior, { ...widest, highest: 9999 }).distribution).toHaveLength(10_000);
    expect(() => predictThetaScore(posterior, { ...widest, highest: 10_000 })).toThrow(
      "Predicting a theta score needs a scale of at most 10000 reportable scores, found 10001 from 0 to 10000 by 1; use a larger step.",
    );
    // Without the limit this scale asks for an array of a trillion scores, which JavaScript refuses with a RangeError.
    expect(() => predictThetaScore(posterior, { ...widest, highest: 1e12 })).toThrow(DataError);
  });

  it("rejects an invalid posterior, scale or options", () => {
    const posterior = { nodes: [0], weights: [1] };
    expect(() => predictThetaScore(posterior, SCALE, { standardError: -0.1 })).toThrow(
      "The option standardError must be a non-negative number, found -0.1.",
    );
    expect(() => predictThetaScore(posterior, SCALE, { standardError: JSON.parse('"0.1"') })).toThrow(DataError);
    expect(() => predictThetaScore(posterior, SCALE, { level: 1.5 })).toThrow(DataError);
    expect(() => predictThetaScore(posterior, SCALE, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => predictThetaScore(posterior, SCALE, JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() => predictThetaScore(posterior, JSON.parse("null"))).toThrow(
      "The score scale must be an object { slope, intercept }, found null.",
    );
    expect(() => predictThetaScore(posterior, { ...SCALE, slope: -1 })).toThrow(DataError);
    expect(() => predictThetaScore({ nodes: [0], weights: [2] }, SCALE)).toThrow(
      "The weights of the posterior must sum to 1, found 2.",
    );
    expect(() => predictThetaScore(JSON.parse('{ "nodes": 1, "weights": [1] }'), SCALE)).toThrow(
      "The nodes of the posterior must be an array, found 1.",
    );
  });
});

describe("drawJointThetas", () => {
  const Correlation: readonly (readonly number[])[] = [
    [1, 0.5, 0.3],
    [0.5, 1, 0.4],
    [0.3, 0.4, 1],
  ];

  /** Returns a posterior of the fixture by its name. */
  const marginNamed = (name: string): ThetaDistribution =>
    defined((joint.margins as Record<string, ThetaDistribution>)[name]);

  /** Returns the matrix with `value` off the diagonal and 1 on it. */
  const equicorrelation = (size: number, value: number): number[][] =>
    Array.from({ length: size }, (_, i) => Array.from({ length: size }, (_, j) => (i === j ? 1 : value)));

  /** Returns the mean and variance of the distribution function of a posterior that is linear within the cell of each node. */
  function cellMoments({ nodes, weights }: ThetaDistribution): { mean: number; variance: number } {
    const last = nodes.length - 1;
    const edges = nodes.map((node, k) => (k === 0 ? node : (defined(nodes[k - 1]) + node) / 2));
    edges[0] = 2 * defined(nodes[0]) - (nodes.length > 1 ? defined(edges[1]) : defined(nodes[0]));
    edges.push(2 * defined(nodes[last]) - (nodes.length > 1 ? defined(edges[last]) : defined(nodes[last])));
    const total = weights.reduce((sum, weight) => sum + weight, 0);
    const cells = weights.map((weight, k) => {
      const lower = defined(edges[k]);
      const upper = defined(edges[k + 1]);
      return { weight: weight / total, center: (lower + upper) / 2, width: upper - lower };
    });
    const mean = cells.reduce((sum, { weight, center }) => sum + weight * center, 0);
    const variance = cells.reduce(
      (sum, { weight, center, width }) => sum + weight * ((center - mean) ** 2 + width ** 2 / 12),
      0,
    );
    return { mean, variance };
  }

  /** Returns the column `k` of draws. */
  const column = (draws: readonly (readonly number[])[], k: number): number[] => draws.map((draw) => defined(draw[k]));

  /** Returns the ranks of the values, ties sharing the mean of their places. */
  function ranks(values: readonly number[]): number[] {
    const order = values.map((_, i) => i).sort((x, y) => defined(values[x]) - defined(values[y]));
    const result = new Array<number>(values.length);
    for (let start = 0; start < order.length; ) {
      let end = start;
      while (end + 1 < order.length && values[defined(order[end + 1])] === values[defined(order[start])]) end++;
      for (let k = start; k <= end; k++) result[defined(order[k])] = (start + end) / 2;
      start = end + 1;
    }
    return result;
  }

  /** Returns the Spearman correlation of two columns: the Pearson correlation of their ranks. */
  function spearman(x: readonly number[], y: readonly number[]): number {
    const rx = ranks(x);
    const ry = ranks(y);
    const mx = mean(rx);
    const my = mean(ry);
    let cross = 0;
    let vx = 0;
    let vy = 0;
    rx.forEach((value, i) => {
      const dx = value - mx;
      const dy = defined(ry[i]) - my;
      cross += dx * dy;
      vx += dx * dx;
      vy += dy * dy;
    });
    return cross / Math.sqrt(vx * vy);
  }

  describe("against base R", () => {
    joint.inverse.forEach((reference) => {
      it(`inverts the distribution function of the posterior ${reference.name} as approx and a search of the cells do`, () => {
        const margin = marginOf({ nodes: reference.nodes, weights: reference.weights }, "posterior");
        reference.u.forEach((u, k) => {
          expectClose(quantile(margin, u), defined(reference.theta[k]), JOINT_TRANSFORM_TOLERANCE, `u = ${u}`);
        });
      });
    });

    it("maps the normal scores of mvtnorm's rmvnorm to the thetas that base R gives", () => {
      const margins = joint.transform.margins.map((name) => marginOf(marginNamed(name), name));
      joint.transform.scores.forEach((scores, row) => {
        const thetas = thetasOf(margins, Float64Array.from(scores));
        joint.transform.thetas[row]?.forEach((theta, k) => {
          expectClose(defined(thetas[k]), theta, JOINT_TRANSFORM_TOLERANCE, `row ${row}, posterior ${k + 1}`);
        });
      });
    });

    joint.corners.forEach((reference, c) => {
      it(`gives the joint distribution function of pmvnorm at the corners of case ${c + 1} within ${MONTE_CARLO_ERRORS} standard errors`, () => {
        const posteriors = reference.margins.map(marginNamed);
        const draws = drawJointThetas(posteriors, reference.correlation, CORNER_DRAWS, createRandom(20261029 + c));
        for (const { corner, probability, error } of reference.corners) {
          const share =
            draws.filter((draw) => draw.every((theta, k) => theta <= defined(corner[k]))).length / CORNER_DRAWS;
          const standardError = Math.sqrt((probability * (1 - probability)) / CORNER_DRAWS);
          expect(Math.abs(share - probability), `corner ${corner}`).toBeLessThanOrEqual(
            MONTE_CARLO_ERRORS * standardError + error,
          );
        }
      });
    });

    joint.spearman.forEach((reference) => {
      it(`gives the rank correlation (6 / pi) asin(r / 2) of the normal scores at r = ${reference.correlation}, whatever the posteriors`, () => {
        const draws = drawJointThetas(
          [marginNamed("fine"), marginNamed("uneven")],
          equicorrelation(2, reference.correlation),
          reference.draws,
          createRandom(7),
        );
        expect(Math.abs(spearman(column(draws, 0), column(draws, 1)) - reference.expected)).toBeLessThanOrEqual(
          MONTE_CARLO_ERRORS * reference.standardError,
        );
      });
    });

    it("factors a correlation so that the factor times its transpose is the correlation", () => {
      const cases = [joint.transform.correlation, Correlation, equicorrelation(6, 1), equicorrelation(4, 0), [[1]]];
      for (const correlation of cases) {
        const size = correlation.length;
        const factor = correlationFactor(correlation, size);
        for (let i = 0; i < size; i++) {
          for (let j = 0; j < size; j++) {
            let product = 0;
            for (let k = 0; k < size; k++) product += defined(factor[i * size + k]) * defined(factor[j * size + k]);
            expectClose(product, defined(correlation[i]?.[j]), 1e-12, `entry (${i + 1}, ${j + 1}) of ${size}`);
          }
        }
      }
    });
  });

  describe("draws", () => {
    it("follow each posterior, with the variance of cells that are linear, as the mean and variance of a draw say", () => {
      const names = ["fine", "default", "uneven"];
      const draws = drawJointThetas(names.map(marginNamed), Correlation, CORNER_DRAWS, createRandom(11));
      names.forEach((name, k) => {
        const { mean: expectedMean, variance: expectedVariance } = cellMoments(marginNamed(name));
        const values = column(draws, k);
        const observedMean = mean(values);
        const deviations = values.map((value) => value - observedMean);
        const observedVariance = mean(deviations.map((d) => d * d));
        const fourth = mean(deviations.map((d) => d ** 4));
        const meanError = Math.sqrt(observedVariance / CORNER_DRAWS);
        const varianceError = Math.sqrt((fourth - observedVariance ** 2) / CORNER_DRAWS);
        expect(Math.abs(observedMean - expectedMean), `mean of ${name}`).toBeLessThanOrEqual(
          MONTE_CARLO_ERRORS * meanError,
        );
        expect(Math.abs(observedVariance - expectedVariance), `variance of ${name}`).toBeLessThanOrEqual(
          MONTE_CARLO_ERRORS * varianceError,
        );
      });
    });

    it("add h squared over 12 to the variance of a posterior on points h apart", () => {
      const { nodes, weights } = marginNamed("default");
      const total = weights.reduce((sum, weight) => sum + weight, 0);
      const nodeMean = nodes.reduce((sum, node, k) => sum + (node * defined(weights[k])) / total, 0);
      const nodeVariance = nodes.reduce(
        (sum, node, k) => sum + ((node - nodeMean) ** 2 * defined(weights[k])) / total,
        0,
      );
      expectClose(cellMoments(marginNamed("default")).variance - nodeVariance, 0.2 ** 2 / 12, 1e-12);
    });

    it("are independent at the correlation 0, as a chi-square test of independence finds", () => {
      const names = ["fine", "default", "uneven", "empty_cells"];
      const posteriors = names.map(marginNamed);
      const draws = drawJointThetas(posteriors, equicorrelation(4, 0), 20_000, createRandom(3));
      const bins = 5;
      const cuts = posteriors.map((posterior, k) => {
        const margin = marginOf(posterior, defined(names[k]));
        return Array.from({ length: bins - 1 }, (_, b) => quantile(margin, (b + 1) / bins));
      });
      const binOf = (k: number, theta: number): number => defined(cuts[k]).filter((cut) => theta >= cut).length;
      for (let a = 0; a < names.length; a++) {
        for (let b = a + 1; b < names.length; b++) {
          const table = new Float64Array(bins * bins);
          for (const draw of draws) {
            const cell = binOf(a, defined(draw[a])) * bins + binOf(b, defined(draw[b]));
            table[cell] = (table[cell] ?? 0) + 1;
          }
          const expected = draws.length / (bins * bins);
          const statistic = table.reduce((sum, observed) => sum + (observed - expected) ** 2 / expected, 0);
          expect(chiSquareUpperTail(statistic, (bins - 1) ** 2), `${names[a]} and ${names[b]}`).toBeGreaterThan(0.001);
        }
      }
    });

    it("take the same quantile in every posterior at the correlation 1, for two and six posteriors", () => {
      for (const size of [2, 6]) {
        const base = marginNamed("default");
        const shifts = Array.from({ length: size }, (_, k) => k * 1.5 - 2);
        const posteriors = shifts.map((shift) => ({
          nodes: base.nodes.map((node) => node + shift),
          weights: base.weights,
        }));
        const draws = drawJointThetas(posteriors, equicorrelation(size, 1), 1_000, createRandom(5));
        for (const draw of draws) {
          shifts.forEach((shift, k) => {
            expectClose(defined(draw[k]) - shift, defined(draw[0]) - defined(shifts[0]), 1e-12, `${size} posteriors`);
          });
        }
      }
    });

    it("order the thetas of different posteriors alike at the correlation 1", () => {
      const draws = drawJointThetas(["fine", "uneven"].map(marginNamed), equicorrelation(2, 1), 400, createRandom(9));
      for (const a of draws) {
        for (const b of draws) {
          if (defined(a[0]) < defined(b[0])) expect(defined(a[1])).toBeLessThanOrEqual(defined(b[1]) + 1e-12);
        }
      }
    });

    it("mirror each other about the center of a symmetric posterior at the correlation -1", () => {
      const fine = marginNamed("fine");
      const draws = drawJointThetas([fine, fine], equicorrelation(2, -1), 1_000, createRandom(13));
      for (const draw of draws) expectClose(defined(draw[0]) + defined(draw[1]), 0, 1e-12);
    });

    it("are the one node of a posterior of one node", () => {
      const draws = drawJointThetas(
        [marginNamed("one_node"), marginNamed("fine")],
        equicorrelation(2, 0.5),
        200,
        createRandom(1),
      );
      expect(column(draws, 0)).toEqual(new Array(200).fill(1.5));
    });

    it("never reach a node without weight, and stay inside the cells of the nodes with weight", () => {
      const draws = drawJointThetas([marginNamed("empty_cells")], [[1]], 20_000, createRandom(2));
      // The nodes -2, 0, 3 and 4 have no weight, and the nodes are 1 apart: the cells with weight are (-1.5, -0.5) of
      // node -1 and (0.5, 1.5) and (1.5, 2.5) of nodes 1 and 2.
      for (const [theta] of draws) {
        const inside = [
          [-1.5, -0.5],
          [0.5, 2.5],
        ].some(([lower, upper]) => defined(theta) >= defined(lower) && defined(theta) <= defined(upper));
        expect(inside, `theta ${theta}`).toBe(true);
      }
    });

    it("stay inside the outer edges of a posterior whose weights sum to 1 only within the tolerance", () => {
      const draws = drawJointThetas([marginNamed("rounded_sum")], [[1]], 20_000, createRandom(4));
      const thetas = column(draws, 0);
      expect(Math.max(...thetas)).toBeLessThanOrEqual(1.5 + 1e-12);
      expect(Math.min(...thetas)).toBeGreaterThanOrEqual(-0.15 - 1e-12);
      expect(Math.max(...thetas)).toBeGreaterThan(1.4);
    });

    it("use as many normals as there are posteriors for each draw, whatever the correlation", () => {
      for (const correlation of [0, 0.4, 1]) {
        const source = createRandom(8);
        let normals = 0;
        const counting = {
          uniform: () => source.uniform(),
          normal: () => {
            normals++;
            return source.normal();
          },
        };
        drawJointThetas(["fine", "default", "uneven"].map(marginNamed), equicorrelation(3, correlation), 50, counting);
        expect(normals, `correlation ${correlation}`).toBe(150);
      }
    });

    it("repeat for a seed, differ between seeds and leave the inputs unchanged", () => {
      const posteriors = ["fine", "uneven"].map(marginNamed);
      const frozenCopy = JSON.parse(JSON.stringify(posteriors));
      const correlation = equicorrelation(2, 0.4);
      const first = drawJointThetas(posteriors, correlation, 100, createRandom(21));
      expect(drawJointThetas(posteriors, correlation, 100, createRandom(21))).toEqual(first);
      expect(drawJointThetas(posteriors, correlation, 100, createRandom(22))).not.toEqual(first);
      expect(posteriors).toEqual(frozenCopy);
      expect(correlation).toEqual(equicorrelation(2, 0.4));
      // The type forbids writing to a result; the test checks that no draw is an array of the inputs.
      for (const draw of first)
        Object.assign(
          draw,
          Array.from(draw, () => 0),
        );
      expect(posteriors).toEqual(frozenCopy);
    });

    it("draw one posterior alone from a correlation of one entry", () => {
      const draws = drawJointThetas([marginNamed("fine")], [[1]], 10, createRandom(1));
      expect(draws).toHaveLength(10);
      expect(draws.every((draw) => draw.length === 1)).toBe(true);
    });

    it("take posteriors of the practice items, as thetaPosterior returns them", () => {
      const posterior = thetaPosterior([item(1.2, -0.8), item(0.9, 0), item(1.5, 0.9)], [1, 0, 1]);
      const draws = drawJointThetas([posterior, posterior], equicorrelation(2, 0.8), 500, createRandom(6));
      expect(draws).toHaveLength(500);
      expect(draws.flat().every(Number.isFinite)).toBe(true);
    });
  });

  describe("input", () => {
    const posteriors = [marginNamed("uneven"), marginNamed("default")];
    const correlation = equicorrelation(2, 0.5);
    const random = createRandom(1);

    it("rejects posteriors that are not a non-empty array of valid distributions", () => {
      expect(() => drawJointThetas([], [], 10, random)).toThrow(
        "Drawing jointly needs at least one posterior, found none.",
      );
      expect(() => drawJointThetas(JSON.parse("null"), correlation, 10, random)).toThrow(
        "The posteriors must be an array of distributions, found null.",
      );
      expect(() =>
        drawJointThetas([marginNamed("fine"), { nodes: [0], weights: [2] }], correlation, 10, random),
      ).toThrow("The weights of the posterior 2 must sum to 1, found 2.");
      expect(() => drawJointThetas([JSON.parse("null"), marginNamed("fine")], correlation, 10, random)).toThrow(
        "The posterior 1 must be an object { nodes, weights }, found null.",
      );
    });

    it("rejects nodes that do not ascend strictly", () => {
      expect(() => drawJointThetas([{ nodes: [0, 1, 1], weights: [0.2, 0.3, 0.5] }], [[1]], 10, random)).toThrow(
        "The nodes of the posterior 1 must ascend strictly, found 1 after 1 at 2.",
      );
      expect(() => drawJointThetas([{ nodes: [0, -1], weights: [0.5, 0.5] }], [[1]], 10, random)).toThrow(
        "The nodes of the posterior 1 must ascend strictly, found -1 after 0 at 1.",
      );
    });

    it("rejects a correlation that is not a symmetric matrix with 1 on the diagonal and entries from -1 to 1", () => {
      expect(() => drawJointThetas(posteriors, [[1, 0.5]], 10, random)).toThrow(
        "The correlation needs 2 rows, one for each posterior, found 1.",
      );
      expect(() => drawJointThetas(posteriors, [[1, 0.5], [0.5]], 10, random)).toThrow(
        "Row 2 of the correlation needs 2 entries, found 1.",
      );
      expect(() => drawJointThetas(posteriors, [[1, 0.5], JSON.parse("1")], 10, random)).toThrow(
        "Row 2 of the correlation must be an array, found 1.",
      );
      expect(() => drawJointThetas(posteriors, JSON.parse("0.5"), 10, random)).toThrow(
        "The correlation must be an array of rows, found 0.5.",
      );
      expect(() =>
        drawJointThetas(
          posteriors,
          [
            [1, 0.5],
            [0.4, 1],
          ],
          10,
          random,
        ),
      ).toThrow("The correlation must be symmetric, but entry (1, 2) is 0.5 and entry (2, 1) is 0.4.");
      expect(() =>
        drawJointThetas(
          posteriors,
          [
            [1, 0.5],
            [0.5, 0.9],
          ],
          10,
          random,
        ),
      ).toThrow("Entry (2, 2) of the correlation must be 1, found 0.9.");
      expect(() =>
        drawJointThetas(
          posteriors,
          [
            [1, 1.2],
            [1.2, 1],
          ],
          10,
          random,
        ),
      ).toThrow("Entry (1, 2) of the correlation must be a number from -1 to 1, found 1.2.");
      expect(() => drawJointThetas(posteriors, JSON.parse('[[1, "0.5"], [0.5, 1]]'), 10, random)).toThrow(
        'Entry (1, 2) of the correlation must be a number from -1 to 1, found "0.5".',
      );
      expect(() => drawJointThetas(posteriors, JSON.parse("[[1, null], [0.5, 1]]"), 10, random)).toThrow(DataError);
      expect(() =>
        drawJointThetas(
          posteriors,
          [
            [1, Number.NaN],
            [Number.NaN, 1],
          ],
          10,
          random,
        ),
      ).toThrow(DataError);
    });

    it("rejects a correlation that is not positive semidefinite and names the way to fix it", () => {
      const impossible = [
        [1, 0.9, -0.9],
        [0.9, 1, 0.9],
        [-0.9, 0.9, 1],
      ];
      expect(() => drawJointThetas(["fine", "default", "uneven"].map(marginNamed), impossible, 10, random)).toThrow(
        /The correlation must be positive semidefinite, but has the eigenvalue -[0-9.e-]+\. Pass it through smoothCorrelations/,
      );
    });

    it("rejects a number of draws that is not a positive integer, and a random source that is not a generator", () => {
      for (const draws of [0, -1, 1.5, Number.NaN, JSON.parse('"3"'), JSON.parse("null")]) {
        expect(() => drawJointThetas(posteriors, correlation, draws, random)).toThrow(DataError);
      }
      expect(() => drawJointThetas(posteriors, correlation, 0, random)).toThrow(
        "The number of draws must be a positive integer, found 0.",
      );
      expect(() => drawJointThetas(posteriors, correlation, 10, JSON.parse("{}"))).toThrow(
        "A random number generator must be one from createRandom, found an object.",
      );
      expect(() => drawJointThetas(posteriors, correlation, 10, JSON.parse("null"))).toThrow(DataError);
    });
  });
});

describe("the recipe for plausible values", () => {
  const matrix = buildResponseMatrix(plausible.responses, { itemIds: [...plausible.itemIds] });
  const covariate = Float64Array.from(plausible.covariate);

  /**
   * The recipe of scaling.md: `draws` plausible values for each person, drawn from the posterior given the person's
   * responses and own prior, which is the calibration's regression on the covariate if `conditional` is true.
   */
  function plausibleValues(conditional: boolean): { values: number[][]; eap: number[]; coefficient: number } {
    const fit = calibrate(matrix, {
      model: twoParameterLogistic,
      ...(conditional && { covariates: { x: [...covariate] } }),
    });
    const coefficient = fit.regression?.coefficients.x ?? 0;
    const residualSd = fit.regression?.residualSd ?? 1;
    const random = createRandom(2026);
    const eap: number[] = [];
    const values = matrix.responses.map((row, person) => {
      const prior = { mean: coefficient * (covariate[person] ?? 0), sd: residualSd };
      const posterior = thetaPosterior(fit.items, row, { prior, quadrature: { points: 241 } });
      eap.push(posterior.nodes.reduce((sum, node, q) => sum + node * defined(posterior.weights[q]), 0));
      return drawJointThetas([posterior], [[1]], plausible.draws, random).map(([theta]) => defined(theta));
    });
    return { values, eap, coefficient };
  }

  /** The mean, the variance and the slope on the covariate of the values of one draw, averaged over the draws. */
  function moments(values: readonly (readonly number[])[]): { mean: number; variance: number; slope: number } {
    const draws = Array.from({ length: plausible.draws }, (_, draw) =>
      Float64Array.from(values, (person) => defined(person[draw])),
    );
    return {
      mean: mean(draws.map((draw) => mean(draw))),
      variance: mean(draws.map((draw) => variance(draw))),
      slope: mean(draws.map((draw) => covariance(draw, covariate) / variance(covariate))),
    };
  }

  /** Expects the moments of the values to be a reference's, within the tolerance. */
  function expectMoments(
    actual: { mean: number; variance: number; slope: number },
    expected: { mean: number; variance: number; slope: number },
    label: string,
    tolerance = PLAUSIBLE_MOMENT_TOLERANCE,
  ): void {
    expectClose(actual.mean, expected.mean, tolerance, `${label} mean`);
    expectClose(actual.variance, expected.variance, tolerance, `${label} variance`);
    expectClose(actual.slope, expected.slope, tolerance, `${label} slope`);
  }

  const free = plausibleValues(false);
  const conditional = plausibleValues(true);

  it("draws values with the moments of mirt's and TAM's without covariates", () => {
    const actual = moments(free.values);
    expectMoments(actual, plausible.withoutCovariates.mirt, "mirt");
    expectMoments(actual, plausible.withoutCovariates.tam, "TAM");
    expectMoments(actual, plausible.withoutCovariates.exact, "expectation", PLAUSIBLE_EXACT_TOLERANCE);
    const eap = Float64Array.from(free.eap);
    expectClose(
      variance(eap),
      plausible.withoutCovariates.mirt.eap.variance,
      PLAUSIBLE_ESTIMATE_TOLERANCE,
      "EAP variance",
    );
  });

  it("draws values with the moments of mirt's and TAM's given the covariate in the prior", () => {
    const actual = moments(conditional.values);
    expectMoments(actual, plausible.withCovariates.mirt, "mirt");
    expectMoments(actual, plausible.withCovariates.tam, "TAM");
    expectMoments(actual, plausible.withCovariates.exact, "expectation", PLAUSIBLE_EXACT_TOLERANCE);
    expectClose(conditional.coefficient, plausible.withCovariates.coefficient, PLAUSIBLE_ESTIMATE_TOLERANCE, "β");
    const eap = Float64Array.from(conditional.eap);
    expectClose(
      variance(eap),
      plausible.withCovariates.mirt.eap.variance,
      PLAUSIBLE_ESTIMATE_TOLERANCE,
      "EAP variance",
    );
    expectClose(
      covariance(eap, covariate) / variance(covariate),
      plausible.withCovariates.mirt.eap.slope,
      PLAUSIBLE_ESTIMATE_TOLERANCE,
      "EAP slope",
    );
  });

  it("understates the relation to a covariate that the prior leaves out", () => {
    const withoutCovariate = moments(free.values).slope;
    expect(withoutCovariate).toBeLessThan(0.5);
    expect(withoutCovariate).toBeLessThan(plausible.withCovariates.mirt.slope - 0.25);
  });

  it("keeps the values of a person in the posterior's grid, from the lowest to the highest node", () => {
    const lowest = free.values.reduce((low, person) => Math.min(low, ...person), Number.POSITIVE_INFINITY);
    const highest = free.values.reduce((high, person) => Math.max(high, ...person), Number.NEGATIVE_INFINITY);
    expect(lowest).toBeGreaterThanOrEqual(-6);
    expect(highest).toBeLessThanOrEqual(6);
  });

  it("gives the same values for the same seed", () => {
    expect(plausibleValues(true).values.slice(0, 20)).toStrictEqual(conditional.values.slice(0, 20));
  });
});

describe("the recipe for Rubin's rules", () => {
  const average = (values: readonly number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
  const sampleVariance = (values: readonly number[]) => {
    const center = average(values);
    return values.reduce((sum, value) => sum + (value - center) ** 2, 0) / (values.length - 1);
  };

  /** The recipe of scaling.md: Rubin's rules for the estimates of the sets and their sampling variances. */
  function combine(estimates: readonly number[], variances: readonly number[]) {
    const sets = estimates.length;
    const within = average(variances);
    const between = sampleVariance(estimates);
    const total = within + (1 + 1 / sets) * between;
    const relativeIncrease = ((1 + 1 / sets) * between) / within;
    return {
      estimate: average(estimates),
      within,
      between,
      total,
      standardError: Math.sqrt(total),
      degreesOfFreedom: (sets - 1) * (1 + 1 / relativeIncrease) ** 2,
    };
  }

  const covariate = Float64Array.from(plausible.covariate);
  const sets = Array.from({ length: plausible.rubin.plausibleValues[0]?.length ?? 0 }, (_, set) =>
    Float64Array.from(plausible.rubin.plausibleValues, (person) => defined(person[set])),
  );

  /** The least-squares slope of a set on the covariate and the variance of that slope. */
  function slopeOf(values: Float64Array): { estimate: number; variance: number } {
    const persons = values.length;
    const slope = covariance(values, covariate) / variance(covariate);
    const residuals = values.map(
      (value, person) => value - mean(values) - slope * (defined(covariate[person]) - mean(covariate)),
    );
    const residualVariance = residuals.reduce((sum, residual) => sum + residual ** 2, 0) / (persons - 2);
    return { estimate: slope, variance: residualVariance / ((persons - 1) * variance(covariate)) };
  }

  /** Expects the pooled quantities to be those of the reference. */
  function expectPooled(actual: ReturnType<typeof combine>, expected: typeof plausible.rubin.mean.pooled): void {
    for (const name of ["estimate", "within", "between", "total", "degreesOfFreedom"] as const) {
      expectClose(actual[name], expected[name], RUBIN_TOLERANCE * Math.max(1, Math.abs(expected[name])), name);
    }
    expectClose(actual.standardError, Math.sqrt(expected.total), RUBIN_TOLERANCE, "standard error");
  }

  it("pools the mean of the values and its variance as base R's formulas do", () => {
    const estimates = sets.map((values) => average([...values]));
    const variances = sets.map((values) => variance(values) / values.length);
    estimates.forEach((estimate, set) => {
      expectClose(estimate, defined(plausible.rubin.mean.estimates[set]), RUBIN_TOLERANCE, `estimate of set ${set}`);
      expectClose(
        defined(variances[set]),
        defined(plausible.rubin.mean.variances[set]),
        RUBIN_TOLERANCE,
        `variance of set ${set}`,
      );
    });
    expectPooled(combine(estimates, variances), plausible.rubin.mean.pooled);
  });

  it("pools the slope of the values on a covariate and its variance as base R's formulas do", () => {
    const slopes = sets.map(slopeOf);
    slopes.forEach(({ estimate, variance: sampling }, set) => {
      expectClose(estimate, defined(plausible.rubin.slope.estimates[set]), RUBIN_TOLERANCE, `slope of set ${set}`);
      expectClose(sampling, defined(plausible.rubin.slope.variances[set]), RUBIN_TOLERANCE, `variance of set ${set}`);
    });
    expectPooled(
      combine(
        slopes.map(({ estimate }) => estimate),
        slopes.map(({ variance: sampling }) => sampling),
      ),
      plausible.rubin.slope.pooled,
    );
  });

  it("adds to the within variance the between variance of the sets, scaled by 1 + 1 / m", () => {
    const pooled = combine([1, 2, 3, 4], [0.5, 0.5, 0.5, 0.5]);
    expect(pooled.within).toBe(0.5);
    expect(pooled.between).toBeCloseTo(5 / 3, 12);
    expect(pooled.total).toBeCloseTo(0.5 + 1.25 * (5 / 3), 12);
  });

  it("has infinite degrees of freedom and the within variance when the sets agree", () => {
    const pooled = combine([0.3, 0.3, 0.3], [0.04, 0.04, 0.04]);
    expect(pooled.between).toBe(0);
    expect(pooled.total).toBe(0.04);
    expect(pooled.degreesOfFreedom).toBe(Number.POSITIVE_INFINITY);
  });
});

describe("properties", () => {
  it("norm statistics do not depend on the order of the scores", () => {
    fc.assert(
      fc.property(fc.array(fc.integer({ min: -20, max: 40 }), { minLength: 3, maxLength: 60 }), (scores) => {
        fc.pre(new Set(scores).size > 1);
        const forward = normStatistics(scores);
        const backward = normStatistics([...scores].reverse());
        expect(backward.scores.map(({ score }) => score)).toEqual(forward.scores.map(({ score }) => score));
        expect(backward.mean.estimate).toBeCloseTo(forward.mean.estimate, 10);
        forward.scores.forEach((norm, k) => {
          expect(backward.scores[k]?.percentile.estimate).toBe(norm.percentile.estimate);
          expect(backward.scores[k]?.z.standardError).toBeCloseTo(norm.z.standardError, 10);
        });
      }),
      { numRuns: 50 },
    );
  });
  const thetas = fc.double({ min: -8, max: 8, noNaN: true });

  it("scale scores do not decrease with theta", () => {
    fc.assert(
      fc.property(
        thetas,
        thetas,
        fc.double({ min: 0.1, max: 50, noNaN: true }),
        fc.double({ min: 0, max: 20, noNaN: true }),
        (x, y, slope, step) => {
          const scale = { slope, intercept: 100, lowest: 0, highest: 500, step };
          expect(scaleScore(Math.min(x, y), scale)).toBeLessThanOrEqual(scaleScore(Math.max(x, y), scale));
        },
      ),
    );
  });

  it("percentile ranks lie between 0 and 100, do not decrease with the score and put a sample's lowest score below 50", () => {
    const samples = fc.array(fc.integer({ min: -20, max: 20 }), { minLength: 1, maxLength: 60 });
    fc.assert(
      fc.property(
        samples,
        fc.array(fc.double({ min: -25, max: 25, noNaN: true }), { maxLength: 20 }),
        (sample, scores) => {
          const ordered = [...scores].sort((x, y) => x - y);
          const ranks = percentileRanks(sample, ordered);
          ranks.forEach((rank, i) => {
            expect(rank).toBeGreaterThanOrEqual(0);
            expect(rank).toBeLessThanOrEqual(100);
            if (i > 0) expect(rank).toBeGreaterThanOrEqual(defined(ranks[i - 1]));
          });
          expect(defined(percentileRanks(sample, [Math.min(...sample)])[0])).toBeLessThan(50 + 1e-9);
        },
      ),
    );
  });

  it("percentile ranks are the same in any order of the sample", () => {
    fc.assert(
      fc.property(fc.array(fc.integer({ min: 0, max: 30 }), { minLength: 2, maxLength: 40 }), (sample) => {
        const scores = [-1, 0, 7.5, 15, 30, 31];
        expect(percentileRanks([...sample].reverse(), scores)).toEqual(percentileRanks(sample, scores));
      }),
    );
  });

  it("a predicted form score is a distribution with a mean and an interval inside its scores", () => {
    const posteriors = fc.record({
      mean: fc.double({ min: -2, max: 2, noNaN: true }),
      sd: fc.double({ min: 0.2, max: 2, noNaN: true }),
      pattern: fc.tuple(...FORM.map(() => fc.constantFrom(0, 1, null))),
      level: fc.double({ min: 0.05, max: 0.95, noNaN: true }),
    });
    fc.assert(
      fc.property(posteriors, ({ mean, sd, pattern, level }) => {
        const posterior = thetaPosterior(FORM, pattern, { prior: { mean, sd } });
        const prediction = predictFormScore(posterior, FORM, [200, 350, 500, 800], { level });
        const total = prediction.distribution.reduce((sum, { probability }) => sum + probability, 0);
        expectClose(total, 1, 1e-12);
        expect(prediction.distribution.every(({ probability }) => probability >= 0)).toBe(true);
        expect(prediction.mean).toBeGreaterThanOrEqual(200);
        expect(prediction.mean).toBeLessThanOrEqual(800);
        expect(prediction.lower).toBeLessThanOrEqual(prediction.upper);
      }),
    );
  });

  it("a posterior concentrated at one theta predicts the form's own distribution there", () => {
    fc.assert(
      fc.property(thetas, (theta) => {
        const { distribution } = predictFormScore({ nodes: [theta], weights: [1] }, FORM, [0, 1, 2, 3]);
        enumeratedScores(FORM, theta).forEach((probability, score) => {
          expectClose(defined(distribution[score]).probability, probability, 1e-12);
        });
      }),
    );
  });

  it("a predicted theta score is a distribution on the reportable scores", () => {
    fc.assert(
      fc.property(
        fc.array(fc.double({ min: -4, max: 4, noNaN: true }), { minLength: 1, maxLength: 8 }),
        fc.double({ min: 0, max: 1, noNaN: true }),
        fc.double({ min: 0.05, max: 0.95, noNaN: true }),
        (nodes, standardError, level) => {
          const weights = nodes.map(() => 1 / nodes.length);
          const prediction = predictThetaScore({ nodes, weights }, SCALE, { standardError, level });
          const total = prediction.distribution.reduce((sum, { probability }) => sum + probability, 0);
          expectClose(total, 1, 1e-12);
          expect(prediction.mean).toBeGreaterThanOrEqual(200 - 1e-9);
          expect(prediction.mean).toBeLessThanOrEqual(800 + 1e-9);
          expect(prediction.lower).toBeLessThanOrEqual(prediction.upper);
        },
      ),
    );
  });

  it("the conversion table's error is the definition's, for any form", () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.tuple(fc.double({ min: 0.3, max: 2.5, noNaN: true }), fc.double({ min: -2, max: 2, noNaN: true })),
          {
            minLength: 1,
            maxLength: 6,
          },
        ),
        (parameters) => {
          const items = parameters.map(([a, b]) => item(a, b));
          const rows = conversionTable(items, SCALE);
          const conversion = rows.map(({ scaleScore: score }) => score);
          for (const row of rows) {
            expectClose(row.scaleStandardError, enumeratedDeviation(items, conversion, row.theta), 1e-9);
          }
        },
      ),
    );
  });

  it("growth changes by the slope of a linear rescale of the scale and keeps its effect size and ratio", () => {
    const group = fc.record({
      mean: fc.double({ min: -50, max: 50, noNaN: true }),
      sd: fc.double({ min: 0.5, max: 20, noNaN: true }),
    });
    fc.assert(
      fc.property(
        fc.array(group, { minLength: 2, maxLength: 5 }),
        fc.double({ min: 0.1, max: 10, noNaN: true }),
        fc.double({ min: -100, max: 100, noNaN: true }),
        (moments, slope, intercept) => {
          const label = (g: { mean: number; sd: number }, i: number) => ({ group: `grade ${i}`, ...g });
          const before = scaleGrowth(moments.map(label));
          const after = scaleGrowth(
            moments.map((g, i) => label({ mean: slope * g.mean + intercept, sd: slope * g.sd }, i)),
          );
          before.forEach((row, k) => {
            const moved = defined(after[k]);
            expectClose(moved.meanChange, slope * row.meanChange, 1e-9);
            expectClose(moved.sdRatio, row.sdRatio, 1e-12);
            expectClose(moved.effectSize, row.effectSize, 1e-9);
          });
        },
      ),
    );
  });

  describe("of joint draws", () => {
    /** A posterior of 1 to 10 strictly ascending nodes and non-negative weights, one at least above 0, that sum to 1. */
    const posteriors = fc
      .tuple(
        fc.uniqueArray(fc.double({ min: -5, max: 5, noNaN: true }), {
          minLength: 1,
          maxLength: 10,
          comparator: "SameValueZero",
        }),
        fc.array(fc.integer({ min: 0, max: 20 }), { minLength: 10, maxLength: 10 }),
      )
      .map(([values, masses]): ThetaDistribution => {
        const nodes = [...values].sort((x, y) => x - y);
        const used = masses.slice(0, nodes.length);
        if (!used.some((mass) => mass > 0)) used[0] = 1;
        const total = used.reduce((sum, mass) => sum + mass, 0);
        return { nodes, weights: used.map((mass) => mass / total) };
      });
    const probabilities = fc.double({ min: 0, max: 1, noNaN: true });
    const correlations = fc.double({ min: -1, max: 1, noNaN: true });

    /** Returns the lowest and highest theta of a posterior's cells: its outer edges. */
    function outerEdges({ nodes }: ThetaDistribution): [number, number] {
      const first = defined(nodes[0]);
      const last = defined(nodes[nodes.length - 1]);
      if (nodes.length === 1) return [first, last];
      return [first - (defined(nodes[1]) - first) / 2, last + (last - defined(nodes[nodes.length - 2])) / 2];
    }

    it("the theta at a probability does not decrease with the probability and stays within the outer edges", () => {
      fc.assert(
        fc.property(posteriors, probabilities, probabilities, (posterior, x, y) => {
          const margin = marginOf(posterior, "posterior");
          const [lower, upper] = outerEdges(posterior);
          const [low, high] = x <= y ? [x, y] : [y, x];
          expect(quantile(margin, low)).toBeLessThanOrEqual(quantile(margin, high) + 1e-9);
          expect(quantile(margin, low)).toBeGreaterThanOrEqual(lower - 1e-9);
          expect(quantile(margin, high)).toBeLessThanOrEqual(upper + 1e-9);
        }),
      );
    });

    it("draws stay within the outer edges of their posteriors and repeat for a seed, for any correlation", () => {
      fc.assert(
        fc.property(posteriors, posteriors, correlations, fc.integer({ min: 0, max: 2 ** 31 }), (p, q, r, seed) => {
          const correlation = [
            [1, r],
            [r, 1],
          ];
          const draws = drawJointThetas([p, q], correlation, 20, createRandom(seed));
          expect(drawJointThetas([p, q], correlation, 20, createRandom(seed))).toEqual(draws);
          const [lowerP, upperP] = outerEdges(p);
          const [lowerQ, upperQ] = outerEdges(q);
          for (const [x, y] of draws) {
            expect(defined(x)).toBeGreaterThanOrEqual(lowerP - 1e-9);
            expect(defined(x)).toBeLessThanOrEqual(upperP + 1e-9);
            expect(defined(y)).toBeGreaterThanOrEqual(lowerQ - 1e-9);
            expect(defined(y)).toBeLessThanOrEqual(upperQ + 1e-9);
          }
        }),
      );
    });

    it("draws at the correlation 1 order the thetas of two posteriors alike", () => {
      fc.assert(
        fc.property(posteriors, posteriors, fc.integer({ min: 0, max: 2 ** 31 }), (p, q, seed) => {
          const correlation = [
            [1, 1],
            [1, 1],
          ];
          const draws = drawJointThetas([p, q], correlation, 30, createRandom(seed));
          for (const a of draws) {
            for (const b of draws) {
              if (defined(a[0]) < defined(b[0])) expect(defined(a[1])).toBeLessThanOrEqual(defined(b[1]) + 1e-9);
            }
          }
        }),
      );
    });
  });
});

import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  type ConversionTableOptions,
  conditionalScaleError,
  conversionTable,
  createRandom,
  DataError,
  generalizedPartialCredit,
  type IrtItem,
  nominalResponse,
  percentileRanks,
  predictFormScore,
  predictThetaScore,
  type ScorePrediction,
  type ScoreScale,
  scaleConversion,
  scaleScore,
  summedScoreTable,
  type ThetaDistribution,
  thetaPosterior,
  twoParameterLogistic,
} from "../src/index.js";
import { normalCdf } from "../src/math/normal.js";
import conversions from "./fixtures/scaling-conversion.json" with { type: "json" };
import norms from "./fixtures/scaling-norms.json" with { type: "json" };
import predictions from "./fixtures/scaling-prediction.json" with { type: "json" };
import { defined, expectClose, type FixtureItems, fixtureItems } from "./helpers.js";

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

  it("rejects an invalid theta or scale", () => {
    expect(() => scaleScore(Number.NaN, SCALE)).toThrow("Theta must be a finite number, found NaN.");
    expect(() => scaleScore(JSON.parse('"1"'), SCALE)).toThrow('Theta must be a finite number, found "1".');
    expect(() => scaleScore(0, JSON.parse("null"))).toThrow(
      "The score scale must be an object { slope, intercept }, found null.",
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
      const actual = scaleConversion(reference.equivalents, reference.conversion, scaleOf(reference.scale));
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
    expect(() => scaleConversion([0], [1, 2], { step: -5 })).toThrow(DataError);
    expect(() => scaleConversion([0], [1, 2], { lowest: 9, highest: 3 })).toThrow(
      "The options needs a lowest score below the highest, found 9 and 3.",
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

describe("properties", () => {
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
});

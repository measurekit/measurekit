import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  type CalibratedItem,
  calibrate,
  createRandom,
  DataError,
  generalizedPartialCredit,
  type ItemFeatureRegression,
  type ItemFeatures,
  itemFeatureRegression,
  normalPrior,
  oneParameterLogistic,
  predictItemParameter,
  type Random,
  type ResponseMatrix,
  responseMatrix,
  simulateResponses,
  twoParameterLogistic,
} from "../src/index.js";
import fixture from "./fixtures/explanatory-regression.json" with { type: "json" };
import { defined, expectClose } from "./helpers.js";

// Tolerances against R, explained in docs/validation.md.
/** Effects, standard errors, covariances, residual variance, log-likelihood and predictions against base R. */
const REGRESSION_TOLERANCE = 1e-10;

/** A 2PL item of a calibration with standard errors of a and b. */
function calibrated(id: string, b: number, standardError: number, a = 1, slopeError = 0.1): CalibratedItem<unknown> {
  return {
    id,
    model: twoParameterLogistic,
    parameters: { a, b, c: 0, d: 1 },
    standardErrors: { a: slopeError, b: standardError },
  };
}

/** Returns the items of a fixture case as a calibration gives them. */
function fixtureItems(items: (typeof fixture.cases)[number]["items"]): CalibratedItem<unknown>[] {
  return items.map(({ id, parameters, standardErrors }) => ({
    id,
    model: twoParameterLogistic,
    parameters,
    standardErrors,
  }));
}

/** Returns the features of a fixture case by item id. */
function featuresOf(reference: (typeof fixture.cases)[number]): ItemFeatures {
  return JSON.parse(JSON.stringify(reference.features));
}

/** Returns a regression of the case, as the fixture's items and options give it. */
function regressionOf(reference: (typeof fixture.cases)[number]): ItemFeatureRegression {
  return itemFeatureRegression(fixtureItems(reference.items), featuresOf(reference), {
    parameter: reference.parameter,
    intercept: reference.intercept,
  });
}

/** Simulates the responses of persons with the given thetas to 2PL items q1 to qN with slope 1 and these difficulties. */
function simulatedResponses(
  difficulties: readonly number[],
  thetas: readonly number[],
  random: Random,
): ResponseMatrix {
  const items = difficulties.map((b) => ({ model: twoParameterLogistic, parameters: { a: 1, b, c: 0, d: 1 } }));
  const { responses } = simulateResponses(items, thetas, random);
  return responseMatrix(responses, { itemIds: difficulties.map((_, i) => `q${i + 1}`) });
}

/** The effects and residual sd of the simulated banks: b = intercept + steps · steps + open · open + τ e. */
const TRUTH = { intercept: -0.5, steps: 0.4, open: 0.5, tau: 0.3 };

/** Draws items with the features steps and open and the difficulties of the model of TRUTH. */
function simulatedBank(count: number, random: Random): { steps: number; open: number; b: number }[] {
  return Array.from({ length: count }, () => {
    const steps = 1 + Math.floor(random.uniform() * 6);
    const open = random.uniform() < 0.4 ? 1 : 0;
    const b = TRUTH.intercept + TRUTH.steps * steps + TRUTH.open * open + TRUTH.tau * random.normal();
    return { steps, open, b };
  });
}

/** Features of items q1 to qN from a list of values of one feature. */
function oneFeature(values: readonly number[], name = "x"): ItemFeatures {
  return Object.fromEntries(values.map((value, i) => [`q${i + 1}`, { [name]: value }]));
}

describe("itemFeatureRegression", () => {
  for (const reference of fixture.cases) {
    it(`matches the base R regression of ${reference.name}`, () => {
      const actual = regressionOf(reference);
      expect(actual.converged).toBe(true);
      expect(actual.parameter).toBe(reference.parameter);
      expect(actual.scale).toBe(reference.parameter === "a" ? "log" : "identity");
      expect(actual.itemIds).toEqual(reference.items.map(({ id }) => id));
      expect(actual.effects.map(({ name }) => name)).toEqual(reference.effects.map(({ name }) => name));
      reference.effects.forEach((expected, j) => {
        const effect = defined(actual.effects[j]);
        expectClose(effect.estimate, expected.estimate, REGRESSION_TOLERANCE, `${expected.name} estimate`);
        expectClose(effect.standardError, expected.standardError, REGRESSION_TOLERANCE, `${expected.name} error`);
      });
      reference.covariance.forEach((row, j) => {
        row.forEach((value, k) => {
          expectClose(defined(actual.covariance[j]?.[k]), value, REGRESSION_TOLERANCE, `covariance (${j}, ${k})`);
        });
      });
      expectClose(actual.residualSd ** 2, reference.residualSd ** 2, REGRESSION_TOLERANCE, "residual variance");
      expectClose(actual.logLikelihood, reference.logLikelihood, REGRESSION_TOLERANCE, "log-likelihood");
    });
  }

  it("holds the residual variance at 0 where the features explain the variation between items", () => {
    const exact = defined(fixture.cases.find(({ name }) => name.includes("no residual")));
    const actual = regressionOf(exact);
    expect(actual.residualSd).toBe(0);
    expect(actual.converged).toBe(true);
    expect(actual.iterations).toBe(1);
  });

  it("estimates a mean and a residual variance from equal sampling variances as the closed forms give them", () => {
    const y = [0.2, 0.9, 1.4, 2.1, 2.6, 3.0];
    const variance = 0.04;
    const items = y.map((b, i) => calibrated(`q${i + 1}`, b, Math.sqrt(variance)));
    const features = Object.fromEntries(y.map((_, i) => [`q${i + 1}`, {}]));
    const regression = itemFeatureRegression(items, features);
    const mean = y.reduce((sum, value) => sum + value, 0) / y.length;
    const spread = y.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (y.length - 1);
    expectClose(defined(regression.effects[0]).estimate, mean, 1e-9);
    expectClose(regression.residualSd ** 2, spread - variance, 1e-9);
    expectClose(defined(regression.effects[0]).standardError, Math.sqrt(spread / y.length), 1e-9);
    expect(defined(regression.effects[0]).name).toBe("(intercept)");
  });

  it("estimates a line from equal sampling variances as least squares and the residual variance less the sampling one", () => {
    const x = [1, 2, 3, 4, 5, 6];
    const y = [0.3, 0.55, 1.2, 1.35, 2.05, 2.2];
    const variance = 0.01;
    const items = y.map((b, i) => calibrated(`q${i + 1}`, b, Math.sqrt(variance)));
    const regression = itemFeatureRegression(items, oneFeature(x));
    const meanX = x.reduce((sum, value) => sum + value, 0) / x.length;
    const meanY = y.reduce((sum, value) => sum + value, 0) / y.length;
    const sxx = x.reduce((sum, value) => sum + (value - meanX) ** 2, 0);
    const slope = x.reduce((sum, value, i) => sum + (value - meanX) * (defined(y[i]) - meanY), 0) / sxx;
    const intercept = meanY - slope * meanX;
    const residuals = x.map((value, i) => defined(y[i]) - intercept - slope * value);
    const residualVariance = residuals.reduce((sum, value) => sum + value ** 2, 0) / (x.length - 2);
    expectClose(defined(regression.effects[0]).estimate, intercept, 1e-9);
    expectClose(defined(regression.effects[1]).estimate, slope, 1e-9);
    expectClose(regression.residualSd ** 2, residualVariance - variance, 1e-9);
    expect(regression.effects.map(({ name }) => name)).toEqual(["(intercept)", "x"]);
  });

  it("gives the weighted mean and no residual variance when the sampling variances swamp the variation", () => {
    const y = [0.5, 0.52, 0.48, 0.5, 0.51];
    const items = y.map((b, i) => calibrated(`q${i + 1}`, b, 1));
    const regression = itemFeatureRegression(items, Object.fromEntries(y.map((_, i) => [`q${i + 1}`, {}])));
    expect(regression.residualSd).toBe(0);
    expectClose(defined(regression.effects[0]).estimate, y.reduce((sum, value) => sum + value, 0) / y.length, 1e-12);
  });

  it("regresses the slope as its logarithm, with the standard error of the delta method", () => {
    const slopes = [0.8, 1.1, 1.5, 0.9, 1.3, 1.7, 1.0];
    const errors = [0.1, 0.12, 0.2, 0.09, 0.15, 0.25, 0.1];
    const x = [1, 2, 3, 4, 5, 6, 7];
    const onSlopes = slopes.map((a, i) => calibrated(`q${i + 1}`, 0, 0.1, a, defined(errors[i])));
    const onLogs = slopes.map((a, i) => calibrated(`q${i + 1}`, Math.log(a), defined(errors[i]) / a));
    const slope = itemFeatureRegression(onSlopes, oneFeature(x), { parameter: "a" });
    const difficulty = itemFeatureRegression(onLogs, oneFeature(x));
    expect(slope.scale).toBe("log");
    expect(slope.effects).toEqual(difficulty.effects);
    expect(slope.covariance).toEqual(difficulty.covariance);
    expect(slope.residualSd).toBe(difficulty.residualSd);
    expect(slope.logLikelihood).toBe(difficulty.logLikelihood);
  });

  it("regresses a threshold of polytomous items, leaving out items whose model has no such parameter", () => {
    const polytomous = (id: string, shift: number): CalibratedItem<unknown> => ({
      id,
      model: generalizedPartialCredit,
      parameters: { a: 1, b: [-0.5 + shift, 0.6 + shift] },
      standardErrors: { a: 0.1, b1: 0.1, b2: 0.1 },
    });
    const polytomousItems = [polytomous("p1", 0), polytomous("p2", 0.4), polytomous("p3", 0.9), polytomous("p4", 1.3)];
    const dichotomousItems = [
      calibrated("d1", 0, 0.1),
      calibrated("d2", 0.5, 0.1),
      calibrated("d3", 1, 0.1),
      calibrated("d4", 1.4, 0.1),
    ];
    const items = [...polytomousItems, ...dichotomousItems];
    const features = Object.fromEntries(items.map(({ id }, i) => [id, { x: i % 4 }]));
    const thresholds = itemFeatureRegression(items, features, { parameter: "b1" });
    expect(thresholds.itemIds).toEqual(["p1", "p2", "p3", "p4"]);
    expect(thresholds.scale).toBe("identity");
    // The first thresholds -0.5, -0.1, 0.4, 0.8 on x = 0 to 3 have the least squares slope 2.2 / 5.
    expectClose(defined(thresholds.effects[1]).estimate, 0.44, 1e-9);
    const difficulties = itemFeatureRegression(items, features);
    expect(difficulties.itemIds).toEqual(["d1", "d2", "d3", "d4"]);
    expectClose(defined(difficulties.effects[1]).estimate, 0.4, 0.1);
  });

  it("leaves out items without standard errors, such as the anchors of a calibration", () => {
    const random = createRandom(31);
    const truth = Array.from({ length: 12 }, (_, i) => -1.2 + 0.2 * i);
    const bank = truth.map((b, i) => ({ id: `q${i + 1}`, parameters: { a: 1, b, c: 0, d: 1 } }));
    const responses = simulatedResponses(
      truth,
      Array.from({ length: 800 }, () => random.normal()),
      random,
    );
    const fixed = { q1: defined(bank[0]).parameters, q2: defined(bank[1]).parameters };
    const fit = calibrate(responses, { model: twoParameterLogistic, fixed, standardErrors: "cross-product" });
    const features = oneFeature(truth.map((_, i) => i));
    const regression = itemFeatureRegression(fit.items, features);
    expect(regression.itemIds).toEqual(bank.slice(2).map(({ id }) => id));
    expectClose(defined(regression.effects[1]).estimate, 0.2, 0.05);
    expect(() => itemFeatureRegression(fit.items.slice(0, 2), features)).toThrow(
      'Regressing "b" needs more items with standard errors than effects, found 0 items for 1 effects.',
    );
  });

  it("ignores features of items outside the calibration and extra features it does not use for the others", () => {
    const items = [0.1, 0.5, 0.9, 1.4, 1.6].map((b, i) => calibrated(`q${i + 1}`, b, 0.1));
    const features = { ...oneFeature([1, 2, 3, 4, 5]), other: { x: 99 } };
    expect(itemFeatureRegression(items, features).itemIds).toHaveLength(5);
  });

  it("orders the effects by their first appearance among the items", () => {
    const items = [0.1, 0.5, 0.9, 1.4, 1.6, 1.9].map((b, i) => calibrated(`q${i + 1}`, b, 0.1));
    const features = {
      q1: { second: 1, first: 0.5 },
      q2: { first: 2, second: 0 },
      q3: { first: 3, second: 4 },
      q4: { first: 1, second: 2 },
      q5: { first: 4, second: 1 },
      q6: { first: 2, second: 5 },
    };
    expect(itemFeatureRegression(items, features).effects.map(({ name }) => name)).toEqual([
      "(intercept)",
      "second",
      "first",
    ]);
    expect(itemFeatureRegression(items, features, { intercept: false }).effects.map(({ name }) => name)).toEqual([
      "second",
      "first",
    ]);
  });

  it("reports non-convergence within the step limit, and converges faster with a looser tolerance", () => {
    const reference = defined(fixture.cases[0]);
    const items = fixtureItems(reference.items);
    const features = featuresOf(reference);
    const limited = itemFeatureRegression(items, features, { maxIterations: 2 });
    expect(limited).toMatchObject({ converged: false, iterations: 2 });
    const exact = itemFeatureRegression(items, features);
    const loose = itemFeatureRegression(items, features, { tolerance: 1e-3 });
    expect(loose.iterations).toBeLessThan(exact.iterations);
    expectClose(loose.residualSd, exact.residualSd, 1e-3);
  });

  it("recovers the effects and the residual of a bank from a calibration of simulated responses", () => {
    const random = createRandom(20261031);
    const bank = simulatedBank(60, random);
    const responses = simulatedResponses(
      bank.map(({ b }) => b),
      Array.from({ length: 3000 }, () => random.normal()),
      random,
    );
    const fit = calibrate(responses, { model: twoParameterLogistic, standardErrors: "cross-product" });
    const features = Object.fromEntries(bank.map(({ steps, open }, i) => [`q${i + 1}`, { steps, open }]));
    const regression = itemFeatureRegression(fit.items, features);
    const [intercept, steps, open] = regression.effects;
    for (const [effect, expected] of [
      [intercept, TRUTH.intercept],
      [steps, TRUTH.steps],
      [open, TRUTH.open],
    ] as const) {
      const found = defined(effect);
      expect(Math.abs(found.estimate - expected), found.name).toBeLessThan(4 * defined(found.standardError));
    }
    expect(Math.abs(regression.residualSd - TRUTH.tau)).toBeLessThan(0.1);
  });

  it("gives 90% prediction intervals that cover the difficulties of held-out items 85% to 95% of the time", () => {
    const random = createRandom(20261031);
    let covered = 0;
    let total = 0;
    for (let replication = 0; replication < 200; replication++) {
      const bank = simulatedBank(40, random);
      const items = bank.map(({ b }, i) => {
        const error = 0.08 + 0.1 * random.uniform();
        return calibrated(`q${i + 1}`, b + error * random.normal(), error);
      });
      const features = Object.fromEntries(bank.map(({ steps, open }, i) => [`q${i + 1}`, { steps, open }]));
      const regression = itemFeatureRegression(items, features);
      for (const { steps, open, b } of simulatedBank(100, random)) {
        const { mean, sd } = predictItemParameter(regression, { steps, open });
        if (Math.abs(b - mean) <= 1.6448536269514722 * sd) covered++;
        total++;
      }
    }
    expect(covered / total).toBeGreaterThanOrEqual(0.85);
    expect(covered / total).toBeLessThanOrEqual(0.95);
  });

  it("rejects items, features and options that are invalid", () => {
    const items = [0.1, 0.5, 0.9, 1.4].map((b, i) => calibrated(`q${i + 1}`, b, 0.1));
    const features = oneFeature([1, 2, 3, 4]);
    expect(() => itemFeatureRegression(JSON.parse("null"), features)).toThrow(
      "The items must be an array of calibrated items, found null.",
    );
    expect(() => itemFeatureRegression(items, features, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => itemFeatureRegression(items, features, { parameter: JSON.parse("5") })).toThrow(
      "The option parameter must be the name of a parameter, found 5.",
    );
    expect(() => itemFeatureRegression(items, features, { parameter: "" })).toThrow(DataError);
    expect(() => itemFeatureRegression(items, features, { intercept: JSON.parse('"yes"') })).toThrow(
      'The option intercept must be a boolean, found "yes".',
    );
    expect(() => itemFeatureRegression(items, features, { tolerance: -1 })).toThrow(
      "The option tolerance must be a non-negative number, found -1.",
    );
    expect(() => itemFeatureRegression(items, features, { maxIterations: JSON.parse('"9"') })).toThrow(DataError);
    expect(() => itemFeatureRegression(items, JSON.parse("null"))).toThrow(
      "The features must be an object of features by item id, found null.",
    );
    expect(() => itemFeatureRegression(items, { ...features, q2: JSON.parse("5") })).toThrow(
      'Item "q2" needs features, an object of numbers, found 5.',
    );
  });

  it("rejects items that cannot be regressed", () => {
    const items = [0.1, 0.5, 0.9, 1.4].map((b, i) => calibrated(`q${i + 1}`, b, 0.1));
    const features = oneFeature([1, 2, 3, 4]);
    expect(() => itemFeatureRegression([], features)).toThrow('No item has the parameter "b".');
    expect(() => itemFeatureRegression(items, features, { parameter: "c" })).toThrow('No item has the parameter "c".');
    expect(() => itemFeatureRegression([...items, JSON.parse("null")], features)).toThrow(
      "Item 5 needs an id, found null.",
    );
    expect(() => itemFeatureRegression([...items, calibrated("q1", 2, 0.1)], features)).toThrow(
      'Item ids must be unique; "q1" appears twice.',
    );
    expect(() => itemFeatureRegression([...items, JSON.parse('{ "id": 7 }')], features)).toThrow(
      "Item 5 needs an id, found 7.",
    );
    const noError = calibrated("q5", 1.9, 0);
    expect(() => itemFeatureRegression([...items, noError], { ...features, q5: { x: 5 } })).toThrow(
      'The standard error of "b" of item "q5" must be a positive number, found 0.',
    );
    const negativeSlope = calibrated("q5", 1.9, 0.1, -0.4);
    expect(() =>
      itemFeatureRegression([...items, negativeSlope], { ...features, q5: { x: 5 } }, { parameter: "a" }),
    ).toThrow('The parameter "a" of item "q5" must be positive to be regressed as its logarithm, found -0.4.');
    const shared: CalibratedItem<unknown>[] = [1, 2, 3, 4].map((i) => ({
      id: `q${i}`,
      model: oneParameterLogistic,
      parameters: { a: 1.2, b: i / 4, c: 0, d: 1 },
      standardErrors: { a: 0.1, b: 0.1 },
    }));
    expect(() => itemFeatureRegression(shared, features, { parameter: "a" })).toThrow(
      'The parameter "a" of item "q1" is shared by the items of its 1PL model, so it does not vary by item.',
    );
    expect(itemFeatureRegression(shared, features).itemIds).toHaveLength(4);
    expect(() =>
      itemFeatureRegression([{ id: "q1", model: JSON.parse("null"), parameters: {}, standardErrors: null }], features),
    ).toThrow(DataError);
  });

  it("rejects a design with too few items or an effect that other effects determine", () => {
    const items = [0.1, 0.5, 0.9].map((b, i) => calibrated(`q${i + 1}`, b, 0.1));
    expect(() => itemFeatureRegression(items, oneFeature([1, 2, 3]))).not.toThrow();
    expect(() => itemFeatureRegression(items, { q1: { x: 1, y: 2 }, q2: { x: 2, y: 1 }, q3: { x: 3, y: 5 } })).toThrow(
      'Regressing "b" needs more items with standard errors than effects, found 3 items for 3 effects.',
    );
    const four = [...items, calibrated("q4", 1.3, 0.1)];
    expect(() => itemFeatureRegression(four, oneFeature([2, 2, 2, 2]))).toThrow(
      'The effect "x" is all zeros or a linear combination of the effects before it, so the effects have no unique estimates.',
    );
    expect(() => itemFeatureRegression(four, oneFeature([0, 0, 0, 0]), { intercept: false })).toThrow(
      'The effect "x" is all zeros',
    );
    const features = {
      q1: { x: 1, double: 2 },
      q2: { x: 2, double: 4 },
      q3: { x: 3, double: 6 },
      q4: { x: 4, double: 8 },
    };
    expect(() => itemFeatureRegression(four, features)).toThrow(
      'The effect "double" is all zeros or a linear combination',
    );
    expect(() => itemFeatureRegression(four, {}, { intercept: false })).toThrow(
      'Item "q1" needs features, an object of numbers, found undefined.',
    );
    expect(() => itemFeatureRegression(four, { q1: {}, q2: {}, q3: {}, q4: {} }, { intercept: false })).toThrow(
      "The design needs an intercept or at least one feature.",
    );
    expect(itemFeatureRegression(four, { q1: {}, q2: {}, q3: {}, q4: {} }).effects).toHaveLength(1);
  });

  it("rejects features that a regressed item lacks or that are not finite numbers", () => {
    const items = [0.1, 0.5, 0.9, 1.4, 1.8].map((b, i) => calibrated(`q${i + 1}`, b, 0.1));
    const features = oneFeature([1, 2, 3, 4, 5]);
    expect(() => itemFeatureRegression(items, { ...features, q3: {} })).toThrow('Item "q3" lacks the feature "x".');
    expect(() => itemFeatureRegression(items, { ...features, q3: { x: Number.NaN } })).toThrow(
      'The feature "x" of item "q3" must be a finite number, found NaN.',
    );
    expect(() => itemFeatureRegression(items, { ...features, q3: { x: JSON.parse('"3"') } })).toThrow(
      'The feature "x" of item "q3" must be a finite number, found "3".',
    );
    expect(() => itemFeatureRegression(items, { ...features, q3: { "(intercept)": 1, x: 3 } })).toThrow(
      'A feature cannot be named "(intercept)", the name of the intercept.',
    );
    expect(() => itemFeatureRegression(items, oneFeature([1, 2, 3, 4, 5], "constructor"))).not.toThrow();
  });
});

/** A regression with known effects and covariance, as a calibration of items would give it. */
const REGRESSION: ItemFeatureRegression = {
  parameter: "b",
  scale: "identity",
  itemIds: ["q1", "q2", "q3"],
  effects: [
    { name: "(intercept)", estimate: 1, standardError: 0.2 },
    { name: "x", estimate: 2, standardError: 0.3 },
  ],
  covariance: [
    [0.04, 0.01],
    [0.01, 0.09],
  ],
  residualSd: 0.5,
  logLikelihood: -3,
  converged: true,
  iterations: 5,
};

describe("predictItemParameter", () => {
  for (const reference of fixture.cases) {
    it(`matches the base R predictions of new items from the regression of ${reference.name}`, () => {
      const regression = regressionOf(reference);
      for (const expected of reference.predictions) {
        const actual = predictItemParameter(regression, expected.features);
        expectClose(actual.mean, expected.mean, REGRESSION_TOLERANCE, "mean");
        expectClose(actual.sd, expected.sd, REGRESSION_TOLERANCE, "sd");
      }
    });
  }

  it("gives the mean x'η and the sd √(τ² + x' Cov x)", () => {
    const { mean, sd } = predictItemParameter(REGRESSION, { x: 3 });
    expect(mean).toBe(7);
    // 0.25 + [1, 3] Cov [1, 3]ᵀ = 0.25 + 0.04 + 2 · 3 · 0.01 + 9 · 0.09.
    expectClose(sd, Math.sqrt(1.16), 1e-14);
  });

  it("uses no intercept where the regression has none, and ignores features it does not use", () => {
    const withoutIntercept: ItemFeatureRegression = {
      ...REGRESSION,
      effects: [{ name: "x", estimate: 2, standardError: 0.3 }],
      covariance: [[0.09]],
    };
    const { mean, sd } = predictItemParameter(withoutIntercept, { x: 3, unused: 100 });
    expect(mean).toBe(6);
    expectClose(sd, Math.sqrt(0.25 + 9 * 0.09), 1e-14);
  });

  it("gives the residual sd alone for a regression whose effects are known", () => {
    const known: ItemFeatureRegression = {
      ...REGRESSION,
      covariance: [
        [0, 0],
        [0, 0],
      ],
    };
    expect(predictItemParameter(known, { x: 7 })).toEqual({ mean: 15, sd: 0.5 });
  });

  it("never returns a standard deviation that is not a number, whatever the rounding of the covariance", () => {
    const rounded: ItemFeatureRegression = {
      ...REGRESSION,
      residualSd: 0,
      covariance: [
        [-1e-18, 0],
        [0, -1e-18],
      ],
    };
    expect(predictItemParameter(rounded, { x: 1 }).sd).toBe(0);
  });

  it("predicts from a regression that went through JSON", () => {
    const regression = regressionOf(defined(fixture.cases[0]));
    const stored: ItemFeatureRegression = JSON.parse(JSON.stringify(regression));
    const features = { steps: 4, density: 0.2, open: 1 };
    expect(predictItemParameter(stored, features)).toEqual(predictItemParameter(regression, features));
  });

  it("has a standard deviation of at least the residual's, and a mean that is affine in the features", () => {
    const regression = regressionOf(defined(fixture.cases[0]));
    const feature = fc.record({
      steps: fc.double({ min: 0, max: 8, noNaN: true }),
      density: fc.double({ min: -3, max: 3, noNaN: true }),
      open: fc.double({ min: 0, max: 1, noNaN: true }),
    });
    fc.assert(
      fc.property(feature, feature, fc.double({ min: 0, max: 1, noNaN: true }), (first, second, weight) => {
        const a = predictItemParameter(regression, first);
        const b = predictItemParameter(regression, second);
        const mixed = predictItemParameter(regression, {
          steps: weight * first.steps + (1 - weight) * second.steps,
          density: weight * first.density + (1 - weight) * second.density,
          open: weight * first.open + (1 - weight) * second.open,
        });
        expect(a.sd).toBeGreaterThanOrEqual(regression.residualSd - 1e-12);
        expectClose(mixed.mean, weight * a.mean + (1 - weight) * b.mean, 1e-9);
      }),
    );
  });

  it("makes the prior of a calibration of an item that few persons answered", () => {
    const random = createRandom(7);
    const difficulties = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2];
    const all = simulatedResponses(
      difficulties,
      Array.from({ length: 500 }, () => random.normal()),
      random,
    );
    const sparse = responseMatrix(
      all.responses.map((row, person) => row.map((response, item) => (item === 7 && person >= 12 ? null : response))),
      { itemIds: all.itemIds },
    );
    const bank = difficulties.slice(0, 7).map((b, i) => ({
      id: `q${i + 1}`,
      model: twoParameterLogistic,
      parameters: { a: 1, b, c: 0, d: 1 },
      standardErrors: { a: 0.1, b: 0.08 },
    }));
    const { mean, sd } = predictItemParameter(itemFeatureRegression(bank, oneFeature([1, 2, 3, 4, 5, 6, 7])), { x: 8 });
    expectClose(mean, 2, 1e-9);
    expect(sd).toBeGreaterThan(0);
    const fixed = Object.fromEntries(bank.map(({ id, parameters }) => [id, parameters]));
    const options = { model: twoParameterLogistic, fixed } as const;
    const free = calibrate(sparse, options).items[7]?.parameters;
    const informed = calibrate(sparse, { ...options, itemPriors: { q8: { b: normalPrior(mean, sd) } } }).items[7]
      ?.parameters;
    expect(Math.abs(defined(informed).b - mean)).toBeLessThan(Math.abs(defined(free).b - mean));
    const strong = calibrate(sparse, { ...options, itemPriors: { q8: { b: normalPrior(mean, 1e-4) } } }).items[7]
      ?.parameters;
    expectClose(defined(strong).b, mean, 1e-3);
  });

  it("rejects a regression, features or values that are invalid", () => {
    expect(() => predictItemParameter(JSON.parse("null"), { x: 1 })).toThrow(
      "The regression must be an object from itemFeatureRegression, found null.",
    );
    expect(() => predictItemParameter({ ...REGRESSION, effects: JSON.parse("5") }, { x: 1 })).toThrow(
      "The effects of the regression must be an array, found 5.",
    );
    expect(() => predictItemParameter({ ...REGRESSION, effects: [] }, { x: 1 })).toThrow(
      "The regression needs a covariance with a row for each of its effects, found 2 rows for 0 effects.",
    );
    expect(() => predictItemParameter({ ...REGRESSION, covariance: [[0.04, 0.01]] }, { x: 1 })).toThrow(
      "The regression needs a covariance with a row for each of its effects, found 1 rows for 2 effects.",
    );
    expect(() => predictItemParameter({ ...REGRESSION, covariance: JSON.parse("null") }, { x: 1 })).toThrow(DataError);
    expect(() => predictItemParameter({ ...REGRESSION, covariance: [[0.04], [0.01, 0.09]] }, { x: 1 })).toThrow(
      "Row 1 of the covariance of the regression needs 2 entries, found an array.",
    );
    expect(() =>
      predictItemParameter({ ...REGRESSION, covariance: JSON.parse('[[0.04, "a"], [0.01, 0.09]]') }, { x: 1 }),
    ).toThrow('The covariance of the regression needs finite entries, found "a".');
    expect(() => predictItemParameter(REGRESSION, JSON.parse("null"))).toThrow(
      "The features of the item must be an object of numbers by feature name, found null.",
    );
    expect(() => predictItemParameter(REGRESSION, {})).toThrow(
      'The features of the item need a finite number for "x", found undefined.',
    );
    expect(() => predictItemParameter(REGRESSION, { x: Number.NaN })).toThrow(DataError);
    expect(() => predictItemParameter(REGRESSION, { x: JSON.parse('"1"') })).toThrow(
      'The features of the item need a finite number for "x", found "1".',
    );
    expect(() =>
      predictItemParameter({ ...REGRESSION, effects: [JSON.parse("null"), defined(REGRESSION.effects[1])] }, { x: 1 }),
    ).toThrow("Effect 1 of the regression needs a name and a finite estimate, found null.");
    expect(() => predictItemParameter({ ...REGRESSION, residualSd: -1 }, { x: 1 })).toThrow(
      "The residual sd of the regression must be a non-negative number, found -1.",
    );
    expect(() => predictItemParameter({ ...REGRESSION, residualSd: JSON.parse('"0.5"') }, { x: 1 })).toThrow(DataError);
  });
});

describe("properties", () => {
  const banks = fc
    .integer({ min: 5, max: 12 })
    .chain((count) =>
      fc.record({
        estimates: fc.array(fc.double({ min: -3, max: 3, noNaN: true }), { minLength: count, maxLength: count }),
        errors: fc.array(fc.double({ min: 0.05, max: 0.6, noNaN: true }), { minLength: count, maxLength: count }),
        x: fc.array(fc.integer({ min: -4, max: 4 }), { minLength: count, maxLength: count }),
      }),
    )
    .filter(({ x }) => new Set(x).size >= 3);

  it("the estimates do not depend on the order of the items", () => {
    fc.assert(
      fc.property(banks, ({ estimates, errors, x }) => {
        const items = estimates.map((b, i) => calibrated(`q${i + 1}`, b, defined(errors[i])));
        const features = oneFeature(x);
        const forward = itemFeatureRegression(items, features);
        const backward = itemFeatureRegression([...items].reverse(), features);
        expectClose(forward.residualSd ** 2, backward.residualSd ** 2, 1e-8);
        forward.effects.forEach((effect, j) => {
          expectClose(effect.estimate, defined(backward.effects[j]).estimate, 1e-6);
        });
        expectClose(forward.logLikelihood, backward.logLikelihood, 1e-7);
      }),
      { numRuns: 60 },
    );
  });

  it("a shift of the parameter moves the intercept and nothing else", () => {
    fc.assert(
      fc.property(banks, fc.double({ min: -2, max: 2, noNaN: true }), ({ estimates, errors, x }, shift) => {
        const base = itemFeatureRegression(
          estimates.map((b, i) => calibrated(`q${i + 1}`, b, defined(errors[i]))),
          oneFeature(x),
        );
        const shifted = itemFeatureRegression(
          estimates.map((b, i) => calibrated(`q${i + 1}`, b + shift, defined(errors[i]))),
          oneFeature(x),
        );
        expectClose(defined(shifted.effects[0]).estimate, defined(base.effects[0]).estimate + shift, 1e-6);
        expectClose(defined(shifted.effects[1]).estimate, defined(base.effects[1]).estimate, 1e-6);
        expectClose(shifted.residualSd ** 2, base.residualSd ** 2, 1e-7);
        expectClose(shifted.logLikelihood, base.logLikelihood, 1e-7);
      }),
      { numRuns: 60 },
    );
  });

  it("the residual sd is not negative, the covariance is symmetric and the standard errors are its square roots", () => {
    fc.assert(
      fc.property(banks, ({ estimates, errors, x }) => {
        const regression = itemFeatureRegression(
          estimates.map((b, i) => calibrated(`q${i + 1}`, b, defined(errors[i]))),
          oneFeature(x),
        );
        expect(regression.residualSd).toBeGreaterThanOrEqual(0);
        expect(regression.covariance[0]?.[1]).toBe(regression.covariance[1]?.[0]);
        regression.effects.forEach((effect, j) => {
          expectClose(defined(effect.standardError), Math.sqrt(defined(regression.covariance[j]?.[j])), 1e-15);
        });
        expect(Number.isFinite(regression.logLikelihood)).toBe(true);
      }),
      { numRuns: 60 },
    );
  });
});

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
  type LinearChange,
  type LinearChangeOptions,
  type Lltm,
  type LltmOptions,
  linearChange,
  lltm,
  logNormalPrior,
  normalPrior,
  oneParameterLogistic,
  partialCredit,
  predictItemParameter,
  type Random,
  type ResponseMatrix,
  responseMatrix,
  responseRecords,
  simulateResponses,
  simulateThetas,
  type ThresholdItem,
  twoParameterLogistic,
} from "../src/index.js";
import { defined, expectClose } from "./assertions.js";
import changeFixture from "./fixtures/explanatory-linear-change.json" with { type: "json" };
import lltmFixture from "./fixtures/explanatory-lltm.json" with { type: "json" };
import polytomousFixture from "./fixtures/explanatory-lltm-polytomous.json" with { type: "json" };
import fixture from "./fixtures/explanatory-regression.json" with { type: "json" };

// Tolerances against R, explained in docs/validation.md.
/** Effects, standard errors, covariances, residual variance, log-likelihood and predictions against base R. */
const REGRESSION_TOLERANCE = 1e-10;
/** LLTM effects and standard deviation of theta against the base R maximum, which EM at tolerance 1e-10 reaches. */
const LLTM_PARAMETER_TOLERANCE = 1e-7;
/** Their standard errors, relative: numerical Hessians on both sides. */
const LLTM_STANDARD_ERROR_TOLERANCE = 1e-6;
/** LLTM log-likelihoods, information criteria and the likelihood-ratio statistic. */
const LLTM_LOG_LIKELIHOOD_TOLERANCE = 1e-8;
/** Linear change effects against the reference: glm on the persons who changed, or the conditional likelihood in base R. */
const CHANGE_TOLERANCE = 1e-6;
/** Their standard errors, relative: the base R information is a numerical Jacobian. */
const CHANGE_STANDARD_ERROR_TOLERANCE = 1e-5;
/** Conditional log-likelihoods of the linear change cases. */
const CHANGE_LOG_LIKELIHOOD_TOLERANCE = 1e-8;
/** The effects of the polytomous LLTM against TAM, whose quadrature nodes are fixed where the package's follow sigma. */
const TAM_TOLERANCE = 5e-4;
/** EM tolerance of the LLTM comparisons: the base R maxima are exact to 1e-10. */
const LLTM_EM_TOLERANCE = 1e-10;

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

/** The responses of a dataset of the LLTM fixture. */
function lltmData(dataset: number) {
  const { itemIds, responses } = defined(lltmFixture.datasets[dataset]);
  return responseMatrix(responses as (number | null)[][], { itemIds });
}

/** The features of a case of the LLTM fixture. */
function lltmFeatures(reference: (typeof lltmFixture.cases)[number]): ItemFeatures {
  return JSON.parse(JSON.stringify(reference.features));
}

/** Estimates the LLTM of a case of the LLTM fixture, to the tolerance that matches the base R maximum. */
function lltmOf(reference: (typeof lltmFixture.cases)[number], options: Omit<LltmOptions, "model"> = {}): Lltm {
  return lltm(lltmData(reference.dataset), lltmFeatures(reference), {
    intercept: reference.intercept,
    tolerance: LLTM_EM_TOLERANCE,
    maxIterations: 10_000,
    ...options,
  });
}

/** The models of the fixture's cases at the tolerance that matches base R, estimated once. */
const FITTED = new Map<number, Lltm>();

function fitted(index: number): Lltm {
  let model = FITTED.get(index);
  if (model === undefined) {
    model = lltmOf(defined(lltmFixture.cases[index]));
    FITTED.set(index, model);
  }
  return model;
}

interface ChangeReference {
  readonly name: string;
  readonly reference: "glm" | "base R";
  readonly occasions: (number | null)[][][];
  readonly itemIds: string[];
  readonly groups: string[] | null;
  readonly baseline: string | null;
  readonly itemGroups: Record<string, string> | null;
  readonly effects: { itemGroup: string; occasion: number; group: string; estimate: number; standardError: number }[];
  readonly logLikelihood: number;
}
const changeCases: readonly ChangeReference[] = JSON.parse(JSON.stringify(changeFixture.cases));

/** The occasions of a case of the linear change fixture, as response matrices of the same persons and items. */
function changeOccasions({ occasions, itemIds }: ChangeReference): ResponseMatrix[] {
  return occasions.map((rows) => responseMatrix(rows, { itemIds }));
}

/** The options of a case of the linear change fixture. */
function changeOptions({ groups, baseline, itemGroups }: ChangeReference): LinearChangeOptions {
  return {
    ...(groups === null ? {} : { groups, ...(baseline === null ? {} : { baseline }) }),
    ...(itemGroups === null ? {} : { itemGroups }),
  };
}

/** The case of the fixture with this name. */
function changeCase(name: string): ChangeReference {
  return defined(changeCases.find((reference) => reference.name.startsWith(name)));
}

/**
 * Six persons answer 4 items twice. On each item at least one person gains and one loses, so that every effect is
 * finite; the last two persons answer 1 and 0 throughout, and so add nothing.
 */
const SMALL_OCCASIONS = [
  responseMatrix([
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
  ]),
  responseMatrix([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
  ]),
];
const POOLED_ITEMS = { item1: "a", item2: "a", item3: "b", item4: "b" };

describe("linearChange", () => {
  changeCases.forEach((reference) => {
    const source =
      reference.reference === "glm" ? "glm on the persons who changed" : "the conditional likelihood in base R";
    it(`matches ${source} for ${reference.name}`, () => {
      const actual = linearChange(changeOccasions(reference), changeOptions(reference));
      expect(actual.converged).toBe(true);
      expect(actual.effects).toHaveLength(reference.effects.length);
      reference.effects.forEach((expected, i) => {
        const effect = defined(actual.effects[i]);
        const label = `${expected.itemGroup}, occasion ${expected.occasion}, ${expected.group}`;
        expect([effect.itemGroup, effect.occasion, effect.group], label).toEqual([
          expected.itemGroup,
          expected.occasion,
          expected.group,
        ]);
        expectClose(effect.estimate, expected.estimate, CHANGE_TOLERANCE, `${label} estimate`);
        expectClose(
          effect.standardError,
          expected.standardError,
          CHANGE_STANDARD_ERROR_TOLERANCE * expected.standardError,
          `${label} error`,
        );
      });
      expectClose(actual.logLikelihood, reference.logLikelihood, CHANGE_LOG_LIKELIHOOD_TOLERANCE, "log-likelihood");
    });
  });

  it("uses only the persons who changed on an item with two occasions: the others add nothing", () => {
    const base = linearChange(SMALL_OCCASIONS);
    const changers = SMALL_OCCASIONS.map(({ responses }) => responseMatrix(responses.slice(0, 4)));
    const constant = linearChange(changers);
    expect(constant.effects.map(({ estimate }) => estimate)).toEqual(base.effects.map(({ estimate }) => estimate));
    expect(constant.logLikelihood).toBe(base.logLikelihood);
  });

  it("negates every effect when the responses are recoded, 0 as 1", () => {
    const recoded = SMALL_OCCASIONS.map(({ responses }) =>
      responseMatrix(responses.map((row) => row.map((response) => 1 - defined(response)))),
    );
    const base = linearChange(SMALL_OCCASIONS, { itemGroups: POOLED_ITEMS });
    const flipped = linearChange(recoded, { itemGroups: POOLED_ITEMS });
    base.effects.forEach((effect, i) => {
      expectClose(defined(flipped.effects[i]).estimate, -effect.estimate, 1e-9, `effect ${i + 1}`);
    });
    expectClose(flipped.logLikelihood, base.logLikelihood, 1e-9);
  });

  it("moves the baseline to another group as the contrast: the trend gains its effect, which changes sign", () => {
    const reference = changeCase("two occasions");
    const occasions = changeOccasions(reference);
    const groups = defined(reference.groups);
    const control = linearChange(occasions, { groups, baseline: "ctl" });
    const treated = linearChange(occasions, { groups, baseline: "trt" });
    expect(treated.effects.map(({ group }) => group).slice(0, 2)).toEqual(["trt", "ctl"]);
    for (let item = 0; item < 8; item++) {
      const trend = defined(control.effects[2 * item]);
      const effect = defined(control.effects[2 * item + 1]);
      expectClose(
        defined(treated.effects[2 * item]).estimate,
        trend.estimate + effect.estimate,
        1e-6,
        `trend ${item + 1}`,
      );
      expectClose(defined(treated.effects[2 * item + 1]).estimate, -effect.estimate, 1e-6, `effect ${item + 1}`);
    }
    expectClose(treated.logLikelihood, control.logLikelihood, 1e-8);
  });

  it("does not depend on the order of the persons", () => {
    const reference = changeCase("three occasions");
    const occasions = changeOccasions(reference);
    const groups = defined(reference.groups);
    const order = Array.from({ length: groups.length }, (_, p) => (p * 7 + 3) % groups.length);
    expect(new Set(order).size).toBe(groups.length);
    const shuffled = occasions.map(({ responses }) => responseMatrix(order.map((p) => defined(responses[p]))));
    const actual = linearChange(shuffled, { groups: order.map((p) => defined(groups[p])), baseline: "ctl" });
    const expected = linearChange(occasions, { groups, baseline: "ctl" });
    expected.effects.forEach((effect, i) => {
      expectClose(defined(actual.effects[i]).estimate, effect.estimate, 1e-7, `effect ${i + 1}`);
    });
  });

  it("gives the same effects with each item as a set of its own as without itemGroups, under the names of the sets", () => {
    const own = linearChange(SMALL_OCCASIONS, {
      itemGroups: { item1: "own1", item2: "own2", item3: "own3", item4: "own4" },
    });
    const plain = linearChange(SMALL_OCCASIONS);
    expect(own.effects.map(({ estimate }) => estimate)).toEqual(plain.effects.map(({ estimate }) => estimate));
    expect(plain.effects.map(({ itemGroup }) => itemGroup)).toEqual(["item1", "item2", "item3", "item4"]);
    expect(own.effects.map(({ itemGroup }) => itemGroup)).toEqual(["own1", "own2", "own3", "own4"]);
  });

  it("leaves a person out of an item at which a response is missing, as a person who did not change on it", () => {
    const reference = changeCase("two occasions with 10%");
    const occasions = changeOccasions(reference);
    const options = changeOptions(reference);
    const items = defined(occasions[0]).itemIds.length;
    const unchanged = occasions.map(({ responses }, t) =>
      responseMatrix(
        responses.map((row, person) =>
          row.map((response, item) => {
            const missing = occasions.some((occasion) => defined(occasion.responses[person])[item] === null);
            return missing ? t * 0 : response;
          }),
        ),
      ),
    );
    expect(unchanged.every((matrix) => matrix.responses.every((row) => row.length === items))).toBe(true);
    const actual = linearChange(occasions, options);
    const filled = linearChange(unchanged, options);
    actual.effects.forEach((effect, i) => {
      expectClose(defined(filled.effects[i]).estimate, effect.estimate, 1e-9, `effect ${i + 1}`);
    });
    expectClose(filled.logLikelihood, actual.logLikelihood, 1e-9);
  });

  it("names the group of everyone `all`, and takes the first person's group as the baseline", () => {
    expect(new Set(linearChange(SMALL_OCCASIONS).effects.map(({ group }) => group))).toEqual(new Set(["all"]));
    const effects = linearChange(SMALL_OCCASIONS, { groups: ["x", "y", "x", "y", "x", "y"] }).effects;
    expect(effects.map(({ group }) => group)).toEqual(["x", "y", "x", "y", "x", "y", "x", "y"]);
  });

  it("reports an effect at infinity, where every person who changed changed one way, as not converged", () => {
    const before = responseMatrix(Array.from({ length: 40 }, () => [0]));
    const after = responseMatrix(Array.from({ length: 40 }, (_, person) => [person < 20 ? 1 : 0]));
    const actual: LinearChange = linearChange([before, after]);
    expect(actual.converged).toBe(false);
    expect(defined(actual.effects[0]).estimate).toBeGreaterThan(20);
  });

  it("reports non-convergence within the iteration limit, and converges faster at a looser tolerance", () => {
    const reference = changeCase("three occasions");
    const occasions = changeOccasions(reference);
    const options = changeOptions(reference);
    expect(linearChange(occasions, { ...options, maxIterations: 1 })).toMatchObject({
      converged: false,
      iterations: 1,
    });
    const strict = linearChange(occasions, options);
    const loose = linearChange(occasions, { ...options, tolerance: 1e-2 });
    expect(loose.iterations).toBeLessThan(strict.iterations);
  });

  it("rejects options of other types from JavaScript", () => {
    expect(() => linearChange(SMALL_OCCASIONS, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => linearChange(SMALL_OCCASIONS, { tolerance: -1 })).toThrow(
      "The option tolerance must be a non-negative number, found -1.",
    );
    expect(() => linearChange(SMALL_OCCASIONS, { maxIterations: JSON.parse('"5"') })).toThrow(DataError);
    expect(() => linearChange(SMALL_OCCASIONS, { groups: JSON.parse('"ab"') })).toThrow(
      'The option groups must be an array of group names, found "ab".',
    );
    expect(() => linearChange(SMALL_OCCASIONS, { itemGroups: JSON.parse("[]") })).toThrow(
      "The option itemGroups must be an object of set names by item id, found an array.",
    );
  });

  it("rejects occasions that are not 2 to 12 matrices of the same persons and items with responses 0 and 1", () => {
    const [first, second] = SMALL_OCCASIONS;
    expect(() => linearChange(JSON.parse("null"))).toThrow(
      "The occasions must be an array of response matrices, found null.",
    );
    expect(() => linearChange([defined(first)])).toThrow("Linear change needs 2 to 12 occasions, found 1.");
    expect(() => linearChange(Array.from({ length: 13 }, () => defined(first)))).toThrow("found 13");
    expect(() => linearChange([defined(first), JSON.parse("{}")])).toThrow(
      "A response matrix needs the arrays personIds, itemIds and responses.",
    );
    const fewer = responseMatrix(defined(second).responses.slice(0, 5));
    expect(() => linearChange([defined(first), fewer])).toThrow("Occasion 2 lists other persons than occasion 1");
    const narrower = responseMatrix(defined(second).responses.map((row) => row.slice(0, 3)));
    expect(() => linearChange([defined(first), narrower])).toThrow("Occasion 2 lists other items than occasion 1");
    const three = responseMatrix(defined(second).responses.map((row, p) => (p === 1 ? [3, ...row.slice(1)] : row)));
    expect(() => linearChange([defined(first), three])).toThrow(
      'The response of person "person2" to item "item1" at occasion 2 must be 0, 1 or null, found 3.',
    );
  });

  it("rejects groups and sets that do not name every person and item, and a baseline that is nobody's", () => {
    const groups = ["a", "a", "b", "b", "a", "b"];
    expect(() => linearChange(SMALL_OCCASIONS, { groups: groups.slice(0, 5) })).toThrow(
      "The option groups needs a group name for each of the 6 persons.",
    );
    expect(() => linearChange(SMALL_OCCASIONS, { groups, baseline: "c" })).toThrow(
      'The baseline "c" is not the group of any person.',
    );
    expect(() => linearChange(SMALL_OCCASIONS, { itemGroups: { item1: "a" } })).toThrow(
      'The option itemGroups needs a set name for item "item2", found undefined.',
    );
    expect(() => linearChange(SMALL_OCCASIONS, { itemGroups: { ...POOLED_ITEMS, item9: "a" } })).toThrow(
      'The option itemGroups names "item9", which is not an item.',
    );
  });

  it("rejects an effect that no person changed on", () => {
    const before = responseMatrix([
      [0, 0],
      [1, 1],
      [0, 1],
      [1, 0],
    ]);
    const after = responseMatrix([
      [1, 0],
      [1, 1],
      [1, 1],
      [1, 0],
    ]);
    expect(() => linearChange([before, after])).toThrow(
      'The effect of item group "item2" at occasion 2 in group "all" has no unique estimate',
    );
  });
});

interface PolytomousReference {
  readonly name: string;
  readonly model: "partial-credit" | "rating-scale";
  readonly intercept: boolean;
  readonly dataset: number;
  readonly features: ItemFeatures;
  readonly effects: { name: string; estimate: number; standardError: number }[];
  readonly sd: number;
  readonly steps: number[][];
  readonly logLikelihood: number;
  readonly aic: number;
  readonly bic: number;
  readonly raschLogLikelihood: number;
  readonly chiSquare: number;
  readonly degreesOfFreedom: number;
  readonly significance: number;
  readonly tamEstimates: number[];
}
const polytomousCases: readonly PolytomousReference[] = JSON.parse(JSON.stringify(polytomousFixture.cases));

/** The responses of a dataset of the polytomous LLTM fixture. */
function polytomousData(dataset: number): ResponseMatrix {
  const { itemIds, responses } = defined(polytomousFixture.datasets[dataset]);
  return responseMatrix(responses as (number | null)[][], { itemIds });
}

/** Estimates the LLTM of a case of the polytomous fixture, to the tolerance that matches the base R maximum. */
function polytomousModel(reference: PolytomousReference): Lltm<ThresholdItem> {
  return lltm(polytomousData(reference.dataset), reference.features, {
    model: reference.model,
    intercept: reference.intercept,
    tolerance: LLTM_EM_TOLERANCE,
    maxIterations: 10_000,
  });
}

/** The models of the polytomous cases, estimated once. */
const POLYTOMOUS_FITTED = new Map<number, Lltm<ThresholdItem>>();

function polytomousFitted(index: number): Lltm<ThresholdItem> {
  let model = POLYTOMOUS_FITTED.get(index);
  if (model === undefined) {
    model = polytomousModel(defined(polytomousCases[index]));
    POLYTOMOUS_FITTED.set(index, model);
  }
  return model;
}

describe("lltm", () => {
  lltmFixture.cases.forEach((reference, index) => {
    it(`matches the base R maximum for ${reference.name}`, () => {
      const actual = fitted(index);
      expect(actual.converged).toBe(true);
      expect(actual.effects.map(({ name }) => name)).toEqual(reference.effects.map(({ name }) => name));
      reference.effects.forEach((expected, j) => {
        const effect = defined(actual.effects[j]);
        expectClose(effect.estimate, expected.estimate, LLTM_PARAMETER_TOLERANCE, `${expected.name} estimate`);
        expectClose(
          effect.standardError,
          expected.standardError,
          LLTM_STANDARD_ERROR_TOLERANCE * expected.standardError,
          `${expected.name} error`,
        );
      });
      expectClose(actual.sd, reference.sd, LLTM_PARAMETER_TOLERANCE, "sd of theta");
      expect(actual.items.map(({ id }) => id)).toEqual(defined(lltmFixture.datasets[reference.dataset]).itemIds);
      reference.difficulties.forEach((expected, i) => {
        const { model, parameters } = defined(actual.items[i]);
        expect(model).toBe(twoParameterLogistic);
        expect(parameters).toMatchObject({ a: 1, c: 0, d: 1 });
        expectClose(parameters.b, expected, LLTM_PARAMETER_TOLERANCE, `difficulty ${i + 1}`);
      });
      expectClose(actual.logLikelihood, reference.logLikelihood, LLTM_LOG_LIKELIHOOD_TOLERANCE, "log-likelihood");
      expectClose(actual.aic, reference.aic, 2 * LLTM_LOG_LIKELIHOOD_TOLERANCE, "AIC");
      expectClose(actual.bic, reference.bic, 2 * LLTM_LOG_LIKELIHOOD_TOLERANCE, "BIC");
      const { comparison } = actual;
      expectClose(comparison.raschLogLikelihood, reference.raschLogLikelihood, LLTM_LOG_LIKELIHOOD_TOLERANCE, "Rasch");
      expectClose(comparison.chiSquare, reference.chiSquare, 2 * LLTM_LOG_LIKELIHOOD_TOLERANCE, "chi-square");
      expect(comparison.degreesOfFreedom).toBe(reference.degreesOfFreedom);
      expectClose(comparison.significance, reference.significance, 1e-9, "significance");
    });
  });

  it("keeps the features' difficulties against the Rasch model where they hold, and rejects them where they do not", () => {
    const [exact, noisy, withoutIntercept, missing, shifts] = lltmFixture.cases.map(
      (_, index) => fitted(index).comparison,
    );
    expect(defined(exact).significance).toBeGreaterThan(0.05);
    expect(defined(missing).significance).toBeGreaterThan(0.05);
    expect(defined(noisy).significance).toBeLessThan(1e-10);
    expect(defined(withoutIntercept).significance).toBeLessThan(1e-6);
    expect(defined(shifts).significance).toBeLessThan(1e-10);
  });

  it("is nested in the Rasch model: the statistic is not negative and the degrees of freedom are the items less the effects", () => {
    lltmFixture.cases.forEach((reference, index) => {
      const { comparison, effects } = fitted(index);
      expect(comparison.chiSquare).toBeGreaterThanOrEqual(0);
      expect(comparison.degreesOfFreedom).toBe(
        defined(lltmFixture.datasets[reference.dataset]).itemIds.length - effects.length,
      );
    });
  });

  it("converges at the default tolerance to within 1e-3 of the maximum", () => {
    const reference = defined(lltmFixture.cases[0]);
    const actual = lltm(lltmData(0), lltmFeatures(reference));
    expect(actual.converged).toBe(true);
    reference.effects.forEach((expected, j) => {
      expectClose(defined(actual.effects[j]).estimate, expected.estimate, 1e-3, expected.name);
    });
    expectClose(actual.comparison.chiSquare, reference.chiSquare, 1e-3);
  });

  it("gives the same model from response records as from the matrix", () => {
    const reference = defined(lltmFixture.cases[3]);
    const matrix = lltmData(2);
    const records = responseRecords(
      matrix.responses
        .flatMap((row, person) =>
          row.flatMap((response, item) =>
            response === null
              ? []
              : [{ person: defined(matrix.personIds[person]), item: defined(matrix.itemIds[item]), response }],
          ),
        )
        .reverse(),
      { personIds: matrix.personIds, itemIds: matrix.itemIds },
    );
    const options = { tolerance: 1e-6 };
    expect(lltm(records, lltmFeatures(reference), options)).toEqual(lltm(matrix, lltmFeatures(reference), options));
  });

  it("is not changed by persons without responses, which are not counted in the BIC", () => {
    const reference = defined(lltmFixture.cases[0]);
    const matrix = lltmData(0);
    const padded = responseMatrix([...matrix.responses, matrix.itemIds.map(() => null)], { itemIds: matrix.itemIds });
    const options = { tolerance: 1e-6 };
    const base = lltm(matrix, lltmFeatures(reference), options);
    const result = lltm(padded, lltmFeatures(reference), options);
    expect(result.effects).toEqual(base.effects);
    expect(result.logLikelihood).toBe(base.logLikelihood);
    expect(result.bic).toBe(base.bic);
  });

  it("does not depend on the order of the items, the effects being named", () => {
    const reference = defined(lltmFixture.cases[0]);
    const matrix = lltmData(0);
    const order = matrix.itemIds.map((_, i) => (i * 7) % matrix.itemIds.length);
    const permuted = responseMatrix(
      matrix.responses.map((row) => order.map((i) => defined(row[i]))),
      { itemIds: order.map((i) => defined(matrix.itemIds[i])) },
    );
    const options = { tolerance: 1e-10, maxIterations: 10_000 };
    const base = lltm(matrix, lltmFeatures(reference), options);
    const result = lltm(permuted, lltmFeatures(reference), options);
    const byName = (model: Lltm) => Object.fromEntries(model.effects.map(({ name, estimate }) => [name, estimate]));
    for (const [name, estimate] of Object.entries(byName(base))) {
      expectClose(defined(byName(result)[name]), estimate, 1e-7, name);
    }
    expectClose(result.logLikelihood, base.logLikelihood, 1e-8);
    expectClose(result.comparison.chiSquare, base.comparison.chiSquare, 1e-7);
  });

  it("leaves out the standard errors on request, with the same estimates", () => {
    const reference = defined(lltmFixture.cases[0]);
    const none = lltmOf(reference, { standardErrors: "none" });
    const hessian = fitted(0);
    expect(none.effects.every(({ standardError }) => standardError === null)).toBe(true);
    expect(hessian.effects.every(({ standardError }) => (standardError ?? 0) > 0)).toBe(true);
    expect(none.effects.map(({ estimate }) => estimate)).toEqual(hessian.effects.map(({ estimate }) => estimate));
    expect(lltm(lltmData(0), lltmFeatures(reference)).effects[0]?.standardError).not.toBeNull();
  });

  it("estimates without an intercept the mean of theta fixed at 0 and the features' effects only", () => {
    expect(fitted(2).effects.map(({ name }) => name)).toEqual(["steps", "density", "open"]);
    expect(fitted(0).effects.map(({ name }) => name)).toEqual(["(intercept)", "steps", "density", "open"]);
  });

  it("reports non-convergence within the iteration limit, and converges faster at a looser tolerance", () => {
    const reference = defined(lltmFixture.cases[0]);
    const limited = lltm(lltmData(0), lltmFeatures(reference), { maxIterations: 2 });
    expect(limited).toMatchObject({ converged: false, iterations: 2 });
    const loose = lltm(lltmData(0), lltmFeatures(reference), { tolerance: 1e-2 });
    const tight = lltm(lltmData(0), lltmFeatures(reference));
    expect(loose.iterations).toBeLessThan(tight.iterations);
  });

  it("takes the quadrature rule, a coarser Gauss-Hermite rule giving nearly the same model", () => {
    const reference = defined(lltmFixture.cases[0]);
    const rectangular = lltm(lltmData(0), lltmFeatures(reference));
    const hermite = lltm(lltmData(0), lltmFeatures(reference), { quadrature: { method: "gauss-hermite", points: 21 } });
    reference.effects.forEach((_, j) => {
      expectClose(defined(hermite.effects[j]).estimate, defined(rectangular.effects[j]).estimate, 5e-3);
    });
    expectClose(hermite.sd, rectangular.sd, 5e-3);
  });

  it("recovers the effects and the standard deviation of theta of simulated responses", () => {
    const random = createRandom(20261031);
    const steps = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5];
    const open = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1];
    const difficulties = steps.map((count, i) => -0.5 + 0.4 * count + 0.6 * defined(open[i]));
    const matrix = simulatedResponses(
      difficulties,
      Array.from({ length: 2000 }, () => 1.3 * random.normal()),
      random,
    );
    const features = Object.fromEntries(
      steps.map((count, i) => [`q${i + 1}`, { steps: count, open: defined(open[i]) }]),
    );
    const model = lltm(matrix, features);
    const [intercept, stepEffect, openEffect] = model.effects;
    for (const [effect, expected] of [
      [intercept, -0.5],
      [stepEffect, 0.4],
      [openEffect, 0.6],
    ] as const) {
      const found = defined(effect);
      expect(Math.abs(found.estimate - expected), found.name).toBeLessThan(4 * defined(found.standardError));
    }
    expect(Math.abs(model.sd - 1.3)).toBeLessThan(0.1);
    expect(model.comparison.significance).toBeGreaterThan(0.001);
  });

  it("gives items that are the anchors of a calibration on the scale of its theta, N(0, sd²)", () => {
    const model = fitted(0);
    const fixed = Object.fromEntries(model.items.map(({ id, parameters }) => [id, parameters]));
    const calibration = calibrate(lltmData(0), { model: twoParameterLogistic, fixed, latent: "normal" });
    const [group] = calibration.groups;
    expectClose(defined(group).mean, 0, 1e-3);
    expectClose(defined(group).sd, model.sd, 1e-3);
    expectClose(calibration.logLikelihood, model.logLikelihood, 1e-3);
  });

  polytomousCases.forEach((reference, index) => {
    it(`matches the base R maximum, and TAM, for the ${reference.model} model: ${reference.name}`, () => {
      const actual = polytomousFitted(index);
      expect(actual.converged).toBe(true);
      expect(actual.effects.map(({ name }) => name)).toEqual(reference.effects.map(({ name }) => name));
      reference.effects.forEach((expected, j) => {
        const effect = defined(actual.effects[j]);
        expectClose(effect.estimate, expected.estimate, LLTM_PARAMETER_TOLERANCE, `${expected.name} estimate`);
        expectClose(effect.estimate, defined(reference.tamEstimates[j]), TAM_TOLERANCE, `${expected.name} against TAM`);
        expectClose(
          effect.standardError,
          expected.standardError,
          LLTM_STANDARD_ERROR_TOLERANCE * expected.standardError,
          `${expected.name} error`,
        );
      });
      expectClose(actual.sd, reference.sd, LLTM_PARAMETER_TOLERANCE, "sd of theta");
      expect(actual.items.map(({ id }) => id)).toEqual(defined(polytomousFixture.datasets[reference.dataset]).itemIds);
      reference.steps.forEach((expected, i) => {
        const { model, parameters } = defined(actual.items[i]);
        expect(model).toBe(partialCredit);
        expect(parameters.a).toBe(1);
        expect(parameters.b).toHaveLength(expected.length);
        expected.forEach((step, j) => {
          expectClose(defined(parameters.b[j]), step, LLTM_PARAMETER_TOLERANCE, `step ${j + 1} of item ${i + 1}`);
        });
      });
      expectClose(actual.logLikelihood, reference.logLikelihood, LLTM_LOG_LIKELIHOOD_TOLERANCE, "log-likelihood");
      expectClose(actual.aic, reference.aic, 2 * LLTM_LOG_LIKELIHOOD_TOLERANCE, "AIC");
      expectClose(actual.bic, reference.bic, 2 * LLTM_LOG_LIKELIHOOD_TOLERANCE, "BIC");
      const { comparison } = actual;
      expectClose(comparison.raschLogLikelihood, reference.raschLogLikelihood, LLTM_LOG_LIKELIHOOD_TOLERANCE, "Rasch");
      expectClose(comparison.chiSquare, reference.chiSquare, 2 * LLTM_LOG_LIKELIHOOD_TOLERANCE, "chi-square");
      expect(comparison.degreesOfFreedom).toBe(reference.degreesOfFreedom);
      expectClose(comparison.significance, reference.significance, 1e-9, "significance");
    });
  });

  it("keeps the features' locations against the Rasch model where they hold, and rejects them where they do not", () => {
    const [intercept, withoutIntercept, scale, mixed, residual] = polytomousCases.map(
      (_, index) => polytomousFitted(index).comparison,
    );
    expect(defined(intercept).significance).toBeGreaterThan(0.05);
    expect(defined(scale).significance).toBeGreaterThan(0.05);
    expect(defined(mixed).significance).toBeGreaterThan(0.05);
    expect(defined(withoutIntercept).significance).toBeLessThan(1e-3);
    expect(defined(residual).significance).toBeLessThan(1e-10);
  });

  it("gives each item steps whose mean is the location the features give it", () => {
    polytomousCases.forEach((reference, index) => {
      const { effects, items } = polytomousFitted(index);
      Object.entries(reference.features).forEach(([id, features], i) => {
        const location = effects.reduce(
          (sum, { name, estimate }) => sum + estimate * (name === "(intercept)" ? 1 : defined(features[name])),
          0,
        );
        const steps = defined(items[i]).parameters.b;
        expect(steps.reduce((sum, step) => sum + step, 0) / steps.length, id).toBeCloseTo(location, 10);
      });
    });
  });

  it("gives the items of a rating scale the same distances between their steps", () => {
    const { items } = polytomousFitted(2);
    const [first] = items;
    const gaps = (b: readonly number[]): number[] => b.slice(1).map((step, j) => step - defined(b[j]));
    for (const { parameters } of items) {
      gaps(parameters.b).forEach((gap, j) => {
        expect(gap).toBeCloseTo(defined(gaps(defined(first).parameters.b)[j]), 10);
      });
    }
    expect(defined(first).parameters.b).toHaveLength(4);
  });

  it("does not depend on the order of the items under either polytomous model, each item keeping its own steps", () => {
    for (const index of [0, 3]) {
      const reference = defined(polytomousCases[index]);
      const matrix = polytomousData(reference.dataset);
      const order = matrix.itemIds.map((_, i) => matrix.itemIds.length - 1 - i);
      const permuted = responseMatrix(
        matrix.responses.map((row) => order.map((i) => row[i] ?? null)),
        { itemIds: order.map((i) => defined(matrix.itemIds[i])) },
      );
      const base = polytomousFitted(index);
      const result = lltm(permuted, reference.features, {
        model: reference.model,
        intercept: reference.intercept,
        tolerance: LLTM_EM_TOLERANCE,
        maxIterations: 10_000,
      });
      base.effects.forEach(({ estimate }, j) => {
        expectClose(defined(result.effects[j]).estimate, estimate, LLTM_PARAMETER_TOLERANCE, `effect ${j + 1}`);
      });
      expectClose(result.logLikelihood, base.logLikelihood, LLTM_LOG_LIKELIHOOD_TOLERANCE, "log-likelihood");
      order.forEach((original, position) => {
        defined(base.items[original]).parameters.b.forEach((step, j) => {
          expectClose(defined(result.items[position]).parameters.b[j] ?? null, step, LLTM_PARAMETER_TOLERANCE, "step");
        });
      });
    }
  });

  it("gives the dichotomous model from either polytomous model of dichotomous responses", () => {
    const reference = defined(lltmFixture.cases[0]);
    const dichotomous = fitted(0);
    for (const model of ["partial-credit", "rating-scale"] as const) {
      const actual = lltm(lltmData(reference.dataset), lltmFeatures(reference), {
        model,
        intercept: reference.intercept,
        tolerance: LLTM_EM_TOLERANCE,
        maxIterations: 10_000,
      });
      dichotomous.effects.forEach((effect, j) => {
        expectClose(defined(actual.effects[j]).estimate, effect.estimate, LLTM_PARAMETER_TOLERANCE, model);
      });
      expectClose(actual.logLikelihood, dichotomous.logLikelihood, LLTM_LOG_LIKELIHOOD_TOLERANCE, model);
      expect(actual.aic).toBeCloseTo(dichotomous.aic, 6);
      dichotomous.items.forEach(({ parameters }, i) => {
        expectClose(defined(actual.items[i]).parameters.b[0] ?? null, parameters.b, LLTM_PARAMETER_TOLERANCE, model);
      });
    }
  });

  it("counts the free thresholds in the information criteria: one less than the categories, less one, for each set", () => {
    const [partial, , scale, mixed] = [0, 1, 2, 3].map((index) => polytomousFitted(index));
    const parameters = (model: Lltm<ThresholdItem>) => (model.aic - -2 * model.logLikelihood) / 2;
    expect(parameters(defined(partial))).toBe(4 + 1 + 8 * 2);
    expect(parameters(defined(scale))).toBe(4 + 1 + 3);
    expect(parameters(defined(mixed))).toBe(4 + 1 + 5 * 2 + 4 * 1);
  });

  it("recovers the effects and the standard deviation of theta of simulated partial credit responses", () => {
    const random = createRandom(20261105);
    const steps = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 2, 3, 4, 5];
    const open = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1];
    const items = steps.map((count, i) => {
      const location = -0.5 + 0.3 * count + 0.5 * defined(open[i]);
      return { model: partialCredit, parameters: { a: 1, b: [location - 0.7, location + 0.7] } };
    });
    const matrix = simulateResponses(
      items,
      Array.from({ length: 2000 }, () => 1.2 * random.normal()),
      random,
    );
    const features = Object.fromEntries(
      steps.map((count, i) => [matrix.itemIds[i] ?? "", { steps: count, open: defined(open[i]) }]),
    );
    const model = lltm(matrix, features, { model: "partial-credit" });
    for (const [effect, expected] of [
      [model.effects[0], -0.5],
      [model.effects[1], 0.3],
      [model.effects[2], 0.5],
    ] as const) {
      const found = defined(effect);
      expect(Math.abs(found.estimate - expected), found.name).toBeLessThan(4 * defined(found.standardError));
    }
    expect(Math.abs(model.sd - 1.2)).toBeLessThan(0.1);
    expect(model.comparison.significance).toBeGreaterThan(0.001);
    // The first item has one step and is not open: its location is -0.5 + 0.3, and its steps lie 0.7 either side of it.
    defined(model.items[0]).parameters.b.forEach((step, j) => {
      expect(Math.abs(step - defined([-0.9, 0.5][j]))).toBeLessThan(0.25);
    });
  });

  it("rejects a model that is not one of the three, and responses that no model of them takes", () => {
    const reference = defined(polytomousCases[0]);
    const matrix = polytomousData(reference.dataset);
    const features = reference.features;
    expect(() => lltm(matrix, features, { model: JSON.parse('"ordinal"') })).toThrow(
      'The option model must be "dichotomous", "partial-credit" or "rating-scale", found "ordinal".',
    );
    expect(() => lltm(matrix, features, { model: JSON.parse("null") })).toThrow(
      'The option model must be "dichotomous", "partial-credit" or "rating-scale", found null.',
    );
    expect(() => lltm(matrix, features)).toThrow(
      'Item "q01" has a response of 3, but the 2PL model allows responses from 0 to 1.',
    );
    const skipped = responseMatrix(
      matrix.responses.map((row) => row.map((response) => (response === 2 ? 3 : response))),
      { itemIds: matrix.itemIds },
    );
    expect(() => lltm(skipped, features, { model: "partial-credit" })).toThrow(
      'Item "q01" has no response in category 2; calibration needs responses in every category up to the highest.',
    );
  });

  it("rejects a rating scale whose items have different numbers of categories, naming the first that differs", () => {
    const mixed = defined(polytomousCases[3]);
    expect(() => lltm(polytomousData(mixed.dataset), mixed.features, { model: "rating-scale" })).toThrow(
      'The rating scale needs the same number of categories for every item, found 4 for item "q01" and 3 for item "q02".',
    );
  });

  it("rejects options of other types from JavaScript", () => {
    const matrix = lltmData(0);
    const features = lltmFeatures(defined(lltmFixture.cases[0]));
    expect(() => lltm(matrix, features, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => lltm(matrix, features, { intercept: JSON.parse('"yes"') })).toThrow(
      'The option intercept must be a boolean, found "yes".',
    );
    expect(() => lltm(matrix, features, { standardErrors: JSON.parse('"full"') })).toThrow(
      'The option standardErrors must be "none" or "hessian", found "full".',
    );
    expect(() => lltm(matrix, features, { tolerance: -1 })).toThrow(
      "The option tolerance must be a non-negative number, found -1.",
    );
    expect(() => lltm(matrix, features, { maxIterations: JSON.parse('"5"') })).toThrow(DataError);
    expect(() => lltm(matrix, features, { quadrature: { points: 1 } })).toThrow(DataError);
    expect(() => lltm(matrix, JSON.parse("null"))).toThrow(
      "The features must be an object of features by item id, found null.",
    );
    expect(() => lltm(JSON.parse("null"), features)).toThrow(DataError);
  });

  it("rejects responses that are not dichotomous with both categories for every item", () => {
    const features = oneFeature([1, 2, 3, 4]);
    const rows = [
      [1, 0, 1, 0],
      [0, 1, 1, 1],
      [1, 1, 0, 0],
      [0, 0, 1, 1],
      [1, 0, 0, 1],
    ];
    const polytomous = responseMatrix(
      rows.map((row, person) => (person === 0 ? [row[0] ?? 0, 2, row[2] ?? 0, row[3] ?? 0] : row)),
      { itemIds: ["q1", "q2", "q3", "q4"] },
    );
    expect(() => lltm(polytomous, features)).toThrow(
      'Item "q2" has a response of 2, but the 2PL model allows responses from 0 to 1.',
    );
    const allCorrect = responseMatrix(
      rows.map(([, ...rest]) => [1, ...rest]),
      { itemIds: ["q1", "q2", "q3", "q4"] },
    );
    expect(() => lltm(allCorrect, features)).toThrow('Item "q1" has no response in category 0');
    expect(() => lltm(responseMatrix([[null, null]]), oneFeature([1, 2]))).toThrow(
      'Item "item1" has responses in fewer than two categories; calibration needs at least two.',
    );
  });

  it("rejects features that are missing, collinear or too many for the items", () => {
    const matrix = lltmData(0);
    const reference = defined(lltmFixture.cases[0]);
    const features = lltmFeatures(reference);
    expect(() => lltm(matrix, { ...features, q03: JSON.parse("null") })).toThrow(
      'Item "q03" needs features, an object of numbers, found null.',
    );
    expect(() => lltm(matrix, { ...features, q03: { steps: 1 } })).toThrow('Item "q03" lacks the feature "density".');
    const itemIds = ["q1", "q2", "q3", "q4", "q5"];
    const random = createRandom(9);
    const small = simulatedResponses(
      [-1, -0.4, 0, 0.5, 1],
      Array.from({ length: 300 }, () => random.normal()),
      random,
    );
    const names = Object.fromEntries(
      itemIds.map((id, i) => [id, { first: i, second: i * i, third: (i * 3) % 4, fourth: i % 2 }]),
    );
    expect(() => lltm(small, names)).toThrow("An LLTM needs more items than effects, found 5 items for 5 effects.");
    const collinear = Object.fromEntries(itemIds.map((id, i) => [id, { x: i, double: 2 * i }]));
    expect(() => lltm(small, collinear)).toThrow('The effect "double" is all zeros or a linear combination');
    const constant = Object.fromEntries(itemIds.map((id) => [id, { x: 3 }]));
    expect(() => lltm(small, constant)).toThrow('The effect "x" is all zeros or a linear combination');
    expect(lltm(small, constant, { intercept: false }).effects).toHaveLength(1);
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
    const constructorEffect = { ...defined(REGRESSION.effects[1]), name: "constructor" };
    expect(() =>
      predictItemParameter({ ...REGRESSION, effects: [defined(REGRESSION.effects[0]), constructorEffect] }, {}),
    ).toThrow('The features of the item need a finite number for "constructor", found undefined.');
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

describe("documentation examples", () => {
  const item = (id: string, b: number, se: number) => ({
    id,
    model: twoParameterLogistic,
    parameters: { a: 1, b, c: 0, d: 1 },
    standardErrors: { a: 0.1, b: se },
  });
  const bank = [
    item("q1", -1.1, 0.12),
    item("q2", -0.4, 0.1),
    item("q3", 0.2, 0.11),
    item("q4", 0.9, 0.1),
    item("q5", 1.6, 0.13),
  ];
  const stepFeatures = { q1: { steps: 1 }, q2: { steps: 2 }, q3: { steps: 3 }, q4: { steps: 4 }, q5: { steps: 5 } };

  it("regresses the difficulties of a bank of five items on their steps, as the reference of itemFeatureRegression shows", () => {
    const regression = itemFeatureRegression(bank, stepFeatures);
    expect(regression.effects.map(({ name }) => name)).toEqual(["(intercept)", "steps"]);
    expect(defined(regression.effects[0]).estimate).toBeCloseTo(-1.76, 2);
    expect(defined(regression.effects[1]).estimate).toBeCloseTo(0.67, 2);
  });

  it("predicts the sixth step from that regression, as the reference of predictItemParameter shows", () => {
    const { mean, sd } = predictItemParameter(itemFeatureRegression(bank, stepFeatures), { steps: 6 });
    expect(mean).toBeCloseTo(2.24, 2);
    expect(sd).toBeCloseTo(0.12, 2);
  });

  it("explains the difficulties of six items by their steps from 500 simulated persons, as the LLTM example shows", () => {
    const random = createRandom(3);
    const items = [1, 2, 3, 4, 5, 6].map((steps) => ({
      model: twoParameterLogistic,
      parameters: { a: 1, b: -1.5 + 0.5 * steps, c: 0, d: 1 },
    }));
    const responses = simulateResponses(items, simulateThetas(500, random), random);
    const features = {
      item1: { steps: 1 },
      item2: { steps: 2 },
      item3: { steps: 3 },
      item4: { steps: 4 },
      item5: { steps: 5 },
      item6: { steps: 6 },
    };
    const model = lltm(responses, features);
    const [intercept, steps] = model.effects;
    expect([defined(intercept).estimate, defined(intercept).standardError]).toEqual([
      expect.closeTo(-1.56, 2),
      expect.closeTo(0.11, 2),
    ]);
    expect([defined(steps).estimate, defined(steps).standardError]).toEqual([
      expect.closeTo(0.5, 2),
      expect.closeTo(0.03, 2),
    ]);
    expect(model.sd).toBeCloseTo(0.98, 2);
    expect(model.comparison.raschLogLikelihood).toBeCloseTo(-1831.8, 1);
    expect(model.comparison).toMatchObject({ degreesOfFreedom: 4 });
    expect(model.comparison.chiSquare).toBeCloseTo(2.43, 2);
    expect(model.comparison.significance).toBeCloseTo(0.66, 2);
    expect(defined(model.items[0]).parameters).toEqual({ a: 1, b: expect.closeTo(-1.06, 2), c: 0, d: 1 });
  });

  it("estimates the gain of two groups on four items from 800 simulated persons, as the linear change example shows", () => {
    const random = createRandom(5);
    const thetas = simulateThetas(800, random);
    const group = thetas.map((_, person) => (person < 400 ? "control" : "treated"));
    const items = [-0.5, 0, 0.5, 1].map((b) => ({ model: twoParameterLogistic, parameters: { a: 1, b, c: 0, d: 1 } }));
    const before = simulateResponses(items, thetas, random);
    const gain = (person: number) => (group[person] === "treated" ? 0.8 : 0.3);
    const after = simulateResponses(
      items,
      thetas.map((theta, person) => theta + gain(person)),
      random,
    );
    const change = linearChange([before, after], {
      groups: group,
      baseline: "control",
      itemGroups: Object.fromEntries(before.itemIds.map((id) => [id, "test"])),
    });
    const [control, treated] = change.effects;
    expect([defined(control).group, defined(control).estimate, defined(control).standardError]).toEqual([
      "control",
      expect.closeTo(0.35, 2),
      expect.closeTo(0.08, 2),
    ]);
    expect([defined(treated).group, defined(treated).estimate, defined(treated).standardError]).toEqual([
      "treated",
      expect.closeTo(0.35, 2),
      expect.closeTo(0.12, 2),
    ]);
  });

  it("explains the mean steps of six items of three categories by their steps, as the partial credit example shows", () => {
    const random = createRandom(7);
    const items = [1, 2, 3, 4, 5, 6].map((steps) => ({
      model: partialCredit,
      parameters: { a: 1, b: [-1.5 + 0.5 * steps, -0.5 + 0.5 * steps] },
    }));
    const responses = simulateResponses(items, simulateThetas(800, random), random);
    const features = Object.fromEntries([1, 2, 3, 4, 5, 6].map((steps) => [`item${steps}`, { steps }]));
    const model = lltm(responses, features, { model: "partial-credit" });
    const [intercept, steps] = model.effects;
    expect([defined(intercept).estimate, defined(intercept).standardError]).toEqual([
      expect.closeTo(-1.01, 2),
      expect.closeTo(0.07, 2),
    ]);
    expect([defined(steps).estimate, defined(steps).standardError]).toEqual([
      expect.closeTo(0.51, 2),
      expect.closeTo(0.02, 2),
    ]);
    expect(model.comparison.raschLogLikelihood).toBeCloseTo(-4236.6, 1);
    expect(model.comparison).toMatchObject({ degreesOfFreedom: 4 });
    expect(model.comparison.chiSquare).toBeCloseTo(7.13, 2);
    expect(model.comparison.significance).toBeCloseTo(0.13, 2);
    const [first, second] = defined(model.items[0]).parameters.b;
    expect([defined(first), defined(second)]).toEqual([expect.closeTo(-1.04, 2), expect.closeTo(0.03, 2)]);
  });

  it("estimates the gain of six items answered twice as the effect of the later occasion on virtual items, as the recipe shows", () => {
    const random = createRandom(9);
    const thetas = simulateThetas(600, random);
    const items = [-1, -0.5, 0, 0.5, 1, 1.5].map((b) => ({
      model: twoParameterLogistic,
      parameters: { a: 1, b, c: 0, d: 1 },
    }));
    const before = simulateResponses(items, thetas, random);
    const after = simulateResponses(
      items,
      thetas.map((theta) => theta + 0.5),
      random,
    );
    const ids = before.itemIds;
    const stacked = responseMatrix(
      before.responses.map((row, person) => [...row, ...(after.responses[person] ?? [])]),
      { itemIds: [...ids.map((id) => `${id}.t1`), ...ids.map((id) => `${id}.t2`)] },
    );
    const features = Object.fromEntries(
      stacked.itemIds.map((id, column) => [
        id,
        {
          ...Object.fromEntries(ids.map((name, i) => [name, i === column % ids.length ? 1 : 0])),
          time2: column >= ids.length ? 1 : 0,
        },
      ]),
    );
    const later = lltm(stacked, features, { intercept: false }).effects.at(-1);
    expect(defined(later).name).toBe("time2");
    expect(defined(later).estimate).toBeCloseTo(-0.44, 2);
    expect(defined(later).standardError).toBeCloseTo(0.06, 2);
  });

  it("runs the cold start of the guide: calibrate a bank, regress on features, predict a new item and calibrate it with the prior", () => {
    const rows = [
      { steps: 1, open: 0, noise: 0.2 },
      { steps: 2, open: 0, noise: -0.3 },
      { steps: 3, open: 0, noise: 0.1 },
      { steps: 4, open: 0, noise: 0.25 },
      { steps: 5, open: 0, noise: -0.2 },
      { steps: 6, open: 0, noise: 0.05 },
      { steps: 1, open: 1, noise: -0.1 },
      { steps: 2, open: 1, noise: 0.3 },
      { steps: 3, open: 1, noise: -0.25 },
      { steps: 4, open: 1, noise: 0.15 },
      { steps: 5, open: 1, noise: 0 },
      { steps: 6, open: 1, noise: -0.05 },
      { steps: 2, open: 0, noise: 0.1 },
      { steps: 3, open: 1, noise: -0.2 },
      { steps: 4, open: 0, noise: 0.3 },
      { steps: 5, open: 1, noise: -0.1 },
    ];
    const difficulty = ({ steps, open, noise }: (typeof rows)[number]) => -1.5 + 0.5 * steps + 0.4 * open + noise;
    const bankItems = rows.map((row, i) => ({
      id: `q${i + 1}`,
      model: twoParameterLogistic,
      parameters: { a: 1, b: difficulty(row), c: 0, d: 1 },
    }));
    const features = Object.fromEntries(rows.map(({ steps, open }, i) => [`q${i + 1}`, { steps, open }]));
    const newItem = { id: "new1", model: twoParameterLogistic, parameters: { a: 1, b: 0.9, c: 0, d: 1 } };
    const random = createRandom(11);
    const thetas = Array.from({ length: 1000 }, () => random.normal());
    const { responses } = simulateResponses([...bankItems, newItem], thetas, random);
    const ids = [...bankItems, newItem].map(({ id }) => id);

    const calibration = calibrate(
      responseMatrix(
        responses.map((row) => row.slice(0, 16)),
        { itemIds: ids.slice(0, 16) },
      ),
      { model: twoParameterLogistic, standardErrors: "hessian" },
    );
    const regression = itemFeatureRegression(calibration.items, features);
    const [intercept, steps, open] = regression.effects;
    expect(defined(intercept).estimate).toBeCloseTo(-1.4, 2);
    expect(defined(intercept).standardError).toBeCloseTo(0.14, 2);
    expect(defined(steps).estimate).toBeCloseTo(0.49, 2);
    expect(defined(open).estimate).toBeCloseTo(0.21, 2);
    expect(regression.residualSd).toBeCloseTo(0.18, 2);
    const slopes = itemFeatureRegression(calibration.items, features, { parameter: "a" });

    const itsFeatures = { steps: 4, open: 1 };
    const b = predictItemParameter(regression, itsFeatures);
    const a = predictItemParameter(slopes, itsFeatures);
    expect([b.mean, b.sd]).toEqual([expect.closeTo(0.78, 2), expect.closeTo(0.2, 2)]);
    expect([a.mean, a.sd]).toEqual([expect.closeTo(0.05, 2), expect.closeTo(0.04, 2)]);

    const sparse = responseMatrix(
      responses.map((row, person) => row.map((response, item) => (item === 16 && person >= 25 ? null : response))),
      { itemIds: ids },
    );
    const fixed = Object.fromEntries(calibration.items.map(({ id, parameters }) => [id, parameters]));
    const itemPriors = { new1: { b: normalPrior(b.mean, b.sd), a: logNormalPrior(a.mean, a.sd) } };
    const free = calibrate(sparse, { model: twoParameterLogistic, fixed }).items[16]?.parameters;
    const informed = calibrate(sparse, { model: twoParameterLogistic, fixed, itemPriors }).items[16]?.parameters;
    expect([defined(free).a, defined(free).b]).toEqual([expect.closeTo(1.0, 2), expect.closeTo(0.59, 2)]);
    expect([defined(informed).a, defined(informed).b]).toEqual([expect.closeTo(1.05, 2), expect.closeTo(0.75, 2)]);
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

  /** Two occasions of 3 items for 40 persons in two groups, with a gain and a loss on every item in every group. */
  const changes = fc
    .record({
      before: fc.array(fc.array(fc.integer({ min: 0, max: 1 }), { minLength: 3, maxLength: 3 }), {
        minLength: 40,
        maxLength: 40,
      }),
      after: fc.array(fc.array(fc.integer({ min: 0, max: 1 }), { minLength: 3, maxLength: 3 }), {
        minLength: 40,
        maxLength: 40,
      }),
    })
    .filter(({ before, after }) =>
      [0, 1, 2].every((item) =>
        [0, 1].every((half) => {
          const rows = before.map((row, person) => [defined(row[item]), defined(after[person]?.[item]), person]);
          const inGroup = rows.filter(([, , person]) => Math.floor(defined(person) / 20) === half);
          return (
            inGroup.some(([first, second]) => first === 0 && second === 1) &&
            inGroup.some(([first, second]) => first === 1 && second === 0)
          );
        }),
      ),
    );
  const halves = Array.from({ length: 40 }, (_, person) => (person < 20 ? "first" : "second"));
  /** The responses of persons to 3 items, one triple of 0 and 1 for each person. */
  const triples = (text: string): number[][] => text.split(" ").map((person) => Array.from(person, Number));
  // The Newton steps of this case fall below the digits of the log-likelihood while the estimates are 1e-7 from the
  // maximum, so a line search without allowance for rounding stopped there.
  const stalledSteps = {
    before: triples(
      "101 101 001 110 011 000 000 010 100 001 011 001 011 000 001 000 000 111 001 100 101 010 010 010 100 010 110 000 010 111 000 100 010 100 110 000 101 110 110 000",
    ),
    after: triples(
      "000 000 100 001 110 101 000 001 000 011 011 001 100 110 010 110 101 010 011 001 001 100 010 100 000 100 110 000 000 010 000 100 010 100 011 111 000 001 101 001",
    ),
  };

  it("the effects of change negate under a recoding of the responses and keep their order and count", () => {
    fc.assert(
      fc.property(changes, ({ before, after }) => {
        const occasions = [before, after].map((rows) => responseMatrix(rows));
        const recoded = [before, after].map((rows) =>
          responseMatrix(rows.map((row) => row.map((response) => 1 - response))),
        );
        const base = linearChange(occasions, { groups: halves });
        const flipped = linearChange(recoded, { groups: halves });
        expect(base.converged).toBe(true);
        expect(base.effects).toHaveLength(6);
        base.effects.forEach((effect, i) => {
          expectClose(defined(flipped.effects[i]).estimate, -effect.estimate, 1e-7, `effect ${i + 1}`);
          expect(effect.standardError).toBeGreaterThan(0);
        });
        expectClose(flipped.logLikelihood, base.logLikelihood, 1e-7);
      }),
      { numRuns: 30, examples: [[stalledSteps]] },
    );
  });

  it("the effects of change do not depend on the order of the persons", () => {
    fc.assert(
      fc.property(changes, fc.integer({ min: 1, max: 39 }), ({ before, after }, turn) => {
        const rotate = <T>(rows: readonly T[]): T[] => [...rows.slice(turn), ...rows.slice(0, turn)];
        const base = linearChange(
          [before, after].map((rows) => responseMatrix(rows)),
          { groups: halves },
        );
        const turned = linearChange(
          [before, after].map((rows) => responseMatrix(rotate(rows))),
          {
            groups: rotate(halves),
            baseline: "first",
          },
        );
        base.effects.forEach((effect, i) => {
          expectClose(defined(turned.effects[i]).estimate, effect.estimate, 1e-7, `effect ${i + 1}`);
        });
      }),
      { numRuns: 30 },
    );
  });
});

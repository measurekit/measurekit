import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  type AbilityEstimate,
  abilityEstimates,
  adaptiveTest,
  betaPrior,
  type CalibrationOptions,
  calibrate,
  conditionalStandardError,
  createRandom,
  DataError,
  empiricalReliability,
  fourParameterLogistic,
  generalizedPartialCredit,
  gradedResponse,
  type IrtItem,
  type IrtModel,
  itemInformation,
  jointMaximumLikelihood,
  type LatentDistribution,
  type LogisticParameters,
  logNormalPrior,
  marginalReliability,
  maximumLikelihoodEstimate,
  nominalResponse,
  normalPrior,
  oneParameterLogistic,
  type ParameterPrior,
  partialCredit,
  posteriorMeanEstimate,
  posteriorModeEstimate,
  ratingScale,
  responseMatrix,
  responseRecords,
  simulateAdaptiveTest,
  simulateItems,
  simulateResponses,
  simulateThetas,
  summedScoreTable,
  type ThresholdParameters,
  testInformation,
  testLength,
  threeParameterLogistic,
  twoParameterLogistic,
  weightedLikelihoodEstimate,
} from "../src/index.js";
import crossProduct from "./fixtures/irt-cross-product.json" with { type: "json" };
import latent from "./fixtures/irt-latent.json" with { type: "json" };
import lsat7 from "./fixtures/irt-lsat7.json" with { type: "json" };
import science from "./fixtures/irt-science.json" with { type: "json" };
import scoring from "./fixtures/irt-scoring.json" with { type: "json" };
import simulated1pl from "./fixtures/irt-simulated-1pl.json" with { type: "json" };
import simulated4pl from "./fixtures/irt-simulated-4pl.json" with { type: "json" };
import simulatedMixed from "./fixtures/irt-simulated-mixed.json" with { type: "json" };
import simulatedPcm from "./fixtures/irt-simulated-pcm.json" with { type: "json" };
import testlet from "./fixtures/irt-testlet.json" with { type: "json" };
import {
  arbitraryItem,
  arbitraryLocation,
  arbitraryLocations,
  arbitrarySlope,
  defined,
  definedEntries,
  expectClose,
  expectParameters,
  type FixtureErrors,
  type FixtureParameters,
  MODELS,
} from "./helpers.js";

// Tolerances against R, explained in docs/validation.md.
/** Item parameters against mirt, ltm and the base R posterior mode. */
const PARAMETER_TOLERANCE = 1e-5;
/** Relative tolerance of standard errors: numerical Hessians on both sides. */
const STANDARD_ERROR_TOLERANCE = 1e-4;
/** Log-likelihood, AIC and BIC at the estimates. */
const LOG_LIKELIHOOD_TOLERANCE = 1e-6;
/** The 4PL, polytomous and mixed-format calibrations: mirt's EM stops short on their flat likelihoods. */
const FLAT_PARAMETER_TOLERANCE = 2e-4;
const FLAT_STANDARD_ERROR_TOLERANCE = 1e-3;
const FLAT_LOG_LIKELIHOOD_TOLERANCE = 1e-4;
/** The log-likelihood at given parameters, with the same quadrature as mirt. */
const FIXED_LOG_LIKELIHOOD_TOLERANCE = 1e-8;
/** ML and MAP theta and standard errors against mirt's nlm, which uses numerical derivatives. */
const MODAL_TOLERANCE = 1e-5;
/** EAP and WLE against references with the same quadrature or an exact root. */
const EXACT_ABILITY_TOLERANCE = 1e-10;
/** Information against mirt; absolute, as mirt's 1 - P loses relative precision in the tails. */
const INFORMATION_TOLERANCE = 1e-12;
/** JML estimates, standard errors and log-likelihood against the base R solution of TAM's steps. */
const JOINT_TOLERANCE = 1e-10;
/** EM tolerance of the fixture comparisons, mirt's TOL in validation/psychometrics/irt.R. */
const EM_TOLERANCE = 1e-10;
/** Estimated distributions of theta against the base R maximum: EM stops within 1e-8 of it. */
const LATENT_PARAMETER_TOLERANCE = 1e-7;
/** Relative tolerance of their standard errors: numerical Hessians on both sides. */
const LATENT_STANDARD_ERROR_TOLERANCE = 1e-6;
/** A histogram on fixed items against base R's EM with the same steps. */
const HISTOGRAM_TOLERANCE = 1e-12;
/** EAP summed-score tables against mirt and the base R recursion with the same nodes. */
const TABLE_TOLERANCE = 1e-12;
/** Marginal reliability by 61 nodes against integrate. */
const MARGINAL_RELIABILITY_TOLERANCE = 1e-8;
/** Testlet calibrations against the base R maximum, whose gradient is by central differences: EM stops within 1e-5. */
const TESTLET_PARAMETER_TOLERANCE = 2e-5;
/** Their standard errors, relative: numerical Hessians on both sides. */
const TESTLET_STANDARD_ERROR_TOLERANCE = 1e-4;

/** Models whose calibrations mirt stops short of the optimum. */
const FLAT_MODELS = new Set(["4pl", "grm", "gpcm", "pcm", "rsm", "nrm"]);

function model(name: string): IrtModel {
  return defined(MODELS[name]);
}

/** A distribution of theta named in a fixture. */
function latentDistribution(name: string): LatentDistribution {
  if (name !== "standard-normal" && name !== "normal" && name !== "empirical-histogram") {
    throw new Error(`Unknown distribution ${name}.`);
  }
  return name;
}

/** A prior of the fixtures: a beta prior, or a normal prior on the logit scale. */
interface PriorDescription {
  readonly type: string;
  readonly alpha?: number;
  readonly beta?: number;
  readonly mean?: number;
  readonly sd?: number;
}

/**
 * A normal prior on logit(x), as mirt's `norm` on g and u. Implemented here as user code: its log density as a function
 * of x is -((logit(x) - mean) / sd)² / 2.
 */
function logitNormalPrior(mean: number, sd: number): ParameterPrior {
  const logit = (x: number): number => Math.log(x / (1 - x));
  const variance = sd * sd;
  return {
    logDensity: (x) => (-0.5 * (logit(x) - mean) ** 2) / variance,
    logDensityDerivative: (x) => -(logit(x) - mean) / (variance * x * (1 - x)),
    logDensitySecondDerivative: (x) => -(1 - (logit(x) - mean) * (1 - 2 * x)) / (variance * (x * (1 - x)) ** 2),
  };
}

function prior(description: PriorDescription): ParameterPrior {
  const { type, alpha = 0, beta = 0, mean = 0, sd = 0 } = description;
  return type === "beta" ? betaPrior(alpha, beta) : logitNormalPrior(mean, sd);
}

/** A calibration of a fixture. */
interface CalibrationFixture {
  readonly name: string;
  readonly models: readonly string[];
  readonly quadrature: { readonly method: string; readonly points: number };
  readonly fixed?: Readonly<Record<string, FixtureParameters>>;
  readonly priors?: Readonly<Record<string, PriorDescription>>;
  readonly items: readonly {
    readonly parameters: FixtureParameters;
    readonly standardErrors?: FixtureErrors;
  }[];
  readonly logLikelihood: number;
  readonly aic?: number;
  readonly bic?: number;
}

/** Items with one model per item, as the fixtures give them. */
interface FixturePool {
  readonly models: readonly string[];
  readonly parameters: readonly FixtureParameters[];
}

/** A fixture: responses with calibrations, abilities and information computed in R. */
interface IrtFixture {
  readonly source: string;
  readonly itemIds: readonly string[];
  readonly responses: readonly (readonly (number | null)[])[];
  readonly calibrations: readonly CalibrationFixture[];
  readonly abilities: readonly (FixturePool & {
    readonly patterns: readonly (readonly (number | null)[])[];
    readonly posteriorMean: readonly AbilityEstimate[];
    readonly posteriorMode: readonly AbilityEstimate[];
    readonly maximumLikelihood: readonly AbilityEstimate[];
    readonly weightedLikelihood: readonly AbilityEstimate[];
  })[];
  readonly information?: readonly (FixturePool & {
    readonly theta: readonly number[];
    readonly items: readonly (readonly number[])[];
    readonly test: readonly number[];
  })[];
}

const FIXTURES: readonly IrtFixture[] = [lsat7, simulated1pl, simulated4pl, science, simulatedMixed];

function calibrationOptions(fixture: CalibrationFixture): CalibrationOptions<unknown> {
  return {
    ...modelsAndQuadrature(fixture),
    ...(fixture.fixed && { fixed: fixture.fixed }),
    ...(fixture.priors && {
      priors: Object.fromEntries(Object.entries(fixture.priors).map(([name, value]) => [name, prior(value)])),
    }),
    tolerance: EM_TOLERANCE,
    maxIterations: 10_000,
    standardErrors: "hessian",
  };
}

function modelsAndQuadrature(fixture: CalibrationFixture): CalibrationOptions<unknown> {
  const method = fixture.quadrature.method === "gauss-hermite" ? "gauss-hermite" : "rectangular";
  return { model: fixture.models.map(model), quadrature: { method, points: fixture.quadrature.points } };
}

function fixtureItems({ models, parameters }: FixturePool): IrtItem[] {
  return models.map((name, i) => ({ model: model(name), parameters: defined(parameters[i]) }));
}

/** Names the models of a pool, such as `3PL and GPCM`. */
function modelNames({ models }: FixturePool): string {
  return [...new Set(models)].map((name) => name.toUpperCase()).join(" and ");
}

/** Returns the paths of the numbers in a result that are not finite, skipping the items' models. */
function nonFinite(value: unknown, path = "result"): string[] {
  if (typeof value === "number") return Number.isFinite(value) ? [] : [path];
  if (typeof value !== "object" || value === null) return [];
  return Object.entries(value).flatMap(([key, entry]) => (key === "model" ? [] : nonFinite(entry, `${path}.${key}`)));
}

/** Simulates responses to items with a seeded generator, theta ~ N(0, 1). */
function simulate(seed: number, persons: number, items: readonly IrtItem[]): readonly (readonly (number | null)[])[] {
  const random = createRandom(seed);
  return simulateResponses(items, simulateThetas(persons, random), random).responses;
}

const PARAMETERS_2PL: readonly LogisticParameters[] = [
  { a: 1.2, b: -1, c: 0, d: 1 },
  { a: 0.8, b: 0, c: 0, d: 1 },
  { a: 1.5, b: 0.5, c: 0, d: 1 },
  { a: 1, b: 1.2, c: 0, d: 1 },
];
const ITEMS_2PL: readonly IrtItem<LogisticParameters>[] = PARAMETERS_2PL.map((parameters) => ({
  model: twoParameterLogistic,
  parameters,
}));
const ITEMS_GPCM: readonly IrtItem<ThresholdParameters>[] = [
  { model: generalizedPartialCredit, parameters: { a: 1.1, b: [-1, 0.3, 1.2] } },
  { model: generalizedPartialCredit, parameters: { a: 0.8, b: [-0.5, 0.9] } },
  { model: generalizedPartialCredit, parameters: { a: 1.4, b: [0.2, -0.6, 1.5] } },
];

describe("ability estimation", () => {
  const estimators = [
    ["posteriorMean", posteriorMeanEstimate, EXACT_ABILITY_TOLERANCE],
    ["posteriorMode", posteriorModeEstimate, MODAL_TOLERANCE],
    ["maximumLikelihood", maximumLikelihoodEstimate, MODAL_TOLERANCE],
    ["weightedLikelihood", weightedLikelihoodEstimate, EXACT_ABILITY_TOLERANCE],
  ] as const;

  for (const fixture of FIXTURES) {
    for (const abilities of fixture.abilities) {
      it(`matches R's EAP, MAP, ML and WLE estimates for the ${modelNames(abilities)} items of ${fixture.source}`, () => {
        const pool = fixtureItems(abilities);
        for (const [method, estimate, tolerance] of estimators) {
          abilities.patterns.forEach((pattern, row) => {
            const expected = defined(abilities[method][row]);
            const actual = estimate(pool, pattern);
            expectClose(actual.theta, expected.theta, tolerance, `${method} theta, pattern ${row + 1}`);
            expectClose(actual.standardError, expected.standardError, tolerance, `${method} SE, pattern ${row + 1}`);
          });
        }
      });
    }
  }

  it("puts the ML estimate at the end of the range when every response is in the lowest or highest category", () => {
    const high = maximumLikelihoodEstimate(ITEMS_2PL, [1, 1, 1, 1]);
    expect(high.theta).toBe(6);
    expectClose(high.standardError, conditionalStandardError(ITEMS_2PL, 6), 1e-12);
    expect(maximumLikelihoodEstimate(ITEMS_2PL, [0, 0, null, 0]).theta).toBe(-6);
    expect(maximumLikelihoodEstimate(ITEMS_2PL, [1, 1, 1, 1], { range: [-4, 4] }).theta).toBe(4);
    const mixed = [...ITEMS_2PL.slice(0, 2), ...ITEMS_GPCM];
    expect(maximumLikelihoodEstimate(mixed, [1, 1, 3, 2, 3]).theta).toBe(6);
    expect(maximumLikelihoodEstimate(mixed, [0, null, 0, 0, 0]).theta).toBe(-6);
  });

  it("gives a finite WLE for extreme 2PL patterns, inside the range", () => {
    for (const pattern of [
      [1, 1, 1, 1],
      [0, 0, 0, 0],
    ]) {
      const { theta, standardError } = weightedLikelihoodEstimate(ITEMS_2PL, pattern);
      expect(Math.abs(theta)).toBeLessThan(6);
      expectClose(standardError, conditionalStandardError(ITEMS_2PL, theta), 1e-12);
    }
  });

  it("gives the prior without responses for MAP and EAP, and rejects ML and WLE", () => {
    const none = [null, null, null, null];
    const thetaPrior = { mean: 0.5, sd: 2 };
    const map = posteriorModeEstimate(ITEMS_2PL, none, { prior: thetaPrior });
    expectClose(map.theta, 0.5, 1e-12);
    expectClose(map.standardError, 2, 1e-12);
    const eap = posteriorMeanEstimate(ITEMS_2PL, none, {
      prior: thetaPrior,
      quadrature: { method: "gauss-hermite", points: 21 },
    });
    expectClose(eap.theta, 0.5, 1e-12);
    expectClose(eap.standardError, 2, 1e-12);
    expect(() => maximumLikelihoodEstimate(ITEMS_2PL, none)).toThrow(
      "Maximum likelihood estimation needs at least one response.",
    );
    expect(() => weightedLikelihoodEstimate([], [])).toThrow(DataError);
  });

  it("estimates from a single item", () => {
    const one = [{ model: twoParameterLogistic, parameters: { a: 1, b: 0.5, c: 0, d: 1 } }];
    expect(maximumLikelihoodEstimate(one, [1]).theta).toBe(6);
    const wle = weightedLikelihoodEstimate(one, [1]);
    // For one 2PL item, Warm's equation (x - P) a + a (1 - 2P) / 2 = 0 has the root P = 3 / 4.
    expectClose(1 / (1 + Math.exp(-(wle.theta - 0.5))), 0.75, 1e-12);
  });

  it("estimates every person from response records or rows, by their answered items", () => {
    const items = ITEMS_2PL.map((item, i) => ({ ...item, id: `q${i + 1}` }));
    const rows = [
      [1, 0, null, 1],
      [null, null, null, null],
      [0, 0, 1, null],
    ];
    const matrix = responseMatrix(rows, { itemIds: ["q4", "q3", "q2", "q1"] });
    const records = responseRecords(
      [
        { person: "person1", item: "q4", response: 1 },
        { person: "person3", item: "q2", response: 1 },
        { person: "person1", item: "q1", response: 1 },
        { person: "person3", item: "q4", response: 0 },
        { person: "person1", item: "q3", response: 0 },
        { person: "person3", item: "q3", response: 0 },
      ],
      { personIds: matrix.personIds, itemIds: matrix.itemIds },
    );
    const expected = [
      { id: "person1", ...weightedLikelihoodEstimate([items[3], items[2], items[0]].map(defined), [1, 0, 1]) },
      { id: "person2", theta: null, standardError: null },
      { id: "person3", ...weightedLikelihoodEstimate([items[3], items[2], items[1]].map(defined), [0, 0, 1]) },
    ];
    expect(abilityEstimates(matrix, items, weightedLikelihoodEstimate)).toEqual(expected);
    expect(abilityEstimates(records, items, weightedLikelihoodEstimate)).toEqual(expected);
    expect(abilityEstimates(matrix, items)[0]).toEqual({
      id: "person1",
      ...posteriorMeanEstimate([items[3], items[2], items[0]].map(defined), [1, 0, 1]),
    });
    expect(() => abilityEstimates(matrix, items.slice(1))).toThrow(
      'The data have item "q1", which is not among the items.',
    );
    // Items are matched by id: a repeated id would leave one of them unused.
    expect(() => abilityEstimates(matrix, [...items, defined(items[0])])).toThrow(
      'Item ids must be unique; "q1" appears twice.',
    );
    expect(() =>
      responseRecords([
        { person: "p", item: "q1", response: 1 },
        { person: "p", item: "q1", response: 0 },
      ]),
    ).toThrow('Person "p" has two records of item "q1"; a person answers an item once.');
    expect(() => responseRecords([{ person: "p", item: "q1", response: 0.5 }])).toThrow(DataError);
  });

  it("rejects invalid items, responses, ranges and priors", () => {
    expect(() => maximumLikelihoodEstimate(ITEMS_2PL, [1, 0])).toThrow("Found 2 responses to 4 items.");
    expect(() => maximumLikelihoodEstimate(ITEMS_2PL, [1, 0, 2, 1])).toThrow(
      "The response to item 3 must be null or an integer from 0 to 1, found 2.",
    );
    expect(() => maximumLikelihoodEstimate(ITEMS_GPCM, [1, 3, 0])).toThrow(
      "The response to item 2 must be null or an integer from 0 to 2, found 3.",
    );
    const guessing = [{ model: twoParameterLogistic, parameters: { a: 1, b: 0, c: 0.2, d: 1 } }];
    expect(() => posteriorMeanEstimate(guessing, [1])).toThrow("Item 1: A 2PL item has c = 0, found c = 0.2.");
    expect(() => maximumLikelihoodEstimate(ITEMS_2PL, [1, 0, 1, 0], { range: [2, -2] })).toThrow(DataError);
    expect(() => posteriorModeEstimate(ITEMS_2PL, [1, 0, 1, 0], { prior: { mean: 0, sd: 0 } })).toThrow(DataError);
    expect(() => posteriorMeanEstimate(ITEMS_2PL, [1, 0, 1, 0], { quadrature: { points: 1 } })).toThrow(DataError);
    const withoutModel: IrtItem[] = JSON.parse('[{ "parameters": { "a": 1, "b": 0 } }]');
    expect(() => posteriorMeanEstimate(withoutModel, [1])).toThrow(
      "Item 1 needs a model such as twoParameterLogistic, found undefined.",
    );
    expect(() => abilityEstimates(responseMatrix([[1]]), [], JSON.parse("null"))).toThrow(
      "The estimator must be a function of items and responses, such as posteriorMeanEstimate, found null.",
    );
  });

  it("rejects items, responses and options of other types from JavaScript", () => {
    const responses = [1, 0, 1, 0];
    expect(() => posteriorMeanEstimate(JSON.parse("null"), [])).toThrow("The items must be an array, found null.");
    expect(() => posteriorMeanEstimate(ITEMS_2PL, JSON.parse("null"))).toThrow(
      "The responses must be an array, found null.",
    );
    expect(() => posteriorMeanEstimate(JSON.parse("[null]"), [1])).toThrow(
      "Item 1 must be an item with a model and parameters, found null.",
    );
    const emptyModel: IrtItem[] = JSON.parse('[{ "model": {}, "parameters": {} }]');
    expect(() => posteriorMeanEstimate(emptyModel, [1])).toThrow(
      "Item 1 needs a model such as twoParameterLogistic, found an object.",
    );
    expect(() => posteriorMeanEstimate([{ model: twoParameterLogistic, parameters: JSON.parse("null") }], [1])).toThrow(
      "Item 1: Logistic item parameters need finite a and b",
    );
    expect(() => posteriorMeanEstimate(ITEMS_2PL, responses, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => posteriorMeanEstimate(ITEMS_2PL, responses, { prior: JSON.parse("null") })).toThrow(
      "The option prior must be an object { mean, sd }, found null.",
    );
    expect(() => posteriorMeanEstimate(ITEMS_2PL, responses, { quadrature: JSON.parse("null") })).toThrow(
      "The option quadrature must be an object { method, points }, found null.",
    );
    expect(() => posteriorMeanEstimate(ITEMS_2PL, responses, { testlets: JSON.parse("{}") })).toThrow(
      "The option testlets must be an array of testlets, found an object.",
    );
    expect(() => maximumLikelihoodEstimate(ITEMS_2PL, responses, { range: JSON.parse("null") })).toThrow(
      "The option range must be an array [lower, upper], found null.",
    );
    expect(() => weightedLikelihoodEstimate(ITEMS_2PL, responses, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => posteriorModeEstimate(ITEMS_2PL, responses, { range: JSON.parse('[-6, "6"]') })).toThrow(
      'The upper bound of the range must be a finite number, found "6".',
    );
  });
});

describe("calibrate", () => {
  for (const fixture of FIXTURES) {
    const matrix = responseMatrix(fixture.responses, { itemIds: fixture.itemIds });
    for (const reference of fixture.calibrations) {
      const flat = reference.models.some((name) => FLAT_MODELS.has(name));
      it(`matches ${reference.name} on ${fixture.source}`, () => {
        const result = calibrate(matrix, calibrationOptions(reference));
        expect(result.converged).toBe(true);
        reference.items.forEach((expected, i) => {
          const actual = defined(result.items[i]);
          expect(actual.id).toBe(fixture.itemIds[i]);
          expect(actual.model).toBe(model(defined(reference.models[i])));
          const tolerance = flat ? FLAT_PARAMETER_TOLERANCE : PARAMETER_TOLERANCE;
          expectParameters(actual.parameters, expected.parameters, tolerance, actual.id);
          if (reference.fixed?.[actual.id] !== undefined) {
            expect(actual.parameters).toEqual(reference.fixed[actual.id]);
            expect(actual.standardErrors).toBeNull();
            return;
          }
          const errors = defined(actual.standardErrors);
          expect(Object.keys(errors)).toEqual(Object.keys(defined(expected.standardErrors)));
          for (const [name, value = 0] of Object.entries(defined(expected.standardErrors))) {
            const relative = (flat ? FLAT_STANDARD_ERROR_TOLERANCE : STANDARD_ERROR_TOLERANCE) * value;
            expectClose(defined(errors[name]), value, relative, `${actual.id} SE of ${name}`);
          }
        });
        const tolerance = flat ? FLAT_LOG_LIKELIHOOD_TOLERANCE : LOG_LIKELIHOOD_TOLERANCE;
        expectClose(result.logLikelihood, reference.logLikelihood, tolerance, "log-likelihood");
        if (reference.aic !== undefined) expectClose(result.aic, reference.aic, 2 * tolerance, "AIC");
        if (reference.bic !== undefined) expectClose(result.bic, reference.bic, 2 * tolerance, "BIC");
      });

      it(`gives ${reference.name}'s log-likelihood at its estimates on ${fixture.source}`, () => {
        const fixed = Object.fromEntries(
          reference.items.map(({ parameters }, i) => [defined(fixture.itemIds[i]), parameters]),
        );
        const result = calibrate(matrix, { ...modelsAndQuadrature(reference), fixed });
        expectClose(result.logLikelihood, reference.logLikelihood, FIXED_LOG_LIKELIHOOD_TOLERANCE, "log-likelihood");
      });
    }
  }

  for (const reference of crossProduct.cases) {
    it(`matches the standard errors of ${reference.name}`, () => {
      const matrix = responseMatrix(reference.responses, { itemIds: reference.itemIds });
      const result = calibrate(matrix, { ...calibrationOptions(reference), standardErrors: "cross-product" });
      const flat = reference.models.some((name) => FLAT_MODELS.has(name));
      reference.items.forEach((expected, i) => {
        const errors = defined(defined(result.items[i]).standardErrors);
        for (const [name, value = 0] of Object.entries(defined(expected.standardErrors))) {
          const relative = (flat ? FLAT_STANDARD_ERROR_TOLERANCE : STANDARD_ERROR_TOLERANCE) * value;
          expectClose(defined(errors[name]), value, relative, `item ${i + 1} SE of ${name}`);
        }
      });
    });
  }

  const rows = simulate(20261029, 300, ITEMS_2PL);

  it("gives identical calibrations from response records and from the matrix", () => {
    const sparse = rows.map((row, person) =>
      row.map((response, item) => ((person + item) % 3 === 0 ? null : response)),
    );
    const matrix = responseMatrix(sparse);
    const records = responseRecords(
      matrix.responses
        .flatMap((row, person) =>
          row.flatMap((response, item) =>
            response === null ? [] : [{ person: `person${person + 1}`, item: `item${item + 1}`, response }],
          ),
        )
        .reverse(),
      { personIds: matrix.personIds, itemIds: matrix.itemIds },
    );
    for (const standardErrors of ["hessian", "cross-product"] as const) {
      const options = {
        model: twoParameterLogistic,
        latent: "normal",
        fixed: { item1: defined(PARAMETERS_2PL[0]) },
        standardErrors,
      } as const;
      expect(calibrate(records, options)).toEqual(calibrate(matrix, options));
    }
  });

  it("reports non-convergence within the cycle limit", () => {
    const result = calibrate(responseMatrix(rows), { model: twoParameterLogistic, maxIterations: 2 });
    expect(result).toMatchObject({ converged: false, iterations: 2 });
  });

  it("ignores persons without responses, also in the BIC", () => {
    const base = calibrate(responseMatrix(rows), { model: twoParameterLogistic });
    const padded = calibrate(responseMatrix([...rows, [null, null, null, null]]), { model: twoParameterLogistic });
    expect(padded.items.map((item) => item.parameters)).toEqual(base.items.map((item) => item.parameters));
    expectClose(padded.logLikelihood, base.logLikelihood, 1e-9);
    expectClose(padded.bic, base.bic, 1e-9);
    expectClose(base.bic, -2 * base.logLikelihood + 8 * Math.log(300), 1e-9);
  });

  it("evaluates the log-likelihood when every item is fixed", () => {
    const fixed = Object.fromEntries(PARAMETERS_2PL.map((item, i) => [`item${i + 1}`, item]));
    const result = calibrate(responseMatrix(rows), { model: twoParameterLogistic, fixed });
    expect(result.items.map((item) => item.parameters)).toEqual(PARAMETERS_2PL);
    expect(result.items.every((item) => item.standardErrors === null)).toBe(true);
    expect(result.aic).toBe(-2 * result.logLikelihood);
  });

  it("takes one model per item and shares a only among the 1PL items", () => {
    const models = [oneParameterLogistic, oneParameterLogistic, twoParameterLogistic, twoParameterLogistic];
    const result = calibrate(responseMatrix(rows), { model: models });
    const [first, second, third, fourth] = result.items.map((item) => item.parameters.a);
    expect(first).toBe(second);
    expect(third).not.toBe(fourth);
    expectClose(result.aic, -2 * result.logLikelihood + 2 * 7, 1e-9);
  });

  it("calibrates multiple-choice and constructed-response items together, typed by their parameters", () => {
    const models: IrtModel<LogisticParameters | ThresholdParameters>[] = [
      threeParameterLogistic,
      threeParameterLogistic,
      twoParameterLogistic,
      generalizedPartialCredit,
      generalizedPartialCredit,
    ];
    const pool = [...ITEMS_2PL.slice(0, 3), ...ITEMS_GPCM.slice(0, 2)];
    const result = calibrate(responseMatrix(simulate(20261029, 600, pool)), {
      model: models,
      priors: { c: betaPrior(5, 17) },
      standardErrors: "hessian",
    });
    expect(result.converged).toBe(true);
    expect(result.items.map((item) => Object.keys(item.standardErrors ?? {}))).toEqual([
      ["a", "b", "c"],
      ["a", "b", "c"],
      ["a", "b"],
      ["a", "b1", "b2", "b3"],
      ["a", "b1", "b2"],
    ]);
    expectClose(result.aic, -2 * result.logLikelihood + 2 * 15, 1e-9);
  });

  it("shares a among PCM items, and a and the steps among RSM items", () => {
    const matrix = responseMatrix(science.responses, { itemIds: science.itemIds });
    const pcm = calibrate(matrix, { model: partialCredit, standardErrors: "cross-product" });
    expect(new Set(pcm.items.map((item) => item.parameters.a)).size).toBe(1);
    expect(Object.keys(defined(pcm.items[0]?.standardErrors))).toEqual(["a", "b1", "b2", "b3"]);
    const rsm = calibrate(matrix, { model: ratingScale, standardErrors: "cross-product" });
    expect(new Set(rsm.items.map((item) => item.parameters.a)).size).toBe(1);
    expect(new Set(rsm.items.map((item) => item.parameters.steps.join())).size).toBe(1);
    expect(new Set(rsm.items.map((item) => item.parameters.b)).size).toBe(4);
    expect(Object.keys(defined(rsm.items[0]?.standardErrors))).toEqual(["a", "b", "step1", "step2"]);
    expectClose(rsm.aic, -2 * rsm.logLikelihood + 2 * 7, 1e-9);
  });

  it("applies priors to the parameters they name", () => {
    const weak = calibrate(responseMatrix(rows), { model: twoParameterLogistic });
    const strong = calibrate(responseMatrix(rows), {
      model: twoParameterLogistic,
      priors: { a: logNormalPrior(0, 0.05), b: normalPrior(0, 0.1) },
    });
    for (const [i, item] of strong.items.entries()) {
      const free = defined(weak.items[i]).parameters;
      expect(Math.abs(Math.log(item.parameters.a))).toBeLessThan(Math.abs(Math.log(free.a)));
      expect(Math.abs(item.parameters.b)).toBeLessThan(Math.abs(free.b));
    }
  });

  it("gives the same estimates with any standard errors, and none by default", () => {
    const matrix = responseMatrix(rows);
    const options = {
      model: twoParameterLogistic,
      latent: { all: "normal" },
      groups: rows.map(() => "all"),
      fixed: { item1: defined(PARAMETERS_2PL[0]) },
    } as const;
    const none = calibrate(matrix, options);
    expect(none.items.every((item) => item.standardErrors === null)).toBe(true);
    expect(none.groups[0]?.standardErrors).toBeNull();
    for (const standardErrors of ["hessian", "cross-product"] as const) {
      const result = calibrate(matrix, { ...options, standardErrors });
      expect(result.items.map((item) => item.parameters)).toEqual(none.items.map((item) => item.parameters));
      expect(result.groups[0]?.standardErrors).not.toBeNull();
      expect(result.items[1]?.standardErrors).not.toBeNull();
    }
  });

  it("reports one standard normal group without groups", () => {
    const [group] = calibrate(responseMatrix(rows), { model: twoParameterLogistic }).groups;
    expect(group).toMatchObject({ group: null, distribution: "standard-normal", mean: 0, sd: 1, standardErrors: null });
    expect(defined(group).nodes).toHaveLength(61);
    expectClose(
      defined(group).weights.reduce((sum, weight) => sum + weight, 0),
      1,
      1e-15,
    );
  });

  it("gives finite results for a single item and for two persons", () => {
    const single = calibrate(responseMatrix(rows.map((row) => row.slice(0, 1))), { model: twoParameterLogistic });
    const pair = calibrate(
      responseMatrix([
        [1, 0],
        [0, 1],
      ]),
      { model: twoParameterLogistic, standardErrors: "hessian" },
    );
    for (const result of [single, pair]) expect(nonFinite(result)).toEqual([]);
    // Two persons with opposite responses say nothing about a: the information matrix is singular.
    expect(pair.items.map((item) => item.standardErrors)).toEqual([null, null]);
  });

  it("rejects RSM items with different numbers of categories", () => {
    const matrix = responseMatrix([
      [0, 0],
      [1, 1],
      [2, 2],
      [1, 3],
    ]);
    expect(() => calibrate(matrix, { model: ratingScale })).toThrow(
      'Items "item1" and "item2" of the RSM model share parameters, so they need the same number of categories; found 3 and 4.',
    );
  });

  it("rejects items it cannot calibrate and invalid options", () => {
    const matrix = responseMatrix([
      [0, 0, 1],
      [0, 1, 0],
      [0, null, 2],
    ]);
    const options = { model: twoParameterLogistic };
    expect(() => calibrate(matrix, options)).toThrow(
      'Item "item1" has responses in fewer than two categories; calibration needs at least two.',
    );
    expect(() => calibrate(matrix, { ...options, fixed: { item1: { a: 1, b: 0, c: 0, d: 1 } } })).toThrow(
      'Item "item3" has a response of 2, but the 2PL model allows responses from 0 to 1.',
    );
    const binary = responseMatrix(rows);
    const binaryItem = defined(PARAMETERS_2PL[0]);
    expect(() => calibrate(binary, { ...options, fixed: { q1: binaryItem } })).toThrow(
      'Fixed parameters are given for "q1", which is not an item id.',
    );
    expect(() => calibrate(binary, { ...options, fixed: { item1: { a: 1, b: 0, c: 0.1, d: 1 } } })).toThrow(
      'Fixed item "item1": A 2PL item has c = 0, found c = 0.1.',
    );
    expect(() => calibrate(binary, { ...options, priors: { g: betaPrior(5, 17) } })).toThrow(
      'A prior is given for "g", which no estimated item has.',
    );
    expect(() => calibrate(binary, { model: [twoParameterLogistic] })).toThrow("Found 1 models for 4 items.");
    expect(() => calibrate(binary, { model: JSON.parse("null") })).toThrow(
      "The option model must be an IRT model such as twoParameterLogistic, found null.",
    );
    expect(() => calibrate(binary, { ...options, standardErrors: JSON.parse('"full"') })).toThrow(
      'The option standardErrors must be "none", "hessian" or "cross-product", found "full".',
    );
    const empty = responseMatrix([[null, null]]);
    expect(() => calibrate(empty, { ...options, fixed: { item1: binaryItem, item2: binaryItem } })).toThrow(
      "Calibration needs at least one response.",
    );
    expect(() => calibrate(binary, { ...options, tolerance: -1 })).toThrow(
      "The option tolerance must be a non-negative number, found -1.",
    );
    expect(calibrate(binary, { ...options, tolerance: 0, maxIterations: 2 }).iterations).toBe(2);
    expect(() => calibrate(binary, { ...options, maxIterations: 0.5 })).toThrow(DataError);
    expect(() => calibrate(binary, { ...options, quadrature: JSON.parse('{ "method": "trapezoid" }') })).toThrow(
      'The quadrature method must be "rectangular" or "gauss-hermite", found "trapezoid".',
    );
  });

  it("rejects options of other types from JavaScript", () => {
    const binary = responseMatrix(rows);
    const options = { model: twoParameterLogistic };
    expect(() => calibrate(binary, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => calibrate(binary, { model: JSON.parse('{ "name": "2PL" }') })).toThrow(
      "The option model must be an IRT model such as twoParameterLogistic, found an object.",
    );
    expect(() => calibrate(binary, { ...options, fixed: JSON.parse("null") })).toThrow(
      "The option fixed must be an object of item parameters by item id, found null.",
    );
    expect(() => calibrate(binary, { ...options, priors: JSON.parse('{ "a": null }') })).toThrow(
      'The prior of "a" must be a prior such as normalPrior(0, 1), found null.',
    );
    expect(() => calibrate(binary, { ...options, testlets: JSON.parse("null") })).toThrow(
      "The option testlets must be an object of item id lists by testlet id, found null.",
    );
    expect(() => calibrate(binary, { ...options, maxIterations: JSON.parse('"10"') })).toThrow(
      'The option maxIterations must be a positive integer, found "10".',
    );
  });

  it("rejects an item without responses or without a response in a lower category", () => {
    const matrix = responseMatrix([
      [1, null],
      [1, null],
      [null, null],
    ]);
    expect(() => calibrate(matrix, { model: fourParameterLogistic })).toThrow(
      'Item "item1" has no response in category 0; calibration needs responses in every category up to the highest.',
    );
    const fixed = { item1: defined(PARAMETERS_2PL[0]) };
    expect(() => calibrate(matrix, { model: fourParameterLogistic, fixed })).toThrow(
      'Item "item2" has responses in fewer than two categories; calibration needs at least two.',
    );
    expect(() => calibrate(matrix, { model: gradedResponse, fixed: { item1: { a: 1, b: [0] } } })).toThrow(
      'Item "item2" has responses in fewer than two categories; calibration needs at least two.',
    );
  });
});

describe("gradedResponse", () => {
  it("names a and the thresholds b1 to b(K-1)", () => {
    expect(gradedResponse.parameters(4)).toEqual([{ name: "a" }, { name: "b1" }, { name: "b2" }, { name: "b3" }]);
    expect(gradedResponse.categoryCount({ a: 1, b: [0, 1] })).toBe(3);
  });

  it("starts at thresholds that give a person at θ = 0 the shares of the categories", () => {
    const shares = Float64Array.of(0.1, 0.2, 0.3, 0.4);
    const start = gradedResponse.startingValues(shares);
    const probabilities = new Float64Array(4);
    gradedResponse.probabilities(start, 0, probabilities);
    probabilities.forEach((p, k) => {
      expectClose(p, defined(shares[k]), 1e-15);
    });
  });

  it("keeps the precision of small probabilities in both tails", () => {
    const item = { a: 1, b: [-1, 0, 1] };
    const probabilities = new Float64Array(4);
    gradedResponse.probabilities(item, 30, probabilities);
    // P(X = 0) = 1 / (1 + exp(a (θ - b1))) = 1 / (1 + exp(31)).
    expectClose(defined(probabilities[0]) / (1 / (1 + Math.exp(31))), 1, 1e-14);
    gradedResponse.probabilities(item, -30, probabilities);
    expectClose(defined(probabilities[3]) / (1 / (1 + Math.exp(31))), 1, 1e-14);
  });

  it("rejects thresholds out of order for the sign of a", () => {
    expect(() => gradedResponse.toVector({ a: 1, b: [0, -1] })).toThrow(
      "A GRM item with more than two categories needs a nonzero a and thresholds b strictly increasing when a > 0 and strictly decreasing when a < 0, found a = 1, b = [0, -1].",
    );
    expect([...gradedResponse.toVector({ a: -1, b: [0, -1] })]).toEqual([-1, 0, -1]);
    expect(() => gradedResponse.toVector({ a: 0, b: [0, 1] })).toThrow(DataError);
    expect(() => gradedResponse.toVector({ a: 1, b: [] })).toThrow(
      "A GRM item needs a finite a and at least one finite threshold b, found a = 1, b = [].",
    );
  });
});

describe("information", () => {
  for (const fixture of FIXTURES) {
    for (const reference of fixture.information ?? []) {
      it(`matches mirt's iteminfo and testinfo for the ${modelNames(reference)} items of ${fixture.source}`, () => {
        const pool = fixtureItems(reference);
        reference.theta.forEach((theta, j) => {
          pool.forEach((item, i) => {
            const expected = defined(reference.items[i]?.[j]);
            expectClose(itemInformation(item, theta), expected, INFORMATION_TOLERANCE, `item ${i + 1}`);
          });
          const test = defined(reference.test[j]);
          expectClose(testInformation(pool, theta), test, INFORMATION_TOLERANCE, `test at ${theta}`);
          const standardError = 1 / Math.sqrt(test);
          expectClose(conditionalStandardError(pool, theta), standardError, 1e-9 * standardError, `SE at ${theta}`);
        });
      });
    }
  }

  it("gives an infinite standard error where there is no information", () => {
    const flat = [{ model: twoParameterLogistic, parameters: { a: 0, b: 0, c: 0, d: 1 } }];
    expect(testInformation(flat, 1)).toBe(0);
    expect(conditionalStandardError(flat, 1)).toBe(Number.POSITIVE_INFINITY);
    expect(testInformation([], 0)).toBe(0);
  });

  it("stays finite for steep divide-by-total items whose lowest category underflows", () => {
    // At θ = 0.5 the GPCM's category 2 has logit 1000 more than category 0, whose weight exp(-1000) is 0.
    const gpcm = { model: generalizedPartialCredit, parameters: { a: 1000, b: [-0.5, 0] } };
    const nrm = { model: nominalResponse, parameters: { a: [0, 1000, 2000], c: [0, 0, 0] } };
    for (const item of [gpcm, nrm]) {
      for (const theta of [-0.5, 0.5, 3]) expect(Number.isFinite(itemInformation(item, theta))).toBe(true);
      expect(Number.isFinite(marginalReliability([item]))).toBe(true);
    }
    // Category 1's share is logistic(-500) ≈ 7e-218 at θ = 0.5, so the variance of the category is about that.
    expect(itemInformation(gpcm, 0.5)).toBeCloseTo(0, 10);
  });

  it("rejects invalid theta and items", () => {
    const item = { model: twoParameterLogistic, parameters: { a: 1, b: 0, c: 0, d: 1 } };
    expect(() => itemInformation(item, Number.NaN)).toThrow("Theta must be a finite number, found NaN.");
    expect(() => testInformation([item, { ...item, parameters: { a: 1, b: Number.NaN, c: 0, d: 1 } }], 0)).toThrow(
      "Item 2: Logistic item parameters need finite a and b and 0 ≤ c < d ≤ 1, found a = 1, b = NaN, c = 0, d = 1.",
    );
    expect(() => testInformation(JSON.parse("null"), 0)).toThrow("The items must be an array, found null.");
    expect(() => itemInformation(item, JSON.parse('"0"'))).toThrow('Theta must be a finite number, found "0".');
  });
});

describe("jointMaximumLikelihood", () => {
  const reference = simulatedPcm.jointMaximumLikelihood;

  it(`matches ${reference.name} on ${simulatedPcm.source}`, () => {
    const matrix = responseMatrix(simulatedPcm.responses, { itemIds: simulatedPcm.itemIds });
    const result = jointMaximumLikelihood(matrix, { tolerance: 1e-12, maxIterations: 10_000 });
    expect(result.converged).toBe(true);
    reference.items.forEach((expected, i) => {
      const actual = defined(result.items[i]);
      expect(actual.id).toBe(simulatedPcm.itemIds[i]);
      expect(actual.model).toBe(model(expected.model));
      expectParameters(actual.parameters, expected.parameters, JOINT_TOLERANCE, actual.id);
      const errors = defined(actual.standardErrors);
      expect(Object.keys(errors)).toEqual(Object.keys(expected.standardErrors));
      for (const [name, value] of Object.entries(expected.standardErrors)) {
        expectClose(defined(errors[name]), value, JOINT_TOLERANCE, `${actual.id} SE of ${name}`);
      }
    });
    reference.persons.forEach((expected, p) => {
      const actual = defined(result.persons[p]);
      expectClose(actual.theta, expected.theta, JOINT_TOLERANCE, `person ${p + 1} theta`);
      expectClose(actual.standardError, expected.standardError, JOINT_TOLERANCE, `person ${p + 1} SE`);
    });
    expectClose(result.logLikelihood, reference.logLikelihood, JOINT_TOLERANCE, "log-likelihood");
  });

  const rows = [
    [1, 1, 2],
    [0, 0, 0],
    [1, 0, 1],
    [0, 1, 0],
    [null, null, null],
    [1, null, 2],
    [0, 1, 1],
  ];

  it("gives finite thetas at extreme scores and none to persons without responses", () => {
    const result = jointMaximumLikelihood(responseMatrix(rows));
    const [high, low, , , none, partial] = result.persons;
    expect(none).toEqual({ id: "person5", theta: null, standardError: null });
    for (const person of [high, low, partial]) expect(Number.isFinite(person?.theta)).toBe(true);
    expect(defined(high?.theta)).toBeGreaterThan(defined(low?.theta));
    expect(result.items.map((item) => item.model)).toEqual([oneParameterLogistic, oneParameterLogistic, partialCredit]);
    expect(nonFinite(result)).toEqual([]);
  });

  it("reports non-convergence within the iteration limit", () => {
    expect(jointMaximumLikelihood(responseMatrix(rows), { maxIterations: 1 })).toMatchObject({
      converged: false,
      iterations: 1,
    });
  });

  it("rejects fewer than two items, items it cannot estimate and invalid options", () => {
    expect(() => jointMaximumLikelihood(responseMatrix([[0], [1]]))).toThrow(
      "Joint maximum likelihood needs at least two items, found 1.",
    );
    expect(() => jointMaximumLikelihood(responseMatrix(rows.map(([a, b]) => [a ?? null, b ?? null, 2])))).toThrow(
      'Item "item3" has no response in category 0; calibration needs responses in every category up to the highest.',
    );
    expect(() => jointMaximumLikelihood(responseMatrix(rows), { tolerance: -1 })).toThrow(DataError);
    expect(() => jointMaximumLikelihood(responseMatrix(rows), { maxIterations: 0 })).toThrow(
      "The option maxIterations must be a positive integer, found 0.",
    );
    expect(() => jointMaximumLikelihood(responseMatrix(rows), JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
  });
});

describe("distributions of theta", () => {
  for (const reference of latent.calibrations) {
    it(`matches ${reference.name}`, () => {
      const groups = reference.groups ?? undefined;
      const distributions: Readonly<Record<string, LatentDistribution>> = Object.fromEntries(
        Object.entries(reference.distributions).map(([group, name]) => [group, latentDistribution(name)]),
      );
      const result = calibrate(responseMatrix(reference.responses, { itemIds: reference.itemIds }), {
        model: twoParameterLogistic,
        fixed: definedEntries(reference.fixed),
        ...(groups === undefined ? { latent: defined(distributions.all) } : { groups, latent: distributions }),
        tolerance: EM_TOLERANCE,
        maxIterations: 10_000,
        standardErrors: "hessian",
      });
      expect(result.converged).toBe(true);
      reference.items.forEach((expected, i) => {
        const actual = defined(result.items[i]);
        expectParameters(actual.parameters, expected.parameters, LATENT_PARAMETER_TOLERANCE, actual.id);
        const errors = "standardErrors" in expected ? expected.standardErrors : undefined;
        if (errors === undefined) {
          expect(actual.standardErrors).toBeNull();
          return;
        }
        for (const [name, value] of Object.entries(errors)) {
          const relative = LATENT_STANDARD_ERROR_TOLERANCE * value;
          expectClose(defined(actual.standardErrors?.[name]), value, relative, `${actual.id} SE of ${name}`);
        }
      });
      for (const expected of reference.groupDistributions) {
        const actual = defined(result.groups.find((group) => (group.group ?? "all") === expected.group));
        expect(actual.distribution).toBe("normal");
        expectClose(actual.mean, expected.mean, LATENT_PARAMETER_TOLERANCE, "mean");
        expectClose(actual.sd, expected.sd, LATENT_PARAMETER_TOLERANCE, "sd");
        const errors = defined(actual.standardErrors);
        expectClose(
          errors.mean,
          expected.standardErrors.mean,
          LATENT_STANDARD_ERROR_TOLERANCE * errors.mean,
          "SE of mean",
        );
        expectClose(errors.sd, expected.standardErrors.sd, LATENT_STANDARD_ERROR_TOLERANCE * errors.sd, "SE of sd");
      }
      expectClose(result.logLikelihood, reference.logLikelihood, LOG_LIKELIHOOD_TOLERANCE, "log-likelihood");
      expectClose(result.aic, reference.aic, 2 * LOG_LIKELIHOOD_TOLERANCE, "AIC");
      expectClose(result.bic, reference.bic, 2 * LOG_LIKELIHOOD_TOLERANCE, "BIC");
    });
  }

  it(`matches ${latent.histogram.name}`, () => {
    const { histogram } = latent;
    const result = calibrate(responseMatrix(histogram.responses, { itemIds: histogram.itemIds }), {
      model: twoParameterLogistic,
      fixed: histogram.fixed,
      latent: "empirical-histogram",
      tolerance: histogram.tolerance,
      maxIterations: 10_000,
    });
    expect(result).toMatchObject({ converged: true, iterations: histogram.iterations });
    const group = defined(result.groups[0]);
    group.weights.forEach((weight, q) => {
      expectClose(weight, defined(histogram.weights[q]), HISTOGRAM_TOLERANCE);
    });
    expectClose(result.logLikelihood, histogram.logLikelihood, 1e-8, "log-likelihood");
    expectClose(result.aic, -2 * result.logLikelihood + 2 * 60, 1e-9, "AIC");
    const mean = group.weights.reduce((sum, weight, q) => sum + weight * defined(group.nodes[q]), 0);
    expectClose(group.mean, mean, 1e-15);
  });

  it("recovers pretest items seeded into simulated adaptive tests, with a normal distribution or a histogram", () => {
    const random = createRandom(20261029);
    const pool = simulateItems(twoParameterLogistic, 60, random).map((item, i) => ({ ...item, id: `op${i + 1}` }));
    const pretest = simulateItems(twoParameterLogistic, 4, random);
    const thetas = simulateThetas(3000, random).map((theta) => 0.4 + 1.2 * theta);
    const sessions = simulateAdaptiveTest(adaptiveTest({ pool, stopping: testLength(12) }), thetas, random).persons;
    const seeded = simulateResponses(pretest, thetas, random).responses;
    const itemIds = [...pool.map((item) => item.id), "pre1", "pre2", "pre3", "pre4"];
    // Each session answers its adaptive items and two of the four pretest items.
    const rows = sessions.map(({ state }, person) => {
      const row: (number | null)[] = itemIds.map(() => null);
      state.items.forEach((id, j) => {
        row[itemIds.indexOf(id)] = defined(state.responses[j]);
      });
      for (const k of [person % 4, (person + 1) % 4]) row[pool.length + k] = defined(seeded[person]?.[k]);
      return row;
    });
    const matrix = responseMatrix(rows, { itemIds });
    const fixed = Object.fromEntries(pool.map((item) => [item.id, item.parameters]));
    for (const distribution of ["normal", "empirical-histogram"] as const) {
      const result = calibrate(matrix, {
        model: twoParameterLogistic,
        fixed,
        latent: distribution,
        standardErrors: "cross-product",
      });
      expect(result.converged).toBe(true);
      pretest.forEach((item, k) => {
        const estimate = defined(result.items[pool.length + k]);
        const errors = defined(estimate.standardErrors);
        expect(Math.abs(estimate.parameters.a - item.parameters.a)).toBeLessThan(4 * defined(errors.a));
        expect(Math.abs(estimate.parameters.b - item.parameters.b)).toBeLessThan(4 * defined(errors.b));
      });
      const group = defined(result.groups[0]);
      expect(Math.abs(group.mean - 0.4)).toBeLessThan(0.1);
      expect(Math.abs(group.sd - 1.2)).toBeLessThan(0.1);
    }
  });

  it("gives the estimates of one group to groups that all keep the standard normal", () => {
    const rows = simulate(7, 400, ITEMS_2PL);
    const groups = rows.map((_, person) => (person % 2 === 0 ? "even" : "odd"));
    const one = calibrate(responseMatrix(rows), { model: twoParameterLogistic });
    const two = calibrate(responseMatrix(rows), { model: twoParameterLogistic, groups });
    two.items.forEach((item, i) => {
      expectParameters(item.parameters, defined(one.items[i]).parameters, 1e-12, item.id);
    });
    expect(two.groups.map((group) => group.group)).toEqual(["even", "odd"]);
    expectClose(two.logLikelihood, one.logLikelihood, 1e-9);
  });

  it("estimates the distribution of a group on fixed items alone", () => {
    const rows = simulate(11, 500, ITEMS_2PL);
    const fixed = Object.fromEntries(PARAMETERS_2PL.map((item, i) => [`item${i + 1}`, item]));
    const result = calibrate(responseMatrix(rows), {
      model: twoParameterLogistic,
      fixed,
      latent: "normal",
      standardErrors: "hessian",
    });
    const group = defined(result.groups[0]);
    expect(result.converged).toBe(true);
    expect(Math.abs(group.mean)).toBeLessThan(4 * defined(group.standardErrors).mean);
    expectClose(result.aic, -2 * result.logLikelihood + 4, 1e-9);
  });

  it("estimates a normal distribution whose SD falls below the spacing of the nodes", () => {
    // Four persons whose likelihoods agree on one theta: the SD's maximum likelihood estimate tends to 0.
    const matrix = responseMatrix([
      [1, 0, 1],
      [0, 1, 1],
      [1, 1, 0],
      [0, 0, 1],
    ]);
    const fixed = {
      item1: { a: 1.2, b: 0.32, c: 0, d: 1 },
      item2: { a: 0.99, b: 0.82, c: 0, d: 1 },
      item3: { a: 1.21, b: -0.22, c: 0, d: 1 },
    };
    const result = calibrate(matrix, { model: twoParameterLogistic, fixed, latent: "normal", maxIterations: 2000 });
    const group = defined(result.groups[0]);
    expect(Number.isFinite(group.mean)).toBe(true);
    expect(group.sd).toBeGreaterThan(0);
    expect(group.sd).toBeLessThan(0.2);
    // The SD is below the spacing of the nodes, 0.2: the likelihood has no maximum there.
    expect(result.converged).toBe(false);
    const first = calibrate(matrix, { model: twoParameterLogistic, fixed, latent: "normal", maxIterations: 1 });
    expect(result.logLikelihood).toBeGreaterThan(first.logLikelihood);
  });

  it("keeps a normal distribution whose maximum lies beyond the nodes finite, without standard errors", () => {
    // Three of the focal group's four persons answer the item in its top category: its mean runs to +∞.
    const responses = [[2], [2], [1], [0], [1], [2], [1], [2], [1]];
    const groups = responses.map((_, person) => (person % 2 === 1 ? "focal" : "reference"));
    const result = calibrate(responseMatrix(responses), {
      model: generalizedPartialCredit,
      groups,
      latent: { focal: "normal" },
      standardErrors: "hessian",
    });
    const focal = defined(result.groups[1]);
    expect(Number.isFinite(focal.mean) && Number.isFinite(focal.sd) && Number.isFinite(result.logLikelihood)).toBe(
      true,
    );
    expect(focal.mean).toBeGreaterThan(6);
    expect(focal.standardErrors).toBeNull();
    expect(result.converged).toBe(false);
  });

  it("rejects groups and distributions that do not fit the data or leave the scale free", () => {
    const rows = simulate(3, 50, ITEMS_2PL);
    const matrix = responseMatrix(rows);
    const options = { model: twoParameterLogistic };
    const labels = rows.map((_, person) => (person < 25 ? "a" : "b"));
    expect(() => calibrate(matrix, { ...options, groups: ["a"] })).toThrow("Found 1 groups for 50 persons.");
    expect(() =>
      calibrate(matrix, { ...options, groups: JSON.parse(JSON.stringify(labels.map((_, i) => (i === 3 ? 3 : "a")))) }),
    ).toThrow("The group of person 4 must be a string, found 3.");
    expect(() => calibrate(matrix, { ...options, latent: "normal" })).toThrow(
      "Estimating every group's distribution of theta needs fixed items or a standard normal group to fix the scale.",
    );
    expect(() => calibrate(matrix, { ...options, latent: { a: "normal" } })).toThrow(
      "Distributions of theta by group need the groups option.",
    );
    expect(() => calibrate(matrix, { ...options, groups: labels, latent: { c: "normal" } })).toThrow(
      'A distribution of theta is given for group "c", which no person is in.',
    );
    expect(() => calibrate(matrix, { ...options, groups: labels, latent: JSON.parse('{ "b": "uniform" }') })).toThrow(
      'The distribution of theta of group "b" must be "standard-normal", "normal" or "empirical-histogram", found "uniform".',
    );
    const fixed = { item1: defined(PARAMETERS_2PL[0]) };
    expect(() =>
      calibrate(matrix, { ...options, fixed, latent: "normal", quadrature: { method: "gauss-hermite", points: 21 } }),
    ).toThrow("Estimating a distribution of theta needs the rectangular quadrature rule.");
    const silent = responseMatrix([...rows, [null, null, null, null]]);
    expect(() => calibrate(silent, { ...options, groups: [...labels, "c"] })).toThrow(
      'Group "c" has no person with a response.',
    );
  });
});

describe("logistic models", () => {
  it("name their estimated parameters, with c and d bounded and a shared by 1PL items", () => {
    expect(oneParameterLogistic.parameters(2)).toEqual([{ name: "a", shared: true }, { name: "b" }]);
    expect(twoParameterLogistic.parameters(2).map(({ name }) => name)).toEqual(["a", "b"]);
    expect(fourParameterLogistic.parameters(2)).toEqual([
      { name: "a" },
      { name: "b" },
      { name: "c", lower: 0, upper: 1 },
      { name: "d", lower: 0, upper: 1 },
    ]);
  });

  it("convert parameters to vectors and back", () => {
    const item = { a: 1.4, b: -0.3, c: 0.2, d: 1 };
    expect([...threeParameterLogistic.toVector(item)]).toEqual([1.4, -0.3, 0.2]);
    expect(threeParameterLogistic.fromVector(threeParameterLogistic.toVector(item))).toEqual(item);
    expect(twoParameterLogistic.fromVector(Float64Array.of(1, 2))).toEqual({ a: 1, b: 2, c: 0, d: 1 });
  });

  it("start inside the bounds, with b where the share correct is reached", () => {
    for (const p of [0.05, 0.5, 0.97]) {
      const start = fourParameterLogistic.startingValues(Float64Array.of(1 - p, p));
      expect(start.c).toBeGreaterThan(0);
      expect(start.d).toBeLessThan(1);
      const share = new Float64Array(2);
      fourParameterLogistic.probabilities(start, start.b, share);
      expect(defined(share[1])).toBeCloseTo((start.c + start.d) / 2, 12);
    }
  });

  it("reject parameters outside the model", () => {
    expect(() => twoParameterLogistic.toVector({ a: 1, b: 0, c: 0, d: 0.9 })).toThrow(
      "A 2PL item has d = 1, found d = 0.9.",
    );
    expect(() => threeParameterLogistic.toVector({ a: 1, b: 0, c: 0.3, d: 0.9 })).toThrow(
      "A 3PL item has d = 1, found d = 0.9.",
    );
    expect(() => fourParameterLogistic.toVector({ a: 1, b: 0, c: 0.5, d: 0.5 })).toThrow(DataError);
    expect(() => fourParameterLogistic.toVector({ a: Number.POSITIVE_INFINITY, b: 0, c: 0, d: 1 })).toThrow(DataError);
  });
});

describe("nominalResponse", () => {
  it("names the slopes and intercepts of all categories but the last", () => {
    expect(nominalResponse.parameters(3).map(({ name }) => name)).toEqual(["a1", "a2", "c1", "c2"]);
  });

  it("centers the slopes and intercepts, which leaves the probabilities unchanged", () => {
    const item = { a: [1, 2, 3], c: [0, 1, 5] };
    const centered = nominalResponse.fromVector(nominalResponse.toVector(item));
    expect(centered).toEqual({ a: [-1, 0, 1], c: [-2, -1, 3] });
    const before = new Float64Array(3);
    const after = new Float64Array(3);
    nominalResponse.probabilities(item, 0.7, before);
    nominalResponse.probabilities(centered, 0.7, after);
    before.forEach((p, k) => {
      expectClose(defined(after[k]), p, 1e-15);
    });
  });

  it("rejects slopes and intercepts of different lengths", () => {
    expect(() => nominalResponse.toVector({ a: [0, 1], c: [0, 1, 2] })).toThrow(
      "An NRM item needs finite slopes a and intercepts c, one of each per category and at least two categories, found a = [0, 1], c = [0, 1, 2].",
    );
    expect(() => nominalResponse.toVector({ a: [1], c: [0] })).toThrow(DataError);
  });
});

describe("partial credit models", () => {
  it("name a, shared by PCM items, and the thresholds", () => {
    expect(partialCredit.parameters(3)).toEqual([{ name: "a", shared: true }, { name: "b1" }, { name: "b2" }]);
    expect(generalizedPartialCredit.parameters(3)).toEqual([{ name: "a" }, { name: "b1" }, { name: "b2" }]);
  });

  it("start at thresholds that give a person at θ = 0 the shares of the categories", () => {
    const shares = Float64Array.of(0.4, 0.1, 0.5);
    const probabilities = new Float64Array(3);
    generalizedPartialCredit.probabilities(generalizedPartialCredit.startingValues(shares), 0, probabilities);
    probabilities.forEach((p, k) => {
      expectClose(p, defined(shares[k]), 1e-15);
    });
  });

  it("convert items with more thresholds than a spread of arguments holds", () => {
    const b = Array.from({ length: 200_000 }, () => 0);
    expect(generalizedPartialCredit.toVector({ a: 1, b })).toHaveLength(200_001);
  });

  it("reject items without thresholds", () => {
    expect(() => partialCredit.toVector({ a: 1, b: [] })).toThrow(
      "A PCM item needs a finite a and at least one finite threshold b, found a = 1, b = [].",
    );
    expect(() => generalizedPartialCredit.toVector({ a: 1, b: [Number.NaN] })).toThrow(DataError);
  });
});

describe("priors", () => {
  it("have derivatives consistent with their log densities", () => {
    const cases: [ParameterPrior, number][] = [
      [normalPrior(0.5, 2), -1.3],
      [logNormalPrior(0, 0.5), 1.7],
      [betaPrior(5, 17), 0.23],
    ];
    const h = 1e-5;
    for (const [density, x] of cases) {
      const slope = (density.logDensity(x + h) - density.logDensity(x - h)) / (2 * h);
      expectClose(density.logDensityDerivative(x), slope, 1e-7);
      const curvature = (density.logDensityDerivative(x + h) - density.logDensityDerivative(x - h)) / (2 * h);
      expectClose(density.logDensitySecondDerivative(x), curvature, 1e-6);
    }
  });

  it("are -Infinity outside their support", () => {
    expect(logNormalPrior(0, 1).logDensity(0)).toBe(Number.NEGATIVE_INFINITY);
    expect(betaPrior(2, 2).logDensity(1)).toBe(Number.NEGATIVE_INFINITY);
  });

  it("reject invalid arguments", () => {
    expect(() => normalPrior(0, -1)).toThrow("A normal prior's sd must be a positive number, found -1.");
    expect(() => normalPrior(Number.NaN, 1)).toThrow(DataError);
    expect(() => logNormalPrior(0, 0)).toThrow(DataError);
    expect(() => betaPrior(0, 1)).toThrow(DataError);
    expect(() => normalPrior(JSON.parse('"0"'), 1)).toThrow(
      'A normal prior\'s mean must be a finite number, found "0".',
    );
    expect(() => logNormalPrior(0, JSON.parse("null"))).toThrow(
      "A log-normal prior's sdLog must be a positive number, found null.",
    );
  });
});

describe("ratingScale", () => {
  it("names a and the steps, shared by its items, and the location b", () => {
    expect(ratingScale.parameters(4)).toEqual([
      { name: "a", shared: true },
      { name: "b" },
      { name: "step1", shared: true },
      { name: "step2", shared: true },
    ]);
    expect(ratingScale.parameters(2).map(({ name }) => name)).toEqual(["a", "b"]);
  });

  it("moves the mean of the steps into the location, which leaves the probabilities unchanged", () => {
    const item = { a: 1.2, b: 0.5, steps: [-1, 0, 2] };
    const centered = ratingScale.fromVector(ratingScale.toVector(item));
    expectClose(centered.b, 0.5 + 1 / 3, 1e-15);
    expect(centered.steps.reduce((sum, step) => sum + step, 0)).toBeCloseTo(0, 15);
    const before = new Float64Array(4);
    const after = new Float64Array(4);
    ratingScale.probabilities(item, -0.4, before);
    ratingScale.probabilities(centered, -0.4, after);
    before.forEach((p, k) => {
      expectClose(defined(after[k]), p, 1e-15);
    });
    expect(ratingScale.fromVector(Float64Array.of(1, 0.3))).toEqual({ a: 1, b: 0.3, steps: [0] });
  });

  it("rejects an item without steps or with steps that are not finite", () => {
    expect(() => ratingScale.toVector({ a: 1, b: 0, steps: [] })).toThrow(
      "An RSM item needs finite a and b and at least one step, found a = 1, b = 0, steps = [].",
    );
    expect(() => ratingScale.toVector({ a: 1, b: 0, steps: [0, Number.POSITIVE_INFINITY] })).toThrow(
      "An RSM item needs finite steps, found [0, Infinity].",
    );
  });
});

const arbitraryTheta = fc.double({ min: -8, max: 8, noNaN: true });

/** Returns an item's category probabilities, their theta derivatives and its information at theta. */
function evaluate({ model: itemModel, parameters }: IrtItem, theta: number): number[][] {
  const count = itemModel.categoryCount(parameters);
  const probabilities = new Float64Array(count);
  const first = new Float64Array(count);
  const second = new Float64Array(count);
  itemModel.probabilities(parameters, theta, probabilities);
  itemModel.thetaDerivatives(parameters, theta, first, second);
  return [[...probabilities], [...first], [...second], [itemModel.information(parameters, theta)]];
}

describe("reliability", () => {
  for (const form of scoring.forms) {
    it(`matches mirt's marginal_rxx and base R for the ${modelNames(form)} items`, () => {
      const items = fixtureItems(form);
      for (const { prior, value } of form.marginalReliability) {
        expectClose(marginalReliability(items, { prior }), value, MARGINAL_RELIABILITY_TOLERANCE);
      }
    });
  }

  it("matches mirt's empirical_rxx of EAP estimates", () => {
    const { estimates, value } = scoring.empiricalReliability;
    expectClose(empiricalReliability(estimates), value, 1e-12);
  });

  it("gives 0 for estimates without variance and 1 for estimates without error", () => {
    expect(
      empiricalReliability([
        { theta: 0.5, standardError: 0.4 },
        { theta: 0.5, standardError: 0.3 },
      ]),
    ).toBe(0);
    expect(
      empiricalReliability([
        { theta: -1, standardError: 0 },
        { theta: 1, standardError: 0 },
      ]),
    ).toBe(1);
  });

  it("rejects too few or invalid estimates and forms without items", () => {
    expect(() => empiricalReliability([{ theta: 0, standardError: 1 }])).toThrow(
      "Empirical reliability needs at least two estimates.",
    );
    expect(() =>
      empiricalReliability([
        { theta: 0, standardError: Infinity },
        { theta: 1, standardError: 1 },
      ]),
    ).toThrow("Estimate 1 needs a finite theta and a finite, non-negative standard error, found 0 and Infinity.");
    expect(() =>
      empiricalReliability([
        { theta: 0, standardError: 0 },
        { theta: 0, standardError: 0 },
      ]),
    ).toThrow(DataError);
    expect(() => marginalReliability([])).toThrow("Marginal reliability needs at least one item.");
    expect(() => marginalReliability(ITEMS_2PL, { prior: { mean: 0, sd: -1 } })).toThrow(DataError);
    expect(() => marginalReliability(JSON.parse("null"))).toThrow(
      "Marginal reliability needs an array of items, found null.",
    );
    expect(() => marginalReliability(ITEMS_2PL, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => empiricalReliability(JSON.parse("null"))).toThrow(
      "Empirical reliability needs an array of estimates, found null.",
    );
    expect(() => empiricalReliability(JSON.parse("[null, null]"))).toThrow(
      "Estimate 1 must be an object { theta, standardError }, found null.",
    );
  });
});

describe("summedScoreTable", () => {
  for (const form of scoring.forms) {
    it(`matches mirt's EAPsum and base R for the ${modelNames(form)} items`, () => {
      const items = fixtureItems(form);
      for (const { prior, estimates } of form.tables) {
        const table = summedScoreTable(items, { prior });
        expect(table.map((row) => row.score)).toEqual(estimates.map((row) => row.score));
        table.forEach((row, s) => {
          const expected = defined(estimates[s]);
          expectClose(row.theta, expected.theta, TABLE_TOLERANCE, `theta of ${s}`);
          expectClose(row.standardError, expected.standardError, TABLE_TOLERANCE, `SE of ${s}`);
          expectClose(row.probability, expected.probability, TABLE_TOLERANCE, `probability of ${s}`);
        });
      }
    });
  }

  it("gives the EAP estimate of the single response of a one-item form", () => {
    const item = defined(ITEMS_2PL[0]);
    const table = summedScoreTable([item]);
    for (const row of table) {
      const estimate = posteriorMeanEstimate([item], [row.score]);
      expectClose(row.theta, estimate.theta, 1e-12);
      expectClose(row.standardError, estimate.standardError, 1e-12);
    }
  });

  it("takes the prior and the quadrature but no testlets, which the table does not integrate out", () => {
    const testlets = [{ items: ["item1", "item2"], sd: 1 }];
    // @ts-expect-error: a summed-score table assumes the items independent given theta.
    summedScoreTable(ITEMS_2PL, { testlets });
    // @ts-expect-error: so does marginal reliability.
    marginalReliability(ITEMS_2PL, { testlets });
  });

  it("rejects a form without items and invalid items", () => {
    expect(() => summedScoreTable([])).toThrow("A summed-score table needs at least one item.");
    expect(() => summedScoreTable([{ model: twoParameterLogistic, parameters: { a: 1, b: 0, c: 0.1, d: 1 } }])).toThrow(
      "Item 1: A 2PL item has c = 0, found c = 0.1.",
    );
    expect(() => summedScoreTable(JSON.parse("{}"))).toThrow(
      "A summed-score table needs an array of items, found an object.",
    );
    expect(() => summedScoreTable(ITEMS_2PL, JSON.parse("null"))).toThrow("The options must be an object, found null.");
  });
});

describe("testlets", () => {
  for (const reference of testlet.cases) {
    it(`matches ${reference.name}`, () => {
      const matrix = responseMatrix(reference.responses, { itemIds: reference.itemIds });
      const result = calibrate(matrix, {
        model: twoParameterLogistic,
        testlets: definedEntries(reference.testlets),
        quadrature: { points: testlet.points },
        tolerance: 1e-8,
        maxIterations: 100_000,
        standardErrors: "hessian",
      });
      expect(result.converged).toBe(true);
      reference.items.forEach((expected, i) => {
        const actual = defined(result.items[i]);
        expectParameters(actual.parameters, expected.parameters, TESTLET_PARAMETER_TOLERANCE, actual.id);
        for (const [name, value] of Object.entries(expected.standardErrors)) {
          const relative = TESTLET_STANDARD_ERROR_TOLERANCE * value;
          expectClose(defined(actual.standardErrors?.[name]), value, relative, `${actual.id} SE of ${name}`);
        }
      });
      result.testlets.forEach((actual, t) => {
        expect(actual.items).toEqual(Object.values(definedEntries(reference.testlets))[t]);
        expectClose(actual.sd, defined(reference.sd[t]), TESTLET_PARAMETER_TOLERANCE, `${actual.id} sd`);
        const error = defined(reference.sdStandardErrors[t]);
        expectClose(actual.standardError, error, TESTLET_STANDARD_ERROR_TOLERANCE * error, `${actual.id} SE`);
      });
      expectClose(result.logLikelihood, reference.logLikelihood, FIXED_LOG_LIKELIHOOD_TOLERANCE, "log-likelihood");
      expectClose(result.aic, -2 * result.logLikelihood + 2 * 27, 1e-9, "AIC");
    });

    it(`matches base R's EAP with the testlet effects integrated out on ${reference.name}`, () => {
      const items = reference.items.map(({ parameters }, j) => ({
        id: defined(reference.itemIds[j]),
        model: twoParameterLogistic,
        parameters,
      }));
      const testlets = Object.values(definedEntries(reference.testlets)).map((ids, t) => ({
        items: ids,
        sd: defined(reference.sd[t]),
      }));
      reference.abilities.forEach((expected, r) => {
        const actual = posteriorMeanEstimate(items, defined(reference.responses[r]), { testlets });
        expectClose(actual.theta, expected.theta, EXACT_ABILITY_TOLERANCE, `theta ${r + 1}`);
        expectClose(actual.standardError, expected.standardError, EXACT_ABILITY_TOLERANCE, `SE ${r + 1}`);
      });
    });
  }

  it("keeps a testlet's SD finite when the data leave no room for testlet effects", () => {
    const matrix = responseMatrix([
      [1, 1, 0, 1, 0, 1],
      [0, 0, 1, 1, 1, 0],
      [1, 1, 1, 0, 1, 1],
      [0, 1, 0, 0, 0, 1],
      [1, 0, 1, 1, 1, 1],
      [0, 0, 0, 1, 0, 0],
    ]);
    const result = calibrate(matrix, {
      model: twoParameterLogistic,
      testlets: { first: ["item1", "item2", "item3"], second: ["item4", "item5", "item6"] },
      standardErrors: "hessian",
    });
    expect(nonFinite(result)).toEqual([]);
    for (const { sd } of result.testlets) expect(sd).toBeGreaterThan(0);
  });

  const named = ITEMS_2PL.map((item, i) => ({ ...item, id: `item${i + 1}` }));

  it("gives the EAP without testlets as the effects' SD goes to 0", () => {
    const testlets = [{ items: ["item1", "item2"], sd: 1e-9 }];
    const plain = posteriorMeanEstimate(named, [1, 0, 1, 1]);
    const nested = posteriorMeanEstimate(named, [1, 0, 1, 1], { testlets });
    expectClose(nested.theta, plain.theta, 1e-8);
    expectClose(nested.standardError, plain.standardError, 1e-8);
  });

  it("reports no testlets without them, and rejects testlets that do not fit the items or the options", () => {
    const rows = simulate(5, 200, ITEMS_2PL);
    const matrix = responseMatrix(rows);
    expect(calibrate(matrix, { model: twoParameterLogistic }).testlets).toEqual([]);
    const options = { model: twoParameterLogistic, maxIterations: 3 };
    expect(() => calibrate(matrix, { ...options, testlets: { a: ["item1"] } })).toThrow(
      'Testlet "a" needs at least two items.',
    );
    expect(() => calibrate(matrix, { ...options, testlets: { a: ["item1", "q9"] } })).toThrow(
      'Testlet "a" names "q9", which is not an item id.',
    );
    expect(() => calibrate(matrix, { ...options, testlets: { a: ["item1", "item2"], b: ["item2", "item3"] } })).toThrow(
      'Item "item2" is in more than one testlet.',
    );
    const testlets = { a: ["item1", "item2"] };
    expect(() =>
      calibrate(matrix, { ...options, testlets, quadrature: { method: "gauss-hermite", points: 21 } }),
    ).toThrow("Testlets need the rectangular quadrature rule.");
    expect(() =>
      calibrate(matrix, { ...options, testlets, groups: rows.map((_, p) => (p < 100 ? "a" : "b")) }),
    ).toThrow("Testlets need one group whose theta is standard normal.");
    expect(() => calibrate(matrix, { ...options, testlets, standardErrors: "cross-product" })).toThrow(
      'Testlets need the standard errors "none" or "hessian".',
    );
    const twice = [
      { items: ["item1", "item2"], sd: 1 },
      { items: ["item2", "item3"], sd: 1 },
    ];
    expect(() => posteriorMeanEstimate(named, [1, 0, 1, 1], { testlets: twice })).toThrow(
      'Item "item2" is in a testlet twice.',
    );
    expect(() => posteriorMeanEstimate(ITEMS_2PL, [1, 0, 1, 1], { testlets: [{ items: ["item1"], sd: 1 }] })).toThrow(
      "Item 1 needs a string id, found undefined.",
    );
    expect(() =>
      posteriorMeanEstimate(named, [1, 0, 1, 1], { testlets: [{ items: ["item1", "item2"], sd: 0 }] }),
    ).toThrow("The sd of testlet 1 must be a positive number, found 0.");
    expect(() =>
      posteriorMeanEstimate(named, [1, 0, 1, 1], { testlets: [{ items: JSON.parse("[0, 1]"), sd: 1 }] }),
    ).toThrow("Testlet 1 needs item ids, found 0.");
    expect(() =>
      posteriorMeanEstimate([...named, defined(named[0])], [1, 0, 1, 1, 1], {
        testlets: [{ items: ["item1"], sd: 1 }],
      }),
    ).toThrow('Item ids must be unique; "item1" appears twice.');
  });

  it("scores each person's answered items with the testlets of a calibration, as abilityEstimates gives them", () => {
    // Persons who skip a testlet item are estimated from a subset of the items: testlets name items by id.
    const rows = [
      [null, 1, 1, 0],
      [1, null, 0, 1],
      [1, 1, 1, 1],
    ];
    const testlets = [{ items: ["item1", "item2"], sd: 1.5 }];
    const estimates = abilityEstimates(responseMatrix(rows), named, (items, responses) =>
      posteriorMeanEstimate(items, responses, { testlets }),
    );
    rows.forEach((row, person) => {
      const expected = posteriorMeanEstimate(named, row, { testlets });
      expect(estimates[person]).toEqual({ id: `person${person + 1}`, ...expected });
    });
  });
});

describe("properties", () => {
  it("category probabilities lie in [0, 1] and sum to 1", () => {
    fc.assert(
      fc.property(arbitraryItem, arbitraryTheta, (item, theta) => {
        const [probabilities = []] = evaluate(item, theta);
        expect(probabilities.every((p) => p >= 0 && p <= 1)).toBe(true);
        expectClose(
          probabilities.reduce((sum, p) => sum + p, 0),
          1,
          1e-14,
        );
      }),
    );
  });

  it("information is non-negative and the sum over categories of P'² / P", () => {
    fc.assert(
      fc.property(arbitraryItem, arbitraryTheta, (item, theta) => {
        const information = itemInformation(item, theta);
        expect(information).toBeGreaterThanOrEqual(0);
        const [probabilities = [], first = []] = evaluate(item, theta);
        const sum = probabilities.reduce((total, p, k) => (p > 0 ? total + defined(first[k]) ** 2 / p : total), 0);
        expectClose(information, sum, 1e-9 * Math.max(1, sum));
      }),
    );
  });

  it("derivatives match central differences", () => {
    const h = 1e-6;
    fc.assert(
      fc.property(arbitraryItem, fc.double({ min: -4, max: 4, noNaN: true }), (item, theta) => {
        const { model: itemModel, parameters } = item;
        const count = itemModel.categoryCount(parameters);
        const at = (t: number, values = parameters): Float64Array => {
          const out = new Float64Array(count);
          itemModel.probabilities(values, t, out);
          return out;
        };
        const [, first = [], second = []] = evaluate(item, theta);
        const [, above = []] = evaluate(item, theta + h);
        const [, below = []] = evaluate(item, theta - h);
        const vector = itemModel.toVector(parameters);
        const derivatives = new Float64Array(count * vector.length);
        itemModel.parameterDerivatives(parameters, theta, derivatives);
        for (let k = 0; k < count; k++) {
          const slopeDifference = (defined(at(theta + h)[k]) - defined(at(theta - h)[k])) / (2 * h);
          expectClose(defined(first[k]), slopeDifference, 1e-6, `dP(${k})/dθ`);
          const curvatureDifference = (defined(above[k]) - defined(below[k])) / (2 * h);
          expectClose(defined(second[k]), curvatureDifference, 1e-5, `d²P(${k})/dθ²`);
          vector.forEach((value, j) => {
            const shifted = (step: number): number => {
              const changed = Float64Array.from(vector);
              changed[j] = value + step;
              return defined(at(theta, itemModel.fromVector(changed))[k]);
            };
            const expected = (shifted(h) - shifted(-h)) / (2 * h);
            expectClose(defined(derivatives[k * vector.length + j]), expected, 1e-6, `dP(${k})/dξ(${j})`);
          });
        }
      }),
    );
  });

  it("the GPCM with equal slopes equals the PCM, and with thresholds b + τ_k the RSM", () => {
    fc.assert(
      fc.property(arbitrarySlope, arbitraryLocations, arbitraryLocation, arbitraryTheta, (a, b, shift, theta) => {
        expect(evaluate({ model: partialCredit, parameters: { a, b } }, theta)).toEqual(
          evaluate({ model: generalizedPartialCredit, parameters: { a, b } }, theta),
        );
        const rsm = evaluate({ model: ratingScale, parameters: { a, b: shift, steps: b } }, theta);
        const gpcm = evaluate(
          { model: generalizedPartialCredit, parameters: { a, b: b.map((step) => shift + step) } },
          theta,
        );
        rsm.forEach((values, i) => {
          values.forEach((value, k) => {
            expectClose(value, defined(gpcm[i]?.[k]), 1e-10 * Math.max(1, Math.abs(value)));
          });
        });
      }),
    );
  });

  it("ability estimates are finite and inside the range", () => {
    const pools = fc
      .array(arbitraryItem, { minLength: 1, maxLength: 6 })
      .chain((pool) =>
        fc.tuple(
          fc.constant(pool),
          fc.tuple(
            ...pool.map(({ model: itemModel, parameters }) =>
              fc.option(fc.integer({ min: 0, max: itemModel.categoryCount(parameters) - 1 }), { nil: null }),
            ),
          ),
        ),
      );
    fc.assert(
      fc.property(pools, ([pool, responses]) => {
        const answered = responses.some((response) => response !== null);
        const estimates = [
          posteriorModeEstimate(pool, responses),
          posteriorMeanEstimate(pool, responses),
          ...(answered
            ? [maximumLikelihoodEstimate(pool, responses), weightedLikelihoodEstimate(pool, responses)]
            : []),
        ];
        for (const { theta, standardError } of estimates) {
          expect(theta).toBeGreaterThanOrEqual(-6);
          expect(theta).toBeLessThanOrEqual(6);
          expect(Number.isNaN(standardError)).toBe(false);
          expect(standardError).toBeGreaterThan(0);
        }
      }),
    );
  });

  it("summed-score probabilities sum to 1, and marginal reliability lies in [0, 1)", () => {
    fc.assert(
      fc.property(fc.array(arbitraryItem, { minLength: 1, maxLength: 6 }), (items) => {
        const table = summedScoreTable(items);
        expectClose(
          table.reduce((sum, row) => sum + row.probability, 0),
          1,
          1e-12,
        );
        for (const row of table) expect(Math.abs(defined(row.theta))).toBeLessThan(6);
        const reliability = marginalReliability(items);
        expect(reliability).toBeGreaterThanOrEqual(0);
        expect(reliability).toBeLessThan(1);
      }),
    );
  });

  it("calibration results contain no NaN or infinities", () => {
    const rows = fc
      .integer({ min: 2, max: 4 })
      .chain((items) =>
        fc.array(
          fc.array(fc.option(fc.constantFrom(0, 1, 2), { freq: 9, nil: null }), { minLength: items, maxLength: items }),
          { minLength: 2, maxLength: 30 },
        ),
      );
    const models = fc.constantFrom<IrtModel>(
      oneParameterLogistic,
      twoParameterLogistic,
      threeParameterLogistic,
      gradedResponse,
      generalizedPartialCredit,
      partialCredit,
      ratingScale,
      nominalResponse,
    );
    fc.assert(
      fc.property(rows, models, (data, m) => {
        try {
          expect(nonFinite(calibrate(responseMatrix(data), { model: m, maxIterations: 100 }))).toEqual([]);
        } catch (error) {
          if (!(error instanceof DataError)) throw error;
        }
      }),
      { numRuns: 80 },
    );
  });

  it("the log-likelihood does not decrease across EM cycles", () => {
    const pools: readonly (readonly IrtItem[])[] = [ITEMS_2PL, ITEMS_GPCM];
    fc.assert(
      fc.property(
        fc.integer({ min: 0, max: 2 ** 31 }),
        fc.constantFrom<[IrtModel, number]>(
          [oneParameterLogistic, 0],
          [twoParameterLogistic, 0],
          [gradedResponse, 1],
          [generalizedPartialCredit, 1],
        ),
        (seed, [m, pool]) => {
          const matrix = responseMatrix(simulate(seed, 150, defined(pools[pool])));
          let previous = Number.NEGATIVE_INFINITY;
          for (let cycles = 1; cycles <= 12; cycles++) {
            const { logLikelihood } = calibrate(matrix, { model: m, maxIterations: cycles, tolerance: 1e-12 });
            expect(logLikelihood).toBeGreaterThanOrEqual(previous - 1e-9);
            previous = logLikelihood;
          }
        },
      ),
      { numRuns: 20 },
    );
  });

  it("seeded runs reproduce exactly", () => {
    const pool = [...ITEMS_2PL, ...ITEMS_GPCM];
    const models: IrtModel<LogisticParameters | ThresholdParameters>[] = [
      ...ITEMS_2PL.map(() => threeParameterLogistic),
      ...ITEMS_GPCM.map(() => generalizedPartialCredit),
    ];
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 2 ** 31 }), (seed) => {
        const run = (): unknown => {
          const matrix = responseMatrix(simulate(seed, 300, pool));
          return [
            calibrate(matrix, { model: models, priors: { c: betaPrior(5, 17) }, maxIterations: 50 }),
            jointMaximumLikelihood(matrix),
          ];
        };
        expect(run()).toEqual(run());
      }),
      { numRuns: 10 },
    );
  });
});

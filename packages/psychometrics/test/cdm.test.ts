import { chiSquareUpperTail } from "@measurekit/core";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { additiveMaximizer, type CdmLink, profilesOf, saturatedCoefficients } from "../src/cdm/internal.js";
import { pseudoInverse } from "../src/cdm/pseudo-inverse.js";
import {
  attributeProfiles,
  buildResponseMatrix,
  type CdmCalibration,
  type CdmModel,
  calibrateCdm,
  cdmClassificationAccuracy,
  cdmItemFit,
  cdmModelFit,
  compareItemModels,
  createRandom,
  DataError,
  nonparametricClassification,
  qMatrixIdentifiability,
  simulateCdmResponses,
  validateQMatrix,
} from "../src/index.js";
import { multiply, transpose } from "../src/math/internal.js";
import { defined, expectClose } from "./assertions.js";
import { definedEntries } from "./fixture-items.js";
import { readFixture } from "./fixtures.js";

interface CdmFixture {
  cases: {
    name: string;
    itemIds: string[];
    responses: number[][];
    qmatrix: { item1: string[]; item2: string[]; item3: string[]; item4: string[]; item5: string[] };
    attributes: string[];
    fits: {
      model: string;
      items: number[][];
      guessing: number[];
      slip: number[];
      proportions: number[];
      logLikelihood: number;
      aic: number;
      bic: number;
      classes: number[] | null;
      persons: { mastery: number[]; map: number[]; eap: number[]; posterior: number[] }[] | null;
      accuracy:
        | {
            classification: string;
            accuracy: number;
            profileAccuracy: number[];
            attributeAccuracy: number[];
            consistency: number;
            attributeConsistency: number[];
          }[]
        | null;
      validation: { suggested: string[]; pvaf: number }[] | null;
    }[];
    nonparametric: { gate: string; distance: number[]; ties: number[]; profile: (number[] | null)[] }[];
  }[];
}
const fixture = readFixture<CdmFixture>("cdm.json");

interface CdmAdditiveFixture {
  cases: {
    name: string;
    itemIds: string[];
    responses: number[][];
    qmatrix: { item1: string[]; item2: string[]; item3: string[]; item4: string[]; item5: string[] };
    attributes: string[];
    fits: {
      model: string | string[];
      items: number[][];
      guessing: number[];
      slip: number[];
      proportions: number[];
      logLikelihood: number;
      aic: number;
      bic: number;
      classes: null;
      persons: { mastery: number[]; map: number[]; eap: number[]; posterior: number[] }[];
      accuracy: {
        classification: string;
        accuracy: number;
        profileAccuracy: number[];
        attributeAccuracy: number[];
        consistency: number;
        attributeConsistency: number[];
      }[];
      validation: null;
      coefficients: number[][];
    }[];
  }[];
}
const additiveFixture = readFixture<CdmAdditiveFixture>("cdm-additive.json");

// biome-ignore-start lint/style/useNamingConvention: the keys are those of the fixture file.
interface CdmFitFixture {
  responses: number[][];
  items: string[];
  qMatrix: { item1: string[]; item2: string[]; item3: string[]; item4: string[]; item5: string[] };
  cases: {
    name: string;
    model: string[];
    gdina: { m2: number; degreesOfFreedom: number; reversed: number } | null;
    m2: number;
    degreesOfFreedom: number;
    significance: number;
    rmsea: number;
    rmseaInterval: number[];
    wideInterval: number[];
    srmsr: number;
    exact: {
      observed: number[];
      expected: number[];
      proportionZ: number[];
      correlationZ: number[];
      logOddsZ: number[];
      significance: number[];
    };
    resampled: { proportionZ: number[]; correlationZ: number[]; logOddsZ: number[] };
  }[];
  din: { srmsr: number; correlationZ: number[] };
  rejections: { sets: { responses: string[]; statistic: number; degreesOfFreedom: number }[]; alpha: number };
}
// biome-ignore-end lint/style/useNamingConvention: the keys are those of the fixture file.
const fitFixture = readFixture<CdmFitFixture>("cdm-fit.json");

interface CdmIdentifiabilityFixture {
  cases: {
    qMatrix: { item1: string[]; item2: string[]; item3: string[]; item4: string[]; item5?: string[] };
    singleItems: { a1: boolean; a2: boolean; a3?: boolean; a4?: boolean; a5?: boolean };
    threeItems: { a1: boolean; a2: boolean; a3?: boolean; a4?: boolean; a5?: boolean };
    distinctColumns: boolean;
    dinaIdentified: boolean;
  }[];
}
const identifiabilityFixture = readFixture<CdmIdentifiabilityFixture>("cdm-identifiability.json");

interface CdmItemComparisonFixture {
  models: string[];
  level: number;
  cases: {
    name: string;
    itemIds: string[];
    responses: number[][];
    qmatrix: { item1: string[]; item2: string[]; item3: string[]; item4: string[]; item5: string[] };
    probabilities: number[][];
    compared: string[];
    covariance: number[][][];
    statistics: number[][];
    degreesOfFreedom: number[][];
    pValues: number[][];
    selected: string[];
    truth: string[] | null;
  }[];
  recovery: { items: number; rejected: number; selected: number };
}
const comparisonFixture = readFixture<CdmItemComparisonFixture>("cdm-item-comparison.json");

interface CdmMaximumFixture {
  cases: {
    link: string;
    attributeCount: number;
    kind: string;
    correct: number[];
    persons: number[];
    activeBounds: number;
    delta: number[];
    probabilities: number[];
    value: number;
  }[];
}
const maximumFixture = readFixture<CdmMaximumFixture>("cdm-maximum.json");

// Tolerances against GDINA, explained in docs/validation.md.
/** Success probabilities, proportions and person results: EM stops at a change of 1e-8, within 1e-6 of the maximum. */
const CDM_TOLERANCE = 1e-5;
/** The log-likelihood and the information criteria. */
const LOG_LIKELIHOOD_TOLERANCE = 1e-8;
/** The success probabilities, coefficients and proportions of fits with additive items: GDINA's EM stops at a change of 1e-8. */
const ADDITIVE_TOLERANCE = 1e-5;
/** The maximum of one M-step on fixed counts, against the point that meets the KKT conditions. */
const MAXIMUM_TOLERANCE = 1e-8;
/** The person results from the success probabilities and proportions of GDINA's fit. */
const REFERENCE_PERSON_TOLERANCE = 1e-6;
/** The covariance of the success probabilities, relative to the largest entry of the item's matrix. */
const COVARIANCE_TOLERANCE = 1e-5;
/** Wald statistics, relative to the larger of the statistic and 1. */
const STATISTIC_TOLERANCE = 1e-4;
/** The p-values of the Wald statistics. */
const P_VALUE_TOLERANCE = 1e-5;
/** The EM tolerance of the fits whose model and item fit are compared. */
const FIT_EM_TOLERANCE = 1e-10;
/** M2, its significance, RMSEA, its interval and SRMSR against GDINA, each from its own calibration. */
const M2_TOLERANCE = 1e-3;
/** M2 of the rejection sets, whose calibrations stop at 1e-6. */
const REJECTION_TOLERANCE = 2e-2;
/** The item statistics against their exact sums at GDINA's estimates, and against CDM's exact correlations. */
const ITEM_TOLERANCE = 1e-3;
/** The z statistics against GDINA's, whose expected values simulate 1e6 persons: their Monte Carlo SE is about 0.03. */
const RESAMPLED_TOLERANCE = 0.15;

const MODELS: readonly CdmModel[] = ["gdina", "dina", "dino", "acdm", "llm", "rrum"];
const LINKS: readonly CdmLink[] = ["identity", "logit", "log"];
/** The link of each additive model. */
const LINK_OF: Readonly<Record<string, CdmLink>> = { acdm: "identity", llm: "logit", rrum: "log" };
const PROBABILITY_BOUNDS = [1e-4, 1 - 1e-4] as const;
function cdmModel(name: string): CdmModel {
  const model = MODELS.find((candidate) => candidate === name);
  if (model === undefined) throw new Error(`Unknown model ${name}.`);
  return model;
}

function cdmLink(name: string): CdmLink {
  const link = LINKS.find((candidate) => candidate === name);
  if (link === undefined) throw new Error(`Unknown link ${name}.`);
  return link;
}

/** The models of a fixture's fit: one for every item, or one per item. */
function fitModels(model: string | readonly string[]): CdmModel | CdmModel[] {
  return typeof model === "string" ? cdmModel(model) : model.map(cdmModel);
}

/** The name of a fixture's fit in a test title. */
function fitName(model: string | readonly string[]): string {
  return typeof model === "string" ? model.toUpperCase() : "per-item mix of";
}

/** The success probabilities that calibrateCdm starts from, for an item of `attributeCount` attributes. */
function startingProbabilities(attributeCount: number): number[] {
  return profilesOf(attributeCount).map(
    (profile) => 0.2 + (0.6 * profile.reduce((sum, v) => sum + v, 0)) / attributeCount,
  );
}

/** A link: the identity, the logit or the log of a probability. */
function linked(link: CdmLink, probability: number): number {
  if (link === "identity") return probability;
  return link === "logit" ? Math.log(probability / (1 - probability)) : Math.log(probability);
}

/** The inverse of a link. */
function inverse(link: CdmLink, predictor: number): number {
  if (link === "identity") return predictor;
  return link === "logit" ? 1 / (1 + Math.exp(-predictor)) : Math.exp(predictor);
}

/** The predictor of each latent group of an item with `attributeCount` attributes at `coefficients`. */
function predictors(coefficients: readonly number[], attributeCount: number): number[] {
  return profilesOf(attributeCount).map((profile) =>
    profile.reduce((sum, mastered, k) => sum + mastered * (coefficients[k + 1] ?? 0), coefficients[0] ?? 0),
  );
}

/** Returns the negative log-likelihood of expected counts at success probabilities. */
function negativeLogLikelihood(correct: readonly number[], persons: readonly number[], p: readonly number[]): number {
  return -p.reduce((sum, value, l) => {
    const r = correct[l] ?? 0;
    return sum + r * Math.log(value) + ((persons[l] ?? 0) - r) * Math.log1p(-value);
  }, 0);
}

/** The calibration GDINA's fit holds: a calibration of the fixture's data with GDINA's success probabilities and proportions. */
function referenceCalibration(
  reference: (typeof additiveFixture.cases)[number],
  fit: (typeof additiveFixture.cases)[number]["fits"][number],
): CdmCalibration {
  const matrix = buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
  const calibration = calibrateCdm(matrix, definedEntries(reference.qmatrix), { maxIterations: 1 });
  return {
    ...calibration,
    items: calibration.items.map((item, j) => ({
      ...item,
      probabilities: defined(fit.items[j]),
      guessing: defined(fit.guessing[j]),
      slip: defined(fit.slip[j]),
      coefficients: defined(fit.coefficients[j]),
    })),
    profiles: calibration.profiles.map((entry, c) => ({ ...entry, proportion: defined(fit.proportions[c]) })),
  };
}

describe("cdmClassificationAccuracy", () => {
  for (const reference of fixture.cases) {
    const matrix = buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
    for (const fit of reference.fits) {
      for (const accuracy of fit.accuracy ?? []) {
        const classification = accuracy.classification === "eap" ? "eap" : "map";
        it(`matches GDINA's CA of the ${fit.model.toUpperCase()} model by ${classification.toUpperCase()} on ${reference.name} data`, () => {
          const calibration = calibrateCdm(matrix, definedEntries(reference.qmatrix), {
            model: cdmModel(fit.model),
            tolerance: 1e-10,
            maxIterations: 10_000,
          });
          const result = cdmClassificationAccuracy(matrix, calibration, { classification });
          expectClose(result.accuracy, accuracy.accuracy, CDM_TOLERANCE, "τ");
          expectClose(result.consistency, accuracy.consistency, CDM_TOLERANCE, "γ");
          result.attributeAccuracy.forEach((value, k) => {
            expectClose(value, defined(accuracy.attributeAccuracy[k]), CDM_TOLERANCE, `τ_${k}`);
          });
          result.attributeConsistency.forEach((value, k) => {
            expectClose(value, defined(accuracy.attributeConsistency[k]), CDM_TOLERANCE, `γ_${k}`);
          });
          result.profileAccuracy.forEach((value, c) => {
            expectClose(value, defined(accuracy.profileAccuracy[c]), CDM_TOLERANCE, `τ_c ${c}`);
          });
        });
      }
    }
  }

  for (const reference of additiveFixture.cases) {
    const matrix = buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
    for (const fit of reference.fits) {
      for (const accuracy of fit.accuracy ?? []) {
        const classification = accuracy.classification === "eap" ? "eap" : "map";
        it(`matches GDINA's CA by ${classification.toUpperCase()} for ${fitName(fit.model)} items on ${reference.name} data`, () => {
          const result = cdmClassificationAccuracy(matrix, referenceCalibration(reference, fit), { classification });
          expectClose(result.accuracy, accuracy.accuracy, REFERENCE_PERSON_TOLERANCE, "τ");
          expectClose(result.consistency, accuracy.consistency, REFERENCE_PERSON_TOLERANCE, "γ");
          result.attributeAccuracy.forEach((value, k) => {
            expectClose(value, defined(accuracy.attributeAccuracy[k]), REFERENCE_PERSON_TOLERANCE, `τ_${k}`);
          });
          result.attributeConsistency.forEach((value, k) => {
            expectClose(value, defined(accuracy.attributeConsistency[k]), REFERENCE_PERSON_TOLERANCE, `γ_${k}`);
          });
          result.profileAccuracy.forEach((value, c) => {
            expectClose(value, defined(accuracy.profileAccuracy[c]), REFERENCE_PERSON_TOLERANCE, `τ_c ${c}`);
          });
        });
      }
    }
  }

  it("rejects persons without responses alone, whose accuracy is undefined, and invalid options", () => {
    const qMatrix = { item1: ["a"], item2: ["b"], item3: ["a", "b"] };
    const calibration = calibrateCdm(
      buildResponseMatrix([
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 1],
      ]),
      qMatrix,
    );
    expect(() => cdmClassificationAccuracy(buildResponseMatrix([[null, null, null]]), calibration)).toThrow(
      "Classification accuracy needs at least one person with a response.",
    );
    const matrix = buildResponseMatrix([[1, 0, 1]]);
    expect(() => cdmClassificationAccuracy(matrix, calibration, { classification: JSON.parse('"mle"') })).toThrow(
      'The option classification must be "map" or "eap", found "mle".',
    );
    expect(() => cdmClassificationAccuracy(matrix, calibration, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => cdmClassificationAccuracy(matrix, calibration, JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
    expect(() => cdmClassificationAccuracy(JSON.parse("null"), calibration)).toThrow(
      "A response matrix needs the arrays personIds, itemIds and responses.",
    );
  });
});

describe("additiveMaximizer", () => {
  maximumFixture.cases.forEach((reference, index) => {
    const link = cdmLink(reference.link);
    const { attributeCount, correct, persons } = reference;
    const name = `case ${index + 1}, the ${reference.link} link, ${attributeCount} attributes, ${reference.kind} counts`;
    it(`matches the maximum that meets the KKT conditions: ${name}`, () => {
      const step = additiveMaximizer(link, attributeCount, PROBABILITY_BOUNDS, startingProbabilities(attributeCount));
      const probabilities = step.maximize(Float64Array.from(correct), Float64Array.from(persons));
      step.coefficients.forEach((value, i) => {
        expectClose(value, defined(reference.delta[i]), MAXIMUM_TOLERANCE, `coefficient ${i}`);
      });
      probabilities.forEach((value, l) => {
        expectClose(value, defined(reference.probabilities[l]), MAXIMUM_TOLERANCE, `group ${l}`);
      });
    });

    it(`reaches the same maximum from a reversed start and from coefficients outside the bounds: ${name}`, () => {
      const reversed = startingProbabilities(attributeCount).reverse();
      const step = additiveMaximizer(link, attributeCount, PROBABILITY_BOUNDS, reversed);
      step.maximize(Float64Array.from(correct), Float64Array.from(persons));
      step.coefficients.forEach((value, i) => {
        expectClose(value, defined(reference.delta[i]), MAXIMUM_TOLERANCE, `coefficient ${i}`);
      });
      step.coefficients.fill(1e6);
      step.maximize(Float64Array.from(correct), Float64Array.from(persons));
      step.coefficients.forEach((value, i) => {
        expectClose(value, defined(reference.delta[i]), MAXIMUM_TOLERANCE, `coefficient ${i} after reset`);
      });
    });
  });

  it("stays at the maximum when run again on the same counts", () => {
    const reference = defined(maximumFixture.cases.find((entry) => entry.activeBounds > 0));
    const step = additiveMaximizer(
      cdmLink(reference.link),
      reference.attributeCount,
      PROBABILITY_BOUNDS,
      startingProbabilities(reference.attributeCount),
    );
    const correct = Float64Array.from(reference.correct);
    const persons = Float64Array.from(reference.persons);
    step.maximize(correct, persons);
    const first = Array.from(step.coefficients);
    step.maximize(correct, persons);
    step.coefficients.forEach((value, i) => {
      expectClose(value, defined(first[i]), 1e-12, `coefficient ${i}`);
    });
  });

  it("keeps the probabilities within the bounds for groups without persons and groups answered by all or none", () => {
    for (const link of LINKS) {
      const step = additiveMaximizer(link, 2, PROBABILITY_BOUNDS, startingProbabilities(2));
      const probabilities = step.maximize(Float64Array.of(0, 20, 0, 0), Float64Array.of(0, 20, 30, 0));
      expect(probabilities.every((p) => p >= PROBABILITY_BOUNDS[0] && p <= PROBABILITY_BOUNDS[1])).toBe(true);
      expect(step.coefficients.every(Number.isFinite)).toBe(true);
      expect(probabilities[1]).toBeGreaterThan(0.9);
      expect(probabilities[2]).toBeLessThan(0.3);
    }
  });

  it("fits an item of one attribute exactly as its two group proportions", () => {
    for (const link of LINKS) {
      const step = additiveMaximizer(link, 1, PROBABILITY_BOUNDS, startingProbabilities(1));
      const probabilities = step.maximize(Float64Array.of(30, 90), Float64Array.of(100, 120));
      expectClose(defined(probabilities[0]), 0.3, 1e-9);
      expectClose(defined(probabilities[1]), 0.75, 1e-9);
    }
  });

  it("returns the coefficients of a saturated item from its probabilities", () => {
    // P = [0.2, 0.5, 0.4, 0.9] over 00, 10, 01, 11: 0.2, 0.5 - 0.2, 0.4 - 0.2 and 0.9 - 0.5 - 0.4 + 0.2.
    const [intercept, first, second, interaction] = saturatedCoefficients([0.2, 0.5, 0.4, 0.9], 2);
    expectClose(defined(intercept), 0.2, 1e-15);
    expectClose(defined(first), 0.3, 1e-15);
    expectClose(defined(second), 0.2, 1e-15);
    expectClose(defined(interaction), 0.2, 1e-15);
    expect(saturatedCoefficients([0.3, 0.8], 1).map((value) => Math.round(value * 1e12) / 1e12)).toEqual([0.3, 0.5]);
  });
});

describe("calibrateCdm", () => {
  for (const reference of fixture.cases) {
    const matrix = buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
    for (const fit of reference.fits) {
      it(`matches GDINA's ${fit.model.toUpperCase()} model on ${reference.name} data`, () => {
        const result = calibrateCdm(matrix, definedEntries(reference.qmatrix), {
          model: cdmModel(fit.model),
          tolerance: 1e-8,
          maxIterations: 10_000,
        });
        expect(result.converged).toBe(true);
        expect(result.attributes).toEqual(reference.attributes);
        result.items.forEach((item, j) => {
          item.probabilities.forEach((p, l) => {
            expectClose(p, defined(fit.items[j]?.[l]), CDM_TOLERANCE, `${item.id} group ${l}`);
          });
          expectClose(item.guessing, defined(fit.guessing[j]), CDM_TOLERANCE, `${item.id} guessing`);
          expectClose(item.slip, defined(fit.slip[j]), CDM_TOLERANCE, `${item.id} slip`);
        });
        // Profiles of one DINA class, alike in their ideal responses, have only their sum identified.
        const classes = fit.classes ?? result.profiles.map((_, c) => c);
        const sums = new Map<number, number>();
        result.profiles.forEach(({ proportion }, c) => {
          const key = defined(classes[c]);
          sums.set(key, (sums.get(key) ?? 0) + proportion - defined(fit.proportions[c]));
        });
        for (const [key, difference] of sums) expectClose(difference, 0, CDM_TOLERANCE, `class ${key}`);
        expectClose(result.logLikelihood, fit.logLikelihood, LOG_LIKELIHOOD_TOLERANCE, "log-likelihood");
        expectClose(result.aic, fit.aic, 2 * LOG_LIKELIHOOD_TOLERANCE, "AIC");
        expectClose(result.bic, fit.bic, 2 * LOG_LIKELIHOOD_TOLERANCE, "BIC");
      });
    }
  }

  for (const reference of additiveFixture.cases) {
    const matrix = buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
    for (const fit of reference.fits) {
      it(`matches GDINA's ${fitName(fit.model)} models on ${reference.name} data`, () => {
        const result = calibrateCdm(matrix, definedEntries(reference.qmatrix), {
          model: fitModels(fit.model),
          tolerance: 1e-10,
          maxIterations: 10_000,
        });
        expect(result.converged).toBe(true);
        expect(result.attributes).toEqual(reference.attributes);
        result.items.forEach((item, j) => {
          expect(item.model).toBe(typeof fit.model === "string" ? fit.model : fit.model[j]);
          item.probabilities.forEach((p, l) => {
            expectClose(p, defined(fit.items[j]?.[l]), ADDITIVE_TOLERANCE, `${item.id} group ${l}`);
          });
          item.coefficients.forEach((value, i) => {
            expectClose(value, defined(fit.coefficients[j]?.[i]), ADDITIVE_TOLERANCE, `${item.id} coefficient ${i}`);
          });
          expectClose(item.guessing, defined(fit.guessing[j]), ADDITIVE_TOLERANCE, `${item.id} guessing`);
          expectClose(item.slip, defined(fit.slip[j]), ADDITIVE_TOLERANCE, `${item.id} slip`);
        });
        result.profiles.forEach(({ proportion }, c) => {
          expectClose(proportion, defined(fit.proportions[c]), ADDITIVE_TOLERANCE, `profile ${c}`);
        });
        // GDINA's M-steps stop short of the maximum, so the exact M-step reaches at least its log-likelihood.
        expect(result.logLikelihood).toBeGreaterThanOrEqual(fit.logLikelihood - LOG_LIKELIHOOD_TOLERANCE);
        // Both criteria add the same parameters to -2 logLikelihood.
        expectClose(result.aic + 2 * result.logLikelihood, fit.aic + 2 * fit.logLikelihood, 1e-6, "AIC parameters");
        expectClose(result.bic + 2 * result.logLikelihood, fit.bic + 2 * fit.logLikelihood, 1e-6, "BIC parameters");
      });
    }
  }

  it("gives coefficients that reproduce each item's success probabilities under its model", () => {
    const reference = defined(additiveFixture.cases[1]);
    const matrix = buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
    const models = defined(reference.fits[3]).model;
    const result = calibrateCdm(matrix, definedEntries(reference.qmatrix), {
      model: fitModels(models),
      maxIterations: 50,
    });
    const rates = new Set<CdmModel>();
    result.items.forEach((item) => {
      rates.add(item.model);
      const groups = profilesOf(item.attributes.length);
      expect(item.coefficients).toHaveLength(
        item.model === "gdina"
          ? groups.length
          : item.model === "dina" || item.model === "dino"
            ? 2
            : item.attributes.length + 1,
      );
      item.probabilities.forEach((p, l) => {
        const profile = defined(groups[l]);
        if (item.model === "gdina") {
          // The sum of the effects of the groups whose attributes the profile masters.
          const sum = groups.reduce(
            (total, other, a) =>
              other.every((v, k) => v <= (profile[k] ?? 0)) ? total + defined(item.coefficients[a]) : total,
            0,
          );
          expectClose(sum, p, 1e-12, `${item.id} group ${l}`);
        } else if (item.model === "dina" || item.model === "dino") {
          const raised = item.model === "dina" ? l === groups.length - 1 : l > 0;
          expectClose(
            defined(item.coefficients[0]) + (raised ? defined(item.coefficients[1]) : 0),
            p,
            1e-12,
            `${item.id} group ${l}`,
          );
        } else {
          const link = defined(LINK_OF[item.model]);
          expectClose(
            inverse(link, defined(predictors(item.coefficients, item.attributes.length)[l])),
            p,
            1e-9,
            `${item.id} group ${l}`,
          );
        }
      });
    });
    expect([...rates].sort()).toEqual([...MODELS].sort());
  });

  it("fits items of one attribute as the G-DINA does under every model", () => {
    const reference = defined(additiveFixture.cases[1]);
    const itemIds = reference.itemIds.slice(0, 10);
    const matrix = buildResponseMatrix(
      reference.responses.map((row) => row.slice(0, 10)),
      { itemIds },
    );
    const qMatrix = Object.fromEntries(Object.entries(definedEntries(reference.qmatrix)).slice(0, 10));
    const options = { tolerance: 1e-10, maxIterations: 10_000 };
    const saturated = calibrateCdm(matrix, qMatrix, { ...options, model: "gdina" });
    for (const model of ["acdm", "llm", "rrum"] as const) {
      const result = calibrateCdm(matrix, qMatrix, { ...options, model });
      result.items.forEach((item, j) => {
        item.probabilities.forEach((p, l) => {
          expectClose(p, defined(saturated.items[j]?.probabilities[l]), 1e-6, `${model} ${item.id} group ${l}`);
        });
      });
    }
  });

  it("fits no better than the G-DINA, which contains the additive models", () => {
    const reference = defined(additiveFixture.cases[1]);
    const matrix = buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
    const qMatrix = definedEntries(reference.qmatrix);
    const saturated = calibrateCdm(matrix, qMatrix, { tolerance: 1e-8, maxIterations: 10_000 });
    for (const model of ["acdm", "llm", "rrum"] as const) {
      const result = calibrateCdm(matrix, qMatrix, { model, tolerance: 1e-8, maxIterations: 10_000 });
      expect(result.logLikelihood).toBeLessThan(saturated.logLikelihood);
    }
  });

  it("lists an item's attributes in any order without changing the calibration", () => {
    const matrix = buildResponseMatrix([
      [1, 1, 0],
      [0, 0, 1],
      [1, 0, 0],
      [1, 1, 1],
      [0, 1, 0],
      [0, 0, 0],
      [1, 1, 1],
      [1, 0, 1],
    ]);
    const first = calibrateCdm(matrix, { item1: ["a"], item2: ["a", "b"], item3: ["b"] }, { model: "llm" });
    const second = calibrateCdm(matrix, { item1: ["a"], item2: ["b", "a"], item3: ["b"] }, { model: "llm" });
    expect(second).toEqual(first);
    expect(defined(first.items[1]).attributes).toEqual(["a", "b"]);
  });

  it("calibrates additive items that every person answered correctly or incorrectly", () => {
    const matrix = buildResponseMatrix([
      [1, 0, 1, 1],
      [1, 0, 0, 0],
      [1, 0, 1, 1],
      [1, 0, 0, 1],
      [1, 0, 1, 0],
    ]);
    const qMatrix = { item1: ["a", "b"], item2: ["a", "b"], item3: ["a"], item4: ["b"] };
    for (const model of ["acdm", "llm", "rrum"] as const) {
      const result = calibrateCdm(matrix, qMatrix, { model });
      const [correct, incorrect] = result.items;
      expect(defined(correct).probabilities.every((p) => p > 0.99 && p <= 1 - 1e-4)).toBe(true);
      expect(defined(incorrect).probabilities.every((p) => p < 0.01 && p >= 1e-4)).toBe(true);
      for (const item of result.items) expect(item.coefficients.every(Number.isFinite)).toBe(true);
    }
  });

  for (const reference of comparisonFixture.cases) {
    it(`matches GDINA's incomplete covariance of the success probabilities on ${reference.name}`, () => {
      const result = comparisonCalibration(reference);
      reference.compared.forEach((id, i) => {
        const item = defined(result.items.find((candidate) => candidate.id === id));
        const expected = defined(reference.covariance[i]);
        const scale = Math.max(...expected.flat().map(Math.abs));
        expected.forEach((row, a) => {
          expectClose(
            defined(item.standardErrors?.[a]),
            Math.sqrt(defined(row[a])),
            COVARIANCE_TOLERANCE * Math.sqrt(scale),
            `${id} SE ${a}`,
          );
          row.forEach((value, b) => {
            expectClose(
              defined(item.covariance?.[a]?.[b]),
              value,
              COVARIANCE_TOLERANCE * scale,
              `${id} covariance ${a} ${b}`,
            );
          });
        });
      });
    });
  }

  it("leaves out the standard errors and the covariances unless the option asks for them", () => {
    const [reference] = comparisonFixture.cases;
    const matrix = buildResponseMatrix(defined(reference).responses, { itemIds: defined(reference).itemIds });
    const qMatrix = definedEntries(defined(reference).qmatrix);
    for (const options of [{}, { standardErrors: "none" as const }]) {
      for (const item of calibrateCdm(matrix, qMatrix, options).items) {
        expect(item.standardErrors).toBeUndefined();
        expect(item.covariance).toBeUndefined();
      }
    }
  });

  it("gives the standard errors and covariances of a calibration whose persons are fewer than its parameters", () => {
    const matrix = buildResponseMatrix([
      [1, 0, 1],
      [0, 1, 1],
      [1, 1, 0],
    ]);
    const result = calibrateCdm(
      matrix,
      { item1: ["a", "b"], item2: ["a", "b"], item3: ["a", "b"] },
      { standardErrors: "incomplete" },
    );
    for (const item of result.items) {
      expect(item.covariance).toHaveLength(4);
      expect(item.standardErrors?.every((error) => Number.isFinite(error) && error >= 0)).toBe(true);
    }
  });

  it("rejects incomplete standard errors for items that are not G-DINA, and other values of the option", () => {
    const matrix = buildResponseMatrix([
      [1, 0],
      [0, 1],
      [1, 1],
    ]);
    const qMatrix = { item1: ["a"], item2: ["a", "b"] };
    expect(() => calibrateCdm(matrix, qMatrix, { model: ["gdina", "dina"], standardErrors: "incomplete" })).toThrow(
      'The option standardErrors "incomplete" needs the G-DINA model for every item, found "dina" for item "item2".',
    );
    expect(() => calibrateCdm(matrix, qMatrix, { standardErrors: JSON.parse('"hessian"') })).toThrow(
      'The option standardErrors must be "none" or "incomplete", found "hessian".',
    );
    expect(() => calibrateCdm(matrix, qMatrix, { standardErrors: JSON.parse("null") })).toThrow(DataError);
  });

  it("refuses incomplete standard errors past 3,162 success probabilities, before it calibrates", () => {
    // One item of 12 attributes has 4,096 success probabilities and 12 items of one attribute 2 each: 4,120.
    const attributes = Array.from({ length: 12 }, (_, k) => `a${k}`);
    const qMatrix = {
      wide: attributes,
      ...Object.fromEntries(attributes.map((attribute) => [attribute, [attribute]])),
    };
    const itemIds = Object.keys(qMatrix);
    // No person answers. Calibration refuses that next, so a call that the bound does not stop ends at once with the
    // other message, where one with responses would go on to invert a matrix of 4,120 by 4,120 for minutes.
    const matrix = buildResponseMatrix([itemIds.map(() => null)], { itemIds });
    expect(() => calibrateCdm(matrix, qMatrix, { standardErrors: "incomplete" })).toThrow(
      'The option standardErrors "incomplete" takes at most 3162 success probabilities over all items, 2^k for an item of k attributes, found 4120.',
    );
  }, 5_000);

  it("refuses more than 15 attributes, 2^22 cells of items by profiles or 2^18 latent groups, before it calibrates", () => {
    const attributes = (count: number) => Array.from({ length: count }, (_, k) => `a${k}`);
    const design = (entries: readonly (readonly string[])[]) => {
      const qMatrix = Object.fromEntries(entries.map((listed, j) => [`item${j}`, listed]));
      const itemIds = Object.keys(qMatrix);
      return [buildResponseMatrix([itemIds.map(() => 0), itemIds.map(() => 1)], { itemIds }), qMatrix] as const;
    };
    // 20 items of one of 16 attributes, 129 items of one of 15, and 9 items of all 15: each passes the other limits.
    const sixteen = attributes(16);
    expect(() => calibrateCdm(...design(Array.from({ length: 20 }, (_, j) => [sixteen[j % 16] ?? ""])))).toThrow(
      "A design has at most 15 attributes, found 16.",
    );
    const fifteen = attributes(15);
    expect(() => calibrateCdm(...design(Array.from({ length: 129 }, (_, j) => [fifteen[j % 15] ?? ""])))).toThrow(
      "A design has at most 4194304 cells of items by profiles, items × 2^attributes, found 4227072 of 129 items and 15 attributes.",
    );
    expect(() => calibrateCdm(...design(Array.from({ length: 9 }, () => fifteen)))).toThrow(
      "A design has at most 262144 latent groups over all items, 2^k for an item of k attributes, found 294912.",
    );
  });

  it("rejects polytomous responses, a Q-matrix that does not fit the items and unknown models", () => {
    const matrix = buildResponseMatrix([
      [1, 0],
      [0, 1],
    ]);
    const qMatrix = { item1: ["a"], item2: ["a", "b"] };
    expect(() => calibrateCdm(buildResponseMatrix([[2, 0]]), qMatrix)).toThrow(
      'Cognitive diagnosis needs dichotomous items, but person "person1" scored 2 on item "item1".',
    );
    expect(() => calibrateCdm(matrix, { item1: ["a"] })).toThrow('Item "item2" needs at least one attribute.');
    expect(() => calibrateCdm(matrix, { item1: [], item2: ["a"] })).toThrow(
      'Item "item1" needs at least one attribute.',
    );
    expect(() => calibrateCdm(matrix, JSON.parse("null"))).toThrow(
      "A Q-matrix must be an object of attribute lists by item id, found null.",
    );
    expect(() => calibrateCdm(matrix, { ...qMatrix, item3: ["a"] })).toThrow(
      'The Q-matrix has item "item3", which is not an item id.',
    );
    expect(() => calibrateCdm(matrix, { item1: ["a", "a"], item2: ["b"] })).toThrow(
      'Item "item1" lists an attribute twice.',
    );
    expect(() => calibrateCdm(matrix, qMatrix, { model: JSON.parse('"mixture"') })).toThrow(
      'The option model must be "gdina", "dina", "dino", "acdm", "llm" or "rrum", found "mixture".',
    );
    expect(() => calibrateCdm(matrix, qMatrix, { model: ["acdm", JSON.parse('"rasch"')] })).toThrow(
      'The model of item 2 must be "gdina", "dina", "dino", "acdm", "llm" or "rrum", found "rasch".',
    );
    expect(() => calibrateCdm(matrix, qMatrix, { model: ["dina"] })).toThrow("Found 1 models for 2 items.");
    expect(() => calibrateCdm(buildResponseMatrix([[null, null]]), qMatrix)).toThrow(
      "Cognitive diagnosis needs at least one response.",
    );
    expect(() => calibrateCdm(matrix, qMatrix, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => calibrateCdm(matrix, qMatrix, JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
    expect(() => calibrateCdm(matrix, qMatrix, { model: JSON.parse("5") })).toThrow(
      "The option model must be a cognitive diagnosis model or an array of one per item, found 5.",
    );
  });
});

/** The calibrations of the comparison fixture's cases, by name: each takes up to a second, and several tests use it. */
const COMPARISON_CALIBRATIONS = new Map<string, CdmCalibration>();

/** A calibration of the fixture's data to the precision of GDINA's fit, with the covariances of the success probabilities. */
function comparisonCalibration(reference: (typeof comparisonFixture.cases)[number]): CdmCalibration {
  const known = COMPARISON_CALIBRATIONS.get(reference.name);
  if (known !== undefined) return known;
  const matrix = buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
  const calibration = calibrateCdm(matrix, definedEntries(reference.qmatrix), {
    standardErrors: "incomplete",
    tolerance: 1e-10,
    maxIterations: 10_000,
  });
  COMPARISON_CALIBRATIONS.set(reference.name, calibration);
  return calibration;
}

/** A G-DINA item of two attributes with given probabilities and independent errors of standard error `error`. */
function twoAttributeCalibration(probabilities: readonly number[], error: number): CdmCalibration {
  const covariance = probabilities.map((_, a) => probabilities.map((__, b) => (a === b ? error * error : 0)));
  return {
    attributes: ["a", "b"],
    items: [
      {
        id: "item1",
        attributes: ["a", "b"],
        model: "gdina",
        probabilities,
        guessing: defined(probabilities[0]),
        slip: 1 - defined(probabilities.at(-1)),
        coefficients: [],
        standardErrors: probabilities.map(() => error),
        covariance,
      },
    ],
    profiles: [],
    logLikelihood: 0,
    aic: 0,
    bic: 0,
    converged: true,
    iterations: 0,
  };
}

describe("compareItemModels", () => {
  for (const reference of comparisonFixture.cases) {
    it(`matches GDINA's Wald statistics, degrees of freedom, p-values and selected models on ${reference.name}`, () => {
      const result = compareItemModels(comparisonCalibration(reference));
      expect(result.map(({ item }) => item)).toEqual(reference.compared);
      result.forEach((comparison, i) => {
        expect(comparison.tests.map(({ model }) => model)).toEqual(comparisonFixture.models);
        comparison.tests.forEach((test, m) => {
          const at = `${comparison.item} ${test.model}`;
          const statistic = defined(reference.statistics[i]?.[m]);
          expect(test.degreesOfFreedom, at).toBe(reference.degreesOfFreedom[i]?.[m]);
          expectClose(test.statistic, statistic, STATISTIC_TOLERANCE * Math.max(statistic, 1), `${at} statistic`);
          expectClose(test.pValue, defined(reference.pValues[i]?.[m]), P_VALUE_TOLERANCE, `${at} p-value`);
        });
        expect(comparison.selected, comparison.item).toBe(reference.selected[i]);
      });
    });
  }

  it("rejects the true model of an item and selects it as often as GDINA's Wald test does on 20 simulated sets", () => {
    let items = 0;
    let rejected = 0;
    let selected = 0;
    for (const reference of comparisonFixture.cases.slice(1)) {
      compareItemModels(comparisonCalibration(reference)).forEach(({ tests, selected: model }, i) => {
        const truth = defined(reference.truth?.[i]);
        items++;
        if (defined(tests.find(({ model: tested }) => tested === truth)?.pValue) <= comparisonFixture.level) rejected++;
        if (model === truth) selected++;
      });
    }
    expect(items).toBe(comparisonFixture.recovery.items);
    expectClose(rejected / items, comparisonFixture.recovery.rejected, 1e-12, "rejected");
    expectClose(selected / items, comparisonFixture.recovery.selected, 1e-12, "selected");
  });

  it("derives the statistics of the DINA, DINO and A-CDM from the success probabilities and their errors", () => {
    // P = (0.2, 0.2, 0.2, 0.8) with independent errors of 0.1: the DINA's two differences are 0; the DINO's are 0 and
    // -0.6, whose covariance is 0.01 [[2, -1], [-1, 2]], so W = 0.36 * 2 / 0.03 = 24; the A-CDM's interaction is
    // 0.2 - 0.2 - 0.2 + 0.8 = 0.6, with variance 0.04, so W = 0.36 / 0.04 = 9.
    const [item] = compareItemModels(twoAttributeCalibration([0.2, 0.2, 0.2, 0.8], 0.1), {
      models: ["dino", "dina", "acdm"],
    });
    expect(item?.tests.map(({ model }) => model)).toEqual(["dino", "dina", "acdm"]);
    expect(item?.tests.map(({ degreesOfFreedom }) => degreesOfFreedom)).toEqual([2, 2, 1]);
    expectClose(defined(item?.tests[0]?.statistic), 24, 1e-9, "DINO");
    expectClose(defined(item?.tests[1]?.statistic), 0, 1e-9, "DINA");
    expectClose(defined(item?.tests[2]?.statistic), 9, 1e-9, "A-CDM");
    expectClose(defined(item?.tests[1]?.pValue), 1, 1e-12, "DINA p-value");
    expectClose(defined(item?.tests[2]?.pValue), 0.0027, 1e-4, "A-CDM p-value");
    expect(item?.selected).toBe("dina");
  });

  it("selects the model with the largest p-value above 0.05, and the G-DINA when every model is rejected", () => {
    // The A-CDM's interaction of 0.6 over its error, against a DINA that the first difference of 0.3 rejects.
    const additive = compareItemModels(twoAttributeCalibration([0.1, 0.4, 0.4, 0.7], 0.1), {
      models: ["dina", "acdm"],
    });
    expect(additive[0]?.selected).toBe("acdm");
    expectClose(defined(additive[0]?.tests[1]?.statistic), 0, 1e-9, "A-CDM");
    const rejected = compareItemModels(twoAttributeCalibration([0.1, 0.4, 0.4, 0.95], 0.01), {
      models: ["dina", "dino", "acdm"],
    });
    expect(rejected[0]?.tests.every(({ pValue }) => defined(pValue) < 0.05)).toBe(true);
    expect(rejected[0]?.selected).toBe("gdina");
  });

  it("tests the models of the option in its order, and selects among them only", () => {
    const [full] = compareItemModels(comparisonCalibration(defined(comparisonFixture.cases[0])));
    const [only] = compareItemModels(comparisonCalibration(defined(comparisonFixture.cases[0])), {
      models: ["rrum", "dina"],
    });
    expect(only?.tests.map(({ model }) => model)).toEqual(["rrum", "dina"]);
    expect(only?.tests[0]).toEqual(full?.tests[4]);
    expect(only?.tests[1]).toEqual(full?.tests[0]);
    expect(["rrum", "dina", "gdina"]).toContain(only?.selected);
  });

  it("leaves out items of one attribute, and has nothing to compare when every item has one", () => {
    const calibration = comparisonCalibration(defined(comparisonFixture.cases[0]));
    expect(compareItemModels(calibration).map(({ item }) => item)).not.toContain("item1");
    const matrix = buildResponseMatrix([
      [1, 0],
      [0, 1],
      [1, 1],
      [0, 0],
    ]);
    const single = calibrateCdm(matrix, { item1: ["a"], item2: ["b"] }, { standardErrors: "incomplete" });
    expect(compareItemModels(single)).toEqual([]);
  });

  it("gives a statistic of 0 and a p-value of 1 to an item that every reduced model fits exactly", () => {
    const [item] = compareItemModels(twoAttributeCalibration([0.3, 0.3, 0.3, 0.3], 0.05), { models: ["dina", "dino"] });
    expect(item?.tests.map(({ statistic }) => statistic)).toEqual([0, 0]);
    expect(item?.tests.map(({ pValue }) => pValue)).toEqual([1, 1]);
    expect(item?.selected).toBe("dina");
  });

  it("rejects calibrations that lack G-DINA items with their covariance, and invalid options", () => {
    const valid = twoAttributeCalibration([0.2, 0.2, 0.2, 0.8], 0.1);
    const [item] = valid.items;
    const withItem = (changes: object): CdmCalibration => ({ ...valid, items: [{ ...defined(item), ...changes }] });
    expect(() => compareItemModels(JSON.parse("null"))).toThrow(
      "A calibration must be an object from calibrateCdm, found null.",
    );
    expect(() => compareItemModels({ ...valid, items: JSON.parse('"items"') })).toThrow(
      'A calibration needs the array items, found "items".',
    );
    expect(() => compareItemModels({ ...valid, items: JSON.parse("[null]") })).toThrow(DataError);
    expect(() => compareItemModels(withItem({ attributes: JSON.parse('"ab"') }))).toThrow(
      'Calibrated item 1 needs the array attributes, found "ab".',
    );
    expect(() => compareItemModels(withItem({ model: "dina" }))).toThrow(
      'Item "item1" was calibrated as "dina"; the test needs the G-DINA estimates.',
    );
    expect(() => compareItemModels(withItem({ probabilities: [0.2, 0.8] }))).toThrow(
      'Item "item1" needs 4 probabilities from 0 to 1, one per latent group.',
    );
    const message =
      'Item "item1" needs the 4 by 4 covariance of its success probabilities, from calibrateCdm with the option standardErrors "incomplete".';
    expect(() => compareItemModels(withItem({ covariance: undefined }))).toThrow(message);
    expect(() => compareItemModels(withItem({ covariance: [[1]] }))).toThrow(message);
    expect(() =>
      compareItemModels(withItem({ covariance: JSON.parse('[[0,0,0,"x"],[0,0,0,0],[0,0,0,0],[0,0,0,0]]') })),
    ).toThrow(message);
    expect(() => compareItemModels(valid, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => compareItemModels(valid, JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
    expect(() => compareItemModels(valid, { models: JSON.parse('"dina"') })).toThrow(
      'The option models must be an array of reduced models, such as ["dina", "acdm"], found "dina".',
    );
    expect(() => compareItemModels(valid, { models: [] })).toThrow("The option models needs at least one model.");
    expect(() => compareItemModels(valid, { models: [JSON.parse('"gdina"')] })).toThrow(
      'Model 1 of the option models must be "dina", "dino", "acdm", "llm" or "rrum", found "gdina".',
    );
    expect(() => compareItemModels(valid, { models: ["dina", "dina"] })).toThrow(
      'The option models lists "dina" twice.',
    );
  });
});

describe("nonparametricClassification", () => {
  for (const reference of fixture.cases) {
    const matrix = buildResponseMatrix(reference.responses.slice(0, 30), { itemIds: reference.itemIds });
    for (const expected of reference.nonparametric) {
      const rule = expected.gate === "AND" ? "conjunctive" : "disjunctive";
      it(`matches NPCD's AlphaNP with the ${expected.gate} gate on ${reference.name} data`, () => {
        nonparametricClassification(matrix, definedEntries(reference.qmatrix), { rule }).forEach((person, i) => {
          expect(person.distance).toBe(expected.distance[i]);
          expect(person.ties).toBe(expected.ties[i]);
          const profile = expected.profile[i];
          if (profile !== null && profile !== undefined) expect(person.profile).toEqual(profile);
        });
      });
    }
  }

  it("takes the first nearest profile on ties and skips missing responses", () => {
    const qMatrix = { item1: ["a"], item2: ["b"], item3: ["a", "b"] };
    const result = nonparametricClassification(
      buildResponseMatrix([
        [1, 0, 0],
        [null, 1, null],
      ]),
      qMatrix,
    );
    expect(result[0]).toEqual({ id: "person1", profile: [1, 0], distance: 0, ties: 1 });
    // Only item 2 counts: profiles 01 and 11 both answer it correctly.
    expect(result[1]).toEqual({ id: "person2", profile: [0, 1], distance: 0, ties: 2 });
  });

  it("classifies over 15 attributes, the most a design takes", () => {
    const attributes = Array.from({ length: 15 }, (_, k) => `a${k}`);
    const qMatrix = Object.fromEntries(attributes.map((attribute, k) => [`item${k + 1}`, [attribute]]));
    const pattern = attributes.map((_, k) => k % 2);
    const [person] = nonparametricClassification(buildResponseMatrix([pattern]), qMatrix);
    expect(person).toEqual({ id: "person1", profile: pattern, distance: 0, ties: 1 });
  });

  it("rejects options of other types from JavaScript", () => {
    expect(() => nonparametricClassification(buildResponseMatrix([[1]]), { item1: ["a"] }, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() =>
      nonparametricClassification(buildResponseMatrix([[1]]), { item1: ["a"] }, JSON.parse('{"verbose":1}')),
    ).toThrow('There is no option "verbose"');
  });
});

describe("attributeProfiles", () => {
  const reference = defined(fixture.cases[1]);
  const matrix = buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
  for (const fit of reference.fits) {
    it(`matches GDINA's personparm under the ${fit.model.toUpperCase()} model`, () => {
      const calibration = calibrateCdm(matrix, definedEntries(reference.qmatrix), {
        model: cdmModel(fit.model),
        tolerance: 1e-10,
        maxIterations: 10_000,
      });
      const persons = attributeProfiles(matrix, calibration, { posterior: true });
      (fit.persons ?? []).forEach((expected, i) => {
        const person = defined(persons[i]);
        person.mastery.forEach((p, k) => {
          expectClose(p, defined(expected.mastery[k]), CDM_TOLERANCE, `mastery ${k}`);
        });
        person.posterior.forEach((p, c) => {
          expectClose(p, defined(expected.posterior[c]), CDM_TOLERANCE, `posterior ${c}`);
        });
        expect(person.eap).toEqual(expected.eap);
        if (expected.map !== null && expected.map !== undefined) expect(person.map).toEqual(expected.map);
      });
    });
  }

  for (const additiveReference of additiveFixture.cases) {
    const additiveMatrix = buildResponseMatrix(additiveReference.responses, { itemIds: additiveReference.itemIds });
    for (const fit of additiveReference.fits) {
      it(`matches GDINA's personparm for ${fitName(fit.model)} items on ${additiveReference.name} data`, () => {
        const persons = attributeProfiles(additiveMatrix, referenceCalibration(additiveReference, fit), {
          posterior: true,
        });
        expect(fit.persons).toHaveLength(20);
        (fit.persons ?? []).forEach((expected, i) => {
          const person = defined(persons[i]);
          person.mastery.forEach((p, k) => {
            expectClose(p, defined(expected.mastery[k]), REFERENCE_PERSON_TOLERANCE, `mastery ${k}`);
          });
          person.posterior.forEach((p, c) => {
            expectClose(p, defined(expected.posterior[c]), REFERENCE_PERSON_TOLERANCE, `posterior ${c}`);
          });
          expect(person.eap).toEqual(expected.eap);
          if (expected.map !== null && expected.map !== undefined) expect(person.map).toEqual(expected.map);
        });
      });
    }
  }

  it("gives a person without responses the calibrated proportions", () => {
    const calibration = calibrateCdm(matrix, definedEntries(reference.qmatrix), { maxIterations: 5 });
    const [person] = attributeProfiles(
      buildResponseMatrix([reference.itemIds.map(() => null)], { itemIds: reference.itemIds }),
      calibration,
      { posterior: true },
    );
    expect(defined(person).posterior).toEqual(calibration.profiles.map((profile) => profile.proportion));
    expect(() => attributeProfiles(buildResponseMatrix([[1]]), calibration)).toThrow(
      "The matrix needs the calibration's items, in its order.",
    );
  });

  it("keeps the id and the place of every person among persons without responses", () => {
    const calibration = calibrateCdm(matrix, definedEntries(reference.qmatrix), { maxIterations: 5 });
    const [first, second] = matrix.responses.map((row) => row.slice());
    const none = reference.itemIds.map(() => null);
    const options = { itemIds: reference.itemIds };
    const alone = attributeProfiles(buildResponseMatrix([defined(first), defined(second)], options), calibration, {
      posterior: true,
    });
    const mixed = attributeProfiles(
      buildResponseMatrix([none, defined(first), none, defined(second)], {
        ...options,
        personIds: ["a", "b", "c", "d"],
      }),
      calibration,
      { posterior: true },
    );
    expect(mixed.map((person) => person.id)).toEqual(["a", "b", "c", "d"]);
    expect(mixed[1]).toEqual({ ...defined(alone[0]), id: "b" });
    expect(mixed[3]).toEqual({ ...defined(alone[1]), id: "d" });
    const proportions = calibration.profiles.map((profile) => profile.proportion);
    expect(defined(mixed[0]).posterior).toEqual(proportions);
    expect(mixed[2]).toEqual({ ...defined(mixed[0]), id: "c" });
  });

  it("places responses that no profile can give on the profiles they contradict least", () => {
    const qMatrix = { item1: ["A"], item2: ["B"], item3: ["A", "B"] };
    const data = buildResponseMatrix([
      [1, 0, 1],
      [0, 0, 1],
    ]);
    const calibration = calibrateCdm(data, qMatrix, { model: "dina", maxIterations: 5 });
    // Deterministic items, as JSON may hold them: a person answers correctly exactly when mastering the attributes.
    const items = calibration.items.map((item) => ({
      ...item,
      probabilities: item.probabilities.map((_, group, all) => (group === all.length - 1 ? 1 : 0)),
    }));
    const [first, second] = attributeProfiles(data, { ...calibration, items }, { posterior: true });
    // Of the profiles [0, 0], [1, 0], [0, 1] and [1, 1], the first person's answers contradict [1, 0] and [1, 1] once
    // and the others more; the second person's contradict [0, 0] once and the others twice.
    const posterior = defined(first).posterior;
    expectClose((posterior[1] ?? 0) + (posterior[3] ?? 0), 1, 1e-12);
    expectClose(defined(second).posterior[0] ?? 0, 1, 1e-12);
    expect(defined(second).map).toEqual([0, 0]);
    expect(Number.isFinite(cdmClassificationAccuracy(data, { ...calibration, items }).accuracy)).toBe(true);
  });

  it("gives the posterior over the profiles only with the option posterior", () => {
    const calibration = calibrateCdm(matrix, definedEntries(reference.qmatrix), { maxIterations: 5 });
    const summaries = attributeProfiles(matrix, calibration);
    const withPosterior = attributeProfiles(matrix, calibration, { posterior: true });
    expect(summaries).toHaveLength(matrix.personIds.length);
    for (const person of summaries) expect(Object.keys(person).sort()).toEqual(["eap", "id", "map", "mastery"]);
    expect(attributeProfiles(matrix, calibration, { posterior: false })).toEqual(summaries);
    expect(
      withPosterior.map(({ posterior, ...rest }) => (posterior.length === calibration.profiles.length ? rest : null)),
    ).toEqual(summaries);
  });

  it("refuses options that are not an object and a posterior that is not a boolean", () => {
    const calibration = calibrateCdm(matrix, definedEntries(reference.qmatrix), { maxIterations: 5 });
    expect(() => attributeProfiles(matrix, calibration, JSON.parse("null"))).toThrow(DataError);
    expect(() => attributeProfiles(matrix, calibration, JSON.parse('{ "posterior": "yes" }'))).toThrow(
      'The option posterior must be a boolean, found "yes".',
    );
    expect(() => attributeProfiles(matrix, calibration, JSON.parse('{ "posterior": null }'))).toThrow(DataError);
  });

  it("rejects a calibration from JSON that does not fit its design", () => {
    const calibration = calibrateCdm(matrix, definedEntries(reference.qmatrix), { maxIterations: 5 });
    const [first, ...rest] = calibration.items;
    const parse = (value: object): CdmCalibration => JSON.parse(JSON.stringify(value));
    expect(() => attributeProfiles(matrix, JSON.parse("null"))).toThrow(
      "A calibration must be an object from calibrateCdm, found null.",
    );
    expect(() => attributeProfiles(matrix, parse({ ...calibration, profiles: "x" }))).toThrow(
      `A calibration over ${calibration.attributes.length} attributes needs its ${calibration.profiles.length} profiles in order.`,
    );
    expect(() =>
      attributeProfiles(matrix, parse({ ...calibration, profiles: [...calibration.profiles].reverse() })),
    ).toThrow(DataError);
    const zero = calibration.profiles.map((profile) => ({ ...profile, proportion: 0 }));
    expect(() => attributeProfiles(matrix, parse({ ...calibration, profiles: zero }))).toThrow(
      "A calibration needs profile proportions that are not all 0.",
    );
    const item = defined(first);
    expect(() =>
      attributeProfiles(matrix, parse({ ...calibration, items: [{ ...item, probabilities: [0.5] }, ...rest] })),
    ).toThrow(`Item "${item.id}" needs ${item.probabilities.length} probabilities from 0 to 1, one per latent group.`);
    expect(() => attributeProfiles(matrix, parse({ ...calibration, attributes: [] }))).toThrow(
      "A calibration's attributes must be its items', in order of first appearance.",
    );
  });
});

describe("validateQMatrix", () => {
  const reference = defined(fixture.cases[1]);
  const fit = defined(reference.fits[0]);
  it("matches GDINA's Qval with the PVAF method", () => {
    const matrix = buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
    const calibration = calibrateCdm(matrix, definedEntries(reference.qmatrix), {
      tolerance: 1e-10,
      maxIterations: 10_000,
    });
    validateQMatrix(matrix, calibration).forEach((item, j) => {
      const expected = defined(fit.validation?.[j]);
      expect(item.suggested).toEqual(expected.suggested);
      expectClose(item.pvaf, expected.pvaf, CDM_TOLERANCE, item.id);
    });
  });

  it("leaves the PVAF of an item without responses undefined and keeps its attributes", () => {
    const qMatrix = { item1: ["a"], item2: ["b"], item3: ["a", "b"] };
    const matrix = buildResponseMatrix([
      [1, 0, null],
      [0, 1, null],
      [1, 1, null],
      [0, 0, null],
    ]);
    const [first, second, third] = validateQMatrix(matrix, calibrateCdm(matrix, qMatrix));
    expect(third).toEqual({ id: "item3", attributes: ["a", "b"], suggested: ["a", "b"], pvaf: null });
    for (const item of [first, second]) expect(Number.isFinite(defined(item).pvaf)).toBe(true);
  });

  it("leaves out the latent groups of profiles without proportion", () => {
    const matrix = buildResponseMatrix(reference.responses, { itemIds: reference.itemIds });
    const calibration = calibrateCdm(matrix, definedEntries(reference.qmatrix));
    const profiles = calibration.profiles.map((entry, c) => (c === 1 ? { ...entry, proportion: 0 } : entry));
    for (const item of validateQMatrix(matrix, { ...calibration, profiles })) {
      expect(item.pvaf).toBeGreaterThan(0);
      expect(item.pvaf).toBeLessThanOrEqual(1 + 1e-12);
    }
  });

  it("validates the items of 15 attributes, in time that grows as 3^K", () => {
    const attributes = Array.from({ length: 15 }, (_, k) => `a${k}`);
    const qMatrix = Object.fromEntries(attributes.map((attribute, k) => [`item${k + 1}`, [attribute]]));
    const itemIds = Object.keys(qMatrix);
    const matrix = buildResponseMatrix(
      Array.from({ length: 4 }, (_, person) => itemIds.map((_, item) => (person + item) % 2)),
      { itemIds },
    );
    const suggestions = validateQMatrix(matrix, calibrateCdm(matrix, qMatrix, { maxIterations: 2 }));
    expect(suggestions).toHaveLength(15);
  });

  it("rejects an eps outside [0, 1] and options of other types", () => {
    const matrix = buildResponseMatrix([
      [1, 0],
      [0, 1],
      [1, 1],
      [0, 0],
    ]);
    const calibration = calibrateCdm(matrix, { item1: ["a"], item2: ["b"] });
    expect(() => validateQMatrix(matrix, calibration, { eps: 2 })).toThrow(
      "The option eps must be a number from 0 to 1, found 2.",
    );
    expect(() => validateQMatrix(matrix, calibration, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => validateQMatrix(matrix, calibration, JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
  });
});

describe("cdmModelFit", () => {
  const matrix = buildResponseMatrix(fitFixture.responses, { itemIds: fitFixture.items });
  const qMatrix = definedEntries(fitFixture.qMatrix);
  const calibrated = (model: readonly string[]) =>
    calibrateCdm(matrix, qMatrix, {
      model: model.map(cdmModel),
      tolerance: FIT_EM_TOLERANCE,
      maxIterations: 100_000,
    });

  for (const fit of fitFixture.cases) {
    it(`matches GDINA's modelfit for ${fit.name}`, () => {
      const result = cdmModelFit(matrix, calibrated(fit.model));
      expectClose(result.m2, fit.m2, M2_TOLERANCE, "M2");
      expect(result.degreesOfFreedom).toBe(fit.degreesOfFreedom);
      expectClose(result.significance, fit.significance, M2_TOLERANCE, "significance");
      expectClose(result.rmsea, fit.rmsea, M2_TOLERANCE, "RMSEA");
      fit.rmseaInterval.forEach((bound, i) => {
        expectClose(result.rmseaInterval?.[i] ?? null, bound, M2_TOLERANCE, `RMSEA interval ${i}`);
      });
      expectClose(result.srmsr, fit.srmsr, M2_TOLERANCE, "SRMSR");
      const wide = cdmModelFit(matrix, calibrated(fit.model), { level: 0.95 });
      fit.wideInterval.forEach((bound, i) => {
        expectClose(wide.rmseaInterval?.[i] ?? null, bound, M2_TOLERANCE, `0.95 interval ${i}`);
      });
    });
  }

  it("uses the whole complement of a Δ whose rank is below its columns, where GDINA's M2 depends on their order", () => {
    for (const fit of fitFixture.cases) {
      const { gdina } = fit;
      if (gdina === null) continue;
      const result = cdmModelFit(matrix, calibrated(fit.model));
      expect(result.degreesOfFreedom).toBeGreaterThan(gdina.degreesOfFreedom);
      expect(Math.abs(gdina.m2 - gdina.reversed)).toBeGreaterThan(0.1);
    }
  });

  it("has CDM's SRMSR for the DINA", () => {
    const dina = defined(fitFixture.cases.find(({ name }) => name === "DINA"));
    expectClose(cdmModelFit(matrix, calibrated(dina.model)).srmsr, fitFixture.din.srmsr, M2_TOLERANCE, "SRMSR");
  });

  it("rejects a true G-DINA model as often as GDINA on the same sets", () => {
    const { sets, alpha } = fitFixture.rejections;
    let rejected = 0;
    let rejectedByGdina = 0;
    for (const { responses, statistic, degreesOfFreedom } of sets) {
      const data = buildResponseMatrix(
        responses.map((row) => Array.from(row, Number)),
        { itemIds: fitFixture.items },
      );
      const result = cdmModelFit(data, calibrateCdm(data, qMatrix, { tolerance: 1e-6, maxIterations: 5000 }));
      expect(result.degreesOfFreedom).toBe(degreesOfFreedom);
      expectClose(result.m2, statistic, REJECTION_TOLERANCE, "M2 of a set");
      if (defined(result.significance) < alpha) rejected++;
      if (chiSquareUpperTail(statistic, degreesOfFreedom) < alpha) rejectedByGdina++;
    }
    expect(Math.abs(rejected - rejectedByGdina) / sets.length).toBeLessThanOrEqual(0.02);
  });

  it("leaves M2 null without degrees of freedom, and still gives SRMSR", () => {
    const small = { item1: ["a", "b"], item2: ["a"], item3: ["b"] };
    const data = buildResponseMatrix(fitFixture.responses.map((row) => row.slice(0, 3)));
    const result = cdmModelFit(data, calibrateCdm(data, small));
    expect(result.degreesOfFreedom).toBeLessThanOrEqual(0);
    expect(result).toMatchObject({ m2: null, significance: null, rmsea: null, rmseaInterval: null });
    expect(result.srmsr).toBeGreaterThanOrEqual(0);
  });

  it("uses the persons who answered every item", () => {
    const calibration = calibrated(defined(fitFixture.cases[0]).model);
    const missing = buildResponseMatrix(
      [...fitFixture.responses, [null, ...(fitFixture.responses[0]?.slice(1) ?? [])]],
      {
        itemIds: fitFixture.items,
      },
    );
    expect(cdmModelFit(missing, calibration)).toEqual(cdmModelFit(matrix, calibration));
  });

  it("refuses more than 60 items and fewer than two persons who answered every item", () => {
    const wide = Object.fromEntries(Array.from({ length: 61 }, (_, j) => [`item${j + 1}`, ["a"]]));
    const rows = [0, 1, 0, 1].map((start) => Array.from({ length: 61 }, (_, j) => (start + j) % 2));
    const many = buildResponseMatrix(rows);
    const calibration = calibrateCdm(many, wide, { maxIterations: 2 });
    expect(() => cdmModelFit(many, calibration)).toThrow(/at most 60 items, found 61/);
    const few = buildResponseMatrix([
      [1, 0],
      [0, null],
      [null, 1],
    ]);
    const pair = calibrateCdm(few, { item1: ["a"], item2: ["a"] }, { maxIterations: 2 });
    expect(() => cdmModelFit(few, pair)).toThrow(/at least two persons who answered every item, found 1/);
  });

  it("refuses arguments of other types", () => {
    const calibration = calibrated(defined(fitFixture.cases[0]).model);
    expect(() => cdmModelFit(JSON.parse("null"), calibration)).toThrow(DataError);
    expect(() => cdmModelFit(matrix, JSON.parse("null"))).toThrow(DataError);
    expect(() => cdmModelFit(matrix, calibration, JSON.parse("null"))).toThrow(DataError);
    expect(() => cdmModelFit(matrix, calibration, JSON.parse('{ "level": 1 }'))).toThrow(/The option level/);
    expect(() => cdmModelFit(matrix, calibration, JSON.parse('{ "level": "0.9" }'))).toThrow(/The option level/);
    const unknown = {
      ...calibration,
      items: calibration.items.map((item) => ({ ...item, model: JSON.parse('"gdm"') })),
    };
    expect(() => cdmModelFit(matrix, unknown)).toThrow(/The model of item "item1"/);
  });
});

describe("cdmItemFit", () => {
  const matrix = buildResponseMatrix(fitFixture.responses, { itemIds: fitFixture.items });
  const qMatrix = definedEntries(fitFixture.qMatrix);

  for (const fit of fitFixture.cases) {
    it(`matches the exact item statistics and GDINA's itemfit for ${fit.name}`, () => {
      const model = fit.model.map(cdmModel);
      const calibration = calibrateCdm(matrix, qMatrix, { model, tolerance: FIT_EM_TOLERANCE, maxIterations: 100_000 });
      const { exact, resampled } = fit;
      cdmItemFit(matrix, calibration).forEach((item, j) => {
        expect(item.id).toBe(fitFixture.items[j]);
        expectClose(item.proportion.observed, defined(exact.observed[j]), 1e-12, "observed");
        expectClose(item.proportion.expected, defined(exact.expected[j]), ITEM_TOLERANCE, "expected");
        expectClose(item.proportion.z, defined(exact.proportionZ[j]), ITEM_TOLERANCE, "z of the proportion");
        expectClose(item.correlationZ, defined(exact.correlationZ[j]), ITEM_TOLERANCE, "z of the correlation");
        expectClose(item.logOddsZ, defined(exact.logOddsZ[j]), ITEM_TOLERANCE, "z of the log odds");
        expectClose(item.significance, defined(exact.significance[j]), ITEM_TOLERANCE, "significance");
        expectClose(item.proportion.z, defined(resampled.proportionZ[j]), RESAMPLED_TOLERANCE, "GDINA's z.prop");
        expectClose(item.correlationZ, defined(resampled.correlationZ[j]), RESAMPLED_TOLERANCE, "GDINA's max[z.r]");
        expectClose(item.logOddsZ, defined(resampled.logOddsZ[j]), RESAMPLED_TOLERANCE, "GDINA's max[z.logOR]");
      });
    });
  }

  it("has CDM's exact z of the residual correlations for the DINA", () => {
    const calibration = calibrateCdm(matrix, qMatrix, {
      model: "dina",
      tolerance: FIT_EM_TOLERANCE,
      maxIterations: 100_000,
    });
    cdmItemFit(matrix, calibration).forEach((item, j) => {
      expectClose(item.correlationZ, defined(fitFixture.din.correlationZ[j]), ITEM_TOLERANCE, item.id);
    });
  });

  it("counts an empty cell of an observed table as one half, and leaves a constant item's correlations out", () => {
    const rows = fitFixture.responses.map((row) => [1, ...row.slice(1)]);
    const data = buildResponseMatrix(rows, { itemIds: fitFixture.items });
    const [constant, ...others] = cdmItemFit(data, calibrateCdm(data, qMatrix));
    expect(constant?.correlationZ).toBeNull();
    expect(Number.isFinite(defined(constant?.logOddsZ))).toBe(true);
    for (const item of others) expect(Number.isFinite(defined(item.correlationZ))).toBe(true);
  });

  it("gives no pair z to a single item", () => {
    const data = buildResponseMatrix([[1], [0], [1], [1]]);
    const [item] = cdmItemFit(data, calibrateCdm(data, { item1: ["a"] }));
    expect(item).toMatchObject({ correlationZ: null, logOddsZ: null, significance: null });
    expect(item?.proportion.z).not.toBeNull();
  });

  it("refuses arguments of other types", () => {
    const calibration = calibrateCdm(matrix, qMatrix, { maxIterations: 2 });
    expect(() => cdmItemFit(JSON.parse("null"), calibration)).toThrow(DataError);
    expect(() => cdmItemFit(matrix, JSON.parse("[]"))).toThrow(DataError);
  });
});

describe("qMatrixIdentifiability", () => {
  it("matches CDM's din_identifiability on random Q-matrices", () => {
    for (const { qMatrix, singleItems, threeItems, distinctColumns, dinaIdentified } of identifiabilityFixture.cases) {
      const result = qMatrixIdentifiability(definedEntries(qMatrix));
      expect(result).toEqual({ singleItems, threeItems, distinctColumns, dinaIdentified });
    }
  });

  it("compares the whole Q-matrix when no attribute has an item of its own", () => {
    const result = qMatrixIdentifiability({ q1: ["a", "b"], q2: ["a", "c"], q3: ["b", "c"] });
    expect(result.distinctColumns).toBe(true);
    expect(result.dinaIdentified).toBe(false);
    expect(qMatrixIdentifiability({ q1: ["a", "b"], q2: ["a", "b"] }).distinctColumns).toBe(false);
  });

  it("gives an empty remainder equal columns", () => {
    expect(qMatrixIdentifiability({ q1: ["a"], q2: ["b"] }).distinctColumns).toBe(false);
    expect(qMatrixIdentifiability({ q1: ["a"] }).distinctColumns).toBe(true);
  });

  it("takes a Q-matrix of many attributes in time with its entries", () => {
    const qMatrix = Object.fromEntries(Array.from({ length: 20_000 }, (_, j) => [`q${j}`, [`a${j}`, `a${j + 1}`]]));
    expect(qMatrixIdentifiability(qMatrix).distinctColumns).toBe(true);
  });

  it("refuses a Q-matrix that is not one", () => {
    expect(() => qMatrixIdentifiability(JSON.parse("null"))).toThrow(/must be an object/);
    expect(() => qMatrixIdentifiability(JSON.parse("[]"))).toThrow(DataError);
    expect(() => qMatrixIdentifiability({})).toThrow(/at least one item/);
    expect(() => qMatrixIdentifiability(JSON.parse('{ "q1": [] }'))).toThrow(/at least one attribute/);
    expect(() => qMatrixIdentifiability(JSON.parse('{ "q1": ["a", "a"] }'))).toThrow(/twice/);
  });
});

describe("properties", () => {
  it("model and item fit do not depend on the order of persons", () => {
    const qMatrix = definedEntries(fitFixture.qMatrix);
    const rows = fitFixture.responses.slice(0, 300);
    const calibration = calibrateCdm(buildResponseMatrix(rows, { itemIds: fitFixture.items }), qMatrix, {
      model: "dina",
      maxIterations: 200,
    });
    fc.assert(
      fc.property(fc.shuffledSubarray(rows, { minLength: rows.length, maxLength: rows.length }), (shuffled) => {
        const original = buildResponseMatrix(rows, { itemIds: fitFixture.items });
        const reordered = buildResponseMatrix(shuffled, { itemIds: fitFixture.items });
        const close = (actual: unknown, expected: unknown): void => {
          expect(
            JSON.stringify(actual, (_, value) => (typeof value === "number" ? value.toPrecision(10) : value)),
          ).toBe(JSON.stringify(expected, (_, value) => (typeof value === "number" ? value.toPrecision(10) : value)));
        };
        close(cdmModelFit(reordered, calibration), cdmModelFit(original, calibration));
        close(cdmItemFit(reordered, calibration), cdmItemFit(original, calibration));
      }),
      { numRuns: 10 },
    );
  });

  it("the identifiability of a Q-matrix does not depend on the order of items and of attributes", () => {
    const attributes = ["a", "b", "c", "d"];
    const entry = fc.subarray(attributes, { minLength: 1, maxLength: 3 });
    fc.assert(
      fc.property(fc.array(entry, { minLength: 1, maxLength: 10 }), fc.integer({ min: 0, max: 1000 }), (rows, seed) => {
        const random = createRandom(seed);
        const order = rows.map((_, j) => ({ j, key: random.uniform() }));
        order.sort((x, y) => x.key - y.key);
        const qMatrix = Object.fromEntries(rows.map((row, j) => [`q${j}`, row]));
        const shuffled = Object.fromEntries(order.map(({ j }) => [`q${j}`, [...(rows[j] ?? [])].reverse()]));
        expect(qMatrixIdentifiability(shuffled)).toEqual(qMatrixIdentifiability(qMatrix));
      }),
    );
  });

  it("profile proportions and each person's posterior sum to 1, and probabilities stay within GDINA's bounds", () => {
    const qMatrix = { item1: ["a"], item2: ["b"], item3: ["a", "b"], item4: ["a"], item5: ["b"] };
    const rows = fc.array(
      fc.array(fc.option(fc.constantFrom(0, 1), { freq: 8, nil: null }), { minLength: 5, maxLength: 5 }),
      {
        minLength: 3,
        maxLength: 40,
      },
    );
    fc.assert(
      fc.property(rows, fc.constantFrom<CdmModel>(...MODELS), (data, model) => {
        try {
          const matrix = buildResponseMatrix(data);
          const result = calibrateCdm(matrix, qMatrix, { model, maxIterations: 50 });
          expectClose(
            result.profiles.reduce((sum, { proportion }) => sum + proportion, 0),
            1,
            1e-12,
          );
          for (const item of result.items) {
            for (const p of item.probabilities) expect(p >= 1e-4 && p <= 1 - 1e-4).toBe(true);
          }
          for (const person of attributeProfiles(matrix, result, { posterior: true })) {
            expectClose(
              person.posterior.reduce((sum, p) => sum + p, 0),
              1,
              1e-12,
            );
          }
        } catch (error) {
          if (!(error instanceof DataError)) throw error;
        }
      }),
      { numRuns: 40 },
    );
  });

  it("has the marginal log-likelihood of the responses that are not missing, summed over the profiles", () => {
    const qMatrix = { item1: ["a"], item2: ["b"], item3: ["a", "b"], item4: ["a"] };
    const matrix = buildResponseMatrix([
      [1, 0, 1, null],
      [0, null, 0, 0],
      [1, 1, 1, 1],
      [null, null, null, null],
      [0, 1, null, 1],
      [null, 0, 1, 0],
    ]);
    for (const model of MODELS) {
      const { items, attributes, profiles, logLikelihood } = calibrateCdm(matrix, qMatrix, { model, maxIterations: 7 });
      const likelihood = (row: readonly (number | null)[], profile: readonly number[]): number =>
        row.reduce((product: number, response, j) => {
          const item = defined(items[j]);
          const reduced = item.attributes.map((name) => defined(profile[attributes.indexOf(name)]));
          const group = profilesOf(reduced.length).findIndex((candidate) =>
            candidate.every((v, k) => v === reduced[k]),
          );
          const p = defined(item.probabilities[group]);
          return response === null ? product : product * (response === 1 ? p : 1 - p);
        }, 1);
      const expected = matrix.responses
        .filter((row) => row.some((response) => response !== null))
        .reduce(
          (total, row) =>
            total +
            Math.log(profiles.reduce((sum, { profile, proportion }) => sum + proportion * likelihood(row, profile), 0)),
          0,
        );
      expectClose(logLikelihood, expected, 1e-9, model);
    }
  });

  it("returns a point that no feasible change of the coefficients improves, for any counts", () => {
    const group = fc.record({
      persons: fc.oneof(fc.constant(0), fc.double({ min: 0, max: 300, noNaN: true })),
      share: fc.oneof(fc.constantFrom(0, 1), fc.double({ min: 0, max: 1, noNaN: true })),
    });
    fc.assert(
      fc.property(
        fc.constantFrom(...LINKS),
        fc.integer({ min: 1, max: 3 }),
        fc.array(group, { minLength: 8, maxLength: 8 }),
        fc.integer({ min: 1, max: 2 ** 30 }),
        (link, attributeCount, groups, seed) => {
          const used = groups.slice(0, 2 ** attributeCount);
          const persons = used.map((entry) => entry.persons);
          const correct = used.map((entry) => entry.persons * entry.share);
          const maximizer = additiveMaximizer(
            link,
            attributeCount,
            PROBABILITY_BOUNDS,
            startingProbabilities(attributeCount),
          );
          const probabilities = Array.from(maximizer.maximize(Float64Array.from(correct), Float64Array.from(persons)));
          expect(probabilities.every((p) => p >= PROBABILITY_BOUNDS[0] && p <= PROBABILITY_BOUNDS[1])).toBe(true);
          const best = negativeLogLikelihood(correct, persons, probabilities);
          const random = createRandom(seed);
          for (const scale of [1e-6, 1e-4, 1e-2, 0.3]) {
            for (let trial = 0; trial < 25; trial++) {
              const moved = Array.from(maximizer.coefficients, (value) => value + scale * random.normal());
              const changed = predictors(moved, attributeCount).map((eta) => inverse(link, eta));
              if (!changed.every((p) => p >= PROBABILITY_BOUNDS[0] && p <= PROBABILITY_BOUNDS[1])) continue;
              expect(negativeLogLikelihood(correct, persons, changed)).toBeGreaterThanOrEqual(
                best - 1e-9 * (1 + Math.abs(best)),
              );
            }
          }
        },
      ),
      { numRuns: 150 },
    );
  });

  it("classifies persons identically with a calibration of any model after a JSON round trip", () => {
    const qMatrix = { item1: ["a"], item2: ["b"], item3: ["a", "b"], item4: ["a"], item5: ["b"] };
    const rows = fc.array(fc.array(fc.constantFrom(0, 1), { minLength: 5, maxLength: 5 }), {
      minLength: 4,
      maxLength: 30,
    });
    fc.assert(
      fc.property(rows, fc.constantFrom<CdmModel>(...MODELS), (data, model) => {
        const matrix = buildResponseMatrix(data);
        const calibration = calibrateCdm(matrix, qMatrix, { model, maxIterations: 30 });
        const stored: CdmCalibration = JSON.parse(JSON.stringify(calibration));
        expect(attributeProfiles(matrix, stored)).toEqual(attributeProfiles(matrix, calibration));
      }),
      { numRuns: 30 },
    );
  });

  for (const [model, link] of Object.entries(LINK_OF)) {
    it(`recovers the success probabilities of simulated ${model} items`, () => {
      const random = createRandom(20261029);
      // From a success probability of 0.12 mastering none of an item's attributes to 0.88 mastering all of them.
      const intercept = linked(link, 0.12);
      const rise = linked(link, 0.88) - intercept;
      const itemAttributes = [
        ["a"],
        ["b"],
        ["c"],
        ["a", "b"],
        ["b", "c"],
        ["a", "c"],
        ["a"],
        ["b"],
        ["c"],
        ["a", "b", "c"],
      ];
      const items = itemAttributes.map((attributes, j) => ({
        id: `q${j + 1}`,
        attributes,
        probabilities: predictors(
          [intercept, ...attributes.map(() => rise / attributes.length)],
          attributes.length,
        ).map((eta) => inverse(link, eta)),
      }));
      const profiles = Array.from({ length: 4000 }, () => [0, 1, 2].map(() => (random.uniform() < 0.5 ? 1 : 0)));
      const matrix = simulateCdmResponses({ attributes: ["a", "b", "c"], items }, profiles, random);
      const result = calibrateCdm(matrix, Object.fromEntries(items.map((item) => [item.id, item.attributes])), {
        model: cdmModel(model),
      });
      // Five binomial standard errors of a latent group of an eighth of the persons.
      const bound = 5 * Math.sqrt(0.25 / 500);
      result.items.forEach((item, j) => {
        expect(item.model).toBe(model);
        expect(item.coefficients).toHaveLength(item.attributes.length + 1);
        item.probabilities.forEach((p, l) => {
          expect(Math.abs(p - defined(items[j]?.probabilities[l]))).toBeLessThan(bound);
        });
      });
    });
  }

  it("recovers the items of simulated responses", () => {
    const random = createRandom(20261029);
    const model = {
      attributes: ["a", "b", "c"],
      items: [
        ...["a", "b", "c"].map((attribute, j) => ({
          id: `q${j + 1}`,
          attributes: [attribute],
          probabilities: [0.2, 0.85],
        })),
        { id: "q4", attributes: ["a", "b"], probabilities: [0.15, 0.4, 0.45, 0.9] },
        { id: "q5", attributes: ["b", "c"], probabilities: [0.1, 0.5, 0.35, 0.85] },
        { id: "q6", attributes: ["a", "c"], probabilities: [0.2, 0.45, 0.4, 0.8] },
        ...["a", "b", "c"].map((attribute, j) => ({
          id: `q${j + 7}`,
          attributes: [attribute],
          probabilities: [0.25, 0.8],
        })),
      ],
    };
    const profiles = Array.from({ length: 4000 }, () => [0, 1, 2].map(() => (random.uniform() < 0.5 ? 1 : 0)));
    const matrix = simulateCdmResponses(model, profiles, random);
    const qMatrix = Object.fromEntries(model.items.map((item) => [item.id, item.attributes]));
    const result = calibrateCdm(matrix, qMatrix);
    // Five binomial standard errors of a latent group of a quarter of the persons.
    const bound = 5 * Math.sqrt(0.25 / 1000);
    result.items.forEach((item, j) => {
      item.probabilities.forEach((p, l) => {
        expect(Math.abs(p - defined(model.items[j]?.probabilities[l]))).toBeLessThan(bound);
      });
    });
  });
});

describe("pseudoInverse", () => {
  it("is the inverse of a positive definite matrix, derived by hand for a 2 by 2 matrix", () => {
    // [[2, 1], [1, 2]]⁻¹ = [[2, -1], [-1, 2]] / 3
    const inverse = pseudoInverse(Float64Array.from([2, 1, 1, 2]), 2);
    [2 / 3, -1 / 3, -1 / 3, 2 / 3].forEach((value, i) => {
      expectClose(defined(inverse[i]), value, 1e-14, `entry ${i}`);
    });
  });

  it("gives the Moore-Penrose inverse of a matrix of deficient rank", () => {
    // The projection onto (1, 1) / √2 scaled by 4 has the eigenvalues 4 and 0: its pseudo-inverse is a quarter of it.
    const inverse = pseudoInverse(Float64Array.from([2, 2, 2, 2]), 2);
    [0.125, 0.125, 0.125, 0.125].forEach((value, i) => {
      expectClose(defined(inverse[i]), value, 1e-14, `entry ${i}`);
    });
    expect(Array.from(pseudoInverse(new Float64Array(4), 2))).toEqual([0, 0, 0, 0]);
  });

  it("drops the eigenvalues below the square root of the machine epsilon times the largest, and keeps those above", () => {
    const dropped = pseudoInverse(Float64Array.from([1, 0, 0, 1e-9]), 2);
    expect(Array.from(dropped)).toEqual([1, 0, 0, 0]);
    const kept = pseudoInverse(Float64Array.from([1, 0, 0, 1e-6]), 2);
    expectClose(defined(kept[3]), 1e6, 1e-3, "kept");
  });

  it("treats an eigenvalue whose reciprocal overflows as 0", () => {
    expect(Array.from(pseudoInverse(Float64Array.from([5e-324]), 1))).toEqual([0]);
  });

  it("satisfies the four Moore-Penrose conditions for any semidefinite matrix", () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 1, max: 5 }),
        fc.integer({ min: 1, max: 5 }),
        fc.array(
          fc.double({ min: -2, max: 2, noNaN: true }).map((x) => (Math.abs(x) < 0.1 ? 0 : x)),
          { minLength: 25, maxLength: 25 },
        ),
        (size, rank, entries) => {
          // B' B for a rank by size matrix B is semidefinite of rank at most `rank`.
          const factor = entries.slice(0, rank * size);
          const a = new Float64Array(size * size);
          for (let i = 0; i < size; i++) {
            for (let j = 0; j < size; j++) {
              for (let k = 0; k < rank; k++) {
                a[i * size + j] = (a[i * size + j] ?? 0) + (factor[k * size + i] ?? 0) * (factor[k * size + j] ?? 0);
              }
            }
          }
          const g = pseudoInverse(a, size);
          const product = (x: Float64Array, y: Float64Array): Float64Array => {
            const out = new Float64Array(size * size);
            multiply(x, y, size, size, size, out);
            return out;
          };
          // Rounding is relative to the largest entry that a condition involves, which a nearly singular matrix's inverse enlarges.
          const close = (x: Float64Array, y: Float64Array): void => {
            const scale = Math.max(1, ...x.map(Math.abs), ...y.map(Math.abs));
            for (let i = 0; i < x.length; i++)
              expect(Math.abs(defined(x[i]) - defined(y[i]))).toBeLessThan(1e-6 * scale);
          };
          close(product(product(a, g), a), a);
          close(product(product(g, a), g), g);
          close(product(a, g), transpose(product(a, g), size, size));
          close(product(g, a), transpose(product(g, a), size, size));
        },
      ),
    );
  });
});

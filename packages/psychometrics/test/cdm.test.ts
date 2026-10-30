import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  attributeProfiles,
  type CdmCalibration,
  type CdmModel,
  calibrateCdm,
  cdmClassificationAccuracy,
  createRandom,
  DataError,
  nonparametricClassification,
  responseMatrix,
  simulateCdmResponses,
  validateQMatrix,
} from "../src/index.js";
import fixture from "./fixtures/cdm.json" with { type: "json" };
import { defined, definedEntries, expectClose } from "./helpers.js";

// Tolerances against GDINA, explained in docs/validation.md.
/** Success probabilities, proportions and person results: EM stops at a change of 1e-8, within 1e-6 of the maximum. */
const CDM_TOLERANCE = 1e-5;
/** The log-likelihood and the information criteria. */
const LOG_LIKELIHOOD_TOLERANCE = 1e-8;

function cdmModel(name: string): CdmModel {
  if (name !== "gdina" && name !== "dina" && name !== "dino") throw new Error(`Unknown model ${name}.`);
  return name;
}

describe("cdmClassificationAccuracy", () => {
  for (const reference of fixture.cases) {
    const matrix = responseMatrix(reference.responses, { itemIds: reference.itemIds });
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

  it("rejects persons without responses alone, whose accuracy is undefined, and invalid options", () => {
    const qMatrix = { item1: ["a"], item2: ["b"], item3: ["a", "b"] };
    const calibration = calibrateCdm(
      responseMatrix([
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 1],
      ]),
      qMatrix,
    );
    expect(() => cdmClassificationAccuracy(responseMatrix([[null, null, null]]), calibration)).toThrow(
      "Classification accuracy needs at least one person with a response.",
    );
    const matrix = responseMatrix([[1, 0, 1]]);
    expect(() => cdmClassificationAccuracy(matrix, calibration, { classification: JSON.parse('"mle"') })).toThrow(
      'The option classification must be "map" or "eap", found "mle".',
    );
    expect(() => cdmClassificationAccuracy(matrix, calibration, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => cdmClassificationAccuracy(JSON.parse("null"), calibration)).toThrow(
      "A response matrix needs the arrays personIds, itemIds and responses.",
    );
  });
});

describe("calibrateCdm", () => {
  for (const reference of fixture.cases) {
    const matrix = responseMatrix(reference.responses, { itemIds: reference.itemIds });
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

  it("rejects polytomous responses, a Q-matrix that does not fit the items and unknown models", () => {
    const matrix = responseMatrix([
      [1, 0],
      [0, 1],
    ]);
    const qMatrix = { item1: ["a"], item2: ["a", "b"] };
    expect(() => calibrateCdm(responseMatrix([[2, 0]]), qMatrix)).toThrow(
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
    expect(() => calibrateCdm(matrix, qMatrix, { model: JSON.parse('"rrum"') })).toThrow(
      'The option model must be "gdina", "dina" or "dino", found "rrum".',
    );
    expect(() => calibrateCdm(matrix, qMatrix, { model: ["dina"] })).toThrow("Found 1 models for 2 items.");
    expect(() => calibrateCdm(responseMatrix([[null, null]]), qMatrix)).toThrow(
      "Cognitive diagnosis needs at least one response.",
    );
    expect(() => calibrateCdm(matrix, qMatrix, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => calibrateCdm(matrix, qMatrix, { model: JSON.parse("5") })).toThrow(
      "The option model must be a cognitive diagnosis model or an array of one per item, found 5.",
    );
  });
});

describe("nonparametricClassification", () => {
  for (const reference of fixture.cases) {
    const matrix = responseMatrix(reference.responses.slice(0, 30), { itemIds: reference.itemIds });
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
      responseMatrix([
        [1, 0, 0],
        [null, 1, null],
      ]),
      qMatrix,
    );
    expect(result[0]).toEqual({ id: "person1", profile: [1, 0], distance: 0, ties: 1 });
    // Only item 2 counts: profiles 01 and 11 both answer it correctly.
    expect(result[1]).toEqual({ id: "person2", profile: [0, 1], distance: 0, ties: 2 });
  });

  it("classifies over 17 attributes, more profiles than a spread of arguments holds", () => {
    const attributes = Array.from({ length: 17 }, (_, k) => `a${k}`);
    const qMatrix = Object.fromEntries(attributes.map((attribute, k) => [`item${k + 1}`, [attribute]]));
    const pattern = attributes.map((_, k) => k % 2);
    const [person] = nonparametricClassification(responseMatrix([pattern]), qMatrix);
    expect(person).toEqual({ id: "person1", profile: pattern, distance: 0, ties: 1 });
  });

  it("rejects options of other types from JavaScript", () => {
    expect(() => nonparametricClassification(responseMatrix([[1]]), { item1: ["a"] }, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
  });
});

describe("attributeProfiles", () => {
  const reference = defined(fixture.cases[1]);
  const matrix = responseMatrix(reference.responses, { itemIds: reference.itemIds });
  for (const fit of reference.fits) {
    it(`matches GDINA's personparm under the ${fit.model.toUpperCase()} model`, () => {
      const calibration = calibrateCdm(matrix, definedEntries(reference.qmatrix), {
        model: cdmModel(fit.model),
        tolerance: 1e-10,
        maxIterations: 10_000,
      });
      const persons = attributeProfiles(matrix, calibration);
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

  it("gives a person without responses the calibrated proportions", () => {
    const calibration = calibrateCdm(matrix, definedEntries(reference.qmatrix), { maxIterations: 5 });
    const [person] = attributeProfiles(
      responseMatrix([reference.itemIds.map(() => null)], { itemIds: reference.itemIds }),
      calibration,
    );
    expect(defined(person).posterior).toEqual(calibration.profiles.map((profile) => profile.proportion));
    expect(() => attributeProfiles(responseMatrix([[1]]), calibration)).toThrow(
      "The matrix needs the calibration's items, in its order.",
    );
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
    const matrix = responseMatrix(reference.responses, { itemIds: reference.itemIds });
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
    const matrix = responseMatrix([
      [1, 0, null],
      [0, 1, null],
      [1, 1, null],
      [0, 0, null],
    ]);
    const [first, second, third] = validateQMatrix(matrix, calibrateCdm(matrix, qMatrix));
    expect(third).toEqual({ id: "item3", attributes: ["a", "b"], suggested: ["a", "b"], pvaf: null });
    for (const item of [first, second]) expect(Number.isFinite(defined(item).pvaf)).toBe(true);
  });

  it("rejects an eps outside [0, 1] and options of other types", () => {
    const matrix = responseMatrix([
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
  });
});

describe("properties", () => {
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
      fc.property(rows, fc.constantFrom<CdmModel>("gdina", "dina", "dino"), (data, model) => {
        try {
          const matrix = responseMatrix(data);
          const result = calibrateCdm(matrix, qMatrix, { model, maxIterations: 50 });
          expectClose(
            result.profiles.reduce((sum, { proportion }) => sum + proportion, 0),
            1,
            1e-12,
          );
          for (const item of result.items) {
            for (const p of item.probabilities) expect(p >= 1e-4 && p <= 1 - 1e-4).toBe(true);
          }
          for (const person of attributeProfiles(matrix, result)) {
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

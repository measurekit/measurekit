import { expect, test } from "vitest";
import { SLOW_OPTIONS } from "../../../bench/harness.js";
import {
  attributeProfiles,
  type CdmModel,
  calibrateCdm,
  cdmClassificationAccuracy,
  cdmItemFit,
  cdmModelFit,
  compareItemModels,
  nonparametricClassification,
  responseMatrix,
  validateQMatrix,
} from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-cdm.json" with { type: "json" };

const matrix = responseMatrix(fixture.responses, { itemIds: fixture.itemIds });
const { qMatrix } = fixture;

const MODELS = [
  ["cdm.calibrateGdina", "gdina"],
  ["cdm.calibrateDina", "dina"],
  ["cdm.calibrateDino", "dino"],
  ["cdm.calibrateAcdm", "acdm"],
  ["cdm.calibrateLlm", "llm"],
  ["cdm.calibrateRrum", "rrum"],
] as const satisfies readonly (readonly [string, CdmModel])[];

for (const [name, model] of MODELS) {
  test(`calibrates the ${model} model of 1,000 persons, 20 items and 4 attributes`, async (context) => {
    expect(calibrateCdm(matrix, qMatrix, { model }).converged).toBe(true);
    await benchmark(context, name, () => calibrateCdm(matrix, qMatrix, { model }), SLOW_OPTIONS);
  });
}

test("classifies the persons of a calibration, and measures the accuracy of the classification", async (context) => {
  const calibration = calibrateCdm(matrix, qMatrix);
  await benchmark(context, "cdm.attributeProfiles", () => attributeProfiles(matrix, calibration));
  await benchmark(context, "cdm.accuracy", () => cdmClassificationAccuracy(matrix, calibration));
  await benchmark(context, "cdm.validateQMatrix", () => validateQMatrix(matrix, calibration));
});

test("classifies 1,000 persons nonparametrically", async (context) => {
  await benchmark(context, "cdm.nonparametric", () => nonparametricClassification(matrix, qMatrix));
});

test("calibrates the G-DINA model of 1,000 persons, 20 items and 4 attributes with the covariance of its success probabilities", async (context) => {
  await benchmark(
    context,
    "cdm.calibrateGdinaStandardErrors",
    () => calibrateCdm(matrix, qMatrix, { standardErrors: "incomplete" }),
    SLOW_OPTIONS,
  );
});

test("compares the reduced models of every item of a calibration with two or more attributes", async (context) => {
  const calibration = calibrateCdm(matrix, qMatrix, { standardErrors: "incomplete" });
  await benchmark(context, "cdm.compareItemModels", () => compareItemModels(calibration));
});

test("measures the fit of a G-DINA calibration of 1,000 persons, 20 items and 4 attributes", async (context) => {
  const calibration = calibrateCdm(matrix, qMatrix, { model: "gdina" });
  await benchmark(context, "cdm.modelFit", () => cdmModelFit(matrix, calibration));
  await benchmark(context, "cdm.itemFit", () => cdmItemFit(matrix, calibration));
});

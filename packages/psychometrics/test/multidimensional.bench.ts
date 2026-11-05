import { expect, test } from "vitest";
import { SLOW_OPTIONS } from "../../../scripts/bench-harness.js";
import { buildResponseMatrix } from "../src/data/index.js";
import {
  calibrateLongitudinal,
  calibrateMultidimensional,
  multidimensionalEstimates,
  multidimensionalGradedResponse,
  multidimensionalMaximumLikelihoodEstimate,
  multidimensionalPosteriorMeanEstimate,
  multidimensionalThreeParameterLogistic,
  multidimensionalTwoParameterLogistic,
  multidimensionalWeightedLikelihoodEstimate,
  rotateLoadings,
} from "../src/multidimensional/index.js";
import { benchmark } from "./budgets.js";
import { readFixture } from "./fixtures.js";

interface BenchMultidimensionalFixture {
  points: number;
  confirmatory: { itemIds: string[]; responses: number[][]; loadings: Record<string, string[]> };
  graded: { itemIds: string[]; responses: number[][]; loadings: Record<string, string[]> };
  exploratory: { itemIds: string[]; responses: number[][] };
  mhrm: { itemIds: string[]; responses: number[][]; loadings: Record<string, string[]> };
}
const fixture = readFixture<BenchMultidimensionalFixture>("bench-multidimensional.json");

interface BenchMultidimensionalLongitudinalFixture {
  points: number;
  occasions: { personIds: string[]; itemIds: string[]; responses: number[][] }[];
}
const longitudinal = readFixture<BenchMultidimensionalLongitudinalFixture>("bench-multidimensional-longitudinal.json");

interface HeldFixture {
  models: string[];
  itemIds: string[];
  responses: number[][];
  loadings: Record<string, string[]>;
  fixedParameters: Record<string, Record<string, number>>;
}
const held = readFixture<HeldFixture>("multidimensional-fixed-parameters.json");

const { points } = fixture;
const confirmatory = buildResponseMatrix(fixture.confirmatory.responses, { itemIds: fixture.confirmatory.itemIds });
const graded = buildResponseMatrix(fixture.graded.responses, { itemIds: fixture.graded.itemIds });
const exploratory = buildResponseMatrix(fixture.exploratory.responses, { itemIds: fixture.exploratory.itemIds });

const confirmatoryOptions = {
  loadings: fixture.confirmatory.loadings,
  points,
};
const gradedOptions = { loadings: fixture.graded.loadings, points };
const exploratoryOptions = { dimensions: 2, points };

test("calibrates a confirmatory two-dimensional 2PL of 1,000 persons and 24 items", async (context) => {
  expect(
    calibrateMultidimensional(confirmatory, multidimensionalTwoParameterLogistic, confirmatoryOptions).converged,
  ).toBe(true);
  await benchmark(
    context,
    "multidimensional.calibrateConfirmatory",
    () => calibrateMultidimensional(confirmatory, multidimensionalTwoParameterLogistic, confirmatoryOptions),
    SLOW_OPTIONS,
  );
});

test("calibrates a confirmatory two-dimensional 2PL and 3PL of 2,000 persons and 12 items with parameters held", async (context) => {
  const optionsModel = held.models.map((name) =>
    name === "m3pl" ? multidimensionalThreeParameterLogistic : multidimensionalTwoParameterLogistic,
  );
  const options = {
    loadings: held.loadings,
    fixedParameters: held.fixedParameters,
    points,
  };
  const data = buildResponseMatrix(held.responses, { itemIds: held.itemIds });
  expect(calibrateMultidimensional(data, optionsModel, options).converged).toBe(true);
  await benchmark(
    context,
    "multidimensional.calibrateHeld",
    () => calibrateMultidimensional(data, optionsModel, options),
    SLOW_OPTIONS,
  );
});

test("calibrates a confirmatory two-dimensional graded model of 1,000 persons and 20 items", async (context) => {
  expect(calibrateMultidimensional(graded, multidimensionalGradedResponse, gradedOptions).converged).toBe(true);
  await benchmark(
    context,
    "multidimensional.calibrateGraded",
    () => calibrateMultidimensional(graded, multidimensionalGradedResponse, gradedOptions),
    SLOW_OPTIONS,
  );
});

test("calibrates an exploratory two-dimensional 2PL of 1,000 persons and 20 items", async (context) => {
  expect(
    calibrateMultidimensional(exploratory, multidimensionalTwoParameterLogistic, exploratoryOptions).converged,
  ).toBe(true);
  await benchmark(
    context,
    "multidimensional.calibrateExploratory",
    () => calibrateMultidimensional(exploratory, multidimensionalTwoParameterLogistic, exploratoryOptions),
    SLOW_OPTIONS,
  );
});

test("calibrates a confirmatory two-dimensional 2PL of 1,000 persons in two groups and 24 items", async (context) => {
  const groups = confirmatory.personIds.map((_, person) => (person < confirmatory.personIds.length / 2 ? "A" : "B"));
  const options = { ...confirmatoryOptions, groups };
  expect(calibrateMultidimensional(confirmatory, multidimensionalTwoParameterLogistic, options).converged).toBe(true);
  await benchmark(
    context,
    "multidimensional.calibrateGroups",
    () => calibrateMultidimensional(confirmatory, multidimensionalTwoParameterLogistic, options),
    SLOW_OPTIONS,
  );
});

test("calibrates a trait over three occasions of 1,000 persons and 10 items", async (context) => {
  const occasions = longitudinal.occasions.map(({ personIds, itemIds, responses }) =>
    buildResponseMatrix(responses, { personIds, itemIds }),
  );
  const options = { points: longitudinal.points };
  expect(calibrateLongitudinal(occasions, options).converged).toBe(true);
  await benchmark(
    context,
    "multidimensional.calibrateLongitudinal",
    () => calibrateLongitudinal(occasions, options),
    SLOW_OPTIONS,
  );
});

test("calibrates a five-dimensional graded model of 800 persons and 25 items by MH-RM", async (context) => {
  const matrix = buildResponseMatrix(fixture.mhrm.responses, { itemIds: fixture.mhrm.itemIds });
  const options = { loadings: fixture.mhrm.loadings, method: "mhrm" } as const;
  expect(calibrateMultidimensional(matrix, multidimensionalGradedResponse, options).converged).toBe(true);
  await benchmark(
    context,
    "multidimensional.calibrateMhrm",
    () => calibrateMultidimensional(matrix, multidimensionalGradedResponse, options),
    SLOW_OPTIONS,
  );
});

test("estimates the abilities of 1,000 persons on 24 items in two dimensions", async (context) => {
  const { items } = calibrateMultidimensional(confirmatory, multidimensionalTwoParameterLogistic, confirmatoryOptions);
  await benchmark(context, "multidimensional.estimateMode", () => multidimensionalEstimates(confirmatory, items));
  await benchmark(context, "multidimensional.estimateMean", () =>
    multidimensionalEstimates(confirmatory, items, { estimator: multidimensionalPosteriorMeanEstimate }),
  );
  await benchmark(context, "multidimensional.estimateLikelihood", () =>
    multidimensionalEstimates(confirmatory, items, { estimator: multidimensionalMaximumLikelihoodEstimate }),
  );
  await benchmark(context, "multidimensional.estimateWeighted", () =>
    multidimensionalEstimates(confirmatory, items, { estimator: multidimensionalWeightedLikelihoodEstimate }),
  );
});

test("rotates the loadings of a calibration of 20 items and 2 dimensions", async (context) => {
  const calibration = calibrateMultidimensional(exploratory, multidimensionalTwoParameterLogistic, exploratoryOptions);
  await benchmark(context, "multidimensional.rotateOblimin", () => rotateLoadings(calibration, { method: "oblimin" }));
  await benchmark(context, "multidimensional.rotateVarimax", () => rotateLoadings(calibration, { method: "varimax" }));
});

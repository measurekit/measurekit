import { expect, test } from "vitest";
import { SLOW_OPTIONS } from "../../../bench/harness.js";
import {
  calibrateLongitudinal,
  calibrateMultidimensional,
  multidimensionalEstimates,
  multidimensionalGradedResponse,
  multidimensionalPosteriorMeanEstimate,
  multidimensionalTwoParameterLogistic,
  responseMatrix,
  rotateLoadings,
} from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-multidimensional.json" with { type: "json" };
import longitudinal from "./fixtures/bench-multidimensional-longitudinal.json" with { type: "json" };

const { points } = fixture;
const confirmatory = responseMatrix(fixture.confirmatory.responses, { itemIds: fixture.confirmatory.itemIds });
const graded = responseMatrix(fixture.graded.responses, { itemIds: fixture.graded.itemIds });
const exploratory = responseMatrix(fixture.exploratory.responses, { itemIds: fixture.exploratory.itemIds });

const confirmatoryOptions = {
  model: multidimensionalTwoParameterLogistic,
  loadings: fixture.confirmatory.loadings,
  points,
};
const gradedOptions = { model: multidimensionalGradedResponse, loadings: fixture.graded.loadings, points };
const exploratoryOptions = { model: multidimensionalTwoParameterLogistic, dimensions: 2, points };

test("calibrates a confirmatory two-dimensional 2PL of 1,000 persons and 24 items", async (context) => {
  expect(calibrateMultidimensional(confirmatory, confirmatoryOptions).converged).toBe(true);
  await benchmark(
    context,
    "multidimensional.calibrateConfirmatory",
    () => calibrateMultidimensional(confirmatory, confirmatoryOptions),
    SLOW_OPTIONS,
  );
});

test("calibrates a confirmatory two-dimensional graded model of 1,000 persons and 20 items", async (context) => {
  expect(calibrateMultidimensional(graded, gradedOptions).converged).toBe(true);
  await benchmark(
    context,
    "multidimensional.calibrateGraded",
    () => calibrateMultidimensional(graded, gradedOptions),
    SLOW_OPTIONS,
  );
});

test("calibrates an exploratory two-dimensional 2PL of 1,000 persons and 20 items", async (context) => {
  expect(calibrateMultidimensional(exploratory, exploratoryOptions).converged).toBe(true);
  await benchmark(
    context,
    "multidimensional.calibrateExploratory",
    () => calibrateMultidimensional(exploratory, exploratoryOptions),
    SLOW_OPTIONS,
  );
});

test("calibrates a confirmatory two-dimensional 2PL of 1,000 persons in two groups and 24 items", async (context) => {
  const groups = confirmatory.personIds.map((_, person) => (person < confirmatory.personIds.length / 2 ? "A" : "B"));
  const options = { ...confirmatoryOptions, groups };
  expect(calibrateMultidimensional(confirmatory, options).converged).toBe(true);
  await benchmark(
    context,
    "multidimensional.calibrateGroups",
    () => calibrateMultidimensional(confirmatory, options),
    SLOW_OPTIONS,
  );
});

test("calibrates a trait over three occasions of 1,000 persons and 10 items", async (context) => {
  const occasions = longitudinal.occasions.map(({ personIds, itemIds, responses }) =>
    responseMatrix(responses, { personIds, itemIds }),
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
  const matrix = responseMatrix(fixture.mhrm.responses, { itemIds: fixture.mhrm.itemIds });
  const options = { model: multidimensionalGradedResponse, loadings: fixture.mhrm.loadings, method: "mhrm" } as const;
  expect(calibrateMultidimensional(matrix, options).converged).toBe(true);
  await benchmark(
    context,
    "multidimensional.calibrateMhrm",
    () => calibrateMultidimensional(matrix, options),
    SLOW_OPTIONS,
  );
});

test("estimates the abilities of 1,000 persons on 24 items in two dimensions", async (context) => {
  const { items } = calibrateMultidimensional(confirmatory, confirmatoryOptions);
  await benchmark(context, "multidimensional.estimateMode", () => multidimensionalEstimates(confirmatory, items));
  await benchmark(context, "multidimensional.estimateMean", () =>
    multidimensionalEstimates(confirmatory, items, multidimensionalPosteriorMeanEstimate),
  );
});

test("rotates the loadings of a calibration of 20 items and 2 dimensions", async (context) => {
  const calibration = calibrateMultidimensional(exploratory, exploratoryOptions);
  await benchmark(context, "multidimensional.rotateOblimin", () => rotateLoadings(calibration, { method: "oblimin" }));
  await benchmark(context, "multidimensional.rotateVarimax", () => rotateLoadings(calibration, { method: "varimax" }));
});

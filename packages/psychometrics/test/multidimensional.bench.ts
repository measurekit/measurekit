import { expect, test } from "vitest";
import { SLOW_OPTIONS } from "../../../bench/harness.js";
import {
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

test("estimates the abilities of 1,000 persons on 24 items in two dimensions", async (context) => {
  const { items } = calibrateMultidimensional(confirmatory, confirmatoryOptions);
  await benchmark(context, "multidimensional.estimatePosteriorMode", () =>
    multidimensionalEstimates(confirmatory, items),
  );
  await benchmark(context, "multidimensional.estimatePosteriorMean", () =>
    multidimensionalEstimates(confirmatory, items, multidimensionalPosteriorMeanEstimate),
  );
});

test("rotates the loadings of a calibration of 20 items and 2 dimensions", async (context) => {
  const calibration = calibrateMultidimensional(exploratory, exploratoryOptions);
  await benchmark(context, "multidimensional.rotateOblimin", () => rotateLoadings(calibration, { method: "oblimin" }));
  await benchmark(context, "multidimensional.rotateVarimax", () => rotateLoadings(calibration, { method: "varimax" }));
});

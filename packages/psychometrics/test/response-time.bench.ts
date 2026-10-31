import { expect, test } from "vitest";
import { SLOW_OPTIONS } from "../../../bench/harness.js";
import {
  calibrateResponseTimes,
  calibrateSpeedAccuracy,
  responseMatrix,
  speedEstimates,
  twoParameterLogistic,
} from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-response-time.json" with { type: "json" };

const matrix = responseMatrix(fixture.responses, { itemIds: fixture.itemIds, times: fixture.times });

test("calibrates the lognormal model of the times of 1,000 persons on 20 items", async (context) => {
  expect(calibrateResponseTimes(matrix).converged).toBe(true);
  await benchmark(context, "responseTime.calibrateResponseTimes", () => calibrateResponseTimes(matrix));
});

test("calibrates the joint model of speed and accuracy of 1,000 persons on 20 items", async (context) => {
  const options = { model: twoParameterLogistic };
  expect(calibrateSpeedAccuracy(matrix, options).converged).toBe(true);
  await benchmark(
    context,
    "responseTime.calibrateSpeedAccuracy",
    () => calibrateSpeedAccuracy(matrix, options),
    SLOW_OPTIONS,
  );
});

test("estimates the speed of 1,000 persons from the times of 20 items", async (context) => {
  const { items } = calibrateResponseTimes(matrix);
  await benchmark(context, "responseTime.speedEstimates", () => speedEstimates(matrix, items));
});

import { expect, test } from "vitest";
import { SLOW_OPTIONS } from "../../../scripts/bench-harness.js";
import {
  buildResponseMatrix,
  calibrateResponseTimes,
  calibrateSpeedAccuracy,
  speedEstimates,
  twoParameterLogistic,
} from "../src/index.js";
import { benchmark } from "./budgets.js";
import { readFixture } from "./fixtures.js";

interface BenchResponseTimeFixture {
  itemIds: string[];
  responses: number[][];
  times: number[][];
}
const fixture = readFixture<BenchResponseTimeFixture>("bench-response-time.json");

const matrix = buildResponseMatrix(fixture.responses, { itemIds: fixture.itemIds, times: fixture.times });

test("calibrates the lognormal model of the times of 1,000 persons on 20 items", async (context) => {
  expect(calibrateResponseTimes(matrix).converged).toBe(true);
  await benchmark(context, "responseTime.calibrateResponseTimes", () => calibrateResponseTimes(matrix));
});

test("calibrates the joint model of speed and accuracy of 1,000 persons on 20 items", async (context) => {
  expect(calibrateSpeedAccuracy(matrix, twoParameterLogistic).converged).toBe(true);
  await benchmark(
    context,
    "responseTime.calibrateSpeedAccuracy",
    () => calibrateSpeedAccuracy(matrix, twoParameterLogistic),
    SLOW_OPTIONS,
  );
});

test("estimates the speed of 1,000 persons from the times of 20 items", async (context) => {
  const { items } = calibrateResponseTimes(matrix);
  await benchmark(context, "responseTime.speedEstimates", () => speedEstimates(matrix, items));
});

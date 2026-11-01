import { createRandom } from "@measurekit/core";
import { expect, test } from "vitest";
import { SLOW_OPTIONS } from "../../../bench/harness.js";
import {
  adaptiveTest,
  attributeProfiles,
  calibrateCdm,
  diagnosticAdaptiveTest,
  maximumInformation,
  multidimensionalAdaptiveTest,
  multidimensionalTwoParameterLogistic,
  multistageCuts,
  posteriorMeanEstimate,
  responseMatrix,
  simulateAdaptiveTest,
  simulateDiagnosticTest,
  simulateMultidimensionalAdaptiveTest,
  type TestItem,
  testLength,
  threeParameterLogistic,
} from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-cat.json" with { type: "json" };
import cdm from "./fixtures/bench-cdm.json" with { type: "json" };
import { defined } from "./helpers.js";

const SEED = 20261029;

test("simulates adaptive tests of 30 items for 200 persons on a pool of 500 3PL items", async (context) => {
  const pool: TestItem[] = fixture.pool.a.map((a, i) => ({
    id: `item${i + 1}`,
    model: threeParameterLogistic,
    parameters: { a, b: defined(fixture.pool.b[i]), c: defined(fixture.pool.c[i]), d: 1 },
  }));
  const test = adaptiveTest({
    pool,
    selection: maximumInformation,
    estimator: posteriorMeanEstimate,
    stopping: testLength(fixture.length),
  });
  expect(simulateAdaptiveTest(test, fixture.thetas, createRandom(SEED)).persons).toHaveLength(fixture.thetas.length);
  await benchmark(
    context,
    "cat.simulateAdaptive",
    () => simulateAdaptiveTest(test, fixture.thetas, createRandom(SEED)),
    SLOW_OPTIONS,
  );
});

test("simulates adaptive tests of 30 items that select testlets, for 200 persons on a pool of 500 3PL items", async (context) => {
  const pool: TestItem[] = fixture.pool.a.map((a, i) => ({
    id: `item${i + 1}`,
    model: threeParameterLogistic,
    parameters: { a, b: defined(fixture.pool.b[i]), c: defined(fixture.pool.c[i]), d: 1 },
    ...(fixture.testlets[i] !== null && { testlet: fixture.testlets[i] }),
  }));
  const test = adaptiveTest({
    pool,
    selection: maximumInformation,
    estimator: posteriorMeanEstimate,
    stopping: testLength(fixture.length),
  });
  expect(simulateAdaptiveTest(test, fixture.thetas, createRandom(SEED)).persons).toHaveLength(fixture.thetas.length);
  await benchmark(
    context,
    "cat.simulateTestlets",
    () => simulateAdaptiveTest(test, fixture.thetas, createRandom(SEED)),
    SLOW_OPTIONS,
  );
});

test("derives the routing cuts of a panel of 1-3-3-3 modules of 20 items from a pool of 500 3PL items", async (context) => {
  const pool: TestItem[] = fixture.pool.a.map((a, i) => ({
    id: `item${i + 1}`,
    model: threeParameterLogistic,
    parameters: { a, b: defined(fixture.pool.b[i]), c: defined(fixture.pool.c[i]), d: 1 },
  }));
  const stages = fixture.multistage.stages.map((modules, stage) =>
    modules.map((items, module) => ({
      id: `stage${stage + 1}module${module + 1}`,
      items: items.map((index) => defined(pool[index])),
    })),
  );
  expect(multistageCuts(stages)).toHaveLength(7);
  await benchmark(context, "cat.multistageCuts", () => multistageCuts(stages));
});

test("simulates diagnostic adaptive tests of 10 items for 200 persons on a pool of 20 items", async (context) => {
  const matrix = responseMatrix(cdm.responses, { itemIds: cdm.itemIds });
  const calibration = calibrateCdm(matrix, cdm.qMatrix);
  const profiles = attributeProfiles(matrix, calibration)
    .slice(0, fixture.diagnostic.persons)
    .map(({ map }) => [...map]);
  const test = diagnosticAdaptiveTest({
    pool: calibration.items,
    attributes: calibration.attributes,
    prior: calibration.profiles,
    stopping: testLength(fixture.diagnostic.length),
  });
  expect(simulateDiagnosticTest(test, profiles, createRandom(SEED)).persons).toHaveLength(profiles.length);
  await benchmark(context, "cat.simulateDiagnostic", () => simulateDiagnosticTest(test, profiles, createRandom(SEED)));
});

test("simulates multidimensional adaptive tests of 20 items for 100 persons on a pool of 200 items", async (context) => {
  const { slopes, intercepts, thetas, length } = fixture.multidimensional;
  const pool = slopes.map((a, i) => ({
    id: `item${i + 1}`,
    model: multidimensionalTwoParameterLogistic,
    parameters: { a, intercept: defined(intercepts[i]), c: 0, d: 1 },
  }));
  const test = multidimensionalAdaptiveTest({ pool, stopping: testLength(length) });
  expect(simulateMultidimensionalAdaptiveTest(test, thetas, createRandom(SEED)).persons).toHaveLength(thetas.length);
  await benchmark(
    context,
    "cat.simulateMultidimensional",
    () => simulateMultidimensionalAdaptiveTest(test, thetas, createRandom(SEED)),
    SLOW_OPTIONS,
  );
});

import { createRandom } from "@measurekit/core";
import { expect, test } from "vitest";
import { SLOW_OPTIONS } from "../../../bench/harness.js";
import {
  adaptiveTest,
  assembleForms,
  attributeProfiles,
  calibrateCdm,
  contentBounds,
  diagnosticAdaptiveTest,
  maximumInformation,
  maximumPriorityIndex,
  multidimensionalAdaptiveTest,
  multidimensionalTwoParameterLogistic,
  multistageCuts,
  type PoolItem,
  posteriorMeanEstimate,
  responseMatrix,
  simulateAdaptiveTest,
  simulateDiagnosticTest,
  simulateMultidimensionalAdaptiveTest,
  type TestItem,
  testLength,
  threeParameterLogistic,
} from "../src/index.js";
import { defined } from "./assertions.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-cat.json" with { type: "json" };
import boundsFixture from "./fixtures/bench-cat-bounds.json" with { type: "json" };
import formsFixture from "./fixtures/bench-cat-forms.json" with { type: "json" };
import cdm from "./fixtures/bench-cdm.json" with { type: "json" };

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

test("selects an item under content bounds on a pool of 1,000 3PL items in 12 content areas", async (context) => {
  const { pool: parameters, areas, constraints, length, given, theta, standardError } = boundsFixture;
  const pool: PoolItem[] = parameters.a.map((a, i) => ({
    id: `item${i + 1}`,
    model: threeParameterLogistic,
    parameters: { a, b: defined(parameters.b[i]), c: defined(parameters.c[i]), d: 1 },
    contentAreas: defined(areas[i]),
  }));
  const rule = contentBounds(maximumPriorityIndex(maximumInformation, constraints, length), constraints, length);
  const eligible = new Uint8Array(pool.length).fill(1);
  for (const index of given) eligible[index] = 0;
  const selection = {
    pool,
    administered: given,
    responses: boundsFixture.responses,
    estimate: { theta, standardError },
    eligible,
    random: createRandom(SEED),
  };
  const scores = new Float64Array(pool.length);
  rule.scores(selection, scores);
  expect(scores.filter((score) => score > Number.NEGATIVE_INFINITY).length).toBeGreaterThan(0);
  await benchmark(context, "cat.contentBoundsStep", () => rule.scores(selection, scores));
});

test("assembles 4 forms of 40 items from a pool of 1,000 3PL items in 5 content areas, with 30 pairs of enemies", async (context) => {
  const { pool: parameters, areas, constraints, target, enemies, forms, length } = formsFixture;
  const pool: PoolItem[] = parameters.a.map((a, i) => ({
    id: `item${i + 1}`,
    model: threeParameterLogistic,
    parameters: { a, b: defined(parameters.b[i]), c: defined(parameters.c[i]), d: 1 },
    contentAreas: [defined(areas[i])],
  }));
  const options = {
    pool,
    length,
    content: constraints,
    target,
    forms,
    enemies: enemies.map((pair) => pair.map((index) => `item${index + 1}`)),
  };
  expect(assembleForms(options)).toHaveLength(forms);
  await benchmark(context, "cat.assembleForms", () => assembleForms(options));
});

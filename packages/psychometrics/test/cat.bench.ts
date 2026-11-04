import { createRandom } from "@measurekit/core";
import { expect, test } from "vitest";
import { SLOW_OPTIONS } from "../../../scripts/bench-harness.js";
import {
  anyOf,
  assembleForms,
  attributeProfiles,
  buildResponseMatrix,
  calibrateCdm,
  classificationThreshold,
  closestDifficulty,
  contentBounds,
  createAdaptiveTest,
  createDiagnosticAdaptiveTest,
  createMultidimensionalAdaptiveTest,
  discriminationIndex,
  expectedInformation,
  expectedPosteriorVariance,
  type ItemSelectionRule,
  kullbackLeibler,
  likelihoodWeightedInformation,
  maximumInformation,
  maximumPriorityIndex,
  multidimensionalTwoParameterLogistic,
  multistageCuts,
  type PoolItem,
  posteriorDiscriminationIndex,
  posteriorKullbackLeibler,
  posteriorMeanEstimate,
  posteriorWeightedInformation,
  progressive,
  proportional,
  type StoppingRule,
  sequentialProbabilityRatio,
  simulateAdaptiveTest,
  simulateDiagnosticTest,
  simulateMultidimensionalAdaptiveTest,
  type TestItem,
  testLength,
  threeParameterLogistic,
} from "../src/index.js";
import { defined } from "./assertions.js";
import { benchmark } from "./budgets.js";
import { type BenchCdmFixture, readFixture } from "./fixtures.js";

interface BenchCatFixture {
  length: number;
  pool: { a: number[]; b: number[]; c: number[] };
  thetas: number[];
  testlets: (string | null)[];
  multidimensional: { length: number; slopes: number[][]; intercepts: number[]; thetas: number[][] };
  diagnostic: { length: number; persons: number };
  multistage: { stages: number[][][] };
}
const fixture = readFixture<BenchCatFixture>("bench-cat.json");

interface BenchCatBoundsFixture {
  pool: { a: number[]; b: number[]; c: number[] };
  areas: string[][];
  constraints: { area: string; minimum: number; maximum: number }[];
  length: number;
  given: number[];
  responses: number[];
  theta: number;
  standardError: number;
}
const boundsFixture = readFixture<BenchCatBoundsFixture>("bench-cat-bounds.json");

interface BenchCatFormsFixture {
  pool: { a: number[]; b: number[]; c: number[] };
  areas: string[];
  constraints: { area: string; minimum: number; maximum: number }[];
  target: { theta: number; information: number }[];
  enemies: number[][];
  forms: number;
  length: number;
}
const formsFixture = readFixture<BenchCatFormsFixture>("bench-cat-forms.json");

interface BenchCatRulesFixture {
  administered: number[];
  responses: number[];
  theta: number;
  standardError: number;
}
const rulesFixture = readFixture<BenchCatRulesFixture>("bench-cat-rules.json");

const cdm = readFixture<BenchCdmFixture>("bench-cdm.json");

const SEED = 20261029;

/** The 3PL pool of 500 items of the simulations. */
const pool: TestItem[] = fixture.pool.a.map((a, i) => ({
  id: `item${i + 1}`,
  model: threeParameterLogistic,
  parameters: { a, b: defined(fixture.pool.b[i]), c: defined(fixture.pool.c[i]), d: 1 },
}));

test("simulates adaptive tests of 30 items for 200 persons on a pool of 500 3PL items", async (context) => {
  const test = createAdaptiveTest({
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
  const test = createAdaptiveTest({
    pool: pool.map((item, i) => ({ ...item, ...(fixture.testlets[i] !== null && { testlet: fixture.testlets[i] }) })),
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
  const matrix = buildResponseMatrix(cdm.responses, { itemIds: cdm.itemIds });
  const calibration = calibrateCdm(matrix, cdm.qMatrix);
  const profiles = attributeProfiles(matrix, calibration)
    .slice(0, fixture.diagnostic.persons)
    .map(({ map }) => [...map]);
  const test = createDiagnosticAdaptiveTest({
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
  const test = createMultidimensionalAdaptiveTest({ pool, stopping: testLength(length) });
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

test("selects one item of a test of 30 after 15, by each rule, from a pool of 500 3PL items", async (context) => {
  const { administered, responses, theta, standardError } = rulesFixture;
  const eligible = new Uint8Array(pool.length).fill(1);
  for (const index of administered) eligible[index] = 0;
  const selection = { pool, administered, responses, estimate: { theta, standardError }, eligible };
  const scores = new Float64Array(pool.length);
  const rules: readonly (readonly [Parameters<typeof benchmark>[1], ItemSelectionRule])[] = [
    ["cat.likelihoodWeightedInformation", likelihoodWeightedInformation],
    ["cat.posteriorWeightedInformation", posteriorWeightedInformation()],
    ["cat.expectedInformation", expectedInformation()],
    ["cat.expectedPosteriorVariance", expectedPosteriorVariance()],
    ["cat.kullbackLeibler", kullbackLeibler],
    ["cat.posteriorKullbackLeibler", posteriorKullbackLeibler()],
    ["cat.discriminationIndex", discriminationIndex],
    ["cat.posteriorDiscriminationIndex", posteriorDiscriminationIndex()],
    ["cat.closestDifficulty", closestDifficulty],
    ["cat.progressive", progressive({ length: fixture.length })],
    ["cat.proportional", proportional({ length: fixture.length })],
  ];
  for (const [name, rule] of rules) {
    const random = createRandom(SEED);
    rule.scores({ ...selection, random }, scores);
    expect(scores.filter((score, index) => eligible[index] === 1 && Number.isFinite(score)).length).toBe(
      pool.length - administered.length,
    );
    await benchmark(context, name, () => rule.scores({ ...selection, random }, scores));
  }
});

test("simulates adaptive tests of at most 30 items for 200 persons that stop on a classification at a cut", async (context) => {
  const classifying = (stopping: StoppingRule) =>
    createAdaptiveTest({
      pool,
      selection: maximumInformation,
      estimator: posteriorMeanEstimate,
      stopping: anyOf(stopping, testLength(fixture.length)),
    });
  const interval = classifying(classificationThreshold([0]));
  const sprt = classifying(sequentialProbabilityRatio([{ lower: -0.3, upper: 0.3 }]));
  expect(simulateAdaptiveTest(sprt, fixture.thetas, createRandom(SEED)).persons).toHaveLength(fixture.thetas.length);
  await benchmark(
    context,
    "cat.simulateClassification",
    () => simulateAdaptiveTest(interval, fixture.thetas, createRandom(SEED)),
    SLOW_OPTIONS,
  );
  await benchmark(
    context,
    "cat.simulateSprt",
    () => simulateAdaptiveTest(sprt, fixture.thetas, createRandom(SEED)),
    SLOW_OPTIONS,
  );
});

import { shuffle } from "@measurekit/core";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { informationPeak } from "../src/cat/a-stratified.js";
import { bindPool, type DiagnosticTestItem } from "../src/cat/diagnostic-items.js";
import { profileSpace } from "../src/cat/profile-space.js";
import { highestScore } from "../src/cat/selection.js";
import { poolTestlets, unitSelection } from "../src/cat/testlets.js";
import { profileEstimate } from "../src/cdm/persons.js";
import {
  type AbilityEstimate,
  type AbilityEstimator,
  type AdaptiveTest,
  type AdaptiveTestContext,
  type AdaptiveTestState,
  type AssembledForm,
  adaptiveTest,
  allOf,
  anyOf,
  aStratified,
  assembleForms,
  type ContentConstraint,
  calibrateExposure,
  contentBounds,
  contentMinimumsMet,
  createRandom,
  DataError,
  type DiagnosticAdaptiveTest,
  type DiagnosticAdaptiveTestState,
  type DiagnosticPoolItem,
  diagnosticAdaptiveTest,
  diagnosticKullbackLeibler,
  type FormAssemblyOptions,
  gdinaDiscriminationIndex,
  generalizedPartialCredit,
  gradedResponse,
  type ItemSelectionRule,
  informationThreshold,
  itemInformation,
  kullbackLeibler,
  type MultidimensionalAdaptiveTest,
  type MultidimensionalAdaptiveTestState,
  type MultidimensionalEstimate,
  type MultidimensionalPoolItem,
  type MultistageCutsOptions,
  type MultistageModule,
  type MultistageRouting,
  type MultistageTest,
  type MultivariateNormalPrior,
  masteryThreshold,
  maximumDeterminant,
  maximumInformation,
  maximumLikelihoodEstimate,
  maximumPriorityIndex,
  modifiedPosteriorWeightedKullbackLeibler,
  multidimensionalAdaptiveTest,
  multidimensionalInformation,
  multidimensionalPosteriorMeanEstimate,
  multidimensionalPosteriorModeEstimate,
  multidimensionalThreeParameterLogistic,
  multidimensionalTwoParameterLogistic,
  multistageCuts,
  multistageTest,
  type PoolItem,
  type ProfileEstimate,
  posteriorMeanEstimate,
  posteriorModeEstimate,
  posteriorThreshold,
  posteriorWeightedKullbackLeibler,
  type Random,
  randomesque,
  type SelectionContext,
  type StoppingRule,
  shannonEntropy,
  simulateAdaptiveTest,
  simulateCdmResponses,
  simulateDiagnosticTest,
  simulateItems,
  simulateMultidimensionalAdaptiveTest,
  simulateMultistageTest,
  standardErrorThreshold,
  type TestItem,
  testInformation,
  testLength,
  threeParameterLogistic,
  twoParameterLogistic,
  untilMixed,
  type WithinTestlet,
  weightedLikelihoodEstimate,
} from "../src/index.js";
import cat2pl from "./fixtures/cat-2pl.json" with { type: "json" };
import cat3pl from "./fixtures/cat-3pl.json" with { type: "json" };
import catBounds from "./fixtures/cat-bounds.json" with { type: "json" };
import catCdm from "./fixtures/cat-cdm.json" with { type: "json" };
import catContent from "./fixtures/cat-content.json" with { type: "json" };
import catGpcm from "./fixtures/cat-gpcm.json" with { type: "json" };
import catMirt from "./fixtures/cat-mirt.json" with { type: "json" };
import cutsFixture from "./fixtures/cat-multistage-cuts.json" with { type: "json" };
import catTestlets from "./fixtures/cat-testlets.json" with { type: "json" };
import catForms from "./fixtures/form-assembly.json" with { type: "json" };
import { arbitraryItem, defined, expectClose, item2pl, MODELS, mean } from "./helpers.js";

// Tolerances against catR, explained in docs/validation.md.
/** ML, MAP and WLE theta and standard error: catR finds roots with uniroot's default tolerance, about 1.2e-4. */
const CATR_MODAL_TOLERANCE = 1e-4;
/**
 * EAP theta and standard error: catR's trapezoid rule halves the end weights of the 61 points of
 * `posteriorMeanEstimate`.
 */
const CATR_EAP_TOLERANCE = 1e-6;
/** Information and Kullback-Leibler scores, relative to max(1, |score|). */
const SCORE_TOLERANCE = 1e-12;
/** EAP theta and standard error against the written-out testlet replay, which sums over the same 61 points. */
const REPLAY_TOLERANCE = 1e-12;
/** The theta of an item's largest information against catR's closed form in thOpt. */
const PEAK_TOLERANCE = 1e-10;
/** Routing cuts against the roots of base R's uniroot at a tolerance of 1e-13. */
const CUTS_TOLERANCE = 1e-6;
/**
 * The cuts of a population of nodes 0.02 apart against the quantiles of its density: a linear distribution function in
 * each cell errs by an amount that falls with the square of the spacing, 6e-6 here.
 */
const DISCRETIZATION_TOLERANCE = 1e-5;

/** catR's estimate: its method, `null` before the first response, when theta is the prior mean 0. */
interface CatEstimate {
  readonly method: string | null;
  readonly theta: number;
  readonly standardError: number | null;
}

/** A step of a catR run: the estimate from the responses so far, catR's next item and the response to it. */
interface CatStep extends CatEstimate {
  readonly nextItem: string;
  readonly response: number;
  readonly scores?: readonly (number | null)[];
  readonly administeredProportions?: readonly number[];
}

/** An adaptive test run step by step in catR. */
interface CatRun {
  readonly name: string;
  readonly selection: string;
  readonly estimators: readonly string[];
  readonly steps: readonly CatStep[];
  readonly final: CatEstimate;
}

const LENGTH = 15;
/** The seed of the simulations that compare the content bounds with the shadow test. */
const SEED_BOUNDS = 20261029;
/** The estimate at which the rules of content bounds are scored directly. */
const BOUNDS_ESTIMATE: AbilityEstimate = { theta: 0, standardError: 1 };
// By catR's names.
const ESTIMATORS = new Map<string, AbilityEstimator>([
  ["BM", posteriorModeEstimate],
  ["EAP", posteriorMeanEstimate],
  ["ML", maximumLikelihoodEstimate],
  ["WL", weightedLikelihoodEstimate],
]);
const SELECTIONS = new Map<string, ItemSelectionRule>([
  ["MFI", maximumInformation],
  ["KL", kullbackLeibler],
  // A single stratum: the item whose information peaks closest to theta.
  ["thOpt", aStratified([LENGTH])],
]);
const STOPPING_RULES = new Map<string, (threshold: number) => StoppingRule>([
  ["length", testLength],
  ["precision", standardErrorThreshold],
  ["minInfo", informationThreshold],
]);

const POOL_3PL: readonly PoolItem[] = cat3pl.items.map(({ id, ...parameters }) => ({
  id,
  model: threeParameterLogistic,
  parameters,
}));
const POOL_GPCM: readonly PoolItem[] = catGpcm.items.map(({ id, ...parameters }) => ({
  id,
  model: generalizedPartialCredit,
  parameters,
}));
const POOL_CONTENT: readonly PoolItem[] = catContent.items.map(({ id, contentArea, ...parameters }) => ({
  id,
  contentAreas: [contentArea],
  model: threeParameterLogistic,
  parameters,
}));
const FIXTURES: readonly { readonly source: string; readonly pool: readonly PoolItem[]; readonly runs: CatRun[] }[] = [
  { source: cat3pl.source, pool: POOL_3PL, runs: cat3pl.runs },
  { source: catGpcm.source, pool: POOL_GPCM, runs: catGpcm.runs },
];

function runEstimator([initial = "", mixed = ""]: readonly string[]): AbilityEstimator {
  const first = defined(ESTIMATORS.get(initial));
  return initial === mixed ? first : untilMixed(first, defined(ESTIMATORS.get(mixed)));
}

/** Expects an estimate within the tolerance for catR's method; before the first response, theta 0. */
function expectEstimate(actual: AbilityEstimate, expected: CatEstimate, label: string): void {
  const tolerance = expected.method === "EAP" || expected.method === null ? CATR_EAP_TOLERANCE : CATR_MODAL_TOLERANCE;
  expectClose(actual.theta, expected.theta, tolerance, `${label} theta`);
  if (expected.standardError !== null) {
    expectClose(actual.standardError, expected.standardError, tolerance, `${label} standard error`);
  }
}

function poolIndex(pool: readonly PoolItem[], id: string): number {
  const index = pool.findIndex((item) => item.id === id);
  expect(index, id).toBeGreaterThanOrEqual(0);
  return index;
}

/** The selection context after the first `count` steps of a run, at `estimate`. */
function contextAfter(
  pool: readonly PoolItem[],
  steps: readonly CatStep[],
  count: number,
  estimate: AbilityEstimate,
  random: Random = createRandom(0),
): SelectionContext {
  const administered = steps.slice(0, count).map((step) => poolIndex(pool, step.nextItem));
  const eligible = new Uint8Array(pool.length).fill(1);
  for (const index of administered) eligible[index] = 0;
  const responses = steps.slice(0, count).map((step) => step.response);
  return { pool, administered, responses, estimate, eligible, random };
}

/** Expects the scores a rule gives the eligible items at each step of a run, at catR's estimate, to be catR's. */
function expectScores(pool: readonly PoolItem[], run: CatRun, rule: ItemSelectionRule): void {
  run.steps.forEach((step, k) => {
    const out = new Float64Array(pool.length);
    rule.scores(contextAfter(pool, run.steps, k, { theta: step.theta, standardError: 1 }), out);
    defined(step.scores).forEach((expected, i) => {
      if (expected === null) return;
      expectClose(out[i] ?? 0, expected, SCORE_TOLERANCE * Math.max(1, Math.abs(expected)), `step ${k}, item ${i + 1}`);
    });
  });
}

/** Runs an adaptive test to its end with the response `respond` gives each item, and returns the final state. */
function administer(test: AdaptiveTest, respond: (id: string) => number, random = createRandom(1)): AdaptiveTestState {
  let state = test.start(random);
  while (state.next !== null) state = test.answer(state, respond(state.next), random);
  return state;
}

/** A context of an adaptive test after `count` items of a pool, at an estimate. */
function contextOf(pool: readonly PoolItem[], count: number, estimate: AbilityEstimate): AdaptiveTestContext {
  const administered = Array.from({ length: count }, (_, i) => i);
  const eligible = Uint8Array.from(pool, (_, i) => (i < count ? 0 : 1));
  return { pool, administered, responses: administered.map(() => 1), estimate, eligible };
}

describe("aStratified", () => {
  const pool2pl: readonly PoolItem[] = cat2pl.items.map(({ id, ...parameters }) => ({
    id,
    model: twoParameterLogistic,
    parameters,
  }));
  const strataCount = Math.max(...cat2pl.strata);

  it("splits 2PL items into catR's aStratified strata by a", () => {
    const rule = aStratified(Array.from({ length: strataCount }, () => 1));
    for (let stratum = 1; stratum <= strataCount; stratum++) {
      // After stratum - 1 items, the rule scores the items of the scheduled stratum only.
      const context = contextOf(pool2pl, 0, { theta: 0, standardError: 1 });
      const out = new Float64Array(pool2pl.length);
      rule.scores(
        { ...context, administered: Array.from({ length: stratum - 1 }, () => 0), random: createRandom(1) },
        out,
      );
      const scored = Array.from(out, (score) => score > Number.NEGATIVE_INFINITY);
      expect(scored).toEqual(cat2pl.strata.map((s) => s === stratum));
    }
  });

  it("selects Chang and Ying's items: the scheduled stratum's difficulty closest to the estimate, catR's bOpt", () => {
    const run: CatRun = cat2pl.run;
    const lengths = Array.from({ length: strataCount }, () => cat2pl.perStratum);
    const test = adaptiveTest({
      pool: pool2pl,
      selection: aStratified(lengths),
      estimator: posteriorModeEstimate,
      stopping: testLength(LENGTH),
    });
    const random = createRandom(1);
    let state = test.start(random);
    run.steps.forEach((step, k) => {
      expect(state.next, `step ${k}`).toBe(step.nextItem);
      expectEstimate(state.estimate, step, `step ${k}`);
      state = test.answer(state, step.response, random);
    });
    expectEstimate(state.estimate, run.final, "final");
  });

  it("finds the theta of each 3PL item's largest information, as catR's thOpt", () => {
    cat3pl.items.forEach(({ id, ...parameters }, i) => {
      const peak = informationPeak({ model: threeParameterLogistic, parameters });
      expectClose(peak, defined(cat3pl.peaks[i]), PEAK_TOLERANCE, id);
    });
  });

  it("finds the higher of two maxima of a graded item's information a few hundredths of theta apart", () => {
    // The information peaks at 0.450 and at 0.512, the first higher by 1e-9 of itself, and dips between them by 1e-6:
    // a grid of 0.25 steps over the first and finds the second. The second item mirrors the first with a positive slope.
    const negative = { a: -2.844320553297738, b: [1.4756847516587377, 0.7572582335927899, 0.11278932628252365, 0] };
    const positive = { a: 2.844320553297738, b: [-1.4756847516587377, -0.7572582335927899, -0.11278932628252365, 0] };
    for (const parameters of [negative, positive]) {
      const information = (theta: number): number => gradedResponse.information(parameters, theta);
      let largest = 0;
      for (let step = 0; step <= 12_000; step++) largest = Math.max(largest, information(-6 + step / 1000));
      const peak = informationPeak({ model: gradedResponse, parameters });
      expect(information(peak)).toBeGreaterThanOrEqual(largest * (1 - 1e-12));
      expect(Math.abs(Math.abs(peak) - 0.45)).toBeLessThan(0.005);
    }
  });

  it("takes the first items from the stratum of the lowest slopes of 2PL items, with b closest to theta", () => {
    // Strata by a: {a1, a2}, {a3, a4}, {a5, a6}.
    const pool = [
      item2pl("a5", 1.8, 0.2),
      item2pl("a1", 0.5, -1),
      item2pl("a3", 1, 0.1),
      item2pl("a2", 0.6, 0.4),
      item2pl("a6", 2.2, -0.5),
      item2pl("a4", 1.2, -2),
    ];
    const test = adaptiveTest({ pool, selection: aStratified([2, 2, 2]), stopping: testLength(6) });
    // At θ = 0, b = 0.4 of a2 is closer than b = -1 of a1.
    expect(test.start(createRandom(1)).next).toBe("a2");
    const { items } = administer(test, () => 1);
    expect(items.slice(0, 2).sort()).toEqual(["a1", "a2"]);
    expect(items.slice(2, 4).sort()).toEqual(["a3", "a4"]);
    expect(items.slice(4).sort()).toEqual(["a5", "a6"]);
  });

  it("uses the next stratum when the scheduled one has no eligible item, and an earlier one after the last", () => {
    // Every peak is at θ = 0, so ties go to the first item in pool order.
    const pool = [item2pl("low1", 0.5, 0), item2pl("low2", 0.6, 0), item2pl("high1", 2, 0), item2pl("high2", 2.1, 0)];
    const early = adaptiveTest({ pool, selection: aStratified([3, 1]), stopping: testLength(4) });
    expect(administer(early, () => 1).items).toEqual(["low1", "low2", "high1", "high2"]);
    const late = adaptiveTest({ pool, selection: aStratified([1, 1]), stopping: testLength(4) });
    expect(administer(late, () => 0).items).toEqual(["low1", "high1", "high2", "low2"]);
  });

  it("rejects lengths that are not positive integers", () => {
    expect(() => aStratified([])).toThrow(DataError);
    expect(() => aStratified([5, 0])).toThrow("The length of stratum 2 must be a positive integer, found 0.");
    expect(() => aStratified([2.5])).toThrow(DataError);
    expect(() => aStratified(JSON.parse("null"))).toThrow(
      "The lengths of a-stratified selection must be an array, found null.",
    );
  });
});

describe("adaptiveTest", () => {
  for (const { source, pool, runs } of FIXTURES) {
    for (const run of runs) {
      it(`selects catR's items with catR's estimates at every step: ${run.name}, ${source}`, () => {
        const test = adaptiveTest({
          pool,
          selection: defined(SELECTIONS.get(run.selection)),
          estimator: runEstimator(run.estimators),
          stopping: testLength(LENGTH),
        });
        const random = createRandom(1);
        let state = test.start(random);
        run.steps.forEach((step, k) => {
          expect(state.next, `step ${k}`).toBe(step.nextItem);
          expectEstimate(state.estimate, step, `step ${k}`);
          state = test.answer(state, step.response, random);
        });
        expect(state.next).toBeNull();
        expectEstimate(state.estimate, run.final, "final");
      });
    }
  }

  it("resumes from a state stored as JSON", () => {
    const test = adaptiveTest({
      pool: POOL_3PL.map((item, i) => ({ ...item, exposure: i % 3 === 0 ? 0.5 : 1 })),
      selection: randomesque(kullbackLeibler, 3),
      stopping: testLength(8),
    });
    const respond = (id: string): number => Number(/[147]$/.test(id));
    const direct = administer(test, respond, createRandom(7));
    const random = createRandom(7);
    let stored = JSON.stringify(test.start(random));
    for (;;) {
      const state = JSON.parse(stored) as AdaptiveTestState;
      if (state.next === null) break;
      stored = JSON.stringify(test.answer(state, respond(state.next), random));
    }
    expect(JSON.parse(stored)).toEqual(direct);
  });

  it("estimates with its estimator before the first response", () => {
    const state = adaptiveTest({ pool: POOL_GPCM, stopping: testLength(1) }).start(createRandom(1));
    expect(state).toMatchObject({ items: [], responses: [], withheld: [], estimate: posteriorMeanEstimate([], []) });
    expect(state.next).not.toBeNull();
  });

  it("ends before the first item when the stopping rule holds without responses", () => {
    const test = adaptiveTest({ pool: POOL_3PL, stopping: standardErrorThreshold(1.5) });
    expect(test.start(createRandom(1))).toMatchObject({ items: [], next: null });
  });

  it("ends when no item is left", () => {
    const pool = POOL_GPCM.slice(0, 3);
    const state = administer(adaptiveTest({ pool, stopping: testLength(10) }), () => 1);
    expect(state.items.slice().sort()).toEqual(pool.map((item) => item.id).sort());
    expect(state.next).toBeNull();
  });

  it("administers a pool of one item", () => {
    const pool = [item2pl("only", 1, 0)];
    const state = administer(
      adaptiveTest({ pool, estimator: posteriorModeEstimate, stopping: testLength(5) }),
      () => 1,
    );
    expect(state.items).toEqual(["only"]);
    expect(state.estimate).toEqual(posteriorModeEstimate(pool, [1]));
  });

  it("rejects answers after the end, unknown items and responses outside the item's categories", () => {
    const test = adaptiveTest({ pool: POOL_GPCM, stopping: testLength(2) });
    const random = createRandom(1);
    const state = test.start(random);
    const next = defined(state.next);
    const item = defined(POOL_GPCM[poolIndex(POOL_GPCM, next)]);
    const count = item.model.categoryCount(item.parameters);
    expect(() => test.answer(state, count, random)).toThrow(
      `The response to item "${next}" must be an integer from 0 to ${count - 1}, found ${count}.`,
    );
    expect(() => test.answer(state, 0.5, random)).toThrow(DataError);
    expect(() => test.answer({ ...state, next: "unknown" }, 0, random)).toThrow('The test has no item "unknown".');
    expect(() => test.answer({ ...state, withheld: ["unknown"] }, 0, random)).toThrow(DataError);
    expect(() => test.answer({ ...state, next: null }, 0, random)).toThrow(
      "The test has ended; it takes no more responses.",
    );
  });

  it("rejects stored states that repeat or withhold the next item, or are malformed", () => {
    const test = adaptiveTest({ pool: POOL_GPCM, stopping: testLength(4) });
    const random = createRandom(1);
    const state = test.answer(test.start(random), 1, random);
    const first = defined(state.items[0]);
    const twice = `A test state names item "${first}" twice; an item is administered or withheld once.`;
    expect(() => test.answer({ ...state, next: first }, 0, random)).toThrow(twice);
    expect(() => test.answer({ ...state, items: [first, first], responses: [1, 1] }, 0, random)).toThrow(twice);
    expect(() => test.answer({ ...state, withheld: [defined(state.next)] }, 0, random)).toThrow(
      `A test state names item "${state.next}" twice; an item is administered or withheld once.`,
    );
    expect(() => test.answer({ ...state, responses: [] }, 0, random)).toThrow(
      "A test state needs one response per item, found 0 responses to 1 items.",
    );
    expect(() => test.answer({ ...state, responses: [9] }, 0, random)).toThrow(
      `The response to item "${first}" must be an integer from 0 to`,
    );
    const parse = (json: string): AdaptiveTestState => JSON.parse(json);
    for (const json of ["null", "{}", `{"next": "${state.next}"}`, '{"items": 5, "responses": []}']) {
      expect(() => test.answer(parse(json), 0, random)).toThrow("A test state needs the arrays items and responses.");
    }
    const missing: readonly string[] = JSON.parse("null");
    expect(() => test.answer({ ...state, withheld: missing }, 0, random)).toThrow(
      "An adaptive test state needs the array withheld.",
    );
  });

  it("rejects invalid pools and an estimator without an estimate before the first response", () => {
    const item = item2pl("q1", 1, 0);
    const stopping = testLength(1);
    expect(() => adaptiveTest({ pool: [item, item], stopping })).toThrow(
      'Item ids must be unique; "q1" appears twice.',
    );
    expect(() => adaptiveTest({ pool: [{ ...item, parameters: { a: 1, b: 0, c: 0.2, d: 1 } }], stopping })).toThrow(
      'Item "q1": A 2PL item has c = 0, found c = 0.2.',
    );
    expect(() => adaptiveTest({ pool: [{ ...item, exposure: 1.5 }], stopping })).toThrow(
      'The exposure parameter of item "q1" must be a number from 0 to 1, found 1.5.',
    );
    const test = adaptiveTest({ pool: [item], estimator: maximumLikelihoodEstimate, stopping });
    expect(() => test.start(createRandom(1))).toThrow("Maximum likelihood estimation needs at least one response.");
  });

  it("rejects a stopping rule, a selection rule or an estimator of another kind", () => {
    const pool = [item2pl("q1", 1, 0)];
    const stopping = testLength(1);
    expect(() => adaptiveTest(JSON.parse("{}"))).toThrow(
      "The option stopping must be a stopping rule, such as testLength(20), found undefined.",
    );
    expect(() => adaptiveTest({ pool, stopping: JSON.parse("{}") })).toThrow(
      "The option stopping must be a stopping rule, such as testLength(20), found an object.",
    );
    expect(() => adaptiveTest({ pool, stopping, selection: JSON.parse("null") })).toThrow(
      "The option selection must be a selection rule, such as maximumInformation, found null.",
    );
    expect(() => adaptiveTest({ pool, stopping, estimator: JSON.parse('"eap"') })).toThrow(
      'The option estimator must be a function of items and responses, such as posteriorMeanEstimate, found "eap".',
    );
  });

  it("rejects options, pools and random number generators of other types from JavaScript", () => {
    const stopping = testLength(1);
    expect(() => adaptiveTest(JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => adaptiveTest({ pool: JSON.parse("null"), stopping })).toThrow(
      "The items must be an array, found null.",
    );
    expect(() => adaptiveTest({ pool: JSON.parse("[null]"), stopping })).toThrow(
      "Item 1 must be an item with an id, a model and parameters, found null.",
    );
    expect(() => adaptiveTest({ pool: [item2pl("q1", 1, 0)], stopping, enemies: JSON.parse("null") })).toThrow(
      "The option enemies must be an array of enemy sets, found null.",
    );
    const test = adaptiveTest({ pool: [item2pl("q1", 1, 0), item2pl("q2", 1, 1)], stopping });
    expect(() => test.start(JSON.parse("null"))).toThrow(
      "A random number generator must be one from createRandom, found null.",
    );
    const state = test.start(createRandom(1));
    expect(() => test.answer(state, 1, JSON.parse("{}"))).toThrow(
      "A random number generator must be one from createRandom, found an object.",
    );
  });

  it("ends the test when the selection rule scores every eligible item -Infinity", () => {
    const none: ItemSelectionRule = { scores: (_, out) => out.fill(Number.NEGATIVE_INFINITY) };
    const test = adaptiveTest({ pool: POOL_3PL, selection: none, stopping: testLength(5) });
    expect(test.start(createRandom(1)).next).toBeNull();
    const afterTwo: ItemSelectionRule = {
      scores: ({ administered, eligible }, out) => {
        eligible.forEach((flag, i) => {
          if (flag === 1) out[i] = administered.length < 2 ? 1 : Number.NEGATIVE_INFINITY;
        });
      },
    };
    const short = adaptiveTest({ pool: POOL_3PL, selection: afterTwo, stopping: testLength(5) });
    expect(administer(short, () => 1).items).toHaveLength(2);
  });
});

describe("content balancing", () => {
  const areas = Object.keys(catContent.targets);
  const targets = Object.values(catContent.targets);
  const test = adaptiveTest({
    pool: POOL_CONTENT,
    estimator: posteriorModeEstimate,
    stopping: testLength(LENGTH),
    contentTargets: catContent.targets,
  });
  const steps: readonly CatStep[] = catContent.run.steps;

  it("selects catR's item where catR's choice of area is not random, and otherwise an area catR could choose", () => {
    steps.forEach((step, k) => {
      const random = createRandom(k);
      // The state after catR's first k - 1 items with its k-th next, as a server stores it, answered.
      const previous = steps[k - 1];
      const state =
        previous === undefined
          ? test.start(random)
          : test.answer(
              {
                items: steps.slice(0, k - 1).map(({ nextItem }) => nextItem),
                responses: steps.slice(0, k - 1).map(({ response }) => response),
                withheld: [],
                estimate: { theta: previous.theta, standardError: previous.standardError ?? 1 },
                next: previous.nextItem,
              },
              previous.response,
              random,
            );
      const shares = defined(step.administeredProportions);
      const gaps = targets.map((target, a) => target - (shares[a] ?? 0));
      const candidates = shares.includes(0)
        ? areas.filter((_, a) => shares[a] === 0)
        : areas.filter((_, a) => gaps[a] === Math.max(...gaps));
      const next = defined(POOL_CONTENT[poolIndex(POOL_CONTENT, defined(state.next))]);
      expect(candidates, `step ${k}`).toContain(next.contentAreas?.[0]);
      if (candidates.length === 1) expect(next.id, `step ${k}`).toBe(step.nextItem);
      const remaining = POOL_CONTENT.filter(
        ({ id, contentAreas }) => contentAreas?.[0] === next.contentAreas?.[0] && !state.items.includes(id),
      );
      const information = remaining.map((item) => itemInformation(item, state.estimate.theta));
      expect(itemInformation(next, state.estimate.theta), `step ${k}`).toBe(Math.max(...information));
    });
  });

  it("takes each area first, then the area furthest below its target", () => {
    const pool = ["x", "y"].flatMap((area) =>
      Array.from({ length: 10 }, (_, i) => ({
        ...item2pl(`${area}${i}`, 1 + i / 10, i / 5 - 1),
        contentAreas: [area],
      })),
    );
    // Targets 3 : 1. One item of each area, then x twice, when both areas meet their targets and the fifth item may
    // be either; 8 items then hold 6 of x.
    const balanced = adaptiveTest({ pool, stopping: testLength(8), contentTargets: { x: 3, y: 1 } });
    for (const seed of [1, 2, 3, 4]) {
      const areasOf = administer(balanced, () => 1, createRandom(seed)).items.map((id) => id.charAt(0));
      expect(areasOf.slice(0, 2).sort()).toEqual(["x", "y"]);
      expect(areasOf.slice(2, 4)).toEqual(["x", "x"]);
      expect(areasOf.filter((area) => area === "x")).toHaveLength(6);
    }
  });

  it("skips areas without eligible items", () => {
    const pool = [
      { ...item2pl("x1", 1, 0), contentAreas: ["x"] },
      { ...item2pl("y1", 1, 0.5), contentAreas: ["y"] },
      { ...item2pl("y2", 1, -0.5), contentAreas: ["y"] },
    ];
    const balanced = adaptiveTest({ pool, stopping: testLength(3), contentTargets: { x: 0.9, y: 0.1 } });
    expect(administer(balanced, () => 0).items).toHaveLength(3);
  });

  it("counts the areas of an item among the targets only, and rejects an item in two of them", () => {
    const pool = [
      { ...item2pl("x1", 1, 0), contentAreas: ["x", "hard"] },
      { ...item2pl("y1", 1, 0.5), contentAreas: ["easy", "y"] },
      { ...item2pl("y2", 1, -0.5), contentAreas: ["y"] },
    ];
    const balanced = adaptiveTest({ pool, stopping: testLength(3), contentTargets: { x: 1, y: 1 } });
    expect(
      new Set(
        administer(balanced, () => 0)
          .items.slice(0, 2)
          .map((id) => id.charAt(0)),
      ),
    ).toEqual(new Set(["x", "y"]));
    const both = [...pool, { ...item2pl("xy", 1, 0), contentAreas: ["x", "y"] }];
    expect(() => adaptiveTest({ pool: both, stopping: testLength(1), contentTargets: { x: 1, y: 1 } })).toThrow(
      'Item "xy" needs exactly one content area among the targets (x, y), found ["x", "y"].',
    );
    const text = [{ ...item2pl("x1", 1, 0), contentAreas: JSON.parse('"x"') }];
    expect(() => adaptiveTest({ pool: text, stopping: testLength(1), contentTargets: { x: 1 } })).toThrow(
      'The contentAreas of item "x1" must be an array of distinct non-empty strings, found "x".',
    );
  });

  it("rejects missing and invalid targets", () => {
    const stopping = testLength(1);
    expect(() => adaptiveTest({ pool: POOL_CONTENT, stopping, contentTargets: {} })).toThrow(DataError);
    expect(() =>
      adaptiveTest({ pool: POOL_CONTENT, stopping, contentTargets: { ...catContent.targets, algebra: 0 } }),
    ).toThrow('The target of content area "algebra" must be a positive number, found 0.');
    expect(() => adaptiveTest({ pool: POOL_CONTENT, stopping, contentTargets: { algebra: 1 } })).toThrow(
      'Item "item001" needs exactly one content area among the targets (algebra), found ["statistics"].',
    );
    expect(() => adaptiveTest({ pool: POOL_3PL, stopping, contentTargets: { algebra: 1 } })).toThrow(DataError);
    expect(() => adaptiveTest({ pool: POOL_3PL, stopping, contentTargets: JSON.parse("null") })).toThrow(
      "The option contentTargets must be an object of target proportions by area, found null.",
    );
  });
});

describe("content bounds", () => {
  const indicesOf = (ids: readonly string[], pool: readonly PoolItem[]): readonly number[] =>
    ids.map((id) => poolIndex(pool, id));
  /** Answers right and wrong in turn, so that theta stays near 0, where the items' information falls with their rank. */
  const alternating = (): ((id: string) => number) => {
    let answered = 0;
    return () => answered++ % 2;
  };
  /** A 2PL item in content areas whose information at theta 0 falls as `rank` rises. */
  const areaItem = (id: string, areas: readonly string[], rank = 0): PoolItem => ({
    ...item2pl(id, 2 - rank / 10, 0),
    contentAreas: areas,
  });
  /** The selection context after the items with the given ids, the others eligible except those in `ineligible`. */
  const contextAfterIds = (
    pool: readonly PoolItem[],
    given: readonly string[],
    ineligible: readonly string[] = [],
  ): SelectionContext => {
    const administered = indicesOf(given, pool);
    const eligible = new Uint8Array(pool.length).fill(1);
    for (const index of [...administered, ...indicesOf(ineligible, pool)]) eligible[index] = 0;
    return {
      pool,
      administered,
      responses: administered.map(() => 1),
      estimate: BOUNDS_ESTIMATE,
      eligible,
      random: createRandom(0),
    };
  };
  const scoresOf = (rule: ItemSelectionRule, context: SelectionContext): Float64Array => {
    const out = new Float64Array(context.pool.length).fill(Number.NaN);
    rule.scores(context, out);
    return out;
  };
  /** The ids of the eligible items that a rule scores above -Infinity. */
  const allowed = (rule: ItemSelectionRule, context: SelectionContext): string[] => {
    const out = scoresOf(rule, context);
    return context.pool.flatMap((item, i) => (context.eligible[i] === 1 && (out[i] ?? 0) > -Infinity ? [item.id] : []));
  };
  const countIn = (items: readonly string[], pool: readonly PoolItem[], area: string): number =>
    items.filter((id) => pool.find((item) => item.id === id)?.contentAreas?.includes(area)).length;

  describe("replays", () => {
    const stoppingOf = (c: (typeof catBounds.cases)[number]): StoppingRule =>
      c.stopping.value === undefined
        ? testLength(c.length)
        : anyOf(
            testLength(c.stopping.length ?? c.length),
            allOf(standardErrorThreshold(c.stopping.value), contentMinimumsMet(c.constraints)),
          );
    const selectionOf = (c: (typeof catBounds.cases)[number]): ItemSelectionRule => {
      if (c.rule === "bounds") return contentBounds(maximumInformation, c.constraints, c.length);
      const index = maximumPriorityIndex(maximumInformation, c.constraints, c.length);
      return c.rule === "priority" ? index : contentBounds(index, c.constraints, c.length);
    };

    for (const c of catBounds.cases) {
      it(`takes the items of a replay in base R of the case ${c.name}`, () => {
        const pool: PoolItem[] = c.items.map((item) => ({
          id: item.id,
          model: threeParameterLogistic,
          parameters: { a: item.a, b: item.b, c: item.c, d: item.d },
          contentAreas: item.contentAreas,
          exposure: item.exposure,
          ...("testlet" in item && { testlet: item.testlet }),
        }));
        const test = adaptiveTest({
          pool,
          selection: selectionOf(c),
          stopping: stoppingOf(c),
          enemies: c.enemies,
          withinTestlet: c.within === "adaptive" ? "adaptive" : "in-order",
        });
        c.persons.forEach((person, p) => {
          const respond = (id: string): number => defined(catBounds.responses[p]?.[poolIndex(pool, id)]);
          const random = createRandom(1);
          let state = test.start(random);
          expectClose(state.estimate.theta, person.initial.theta, REPLAY_TOLERANCE, `${c.name}, person ${p}, start`);
          person.administered.forEach((id, k) => {
            expect(state.next, `${c.name}, person ${p}, item ${k + 1}`).toBe(id);
            state = test.answer(state, respond(id), random);
            const expected = defined(person.estimates[k]);
            expectClose(
              state.estimate.theta,
              expected.theta,
              REPLAY_TOLERANCE,
              `${c.name}, person ${p}, theta ${k + 1}`,
            );
            expectClose(
              state.estimate.standardError,
              expected.standardError,
              REPLAY_TOLERANCE,
              `${c.name}, person ${p}, standard error ${k + 1}`,
            );
          });
          expect(state.next, `${c.name}, person ${p}, end`).toBeNull();
          expect(state.items).toEqual(person.administered);
        });
      });
    }
  });

  describe("on persons simulated against the shadow test of mirtCAT", () => {
    for (const blueprint of catBounds.shadowTests) {
      it(`holds every bound of the ${blueprint.name} blueprint with the precision of the shadow test`, () => {
        const pool: PoolItem[] = blueprint.bank.map(({ id, a, b, c, d, contentAreas }) => ({
          id,
          model: threeParameterLogistic,
          parameters: { a, b, c, d },
          contentAreas,
        }));
        const { constraints, length } = blueprint;
        const test = adaptiveTest({
          pool,
          selection: contentBounds(maximumPriorityIndex(maximumInformation, constraints, length), constraints, length),
          stopping: testLength(length),
        });
        const simulation = simulateAdaptiveTest(test, blueprint.thetas, createRandom(SEED_BOUNDS));
        expect(simulation.persons).toHaveLength(blueprint.thetas.length);
        const outside = simulation.persons.filter(({ state }) =>
          constraints.some(({ area, minimum, maximum }) => {
            const count = countIn(state.items, pool, area);
            return count < minimum || count > maximum;
          }),
        );
        expect(outside).toHaveLength(0);
        expect(simulation.meanLength).toBe(length);
        expect(simulation.rootMeanSquareError).toBeLessThanOrEqual(1.05 * blueprint.shadow.rmse);
      });
    }
  });

  describe("contentBounds", () => {
    const run = (
      pool: readonly PoolItem[],
      constraints: readonly ContentConstraint[],
      length: number,
      stopping: StoppingRule = testLength(length),
    ): AdaptiveTestState =>
      administer(
        adaptiveTest({ pool, selection: contentBounds(maximumInformation, constraints, length), stopping }),
        alternating(),
      );

    it("keeps a test within the maximum of an area", () => {
      const pool = [
        ...[0, 1, 2, 3, 4, 5].map((rank) => areaItem(`x${rank + 1}`, ["x"], rank)),
        ...[6, 7, 8, 9, 10, 11].map((rank) => areaItem(`y${rank - 5}`, ["y"], rank)),
      ];
      const items = run(pool, [{ area: "x", minimum: 0, maximum: 2 }], 6).items;
      expect(items).toEqual(["x1", "x2", "y1", "y2", "y3", "y4"]);
    });

    it("takes the items of an area before the places run out for its minimum", () => {
      const pool = [
        ...[0, 1, 2, 3, 4].map((rank) => areaItem(`x${rank + 1}`, ["x"], rank)),
        ...[5, 6, 7, 8, 9].map((rank) => areaItem(`y${rank - 4}`, ["y"], rank)),
      ];
      const constraints = [
        { area: "x", minimum: 0, maximum: 5 },
        { area: "y", minimum: 3, maximum: 5 },
      ];
      expect(run(pool, constraints, 5).items).toEqual(["x1", "x2", "y1", "y2", "y3"]);
    });

    it("scores the items that break a bound -Infinity and leaves the scores of the others as the rule gave them", () => {
      const pool = [
        areaItem("x1", ["x"], 0),
        areaItem("x2", ["x"], 1),
        areaItem("x3", ["x"], 2),
        areaItem("y1", ["y"], 3),
        areaItem("z1", [], 4),
      ];
      const rule = contentBounds(maximumInformation, [{ area: "x", minimum: 0, maximum: 2 }], 5);
      const context = contextAfterIds(pool, ["x1", "x2"]);
      const out = scoresOf(rule, context);
      const reference = scoresOf(maximumInformation, context);
      expect(out[poolIndex(pool, "x3")]).toBe(Number.NEGATIVE_INFINITY);
      for (const id of ["y1", "z1"]) expect(out[poolIndex(pool, id)]).toBe(reference[poolIndex(pool, id)]);
      expect(allowed(rule, contextAfterIds(pool, ["x1"]))).toEqual(["x2", "x3", "y1", "z1"]);
    });

    it("keeps a test within the length: a testlet that does not fit is out", () => {
      const pool = [
        ...[0, 1, 2].map((rank) => ({ ...areaItem(`t${rank + 1}`, [], rank), testlet: "T" })),
        areaItem("s1", [], 3),
        areaItem("s2", [], 4),
      ];
      const rule = contentBounds(maximumInformation, [], 4);
      expect(allowed(rule, contextAfterIds(pool, []))).toEqual(["t1", "t2", "t3", "s1", "s2"]);
      expect(allowed(rule, contextAfterIds(pool, ["s1", "s2"]))).toEqual([]);
      expect(allowed(rule, contextAfterIds(pool, ["s1"]))).toEqual(["t1", "t2", "t3", "s2"]);
      const fitting = contentBounds(maximumInformation, [], 3);
      expect(allowed(fitting, contextAfterIds(pool, ["s1"]))).toEqual(["s2"]);
    });

    it("decides by a greedy cover where the items still needed in two areas sum to more than the places left", () => {
      // Minimums 3 and 3 in a test of 4 need two items in both areas.
      const pool = [
        ...[0, 1, 2, 3].map((rank) => areaItem(`a${rank + 1}`, ["a"], rank)),
        ...[4, 5, 6, 7].map((rank) => areaItem(`b${rank - 3}`, ["b"], rank)),
        areaItem("ab1", ["a", "b"], 8),
        areaItem("ab2", ["a", "b"], 9),
      ];
      const constraints = [
        { area: "a", minimum: 3, maximum: 4 },
        { area: "b", minimum: 3, maximum: 4 },
      ];
      const items = run(pool, constraints, 4).items;
      expect(items).toHaveLength(4);
      expect(countIn(items, pool, "a")).toBeGreaterThanOrEqual(3);
      expect(countIn(items, pool, "b")).toBeGreaterThanOrEqual(3);
      const rule = contentBounds(maximumInformation, constraints, 4);
      // After one item of each area, only an item of both can still meet both minimums.
      expect(allowed(rule, contextAfterIds(pool, ["a1", "b1"]))).toEqual(["ab1", "ab2"]);
    });

    it("takes the units within the maximums that fill the most needs where none can meet the minimums", () => {
      const pool = [
        areaItem("a1", ["a"], 0),
        areaItem("a2", ["a"], 1),
        areaItem("b1", ["b"], 2),
        areaItem("b2", ["b"], 3),
        areaItem("ab", ["a", "b"], 4),
        areaItem("z1", [], 5),
      ];
      const rule = contentBounds(
        maximumInformation,
        [
          { area: "a", minimum: 2, maximum: 3 },
          { area: "b", minimum: 2, maximum: 3 },
        ],
        3,
      );
      expect(allowed(rule, contextAfterIds(pool, ["a1", "b1"]))).toEqual(["ab"]);
      // Without the item of both areas, no item reaches both minimums: those that fill a need stay.
      expect(allowed(rule, contextAfterIds(pool, ["a1", "b1"], ["ab"]))).toEqual(["a2", "b2"]);
    });

    it("lets the testlet in progress pass and judges a testlet that has not begun as a whole", () => {
      const pool = [
        ...[0, 1, 2].map((rank) => ({ ...areaItem(`t${rank + 1}`, ["x"], rank), testlet: "T" })),
        areaItem("x1", ["x"], 3),
        areaItem("y1", ["y"], 4),
      ];
      const rule = contentBounds(maximumInformation, [{ area: "x", minimum: 0, maximum: 3 }], 5);
      // The testlet fits within the maximum of 3 items of x; a single item of x then does not fit.
      expect(allowed(rule, contextAfterIds(pool, []))).toEqual(["t1", "t2", "t3", "x1", "y1"]);
      expect(allowed(rule, contextAfterIds(pool, ["x1"]))).toEqual(["y1"]);
      // With a maximum of 2, the testlet is out whole; after its first item, the rest of it is not judged.
      const tight = contentBounds(maximumInformation, [{ area: "x", minimum: 0, maximum: 2 }], 5);
      expect(allowed(tight, contextAfterIds(pool, []))).toEqual(["x1", "y1"]);
      expect(allowed(tight, contextAfterIds(pool, ["t1"]))).toEqual(["t2", "t3", "x1", "y1"]);
    });

    it("keeps -Infinity where the rule gave it", () => {
      const pool = [areaItem("x1", ["x"], 0), areaItem("x2", ["x"], 1), areaItem("y1", ["y"], 2)];
      const excluding: ItemSelectionRule = {
        scores: (context, out) => {
          maximumInformation.scores(context, out);
          out[poolIndex(pool, "x2")] = Number.NEGATIVE_INFINITY;
        },
      };
      const rule = contentBounds(excluding, [{ area: "x", minimum: 1, maximum: 2 }], 2);
      expect(allowed(rule, contextAfterIds(pool, []))).toEqual(["x1", "y1"]);
    });

    it("ends the test where no item is within the maximums", () => {
      const pool = [
        ...[0, 1, 2, 3].map((rank) => areaItem(`x${rank + 1}`, ["x"], rank)),
        ...[4, 5, 6, 7].map((rank) => areaItem(`y${rank - 3}`, ["y"], rank)),
      ];
      const constraints = [
        { area: "x", minimum: 0, maximum: 2 },
        { area: "y", minimum: 0, maximum: 2 },
      ];
      const state = run(pool, constraints, 8);
      expect(state.items).toHaveLength(4);
      expect(state.next).toBeNull();
    });

    it("takes the order of a testlet's items from the rule while the testlet is in progress", () => {
      const pool = [
        ...[0, 1, 2].map((rank) => ({ ...areaItem(`t${rank + 1}`, ["x"], rank), testlet: "T" })),
        areaItem("y1", ["y"], 3),
      ];
      const test = adaptiveTest({
        pool,
        selection: contentBounds(maximumInformation, [{ area: "x", minimum: 3, maximum: 3 }], 4),
        stopping: testLength(4),
        withinTestlet: "adaptive",
      });
      expect(administer(test, alternating()).items).toEqual(["t1", "t2", "t3", "y1"]);
    });

    it("works inside randomesque and with exposure control, and holds the maximums", () => {
      const pool = [
        ...[0, 1, 2, 3, 4, 5].map((rank) => ({ ...areaItem(`x${rank + 1}`, ["x"], rank), exposure: 0.5 })),
        ...[6, 7, 8, 9, 10, 11].map((rank) => ({ ...areaItem(`y${rank - 5}`, ["y"], rank), exposure: 0.5 })),
      ];
      const constraints = [
        { area: "x", minimum: 2, maximum: 3 },
        { area: "y", minimum: 2, maximum: 3 },
      ];
      const test = adaptiveTest({
        pool,
        selection: randomesque(contentBounds(maximumInformation, constraints, 6), 3),
        stopping: testLength(6),
      });
      // A withheld item can leave an area with no item to take, and the test ends short, within the maximums.
      const lengths = Array.from({ length: 20 }, (_, seed) => {
        const { items } = administer(test, alternating(), createRandom(seed));
        expect(countIn(items, pool, "x")).toBeLessThanOrEqual(3);
        expect(countIn(items, pool, "y")).toBeLessThanOrEqual(3);
        return items.length;
      });
      expect(Math.max(...lengths)).toBe(6);
    });

    it("does not change what a call receives or an earlier call left", () => {
      const pool = [areaItem("x1", ["x"], 0), areaItem("y1", ["y"], 1)];
      const rule = contentBounds(maximumInformation, [{ area: "x", minimum: 0, maximum: 1 }], 2);
      const context = contextAfterIds(pool, []);
      const before = [...context.eligible];
      expect(scoresOf(rule, context)).toEqual(scoresOf(rule, context));
      expect([...context.eligible]).toEqual(before);
    });

    it("throws for a proof that the bounds cannot be met, at the first selection", () => {
      const pool = [
        areaItem("x1", ["x"], 0),
        areaItem("x2", ["x"], 1),
        areaItem("y1", ["y"], 2),
        areaItem("y2", ["y"], 3),
      ];
      const first = (constraints: readonly ContentConstraint[], length: number) => () =>
        adaptiveTest({
          pool,
          selection: contentBounds(maximumInformation, constraints, length),
          stopping: testLength(2),
        }).start(createRandom(1));
      expect(first([{ area: "x", minimum: 3, maximum: 4 }], 4)).toThrow(
        'The pool has 2 items in content area "x", fewer than its minimum 3; add items or lower the minimum.',
      );
      expect(first([{ area: "x", minimum: 2, maximum: 2 }], 1)).toThrow(
        'The minimum 2 of content area "x" is above the test length 1; lower the minimum or raise the length.',
      );
      expect(
        first(
          [
            { area: "x", minimum: 2, maximum: 2 },
            { area: "y", minimum: 2, maximum: 2 },
          ],
          3,
        ),
      ).toThrow(
        "The minimums of the content areas sum to 4, above the test length 3, and no item is in two of the areas; lower a minimum or raise the length.",
      );
      expect(first([{ area: "z", minimum: 1, maximum: 2 }], 3)).toThrow(
        'The pool has 0 items in content area "z", fewer than its minimum 1; add items or lower the minimum.',
      );
      // Minimums that sum to more than the length are no proof where an item is in two of the areas.
      const overlap = [...pool, areaItem("xy", ["x", "y"], 4)];
      const constraints = [
        { area: "x", minimum: 2, maximum: 3 },
        { area: "y", minimum: 2, maximum: 3 },
      ];
      expect(() =>
        adaptiveTest({
          pool: overlap,
          selection: contentBounds(maximumInformation, constraints, 3),
          stopping: testLength(3),
        }).start(createRandom(1)),
      ).not.toThrow();
    });

    it("rejects constraints, lengths, rules and pools of other types and invalid ones", () => {
      const ok = [{ area: "x", minimum: 1, maximum: 2 }];
      expect(() => contentBounds(maximumInformation, JSON.parse("null"), 4)).toThrow(
        "The content constraints must be an array of { area, minimum, maximum }, found null.",
      );
      expect(() => contentBounds(maximumInformation, JSON.parse("[null]"), 4)).toThrow(
        "A content constraint must be an object { area, minimum, maximum }, found null.",
      );
      expect(() => contentBounds(maximumInformation, JSON.parse('[{"area":1,"minimum":0,"maximum":1}]'), 4)).toThrow(
        "A content constraint needs a string area, found 1.",
      );
      expect(() =>
        contentBounds(maximumInformation, JSON.parse('[{"area":"x","minimum":"1","maximum":2}]'), 4),
      ).toThrow('The minimum of the constraint on "x" must be a finite number, found "1".');
      expect(() => contentBounds(maximumInformation, [{ area: "x", minimum: 3, maximum: 2 }], 4)).toThrow(
        'The constraint on "x" needs the minimum at most the maximum, found 3 and 2.',
      );
      expect(() => contentBounds(maximumInformation, [{ area: "x", minimum: 0.5, maximum: 2 }], 4)).toThrow(
        'The minimum of the constraint on "x" must be an integer from 0, found 0.5.',
      );
      expect(() => contentBounds(maximumInformation, [{ area: "x", minimum: -1, maximum: 2 }], 4)).toThrow(
        'The minimum of the constraint on "x" must be an integer from 0, found -1.',
      );
      expect(() => contentBounds(maximumInformation, [{ area: "x", minimum: 0, maximum: Infinity }], 4)).toThrow(
        DataError,
      );
      expect(() => contentBounds(maximumInformation, [{ area: "x", minimum: 0, maximum: 2, weight: 0 }], 4)).toThrow(
        'The weight of the constraint on "x" must be a positive number, found 0.',
      );
      expect(() => contentBounds(maximumInformation, [...ok, ...ok], 4)).toThrow(
        'Content area "x" has two constraints; give it one.',
      );
      for (const length of [0, 1.5, -2, Number.NaN]) {
        expect(() => contentBounds(maximumInformation, ok, length)).toThrow(
          `The length of contentBounds must be a positive integer, found ${length}.`,
        );
      }
      expect(() => contentBounds(maximumInformation, ok, JSON.parse('"4"'))).toThrow(
        'The length of contentBounds must be a positive integer, found "4".',
      );
      expect(() => contentBounds(JSON.parse("null"), ok, 4)).toThrow(
        "contentBounds needs a selection rule, such as maximumInformation, found null.",
      );
      const pool = [{ ...areaItem("x1", [], 0), contentAreas: JSON.parse('"x"') }];
      const test = adaptiveTest({ pool, selection: contentBounds(maximumInformation, ok, 4), stopping: testLength(1) });
      expect(() => test.start(createRandom(1))).toThrow(
        'The contentAreas of item "x1" must be an array of distinct non-empty strings, found "x".',
      );
      const twice = [{ ...areaItem("x1", [], 0), contentAreas: ["x", "x"] }];
      expect(() =>
        adaptiveTest({
          pool: twice,
          selection: contentBounds(maximumInformation, ok, 4),
          stopping: testLength(1),
        }).start(createRandom(1)),
      ).toThrow('The contentAreas of item "x1" must be an array of distinct non-empty strings, found ["x", "x"].');
    });
  });

  describe("maximumPriorityIndex", () => {
    const pool = [
      areaItem("x1", ["x"], 0),
      areaItem("x2", ["x"], 1),
      areaItem("y1", ["y"], 2),
      areaItem("y2", ["y"], 3),
      areaItem("xy", ["x", "y"], 4),
      areaItem("z1", [], 5),
    ];
    const constraints = [
      { area: "x", minimum: 2, maximum: 4 },
      { area: "y", minimum: 1, maximum: 3, weight: 2 },
    ];

    it("multiplies each score by the weight, the share of the maximum left and the share of the places the others may take", () => {
      const rule = maximumPriorityIndex(maximumInformation, constraints, 6);
      const context = contextAfterIds(pool, ["x1"]);
      const out = scoresOf(rule, context);
      const reference = scoresOf(maximumInformation, context);
      // x: (4 - 1) / 4 times ((6 - 2) - (1 - 1)) / (6 - 2) = 0.75. y: 2 times (3 - 0) / 3 times ((6 - 1) - (1 - 0)) / (6 - 1) = 1.6.
      const factor = (id: string): number =>
        defined(out[poolIndex(pool, id)]) / defined(reference[poolIndex(pool, id)]);
      expectClose(factor("x2"), 0.75, 1e-14);
      expectClose(factor("y1"), 1.6, 1e-14);
      expectClose(factor("xy"), 0.75 * 1.6, 1e-14);
      expectClose(factor("z1"), 1, 1e-14);
    });

    it("gives the weight alone where the items outside an area have used up the places its minimum leaves", () => {
      const rule = maximumPriorityIndex(
        maximumInformation,
        [
          { area: "x", minimum: 4, maximum: 6, weight: 3 },
          { area: "y", minimum: 6, maximum: 6, weight: 5 },
        ],
        6,
      );
      const context = contextAfterIds(pool, ["y1", "y2"]);
      const out = scoresOf(rule, context);
      const reference = scoresOf(maximumInformation, context);
      // x: 6 - 4 = 2 places outside, both used by y1 and y2, so f2 = 0 and f = 1. y: the minimum is the length.
      expectClose(defined(out[poolIndex(pool, "x1")]) / defined(reference[poolIndex(pool, "x1")]), 3, 1e-14);
      expectClose(defined(out[poolIndex(pool, "xy")]) / defined(reference[poolIndex(pool, "xy")]), 15, 1e-14);
    });

    it("scores 0 for an item of an area whose maximum is 0, and keeps -Infinity", () => {
      const excluding: ItemSelectionRule = {
        scores: (context, out) => {
          maximumInformation.scores(context, out);
          out[poolIndex(pool, "y2")] = Number.NEGATIVE_INFINITY;
        },
      };
      const rule = maximumPriorityIndex(
        excluding,
        [
          { area: "x", minimum: 0, maximum: 0 },
          { area: "y", minimum: 0, maximum: 4 },
        ],
        6,
      );
      const out = scoresOf(rule, contextAfterIds(pool, []));
      expect(out[poolIndex(pool, "x1")]).toBe(0);
      expect(out[poolIndex(pool, "y2")]).toBe(Number.NEGATIVE_INFINITY);
      expect(defined(out[poolIndex(pool, "y1")])).toBeGreaterThan(0);
    });

    it("throws for a score below 0, which a product would put in the wrong order", () => {
      const negative: ItemSelectionRule = {
        scores: (context, out) =>
          context.eligible.forEach((_, i) => {
            out[i] = -0.5;
          }),
      };
      const rule = maximumPriorityIndex(negative, constraints, 6);
      expect(() => rule.scores(contextAfterIds(pool, []), new Float64Array(pool.length))).toThrow(
        'The rule scored item "x1" -0.5; maximumPriorityIndex multiplies scores of at least 0, so give a rule that scores below 0 to contentBounds alone.',
      );
      const aStratifiedScores = aStratified([3]);
      const test = adaptiveTest({
        pool,
        selection: contentBounds(aStratifiedScores, constraints, 3),
        stopping: testLength(3),
      });
      expect(administer(test, alternating()).items).toHaveLength(3);
      expect(() =>
        adaptiveTest({
          pool,
          selection: maximumPriorityIndex(aStratifiedScores, constraints, 3),
          stopping: testLength(3),
        }).start(createRandom(1)),
      ).toThrow(DataError);
    });

    it("spreads three areas that compete, where the filter alone takes the most informative", () => {
      const spread = [
        ...[0, 1, 2, 3, 4, 5].map((rank) => areaItem(`a${rank + 1}`, ["a"], rank)),
        ...[6, 7, 8, 9, 10, 11].map((rank) => areaItem(`b${rank - 5}`, ["b"], rank)),
        ...[12, 13, 14, 15, 16, 17].map((rank) => areaItem(`c${rank - 11}`, ["c"], rank)),
      ];
      const bounds = [
        { area: "a", minimum: 2, maximum: 4 },
        { area: "b", minimum: 2, maximum: 4 },
        { area: "c", minimum: 2, maximum: 4 },
      ];
      const filter = administer(
        adaptiveTest({
          pool: spread,
          selection: contentBounds(maximumInformation, bounds, 9),
          stopping: testLength(9),
        }),
        alternating(),
      ).items;
      const index = administer(
        adaptiveTest({
          pool: spread,
          selection: contentBounds(maximumPriorityIndex(maximumInformation, bounds, 9), bounds, 9),
          stopping: testLength(9),
        }),
        alternating(),
      ).items;
      const longestRun = (items: readonly string[]): number => {
        let longest = 0;
        let run = 0;
        items.forEach((id, i) => {
          run = i > 0 && items[i - 1]?.charAt(0) === id.charAt(0) ? run + 1 : 1;
          longest = Math.max(longest, run);
        });
        return longest;
      };
      expect(index).not.toEqual(filter);
      expect(longestRun(index)).toBeLessThan(longestRun(filter));
      for (const items of [filter, index]) {
        expect(items).toHaveLength(9);
        for (const area of ["a", "b", "c"]) {
          expect(countIn(items, spread, area)).toBeGreaterThanOrEqual(2);
          expect(countIn(items, spread, area)).toBeLessThanOrEqual(4);
        }
      }
    });

    it("rejects constraints, lengths and rules of other types and invalid ones", () => {
      expect(() => maximumPriorityIndex(maximumInformation, JSON.parse("{}"), 6)).toThrow(
        "The content constraints must be an array of { area, minimum, maximum }, found an object.",
      );
      expect(() => maximumPriorityIndex(maximumInformation, [{ area: "x", minimum: 2, maximum: 1 }], 6)).toThrow(
        DataError,
      );
      expect(() => maximumPriorityIndex(maximumInformation, constraints, 0)).toThrow(
        "The length of maximumPriorityIndex must be a positive integer, found 0.",
      );
      expect(() => maximumPriorityIndex(JSON.parse("{}"), constraints, 6)).toThrow(
        "maximumPriorityIndex needs a selection rule, such as maximumInformation, found an object.",
      );
    });
  });

  describe("contentMinimumsMet", () => {
    const pool = [areaItem("x1", ["x"], 0), areaItem("x2", ["x"], 1), areaItem("y1", ["y"], 2), areaItem("z1", [], 3)];
    const constraints = [
      { area: "x", minimum: 2, maximum: 3 },
      { area: "y", minimum: 1, maximum: 3 },
    ];

    it("stops once every minimum is met", () => {
      const rule = contentMinimumsMet<PoolItem, AbilityEstimate>(constraints);
      const stops = (given: readonly string[]): boolean => rule.stops(contextAfterIds(pool, given));
      expect(stops([])).toBe(false);
      expect(stops(["x1", "x2"])).toBe(false);
      expect(stops(["x1", "y1", "z1"])).toBe(false);
      expect(stops(["x1", "x2", "y1"])).toBe(true);
    });

    it("stops a test on precision only after the minimums, in allOf", () => {
      const test = adaptiveTest({
        pool,
        selection: contentBounds(maximumInformation, constraints, 4),
        stopping: anyOf(testLength(4), allOf(standardErrorThreshold(100), contentMinimumsMet(constraints))),
      });
      expect(administer(test, alternating()).items).toHaveLength(3);
    });

    it("stops at once without a minimum, and with no constraint", () => {
      expect(
        contentMinimumsMet<PoolItem, AbilityEstimate>([{ area: "x", minimum: 0, maximum: 2 }]).stops(
          contextAfterIds(pool, []),
        ),
      ).toBe(true);
      expect(contentMinimumsMet<PoolItem, AbilityEstimate>([]).stops(contextAfterIds(pool, []))).toBe(true);
    });

    it("rejects constraints of other types and invalid ones", () => {
      expect(() => contentMinimumsMet(JSON.parse("null"))).toThrow(
        "The content constraints must be an array of { area, minimum, maximum }, found null.",
      );
      expect(() => contentMinimumsMet([{ area: "x", minimum: 1.5, maximum: 2 }])).toThrow(DataError);
      const bad = [{ ...areaItem("x1", [], 0), contentAreas: JSON.parse("[1]") }];
      expect(() => contentMinimumsMet<PoolItem, AbilityEstimate>(constraints).stops(contextAfterIds(bad, []))).toThrow(
        'The contentAreas of item "x1" must be an array of distinct non-empty strings, found [1].',
      );
    });
  });
});

describe("enemy sets", () => {
  it("never administers two items of a set, whichever the responses", () => {
    const ids = POOL_3PL.map((item) => item.id);
    const sets = fc.array(fc.uniqueArray(fc.constantFrom(...ids), { minLength: 2, maxLength: 5 }), {
      minLength: 1,
      maxLength: 10,
    });
    fc.assert(
      fc.property(sets, fc.integer({ min: 0, max: 2 ** 31 }), (enemies, seed) => {
        const test = adaptiveTest({ pool: POOL_3PL, stopping: testLength(20), enemies });
        const random = createRandom(seed);
        const state = administer(test, () => (random.uniform() < 0.5 ? 1 : 0), random);
        expect(state.items).toHaveLength(20);
        for (const set of enemies) expect(state.items.filter((id) => set.includes(id)).length).toBeLessThanOrEqual(1);
      }),
      { numRuns: 30 },
    );
  });

  it("selects the best item that is not an enemy of an administered one", () => {
    const pool = [item2pl("first", 2, 0), item2pl("twin", 1.9, 0.1), item2pl("other", 1, 0.2)];
    const free = administer(adaptiveTest({ pool, stopping: testLength(2) }), () => 1);
    expect(free.items).toEqual(["first", "twin"]);
    const enemies = [["first", "twin"]];
    const state = administer(adaptiveTest({ pool, stopping: testLength(3), enemies }), () => 1);
    expect(state).toMatchObject({ items: ["first", "other"], next: null });
  });

  it("rejects sets that are not two or more distinct pool items, and stored states with two enemies", () => {
    const pool = POOL_3PL.slice(0, 4);
    const [a = "", b = "", c = ""] = pool.map((item) => item.id);
    const stopping = testLength(3);
    expect(() => adaptiveTest({ pool, stopping, enemies: [[a]] })).toThrow("Enemy set 1 needs at least two item ids.");
    expect(() => adaptiveTest({ pool, stopping, enemies: [[a, "q999"]] })).toThrow(
      'Enemy set 1 names "q999", which is not a pool item.',
    );
    expect(() =>
      adaptiveTest({
        pool,
        stopping,
        enemies: [
          [a, b],
          [c, c],
        ],
      }),
    ).toThrow("Enemy set 2 names an item twice.");
    const test = adaptiveTest({ pool, stopping, enemies: [[a, b]] });
    const random = createRandom(1);
    const forged = { ...test.start(random), items: [a], responses: [1], next: b };
    expect(() => test.answer(forged, 0, random)).toThrow(
      `A test state administers items "${a}" and "${b}", which are enemies.`,
    );
  });
});

type TestletReference = (typeof catTestlets.cases)[number];

const WITHIN_TESTLET = new Map<string, WithinTestlet>([
  ["in-order", "in-order"],
  ["adaptive", "adaptive"],
]);

/** The pool of a case of the written-out replay: 3PL items, with a testlet where the item has one. */
function testletPool(items: TestletReference["items"]): PoolItem[] {
  return items.map(({ id, testlet, exposure, ...parameters }) => ({
    id,
    model: threeParameterLogistic,
    parameters,
    exposure,
    ...(testlet !== null && { testlet }),
  }));
}

function expectReplayEstimate(actual: AbilityEstimate, expected: AbilityEstimate, label: string): void {
  expectClose(actual.theta, expected.theta, REPLAY_TOLERANCE, `${label} theta`);
  expectClose(actual.standardError, expected.standardError, REPLAY_TOLERANCE, `${label} standard error`);
}

/** Expects each testlet's items to be one block of `items`, in pool order if `inOrder`, and no item of it elsewhere. */
function expectTestletBlocks(
  items: readonly string[],
  pool: readonly { readonly id: string; readonly testlet?: string }[],
  inOrder: boolean,
): void {
  for (const testlet of new Set(pool.flatMap((item) => item.testlet ?? []))) {
    const members = pool.filter((item) => item.testlet === testlet).map((item) => item.id);
    const first = items.findIndex((id) => members.includes(id));
    if (first < 0) continue;
    const block = items.slice(first, first + members.length);
    expect(inOrder ? block : block.slice().sort(), testlet).toEqual(inOrder ? members : members.slice().sort());
    expect(
      items.filter((id) => members.includes(id)),
      testlet,
    ).toHaveLength(members.length);
  }
}

describe("testlets", () => {
  // At theta 0, where every test starts, an item with difficulty 0 has information a² / 4: 0.36 for solo.
  const solo: PoolItem = item2pl("solo", 1.2, 0);
  // Pool order t2, t3, t1: in order the testlet starts with t2, adaptively with t1, its most informative item. The
  // information at theta 0 is 0.0225, 0.0225 and 1, with the mean 0.348333, below solo's 0.36.
  const lopsided: PoolItem[] = [
    { ...item2pl("t2", 0.3, 0), testlet: "T" },
    { ...item2pl("t3", 0.3, 0), testlet: "T" },
    { ...item2pl("t1", 2, 0), testlet: "T" },
  ];
  const stopped = testLength(1);

  for (const reference of catTestlets.cases) {
    it(`selects the items, estimates and withheld testlets of the written-out replay and mirtCAT: ${reference.name}`, () => {
      const pool = testletPool(reference.items);
      const { stopping } = reference;
      const test = adaptiveTest({
        pool,
        stopping: stopping.type === "length" ? testLength(stopping.value) : standardErrorThreshold(stopping.value),
        enemies: reference.enemies,
        withinTestlet: defined(WITHIN_TESTLET.get(reference.within)),
      });
      reference.persons.forEach((person, p) => {
        const label = `person ${p + 1}`;
        const row = defined(catTestlets.responses[p]);
        const respond = (id: string): number => defined(row[poolIndex(pool, id)]);
        const random = createRandom(1);
        let state = test.start(random);
        expectReplayEstimate(state.estimate, person.initial, `${label}, initial`);
        person.estimates.forEach((estimate, k) => {
          expect(state.next, `${label}, step ${k}`).toBe(person.administered[k]);
          state = test.answer(state, respond(defined(state.next)), random);
          expectReplayEstimate(state.estimate, estimate, `${label}, step ${k}`);
        });
        expect(state.next, label).toBeNull();
        expect(state.items, label).toEqual(person.administered);
        expect(state.withheld, label).toEqual(person.withheld);
        // mirtCAT's ordered groups, where the items of a testlet are equally informative.
        if (person.mirtCAT !== null) expect(person.mirtCAT, label).toEqual(person.administered);
        // Enemies leave a testlet partly administered.
        if (reference.enemies.length === 0) expectTestletBlocks(state.items, pool, reference.within === "in-order");
      });
    });
  }

  it("scores a testlet by the mean score of its items, at its first item to be administered", () => {
    const pool = [solo, ...lopsided];
    const testlets = defined(poolTestlets(pool));
    const context: SelectionContext = {
      ...contextOf(pool, 0, { theta: 0, standardError: 1 }),
      random: createRandom(0),
    };
    for (const [within, entry] of [
      ["in-order", 1],
      ["adaptive", 3],
    ] as const) {
      const out = new Float64Array(pool.length);
      unitSelection(maximumInformation, testlets, within).scores(context, out);
      const expected = [0.36, ...[1, 2, 3].map((i) => (i === entry ? 1.045 / 3 : Number.NEGATIVE_INFINITY))];
      expected.forEach((score, i) => {
        if (score === Number.NEGATIVE_INFINITY) expect(out[i], `${within}, item ${i}`).toBe(score);
        else expectClose(out[i] ?? 0, score, 1e-15, `${within}, item ${i}`);
      });
    }
  });

  it("scores only the eligible items of a testlet, and a testlet without scored items -Infinity", () => {
    const pool = [solo, ...lopsided];
    const testlets = defined(poolTestlets(pool));
    // t3 is not eligible: the mean of t2 and t1 is 0.51125, at t1.
    const context: SelectionContext = {
      ...contextOf(pool, 0, { theta: 0, standardError: 1 }),
      eligible: Uint8Array.from([1, 1, 0, 1]),
      random: createRandom(0),
    };
    const out = new Float64Array(pool.length);
    unitSelection(maximumInformation, testlets, "adaptive").scores(context, out);
    expectClose(out[0] ?? 0, 0.36, 1e-15);
    expect(out[1]).toBe(Number.NEGATIVE_INFINITY);
    expectClose(out[3] ?? 0, 0.51125, 1e-15);
    const noScores: ItemSelectionRule = {
      scores(_, scores) {
        scores.fill(Number.NEGATIVE_INFINITY);
      },
    };
    unitSelection(noScores, testlets, "in-order").scores(context, out);
    expect([0, 1, 3].map((i) => out[i])).toEqual([-Infinity, -Infinity, -Infinity]);
  });

  it("selects the testlet whose mean information is highest, not the one holding the most informative item", () => {
    const withTestlet = adaptiveTest({ pool: [solo, ...lopsided], stopping: stopped });
    expect(withTestlet.start(createRandom(1)).next).toBe("solo");
    const withoutTestlet = adaptiveTest({
      pool: [solo, ...lopsided].map(({ testlet, ...item }) => item),
      stopping: stopped,
    });
    expect(withoutTestlet.start(createRandom(1)).next).toBe("t1");
  });

  it("administers the items of a testlet together, in pool order", () => {
    const test = adaptiveTest({ pool: [solo, ...lopsided], stopping: testLength(2), withinTestlet: "in-order" });
    expect(administer(test, () => 1).items).toEqual(["solo", "t2", "t3", "t1"]);
    const defaults = adaptiveTest({ pool: [solo, ...lopsided], stopping: testLength(2) });
    expect(administer(defaults, () => 0).items).toEqual(["solo", "t2", "t3", "t1"]);
  });

  it("administers the items of a testlet together, each next item the most informative at the estimate", () => {
    const low = { ...item2pl("low", 1.5, -3), testlet: "T" };
    const high = { ...item2pl("high", 1.5, 3), testlet: "T" };
    const middle = { ...item2pl("middle", 1.5, 0), testlet: "T" };
    const pool = [high, low, middle, item2pl("far", 1, 4)];
    const adaptive = adaptiveTest({ pool, stopping: testLength(4), withinTestlet: "adaptive" });
    // The testlet starts at its most informative item at theta 0; a correct response moves the estimate up, towards
    // "high", and an incorrect one down, towards "low".
    expect(administer(adaptive, () => 1).items).toEqual(["middle", "high", "low", "far"]);
    expect(administer(adaptive, () => 0).items).toEqual(["middle", "low", "high", "far"]);
    const inOrder = adaptiveTest({ pool, stopping: testLength(4) });
    expect(administer(inOrder, () => 1).items).toEqual(["high", "low", "middle", "far"]);
  });

  it("orders a testlet's items by pool order where the rule scores none above -Infinity", () => {
    const onlyT2: ItemSelectionRule = {
      scores(context, scores) {
        scores.fill(Number.NEGATIVE_INFINITY);
        const index = context.pool.findIndex((item) => item.id === "t2");
        if (context.eligible[index] === 1) scores[index] = 1;
      },
    };
    const pool = [lopsided[2], lopsided[0], lopsided[1]].map((item) => defined(item));
    const state = administer(
      adaptiveTest({ pool, selection: onlyT2, stopping: stopped, withinTestlet: "adaptive" }),
      () => 1,
    );
    expect(state.items).toEqual(["t2", "t1", "t3"]);
  });

  it("ends a test only between testlets", () => {
    const pool = [solo, ...lopsided];
    for (const stopping of [testLength(1), anyOf(testLength(1), standardErrorThreshold(0.01))]) {
      const state = administer(adaptiveTest({ pool, stopping }), () => 1);
      expect(state.items).toEqual(["solo"]);
    }
    const first = adaptiveTest({ pool: lopsided, stopping: stopped });
    expect(administer(first, () => 1)).toMatchObject({ items: ["t2", "t3", "t1"], next: null });
    const early = adaptiveTest({ pool: lopsided, stopping: standardErrorThreshold(1.5) });
    expect(early.start(createRandom(1))).toMatchObject({ items: [], next: null });
  });

  it("behaves as without testlets where every testlet has one item", () => {
    const pool = POOL_3PL.map((item, i) => ({ ...item, testlet: `single${i}` }));
    for (const withinTestlet of ["in-order", "adaptive"] as const) {
      const random = createRandom(5);
      const responses = POOL_3PL.map(() => (random.uniform() < 0.5 ? 1 : 0));
      const respond = (id: string): number => defined(responses[poolIndex(POOL_3PL, id)]);
      const alone = administer(adaptiveTest({ pool: POOL_3PL, stopping: testLength(12) }), respond);
      const single = administer(adaptiveTest({ pool, stopping: testLength(12), withinTestlet }), respond);
      expect(single, withinTestlet).toEqual(alone);
    }
  });

  it("withholds all items of a testlet when exposure control withholds its first item", () => {
    const pool = [
      { ...item2pl("t1", 2, 0), testlet: "T", exposure: 0 },
      { ...item2pl("t2", 1.5, 0), testlet: "T" },
      { ...item2pl("t3", 1.5, 0), testlet: "T" },
      item2pl("other", 1, 0),
    ];
    const state = administer(adaptiveTest({ pool, stopping: testLength(4) }), () => 1);
    expect(state).toMatchObject({ items: ["other"], withheld: ["t1", "t2", "t3"], next: null });
  });

  it("never withholds the last eligible testlet", () => {
    const pool = lopsided.map((item) => ({ ...item, exposure: 0 }));
    const state = administer(adaptiveTest({ pool, stopping: testLength(3) }), () => 1);
    expect(state).toMatchObject({ items: ["t2", "t3", "t1"], withheld: [], next: null });
  });

  it("draws exposure for the testlet's first item only", () => {
    // t3 has exposure 0, but only the first item to be administered, t2, is subject to control.
    const pool = lopsided.map((item) => ({ ...item, exposure: item.id === "t3" ? 0 : 1 }));
    const state = administer(adaptiveTest({ pool: [solo, ...pool], stopping: testLength(2) }), () => 1);
    expect(state).toMatchObject({ items: ["solo", "t2", "t3", "t1"], withheld: [] });
    const adaptive = administer(
      adaptiveTest({ pool: [solo, ...pool], stopping: testLength(2), withinTestlet: "adaptive" }),
      () => 1,
    );
    // Adaptively, t1 is first to be administered and has exposure 1.
    expect(adaptive.items[1]).toBe("t1");
    expect(adaptive.withheld).toEqual([]);
  });

  it("administers a testlet at the rate of its first item's exposure parameter", () => {
    const pool = [
      { ...item2pl("t1", 3, 0), testlet: "T", exposure: 0.3 },
      { ...item2pl("t2", 2.8, 0.1), testlet: "T" },
      { ...item2pl("t3", 2.8, -0.1), testlet: "T" },
      ...POOL_3PL.slice(0, 6),
    ];
    const random = createRandom(21);
    const thetas = Array.from({ length: 4000 }, () => random.normal());
    const { persons } = simulateAdaptiveTest(adaptiveTest({ pool, stopping: testLength(3) }), thetas, random);
    let administered = 0;
    for (const { state } of persons) {
      // At theta 0 the testlet is the best unit, so each test selects it first: it is administered whole, or withheld whole.
      const reached = state.items.includes("t1");
      expect(state.items.includes("t2"), "t2").toBe(reached);
      expect(state.items.includes("t3"), "t3").toBe(reached);
      expect(state.withheld.includes("t1"), "withheld").toBe(!reached);
      if (!reached) expect(state.withheld).toEqual(expect.arrayContaining(["t2", "t3"]));
      if (reached) administered++;
    }
    const rate = administered / thetas.length;
    expect(Math.abs(rate - 0.3)).toBeLessThan(4 * Math.sqrt((0.3 * 0.7) / thetas.length));
  });

  it("skips an enemy of an administered item inside the testlet and goes on with the rest", () => {
    const state = administer(adaptiveTest({ pool: lopsided, stopping: stopped, enemies: [["t2", "t3"]] }), () => 1);
    expect(state.items).toEqual(["t2", "t1"]);
  });

  it("skips the enemies of administered items in other testlets, so that a testlet may be smaller", () => {
    const other = [
      { ...item2pl("u1", 0.4, 0), testlet: "U" },
      { ...item2pl("u2", 0.4, 0.5), testlet: "U" },
    ];
    const test = adaptiveTest({ pool: [...lopsided, ...other], stopping: testLength(4), enemies: [["t1", "u1"]] });
    // T, whose mean information is higher, first; u1 is the enemy of t1, so U has u2 only.
    expect(administer(test, () => 1).items).toEqual(["t2", "t3", "t1", "u2"]);
  });

  it("balances content areas between testlets", () => {
    const area = (testlet: string, ids: readonly string[], contentArea: string) =>
      ids.map((id, i) => ({ ...item2pl(id, 1 + i / 10, i / 4), testlet, contentAreas: [contentArea] }));
    const pool = [...area("X", ["x1", "x2", "x3"], "x"), ...area("Y", ["y1", "y2", "y3"], "y")];
    const test = adaptiveTest({ pool, stopping: testLength(4), contentTargets: { x: 1, y: 1 } });
    const firsts = new Set<string>();
    for (const seed of [1, 2, 3, 4, 5, 6]) {
      const { items } = administer(test, () => 1, createRandom(seed));
      expect(items).toHaveLength(6);
      expectTestletBlocks(items, pool, false);
      firsts.add(defined(items[0]).charAt(0));
    }
    expect([...firsts].sort()).toEqual(["x", "y"]);
  });

  it("rejects the testlets of a content balanced test whose items are in several areas", () => {
    const pool = [
      { ...item2pl("a", 1, 0), testlet: "T", contentAreas: ["x"] },
      { ...item2pl("b", 1, 0), testlet: "T", contentAreas: ["y"] },
    ];
    expect(() => adaptiveTest({ pool, stopping: stopped, contentTargets: { x: 1, y: 1 } })).toThrow(
      'The items of testlet "T" are in content areas "x" and "y"; with content targets they must share one area.',
    );
    expect(() => adaptiveTest({ pool, stopping: stopped })).not.toThrow();
  });

  it("resumes from a state stored as JSON at every step of a testlet", () => {
    const pool = [solo, ...lopsided];
    const test = adaptiveTest({ pool, stopping: stopped, withinTestlet: "adaptive" });
    const random = createRandom(3);
    let state = test.start(random);
    const direct = administer(test, (id) => Number(id !== "t2"), createRandom(3));
    while (state.next !== null) {
      const stored = JSON.parse(JSON.stringify(state)) as AdaptiveTestState;
      state = test.answer(stored, Number(stored.next !== "t2"), random);
    }
    expect(state).toEqual(direct);
  });

  it("continues a testlet from its first item when the test resumes from a stored state", () => {
    const test = adaptiveTest({ pool: [solo, ...lopsided], stopping: stopped, withinTestlet: "adaptive" });
    const estimate = { theta: 0, standardError: 1 };
    const stored = { items: [], responses: [], withheld: [], estimate, next: "t1" };
    const state = test.answer(stored, 1, createRandom(1));
    expect(state).toMatchObject({ items: ["t1"], next: "t2" });
  });

  it("estimates with the testlet effects integrated out when the estimator does", () => {
    const pool = [solo, ...lopsided];
    const testlets = [{ items: ["t2", "t3", "t1"], sd: 0.8 }];
    const estimator: AbilityEstimator = (items, responses) => posteriorMeanEstimate(items, responses, { testlets });
    const state = administer(adaptiveTest({ pool, stopping: testLength(2), estimator }), () => 1);
    expect(state.items).toEqual(["solo", "t2", "t3", "t1"]);
    const expected = posteriorMeanEstimate(
      state.items.map((id) => defined(pool[poolIndex(pool, id)])),
      [1, 1, 1, 1],
      { testlets },
    );
    expectClose(state.estimate.theta, expected.theta, 1e-12);
    expectClose(state.estimate.standardError, expected.standardError, 1e-12);
    // The testlet's effect makes the estimate less certain than treating the items as independent.
    expect(state.estimate.standardError).toBeGreaterThan(posteriorMeanEstimate(pool, [1, 1, 1, 1]).standardError);
  });

  it("rejects testlets that are not non-empty strings and unknown orders of the items", () => {
    const poolOf = (testlet: string): PoolItem[] => [{ ...item2pl("a", 1, 0), testlet }];
    expect(() => adaptiveTest({ pool: poolOf(""), stopping: stopped })).toThrow(
      'The testlet of item "a" must be a non-empty string, found "".',
    );
    expect(() => adaptiveTest({ pool: poolOf(JSON.parse("5")), stopping: stopped })).toThrow(
      'The testlet of item "a" must be a non-empty string, found 5.',
    );
    expect(() => adaptiveTest({ pool: poolOf(JSON.parse("null")), stopping: stopped })).toThrow(DataError);
    const shuffled: WithinTestlet = JSON.parse('"shuffled"');
    expect(() => adaptiveTest({ pool: lopsided, stopping: stopped, withinTestlet: shuffled })).toThrow(
      'The option withinTestlet must be "in-order" or "adaptive", found "shuffled".',
    );
  });

  it("administers testlets in multidimensional and diagnostic tests", () => {
    const multidimensional = [
      { ...multidimensionalItem("m1", [0.5, 0.5], 0), testlet: "T" },
      { ...multidimensionalItem("m2", [0.5, 0.5], 0.2), testlet: "T" },
      { ...multidimensionalItem("m3", [0.5, 0.5], -0.2), testlet: "T" },
      multidimensionalItem("m4", [2, 0.1], 0),
      multidimensionalItem("m5", [0.1, 2], 0),
    ];
    const items = (pool: readonly MultidimensionalPoolItem[]): readonly string[] =>
      defined(
        administerMultidimensional(multidimensionalAdaptiveTest({ pool, stopping: testLength(5) }), () => 1).at(-1),
      ).items;
    const together = items(multidimensional);
    expectTestletBlocks(together, multidimensional, true);
    expect(together.slice().sort()).toEqual(["m1", "m2", "m3", "m4", "m5"]);
    expect(together).not.toEqual(items(multidimensional.map(({ testlet, ...item }) => item)));

    const diagnostic = [
      { id: "d1", attributes: ["add"], probabilities: [0.2, 0.9], testlet: "S" },
      { id: "d2", attributes: ["add"], probabilities: [0.3, 0.6], testlet: "S" },
      { id: "d3", attributes: ["subtract"], probabilities: [0.15, 0.85] },
      { id: "d4", attributes: ["add", "subtract"], probabilities: [0.1, 0.4, 0.4, 0.9] },
    ];
    const diagnosed = (pool: readonly DiagnosticPoolItem[]): readonly string[] =>
      defined(
        administerDiagnostic(diagnosticAdaptiveTest({ pool, attributes: SKILLS, stopping: testLength(4) }), () => 1).at(
          -1,
        ),
      ).items;
    const grouped = diagnosed(diagnostic);
    expectTestletBlocks(grouped, diagnostic, true);
    expect(grouped).toHaveLength(4);
    expect(grouped).not.toEqual(diagnosed(diagnostic.map(({ testlet, ...item }) => item)));
  });
});

describe("untilMixed", () => {
  it("uses the initial estimator until one response is above its lowest category and one below its highest", () => {
    const calls: string[] = [];
    const tagged =
      (name: string): AbilityEstimator =>
      (items, responses) => {
        calls.push(name);
        return posteriorMeanEstimate(items, responses);
      };
    const estimate = untilMixed(tagged("initial"), tagged("mixed"));
    const gpcm = defined(POOL_GPCM.find((item) => item.model.categoryCount(item.parameters) === 4));
    const dichotomous = item2pl("d", 1, 0);
    const cases: [readonly PoolItem[], readonly (number | null)[], string][] = [
      [[], [], "initial"],
      [[dichotomous, dichotomous], [1, 1], "initial"],
      [[dichotomous, dichotomous], [0, null], "initial"],
      [[dichotomous, dichotomous], [0, 1], "mixed"],
      [[gpcm, dichotomous], [3, 1], "initial"],
      [[gpcm, dichotomous], [0, 0], "initial"],
      [[gpcm], [1], "mixed"],
      [[gpcm, dichotomous], [3, 0], "mixed"],
    ];
    for (const [items, responses, expected] of cases) {
      calls.length = 0;
      expect(estimate(items, responses)).toEqual(posteriorMeanEstimate(items, responses));
      expect(calls, JSON.stringify(responses)).toEqual([expected]);
    }
  });

  it("rejects an estimator that is not a function", () => {
    expect(() => untilMixed(posteriorMeanEstimate, JSON.parse("null"))).toThrow(
      "The mixed estimator must be a function of items and responses, found null.",
    );
  });
});

describe("calibrateExposure", () => {
  const thetas = (seed: number, count: number): number[] => {
    const random = createRandom(seed);
    return Array.from({ length: count }, () => random.normal());
  };

  it("brings every exposure rate near or below the largest rate allowed", () => {
    // With 1,500 persons a rate at 0.3 varies by 0.012; the largest of the rates near it by about 0.03.
    const test = { pool: POOL_3PL.slice(0, 40), stopping: testLength(5) };
    const options = { maxRate: 0.3, tolerance: 0.04 };
    const result = calibrateExposure(test, thetas(5, 1500), createRandom(5), options);
    expect(result.converged).toBe(true);
    expect(result.iterations).toBeGreaterThan(1);
    for (const { rate } of result.exposureRates) expect(rate).toBeLessThanOrEqual(0.34);
    const parameters = result.pool.map((item) => defined(item.exposure));
    expect(Math.min(...parameters)).toBeLessThan(0.5);
    expect(parameters.filter((k) => k === 1).length).toBeGreaterThan(20);
    expect(calibrateExposure(test, thetas(5, 1500), createRandom(5), options)).toEqual(result);
  });

  it("brings the rate of a testlet near or below the largest rate allowed, the same for each of its items", () => {
    const pool = POOL_3PL.slice(0, 24).map((item, i) =>
      i < 12 ? { ...item, testlet: `set${Math.floor(i / 3)}` } : item,
    );
    const result = calibrateExposure({ pool, stopping: testLength(8) }, thetas(9, 1500), createRandom(9), {
      maxRate: 0.4,
      tolerance: 0.04,
    });
    expect(result.converged).toBe(true);
    const rates = new Map(result.exposureRates.map(({ id, rate }) => [id, rate]));
    for (let set = 0; set < 4; set++) {
      const own = result.pool.filter((item) => item.testlet === `set${set}`).map((item) => defined(rates.get(item.id)));
      expect(new Set(own).size).toBe(1);
      expect(own[0]).toBeLessThanOrEqual(0.44);
    }
  });

  it("keeps every parameter at 1 when no rate exceeds the largest allowed", () => {
    const pool = POOL_3PL.slice(0, 20).map((item) => ({ ...item, exposure: 0.5 }));
    const result = calibrateExposure({ pool, stopping: testLength(5) }, thetas(6, 100), createRandom(6), {
      maxRate: 1,
    });
    expect(result).toMatchObject({ converged: true, iterations: 1 });
    expect(result.pool.every((item) => item.exposure === 1)).toBe(true);
  });

  it("reports non-convergence when the test is too long for the rate", () => {
    const pool = POOL_3PL.slice(0, 10);
    const result = calibrateExposure({ pool, stopping: testLength(5) }, thetas(7, 100), createRandom(7), {
      maxRate: 0.2,
      maxIterations: 3,
    });
    expect(result).toMatchObject({ converged: false, iterations: 3 });
  });

  it("rejects an invalid largest rate or stopping rule", () => {
    const test = { pool: POOL_3PL.slice(0, 5), stopping: testLength(2) };
    expect(() => calibrateExposure(test, [0], createRandom(1), { maxRate: 0 })).toThrow(
      "The option maxRate must be a number above 0 and at most 1, found 0.",
    );
    expect(() => calibrateExposure(test, [0], createRandom(1), { maxRate: 0.5, maxIterations: 0 })).toThrow(DataError);
    expect(() => calibrateExposure(test, [], createRandom(1), { maxRate: 0.5 })).toThrow(DataError);
    expect(() => calibrateExposure(test, [0], createRandom(1), JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => calibrateExposure(test, [0], JSON.parse("null"), { maxRate: 0.5 })).toThrow(
      "A random number generator must be one from createRandom, found null.",
    );
    expect(() =>
      calibrateExposure({ ...test, pool: JSON.parse("null") }, [0], createRandom(1), { maxRate: 0.5 }),
    ).toThrow("The items must be an array, found null.");
  });
});

describe("assembleForms", () => {
  const random = createRandom(20261029);
  const areas = ["algebra", "geometry", "number"];
  const pool: PoolItem[] = simulateItems(twoParameterLogistic, 300, random).map((item, i) => ({
    ...item,
    id: `q${i + 1}`,
    contentAreas: [defined(areas[i % 3])],
  }));
  /** The one form of a call that asks for one. */
  const first = (options: FormAssemblyOptions): AssembledForm => defined(assembleForms(options)[0]);
  const idsOf = (form: AssembledForm): string[] => form.items.map((item) => item.id);
  /** A pool of 2PL items whose information falls with their rank, alone or in areas. */
  const ranked = (count: number, extra: (rank: number) => Partial<PoolItem> = () => ({})): PoolItem[] =>
    Array.from({ length: count }, (_, rank) => ({ ...item2pl(`r${rank}`, 2 - rank / 40, 0), ...extra(rank) }));

  describe("replays", () => {
    for (const c of catForms.replays) {
      it(`takes the items of a replay in base R of the case ${c.name}`, () => {
        const items: PoolItem[] = c.items.map((item) => ({
          id: item.id,
          model: threeParameterLogistic,
          parameters: { a: item.a, b: item.b, c: item.c, d: item.d },
          contentAreas: item.contentAreas,
          ...("testlet" in item && { testlet: item.testlet }),
        }));
        const forms = assembleForms({
          pool: items,
          length: c.length,
          content: c.constraints,
          target: c.target,
          enemies: c.enemies,
          forms: c.forms,
          overlap: c.overlap,
        });
        expect(forms).toHaveLength(c.expected.length);
        c.expected.forEach((expected, f) => {
          const form = defined(forms[f]);
          expect(idsOf(form), `${c.name}, form ${f + 1}`).toEqual(expected.items);
          c.constraints.forEach(({ area }, k) => {
            expect(form.contentCounts[area], `${c.name}, form ${f + 1}, ${area}`).toBe(expected.counts[k]);
          });
          c.target.forEach((_, t) => {
            expectClose(
              defined(form.information[t]),
              defined(expected.information[t]),
              REPLAY_TOLERANCE * 100,
              `${c.name}, form ${f + 1}`,
            );
          });
        });
      });
    }
  });

  describe("on blueprints that a solver proves feasible", () => {
    const blueprints = catForms.pools.flatMap(({ name, items, blueprints: list }) =>
      list.map((blueprint, index) => ({ name, items, blueprint, index })),
    );
    it("meets every bound of 340 forms on 85 blueprints, with all but 3 of them within 5% of the target", () => {
      expect(blueprints).toHaveLength(85);
      let formsMet = 0;
      const gaps: number[] = [];
      for (const { name, items, blueprint, index } of blueprints) {
        const bank: PoolItem[] = items.map(({ id, a, b, c, d, contentAreas }) => ({
          id,
          model: threeParameterLogistic,
          parameters: { a, b, c, d },
          contentAreas,
        }));
        const forms = assembleForms({
          pool: bank,
          length: catForms.length,
          content: blueprint.constraints,
          target: blueprint.target,
          forms: catForms.forms,
        });
        expect(forms, `${name}, blueprint ${index + 1}`).toHaveLength(catForms.forms);
        forms.forEach((form, f) => {
          const label = `${name}, blueprint ${index + 1}, form ${f + 1}`;
          expect(form.items, label).toHaveLength(catForms.length);
          for (const { area, minimum, maximum } of blueprint.constraints) {
            expect(form.contentCounts[area], `${label}, ${area}`).toBeGreaterThanOrEqual(minimum);
            expect(form.contentCounts[area], `${label}, ${area}`).toBeLessThanOrEqual(maximum);
          }
          gaps.push(
            Math.max(
              ...blueprint.target.map(
                ({ information }, t) => Math.abs(defined(form.information[t]) - information) / information,
              ),
            ),
          );
          formsMet++;
        });
        expect(new Set(forms.flatMap(idsOf)).size, `${name}, blueprint ${index + 1}`).toBe(
          catForms.forms * catForms.length,
        );
      }
      expect(formsMet).toBe(340);
      // The rotation is a heuristic: the gaps have a median of 1.4% and a tail; a solver's minimax gaps are 0.2% to 0.3%.
      expect(gaps.filter((gap) => gap >= 0.05).length).toBeLessThanOrEqual(3);
      expect(Math.max(...gaps)).toBeLessThan(0.08);
    });
  });

  it("meets the content constraints and comes near the information target", () => {
    const content = [
      { area: "algebra", minimum: 12, maximum: 14 },
      { area: "geometry", minimum: 8, maximum: 10 },
      { area: "number", minimum: 8, maximum: 10 },
    ];
    const target = [
      { theta: -1, information: 6 },
      { theta: 0, information: 8 },
      { theta: 1, information: 6 },
    ];
    const form = first({ pool, length: 30, content, target });
    expect(new Set(form.items.map((item) => item.id)).size).toBe(30);
    for (const { area, minimum, maximum } of content) {
      const count = form.items.filter((item) => item.contentAreas?.includes(area)).length;
      expect(form.contentCounts[area]).toBe(count);
      expect(count).toBeGreaterThanOrEqual(minimum);
      expect(count).toBeLessThanOrEqual(maximum);
    }
    target.forEach(({ theta, information }, t) => {
      expectClose(defined(form.information[t]), testInformation(form.items, theta), 1e-12);
      expect(Math.abs(defined(form.information[t]) - information) / information).toBeLessThan(0.05);
    });
  });

  it("takes the most informative items for a target above any form's information", () => {
    const form = first({ pool, length: 10, target: [{ theta: 0.5, information: 1000 }] });
    const sorted = [...pool].sort((x, y) => itemInformation(y, 0.5) - itemInformation(x, 0.5));
    expect(new Set(form.items.map((item) => item.id))).toEqual(new Set(sorted.slice(0, 10).map((item) => item.id)));
  });

  it("assembles the whole pool and is deterministic", () => {
    const small = pool.slice(0, 6);
    const form = first({ pool: small, length: 6, content: [{ area: "algebra", minimum: 2, maximum: 2 }] });
    expect(new Set(form.items)).toEqual(new Set(small));
    const options = { pool, length: 12, target: [{ theta: 0, information: 3 }], forms: 2 };
    expect(assembleForms(options)).toEqual(assembleForms(options));
  });

  describe("forms in rotation", () => {
    const items = ranked(12);
    const target = [{ theta: 0, information: 1000 }];

    it("lets the forms choose in order, then in reverse, so that none always chooses first", () => {
      const [one, two] = assembleForms({ pool: items, length: 6, target, forms: 2 });
      expect(idsOf(defined(one))).toEqual(["r0", "r3", "r4", "r7", "r8", "r11"]);
      expect(idsOf(defined(two))).toEqual(["r1", "r2", "r5", "r6", "r9", "r10"]);
    });

    it("gives forms with no item in common unless overlap allows it", () => {
      const forms = assembleForms({ pool: items, length: 4, target, forms: 3 });
      expect(new Set(forms.flatMap(idsOf)).size).toBe(12);
      const shared = assembleForms({ pool: items, length: 8, target, forms: 2, overlap: 4 });
      const [x, y] = shared.map(idsOf);
      expect(defined(x).filter((id) => defined(y).includes(id)).length).toBeLessThanOrEqual(4);
      expect(shared.every((form) => form.items.length === 8)).toBe(true);
    });

    it("lets forms share every item where overlap is the length", () => {
      const [one, two] = assembleForms({ pool: items.slice(0, 5), length: 5, target, forms: 2, overlap: 5 });
      expect(new Set(idsOf(defined(one)))).toEqual(new Set(idsOf(defined(two))));
    });

    it("leaves a form short where the pool, the enemies or overlap leave it no unit", () => {
      const forms = assembleForms({ pool: items, length: 8, target, forms: 2 });
      expect(forms.map((form) => form.items.length)).toEqual([6, 6]);
      expect(forms.map((form) => form.information.length)).toEqual([1, 1]);
      const enemies = [["r0", "r1", "r2", "r3", "r4"]];
      const [only] = assembleForms({ pool: items.slice(0, 6), length: 6, target, enemies });
      expect(defined(only).items).toHaveLength(2);
    });

    it("keeps the enemies of a set on different forms or apart on one", () => {
      const enemies = [
        ["r0", "r1"],
        ["r2", "r3", "r4"],
      ];
      const forms = assembleForms({ pool: items, length: 6, target, enemies, forms: 2 });
      for (const form of forms) {
        const ids = idsOf(form);
        for (const set of enemies) expect(set.filter((id) => ids.includes(id)).length).toBeLessThanOrEqual(1);
      }
      expect(idsOf(defined(forms[0]))).toContain("r0");
      expect(idsOf(defined(forms[1]))).toContain("r1");
    });

    it("takes a testlet whole or not at all, and skips one that does not fit", () => {
      const withTestlets = ranked(12, (rank) =>
        rank >= 1 && rank <= 3 ? { testlet: "T" } : rank >= 8 && rank <= 10 ? { testlet: "U" } : {},
      );
      const forms = assembleForms({ pool: withTestlets, length: 5, target, forms: 2 });
      for (const form of forms) {
        const ids = idsOf(form);
        for (const set of [
          ["r1", "r2", "r3"],
          ["r8", "r9", "r10"],
        ]) {
          expect(set.filter((id) => ids.includes(id)).length % 3).toBe(0);
        }
      }
      // Two places left cannot take a testlet of three.
      const [fits] = assembleForms({ pool: withTestlets, length: 2, target });
      expect(defined(fits).items.every((item) => item.testlet === undefined)).toBe(true);
      const [whole] = assembleForms({ pool: withTestlets, length: 3, target });
      expect([0, 3]).toContain(idsOf(defined(whole)).filter((id) => ["r1", "r2", "r3"].includes(id)).length);
    });

    it("never takes a testlet whose items are enemies", () => {
      const withTestlet = ranked(8, (rank) => (rank < 3 ? { testlet: "T" } : {}));
      const [form] = assembleForms({ pool: withTestlet, length: 4, target, enemies: [["r0", "r2"]] });
      expect(idsOf(defined(form)).filter((id) => ["r0", "r1", "r2"].includes(id))).toEqual([]);
    });

    it("settles ties by a random order of the pool, the same for a seed", () => {
      const twins = Array.from({ length: 20 }, (_, i) => item2pl(`t${i}`, 1, 0));
      const options = { pool: twins, length: 5, target: [{ theta: 0, information: 1 }] };
      expect(idsOf(first(options))).toEqual(["t0", "t1", "t2", "t3", "t4"]);
      const seeded = idsOf(first({ ...options, random: createRandom(5) }));
      expect(seeded).toEqual(idsOf(first({ ...options, random: createRandom(5) })));
      expect(seeded).not.toEqual(["t0", "t1", "t2", "t3", "t4"]);
      expect(idsOf(first({ ...options, random: createRandom(6) }))).not.toEqual(seeded);
    });

    it("leaves its inputs unchanged and returns items of the pool", () => {
      const copy = items.map((item) => ({ ...item }));
      const forms = assembleForms({ pool: items, length: 4, target, forms: 2 });
      expect(items).toEqual(copy);
      for (const item of forms.flatMap((form) => form.items)) expect(items).toContain(item);
    });
  });

  it("rejects options of other types and invalid ones", () => {
    const base = { pool, length: 5 };
    expect(() => assembleForms({ ...base, length: 301 })).toThrow(
      "The option length must be an integer from 1 to the pool size 300, found 301.",
    );
    expect(() => assembleForms({ ...base, content: [{ area: "algebra", minimum: 3, maximum: 2 }] })).toThrow(
      'The constraint on "algebra" needs the minimum at most the maximum, found 3 and 2.',
    );
    expect(() => assembleForms({ ...base, content: [{ area: "algebra", minimum: 1, maximum: 2, weight: 0 }] })).toThrow(
      'The weight of the constraint on "algebra" must be a positive number, found 0.',
    );
    expect(() => assembleForms({ ...base, target: [{ theta: 0, information: Number.NaN }] })).toThrow(
      "The information of the target at theta 0 must be a finite number, found NaN.",
    );
    expect(() => assembleForms({ ...base, target: [{ theta: Infinity, information: 1 }] })).toThrow(DataError);
    expect(() => assembleForms(JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => assembleForms({ ...base, content: JSON.parse("{}") })).toThrow(
      "The option content must be an array of content constraints, found an object.",
    );
    expect(() => assembleForms({ ...base, target: JSON.parse("[null]") })).toThrow(
      "An information target must be an object { theta, information }, found null.",
    );
    for (const forms of [0, 1.5, -1, Number.NaN]) {
      expect(() => assembleForms({ ...base, forms })).toThrow(
        `The option forms must be a positive integer, found ${forms}.`,
      );
    }
    expect(() => assembleForms({ ...base, forms: JSON.parse("null") })).toThrow(
      "The option forms must be a positive integer, found null.",
    );
    expect(() => assembleForms({ ...base, forms: JSON.parse('"2"') })).toThrow(
      'The option forms must be a positive integer, found "2".',
    );
    for (const overlap of [-1, 0.5, Number.NaN]) {
      expect(() => assembleForms({ ...base, overlap })).toThrow(
        `The option overlap must be an integer from 0, found ${overlap}.`,
      );
    }
    expect(() => assembleForms({ ...base, overlap: JSON.parse("null") })).toThrow(
      "The option overlap must be an integer from 0, found null.",
    );
    expect(() => assembleForms({ ...base, enemies: JSON.parse("{}") })).toThrow(
      "The option enemies must be an array of enemy sets, found an object.",
    );
    expect(() => assembleForms({ ...base, enemies: [["q1"]] })).toThrow("Enemy set 1 needs at least two item ids.");
    expect(() => assembleForms({ ...base, enemies: [["q1", "zz"]] })).toThrow(
      'Enemy set 1 names "zz", which is not a pool item.',
    );
    expect(() => assembleForms({ ...base, enemies: [["q1", "q1"]] })).toThrow("Enemy set 1 names an item twice.");
    expect(() => assembleForms({ ...base, random: JSON.parse("null") })).toThrow(
      "A random number generator must be one from createRandom, found null.",
    );
    expect(() => assembleForms({ ...base, random: JSON.parse("{}") })).toThrow(
      "A random number generator must be one from createRandom, found an object.",
    );
    const text = [{ ...defined(pool[0]), contentAreas: JSON.parse('"algebra"') }, ...pool.slice(1)];
    expect(() =>
      assembleForms({ pool: text, length: 5, content: [{ area: "algebra", minimum: 1, maximum: 2 }] }),
    ).toThrow('The contentAreas of item "q1" must be an array of distinct non-empty strings, found "algebra".');
  });
});

describe("kullbackLeibler", () => {
  for (const { source, pool, runs } of FIXTURES) {
    for (const run of runs.filter(({ selection }) => selection === "KL")) {
      it(`matches catR's KL of every eligible item at each step, scaled by the largest likelihood: ${source}`, () => {
        expectScores(pool, run, kullbackLeibler);
      });
    }
  }
});

/** A module of 2PL items with slope 1 and these difficulties, with the cut points when given. */
const module = (id: string, difficulties: readonly number[], cuts?: readonly number[]): MultistageModule => ({
  id,
  items: difficulties.map((b, j) => item2pl(`${id}-${j + 1}`, 1, b)),
  ...(cuts && { cuts }),
});

describe("multistageTest", () => {
  const stages = [
    [module("routing", [-1, -0.3, 0.3, 1], [2, 3])],
    [
      module("easy", [-1.5, -1, -0.5], [3, 5]),
      module("medium", [-0.5, 0, 0.5], [4, 6]),
      module("hard", [0.5, 1, 1.5], [5, 7]),
    ],
    [module("easy-final", [-1.2, -0.8]), module("medium-final", [-0.2, 0.2]), module("hard-final", [0.8, 1.2])],
  ];

  function route(test: MultistageTest, answers: readonly (readonly number[])[]): string[] {
    let state = test.start();
    for (const responses of answers) state = test.answer(state, responses);
    expect(state.next).toBeNull();
    return [...state.modules];
  }

  it("routes by the total score so far, a score at a cut to the higher module", () => {
    const test = multistageTest({ stages, routing: "number-correct" });
    // Score 1, below 2: easy; then 1 + 3 = 4, from 3 to below 5: the middle module.
    expect(
      route(test, [
        [1, 0, 0, 0],
        [1, 1, 1],
        [0, 1],
      ]),
    ).toEqual(["routing", "easy", "medium-final"]);
    // Score 2, at the first cut: medium; then 2 + 3 = 5, from 4 to below 6.
    expect(
      route(test, [
        [1, 1, 0, 0],
        [1, 1, 1],
        [1, 1],
      ]),
    ).toEqual(["routing", "medium", "medium-final"]);
    // Score 3: hard; then 3 + 0 = 3, below 5.
    expect(
      route(test, [
        [1, 1, 1, 0],
        [0, 0, 0],
        [0, 0],
      ]),
    ).toEqual(["routing", "hard", "easy-final"]);
    expect(
      route(test, [
        [1, 1, 1, 1],
        [1, 1, 1],
        [1, 1],
      ]),
    ).toEqual(["routing", "hard", "hard-final"]);
  });

  it("routes by the estimate of theta, an estimate at a cut to the higher module", () => {
    const routing = module("routing", [-1, 0, 1], [0]);
    const second = [module("low", [-1, -0.5]), module("high", [0.5, 1])];
    const test = multistageTest({ stages: [[routing], second], routing: "theta" });
    // Equal slopes make the estimate depend on the score only, and the difficulties are symmetric around 0.
    expect(
      route(test, [
        [1, 1, 0],
        [1, 0],
      ]),
    ).toEqual(["routing", "high"]);
    expect(
      route(test, [
        [0, 1, 0],
        [1, 0],
      ]),
    ).toEqual(["routing", "low"]);
    const { theta } = posteriorMeanEstimate(routing.items, [0, 1, 0]);
    for (const [cut, expected] of [
      [theta, "high"],
      [theta + 1e-9, "low"],
    ] as const) {
      const cutAt = multistageTest({ stages: [[{ ...routing, cuts: [cut] }], second], routing: "theta" });
      expect(
        route(cutAt, [
          [0, 1, 0],
          [1, 0],
        ])[1],
      ).toBe(expected);
    }
  });

  it("estimates theta with its estimator from all responses so far", () => {
    const estimator = untilMixed(posteriorMeanEstimate, weightedLikelihoodEstimate);
    const test = multistageTest({ stages, routing: "number-correct", estimator });
    const start = test.start();
    expect(start).toEqual({
      modules: [],
      items: [],
      responses: [],
      estimate: posteriorMeanEstimate([], []),
      next: "routing",
    });
    const first = test.answer(start, [1, 1, 1, 1]);
    const routingItems = defined(stages[0]?.[0]).items;
    expect(first).toMatchObject({ items: routingItems.map(({ id }) => id), next: "hard" });
    expect(first.estimate).toEqual(posteriorMeanEstimate(routingItems, [1, 1, 1, 1]));
    const items = [...routingItems, ...defined(stages[1]?.[2]).items];
    expect(test.answer(first, [1, 0, 1]).estimate).toEqual(weightedLikelihoodEstimate(items, [1, 1, 1, 1, 1, 0, 1]));
  });

  it("resumes from a state stored as JSON", () => {
    const test = multistageTest({ stages, routing: "theta" });
    const state = test.answer(test.start(), [1, 1, 0, 1]);
    expect(test.answer(JSON.parse(JSON.stringify(state)), [0, 1, 1])).toEqual(test.answer(state, [0, 1, 1]));
  });

  it("administers a panel of one module", () => {
    const test = multistageTest({ stages: [[module("only", [0])]], routing: "theta" });
    expect(route(test, [[1]])).toEqual(["only"]);
  });

  it("rejects invalid panels", () => {
    const routing = "number-correct";
    expect(() => multistageTest({ stages: [], routing })).toThrow(DataError);
    expect(() => multistageTest({ stages: [[module("a", [0])], []], routing })).toThrow(DataError);
    expect(() => multistageTest({ stages: [[module("a", [0]), module("b", [0])]], routing })).toThrow(
      "The first stage of a multistage test has one module, found 2.",
    );
    expect(() => multistageTest({ stages: [[module("a", [0], [])], [module("a", [1])]], routing })).toThrow(
      'Module ids must be unique; "a" appears twice.',
    );
    expect(() => multistageTest({ stages: [[module("a", [])]], routing })).toThrow(
      'Module "a" needs an array of at least one item.',
    );
    const three = [module("b", [1]), module("c", [2]), module("d", [3])];
    expect(() => multistageTest({ stages: [[module("a", [0], [2, 1])], three], routing })).toThrow(
      'Module "a" needs an array of 2 cut points in ascending order to route to the next stage, found [2, 1].',
    );
    expect(() => multistageTest({ stages: [[module("a", [0], [1])]], routing })).toThrow(DataError);
    const shared = { id: "b", items: module("a", [1]).items };
    expect(() => multistageTest({ stages: [[module("a", [0], [])], [shared]], routing })).toThrow(
      'Item ids must be unique; "a-1" appears twice.',
    );
    expect(() => multistageTest({ stages, routing: JSON.parse('"score"') })).toThrow(
      'The option routing must be "number-correct" or "theta", found "score".',
    );
    expect(() => multistageTest({ stages, routing, estimator: JSON.parse("5") })).toThrow(
      "The option estimator must be a function of items and responses, such as posteriorMeanEstimate, found 5.",
    );
  });

  it("rejects panels and responses of other types from JavaScript", () => {
    const routing = "number-correct";
    expect(() => multistageTest(JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => multistageTest({ stages: JSON.parse("null"), routing })).toThrow(
      "The option stages must be an array of stages, found null.",
    );
    expect(() => multistageTest({ stages: JSON.parse("[[null]]"), routing })).toThrow(
      "The modules of stage 1 must be objects { id, items, cuts }, found null.",
    );
    expect(() =>
      multistageTest({ stages: [[{ ...module("a", [0]), cuts: JSON.parse('"1"') }], [module("b", [1])]], routing }),
    ).toThrow('Module "a" needs an array of 0 cut points');
    const test = multistageTest({ stages, routing });
    expect(() => test.answer(test.start(), JSON.parse("null"))).toThrow("The responses must be an array, found null.");
  });

  it("rejects responses that do not match the module and answers after the end", () => {
    const test = multistageTest({ stages, routing: "number-correct" });
    const start = test.start();
    expect(() => test.answer(start, [1, 0])).toThrow('Module "routing" has 4 items, found 2 responses.');
    expect(() => test.answer(start, [1, 0, 2, 0])).toThrow(
      'The response to item "routing-3" must be an integer from 0 to 1, found 2.',
    );
    expect(() => test.answer({ ...start, next: "missing" }, [1])).toThrow('The panel has no module "missing".');
    expect(() => test.answer({ ...start, next: null }, [1])).toThrow("The test has ended; it takes no more responses.");
  });

  it("rejects stored states that skip or repeat a stage, or whose items are not their modules'", () => {
    const test = multistageTest({ stages, routing: "number-correct" });
    const second = test.answer(test.start(), [1, 1, 0, 0]);
    const skipped = "A multistage test state takes one module per stage, in order, with their items; found modules";
    expect(() => test.answer({ ...test.start(), next: "medium" }, [1, 1, 1])).toThrow(
      `${skipped} [] and then "medium".`,
    );
    expect(() => test.answer({ ...second, next: "routing" }, [1, 1, 0, 0])).toThrow(skipped);
    expect(() => test.answer({ ...second, items: [...second.items].reverse() }, [1, 1, 1])).toThrow(skipped);
    expect(() => test.answer({ ...second, modules: JSON.parse("null") }, [1, 1, 1])).toThrow(
      "A multistage test state needs the array modules.",
    );
    expect(() => test.answer(JSON.parse("{}"), [1, 1, 1, 1])).toThrow(
      "A test state needs the arrays items and responses.",
    );
  });
});

describe("multistageCuts", () => {
  const items = new Map<string, TestItem>(
    cutsFixture.items.map(({ id, model, parameters }) => [id, { id, model: defined(MODELS[model]), parameters }]),
  );
  const fixtureStages = (panel: (typeof cutsFixture.panels)[number]): MultistageModule[][] =>
    panel.stages.map((modules) =>
      modules.map(({ id, items: names }) => ({ id, items: names.map((name) => defined(items.get(name))) })),
    );
  // Cuts are thetas or whole scores, so the tolerance holds both.
  const expectCuts = (actual: readonly (readonly number[])[], expected: readonly (readonly number[])[]): void => {
    expect(actual).toHaveLength(expected.length);
    expected.forEach((cuts, m) => {
      const found = defined(actual[m]);
      expect(found, `module ${m}`).toHaveLength(cuts.length);
      cuts.forEach((cut, j) => {
        expectClose(found[j] ?? null, cut, CUTS_TOLERANCE, `module ${m}, cut ${j}`);
      });
    });
  };
  // Modules that translate each other by 1: their information curves are symmetric and cross at the midpoints of the
  // centers, -0.5 and 0.5.
  const translates = [
    [module("routing", [-1, -0.3, 0.3, 1])],
    [module("easy", [-1.5, -1, -0.5]), module("medium", [-0.5, 0, 0.5]), module("hard", [0.5, 1, 1.5])],
    [module("last", [0])],
  ];
  const shared = (shares: readonly number[], options: MultistageCutsOptions = {}) =>
    multistageCuts([[module("routing", [0])], shares.map((_, j) => module(`next${j}`, [j]))], {
      method: "routing-shares",
      shares: [shares],
      ...options,
    });

  for (const panel of cutsFixture.panels) {
    for (const reference of panel.cases) {
      it(`matches the cuts of base R: ${reference.description}, panel ${panel.name}`, () => {
        const options: MultistageCutsOptions = JSON.parse(JSON.stringify(reference.options));
        expectCuts(multistageCuts(fixtureStages(panel), options), reference.cuts);
      });
    }
  }

  it("takes the crossing between the peaks of two modules whose curves also cross in a tail", () => {
    const [tail, between] = cutsFixture.tailCrossings;
    const tails = defined(cutsFixture.panels.find(({ name }) => name === "tails"));
    expect(cutsFixture.tailCrossings).toHaveLength(2);
    const routing = defined(multistageCuts(fixtureStages(tails))[0]);
    expectClose(routing[0] ?? null, defined(between), CUTS_TOLERANCE);
    expect(Math.abs((routing[0] ?? 0) - defined(tail))).toBeGreaterThan(1);
  });

  it("places the cuts of a fine population at the quantiles of the density it discretizes", () => {
    const { population, shares, density } = cutsFixture.discretized;
    const cuts = defined(shared(shares, { population })[0]);
    density.forEach((quantile, j) => {
      expectClose(cuts[j] ?? null, quantile, DISCRETIZATION_TOLERANCE, `cut ${j}`);
    });
  });

  describe("by maximum information", () => {
    it("puts the cuts of modules that translate each other at the midpoints of their centers", () => {
      expectCuts(multistageCuts(translates), [[-0.5, 0.5], [], [], []]);
    });

    it("puts the cut between two single items of opposite difficulty at 0", () => {
      const stages = [[module("routing", [0])], [module("easy", [-1]), module("hard", [1])]];
      expectCuts(multistageCuts(stages), [[0]]);
    });

    it("searches the crossing within the range", () => {
      expectCuts(multistageCuts(translates, { range: [-2, 2] }), [[-0.5, 0.5], [], [], []]);
      const far = [[module("routing", [0])], [module("easy", [7]), module("hard", [9])]];
      expectCuts(multistageCuts(far, { range: [0, 12] }), [[8]]);
      expect(() => multistageCuts(far)).toThrow('The information of modules "easy" and "hard" of stage 2');
    });

    it("has no cuts into a stage of one module, and none for a panel of one stage", () => {
      const single = [[module("routing", [0])], [module("only", [0])]];
      expect(multistageCuts(single)).toEqual([[]]);
      expect(multistageCuts([[module("only", [0])]])).toEqual([]);
    });

    it("ignores the cuts that the modules carry", () => {
      const carrying = [
        [{ ...module("routing", [0]), cuts: JSON.parse('"cuts"') }],
        [module("easy", [-1]), module("hard", [1])],
      ];
      expectCuts(multistageCuts(carrying), [[0]]);
    });

    it("refuses modules of a stage that are not from the easiest to the hardest", () => {
      const reversed = [[module("routing", [0])], [module("hard", [1]), module("easy", [-1])]];
      expect(() => multistageCuts(reversed)).toThrow(
        'The information of modules "hard" and "easy" of stage 2 peaks at theta 1 and -1; a stage needs its modules from the easiest to the hardest',
      );
      const same = [[module("routing", [0])], [module("a", [0]), module("b", [0])]];
      expect(() => multistageCuts(same)).toThrow(DataError);
    });

    it("refuses modules whose information does not cross between their peaks", () => {
      const dominated = [[module("routing", [0])], [module("easy", [-1.5]), module("hard", [0, 0, 0, 0, 0])]];
      expect(() => multistageCuts(dominated)).toThrow(
        'The information of modules "easy" and "hard" of stage 2 does not cross between their peaks at theta -1.5 and 0',
      );
    });
  });

  describe("by routing shares", () => {
    it("sends equal shares of the standard normal population to the modules by default", () => {
      const equal = multistageCuts(translates, { method: "routing-shares" });
      expect(equal[0]?.map((cut) => cut.toFixed(10))).toEqual(["-0.4307272993", "0.4307272993"]);
      expect(equal[1]).toEqual([]);
    });

    it("divides shares by their sum, which may be off 1 by a rounding", () => {
      // The share below the cut is 0.5 / 1.0000004, a little under a half: the cut is a little below 0.
      const [cut] = defined(shared([0.5, 0.5000004])[0]);
      expect(cut).toBeLessThan(0);
      expect(Math.abs(cut ?? 0)).toBeLessThan(1e-6);
    });

    it("puts a cut at the edge between two cells when the shares are the weights of the nodes", () => {
      const population = { nodes: [0, 1, 2], weights: [0.25, 0.5, 0.25] };
      expectCuts(shared([0.25, 0.5, 0.25], { population }), [[0.5, 1.5]]);
      expectCuts(shared([0.5, 0.5], { population }), [[1]]);
      expectCuts(shared([0.125, 0.875], { population }), [[0]]);
    });

    it("puts every cut of a population of one node at it", () => {
      expectCuts(shared([0.2, 0.3, 0.5], { population: { nodes: [0.5], weights: [1] } }), [[0.5, 0.5]]);
    });

    it("sends a population of nodes to the modules in the shares", () => {
      const population = { nodes: [-1, 0, 2, 3], weights: [0.1, 0.4, 0.3, 0.2] };
      const [cuts] = shared([0.3, 0.4, 0.3], { population });
      // Cells: [-1.5, -0.5], [-0.5, 1], [1, 2.5], [2.5, 3.5]; 0.3 is 0.2 into the second cell, 0.7 is 0.2 into the third.
      expectCuts([defined(cuts)], [[-0.5 + (0.2 / 0.4) * 1.5, 1 + (0.2 / 0.3) * 1.5]]);
    });
  });

  describe("on the scale number-correct", () => {
    it("lifts a cut at a whole expected score to that score, not the next", () => {
      // The expected score of the routing module at theta 0 is 2: p(-1) + p(1) and p(-0.3) + p(0.3) are 1 each.
      const stages = [[module("routing", [-1, -0.3, 0.3, 1])], [module("low", [-1]), module("high", [1])]];
      expect(multistageCuts(stages, { method: "routing-shares", scale: "number-correct" })).toEqual([[2]]);
    });

    it("counts the items of the modules that routing by theta takes before the routing module", () => {
      const stages = [
        [module("routing", [0])],
        [module("low", [-1, -1]), module("high", [1, 1])],
        [module("one", [0]), module("two", [0])],
        [module("a", [0]), module("b", [0])],
      ];
      // Every cut into a stage is theta 0, which routing sends to "high". The routing module scores 1 / 2 there, "low"
      // 2 p(1) = 1.462 and "high" 2 p(-1) = 0.538, with p(x) the logistic function, and "one" and "two" 1 / 2.
      // "routing": 0.5 rounds up to 1. "low": 1.962 and "high": 1.038 round up to 2. "one" and "two" are reached through
      // "high": 0.5 + 0.538 + 0.5 = 1.538 rounds up to 2; through "low" it would be 2.462, and 3.
      expect(multistageCuts(stages, { method: "routing-shares", scale: "number-correct" })).toEqual([
        [1],
        [2],
        [2],
        [2],
        [2],
      ]);
    });

    it("refuses cuts whose expected scores fall as theta rises", () => {
      const stages = [
        [module("routing", [0, 0])],
        [module("long", [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]), module("short", [1])],
        [module("x", [0]), module("y", [0])],
        [module("a", [0]), module("b", [0]), module("c", [0]), module("d", [0])],
      ];
      const options = {
        method: "routing-shares",
        scale: "number-correct",
        shares: [
          [0.5, 0.5],
          [0.5, 0.5],
          [0.25, 0.25, 0.25, 0.25],
        ],
      } as const;
      expect(() => multistageCuts(stages, options)).toThrow(
        'The cuts of module "x" on the scale number-correct would be',
      );
      expect(multistageCuts(stages, { ...options, scale: "theta" })).toHaveLength(5);
    });
  });

  describe("results", () => {
    it("are the cuts of a panel that multistageTest takes, one list for each module of the stages but the last", () => {
      for (const scale of ["theta", "number-correct"] as const) {
        const cuts = multistageCuts(translates, { scale });
        expect(cuts).toHaveLength(4);
        let next = 0;
        const withCuts = translates.map((modules, stage) =>
          modules.map((module) => (stage === translates.length - 1 ? module : { ...module, cuts: cuts[next++] })),
        );
        const test = multistageTest({ stages: withCuts, routing: scale });
        const lowest = [[0, 0, 0, 0], [0, 0, 0], [0]].reduce(
          (state, answers) => test.answer(state, answers),
          test.start(),
        );
        expect(lowest.modules).toEqual(["routing", "easy", "last"]);
        const highest = [[1, 1, 1, 1], [1, 1, 1], [1]].reduce(
          (state, answers) => test.answer(state, answers),
          test.start(),
        );
        expect(highest.modules).toEqual(["routing", "hard", "last"]);
      }
    });

    it("leave the stages as they were, and give each module a list of its own", () => {
      const panel = [
        [module("routing", [0])],
        [module("a", [-1]), module("b", [1])],
        [module("c", [-1]), module("d", [1])],
      ];
      const before = JSON.stringify(panel);
      const cuts = multistageCuts(panel);
      expect(JSON.stringify(panel)).toBe(before);
      expect(cuts[1]).toEqual(cuts[2]);
      expect(cuts[1]).not.toBe(cuts[2]);
    });
  });

  describe("input", () => {
    const stages = translates;
    const shares = [[0.3, 0.3, 0.4], [1]];

    it("refuses options of the other method", () => {
      expect(() => multistageCuts(stages, { shares })).toThrow(
        'The option shares belongs to the method "routing-shares", not to "maximum-information".',
      );
      expect(() => multistageCuts(stages, { population: { nodes: [0], weights: [1] } })).toThrow(
        'The option population belongs to the method "routing-shares"',
      );
      expect(() => multistageCuts(stages, { method: "routing-shares", range: [-1, 1] })).toThrow(
        'The option range belongs to the method "maximum-information", not to "routing-shares".',
      );
    });

    it("refuses an unknown method or scale, and options that are not an object", () => {
      expect(() => multistageCuts(stages, JSON.parse("null"))).toThrow("The options must be an object, found null.");
      expect(() => multistageCuts(stages, { method: JSON.parse('"quantile"') })).toThrow(
        'The option method must be "maximum-information" or "routing-shares", found "quantile".',
      );
      expect(() => multistageCuts(stages, { scale: JSON.parse('"score"') })).toThrow(
        'The option scale must be "number-correct" or "theta", found "score".',
      );
      expect(() => multistageCuts(stages, { method: JSON.parse("null") })).toThrow(DataError);
      expect(() => multistageCuts(stages, { scale: JSON.parse("null") })).toThrow(DataError);
    });

    it("refuses a range that is not an interval", () => {
      expect(() => multistageCuts(stages, { range: [1, -1] })).toThrow(
        "The range needs the lower bound below the upper",
      );
      expect(() => multistageCuts(stages, { range: JSON.parse('"wide"') })).toThrow(
        "The option range must be an array [lower, upper], found",
      );
      expect(() => multistageCuts(stages, { range: JSON.parse("null") })).toThrow(DataError);
    });

    it("refuses shares that are not a positive share for each module of each stage after the first, summing to 1", () => {
      const routing = (value: unknown) =>
        multistageCuts(stages, { method: "routing-shares", shares: JSON.parse(JSON.stringify(value)) });
      expect(() => routing([[0.5, 0.5]])).toThrow(
        "The option shares needs a list for each of the 2 stages after the first, found 1.",
      );
      expect(() => routing([[0.5, 0.5], [1]])).toThrow("Stage 2 has 3 modules, found 2 shares.");
      expect(() => routing([[0.5, 0.5, 0], [1]])).toThrow(
        "A share of a module of stage 2 must be a positive number, found 0.",
      );
      expect(() => routing([[0.5, 0.5, 0.5], [1]])).toThrow(
        "The shares of the modules of stage 2 must sum to 1, found 1.5.",
      );
      expect(() => routing([[0.3, 0.3, 0.4], [0.9]])).toThrow(
        "The shares of the modules of stage 3 must sum to 1, found 0.9.",
      );
      expect(() => routing([[0.3, 0.3, 0.4], null])).toThrow(
        "The shares of the modules of stage 3 must be an array, found null.",
      );
      expect(() => routing("shares")).toThrow(
        "The option shares must be an array of the shares of the modules of each stage after the first, found",
      );
      expect(() => routing(null)).toThrow(DataError);
      expect(() => routing([[0.3, 0.3, "0.4"], [1]])).toThrow(DataError);
    });

    it("refuses a population that is not a distribution of nodes in ascending order", () => {
      const population = (value: unknown) =>
        multistageCuts(stages, { method: "routing-shares", population: JSON.parse(JSON.stringify(value)) });
      expect(() => population({ nodes: [1, 0], weights: [0.5, 0.5] })).toThrow(
        "The nodes of the population must be in ascending order, found 1 before 0.",
      );
      expect(() => population({ nodes: [0, 1], weights: [0.5, 0.6] })).toThrow(
        "The weights of the population must sum to 1, found 1.1.",
      );
      expect(() => population({ nodes: [], weights: [] })).toThrow(DataError);
      expect(() => population("normal")).toThrow("The population must be an object { nodes, weights }");
      expect(() => population(null)).toThrow(DataError);
    });

    it("refuses stages and items that multistageTest would refuse", () => {
      expect(() => multistageCuts(JSON.parse("null"))).toThrow(
        "The option stages must be an array of stages, found null.",
      );
      expect(() => multistageCuts([])).toThrow(DataError);
      expect(() => multistageCuts([[module("a", [0])], []])).toThrow(DataError);
      expect(() => multistageCuts([[module("a", [0]), module("b", [0])]])).toThrow(
        "The first stage of a multistage test has one module, found 2.",
      );
      expect(() => multistageCuts([[module("a", [0])], [module("a", [1])]])).toThrow(
        'Module ids must be unique; "a" appears twice.',
      );
      expect(() => multistageCuts([[module("a", [])]])).toThrow('Module "a" needs an array of at least one item.');
      expect(() => multistageCuts([[JSON.parse("null")]])).toThrow(DataError);
      const invalid = { id: "q", model: twoParameterLogistic, parameters: JSON.parse('{"a":"1","b":0,"c":0,"d":1}') };
      expect(() => multistageCuts([[{ id: "a", items: [invalid] }]])).toThrow('Item "q"');
      expect(() =>
        multistageCuts([[{ id: "a", items: [item2pl("q", 1, 0)] }], [{ id: "b", items: [item2pl("q", 1, 0)] }]]),
      ).toThrow('Item ids must be unique; "q" appears twice.');
    });
  });
});

describe("randomesque", () => {
  it("selects only among the eligible items the rule scores highest", () => {
    fc.assert(
      fc.property(
        fc.array(arbitraryItem, { minLength: 1, maxLength: 10 }),
        fc.array(fc.boolean(), { minLength: 10, maxLength: 10 }),
        fc.double({ min: -3, max: 3, noNaN: true }),
        fc.integer({ min: 1, max: 4 }),
        fc.integer(),
        (items, mask, theta, count, seed) => {
          const pool = items.map((item, i) => ({ ...item, id: `q${i}` }));
          const eligible = Uint8Array.from(pool, (_, i) => (i === 0 || mask[i] ? 1 : 0));
          const estimate = { theta, standardError: 1 };
          const context = { pool, administered: [], responses: [], estimate, eligible, random: createRandom(seed) };
          const out = new Float64Array(pool.length);
          randomesque(maximumInformation, count).scores(context, out);
          const chosen = highestScore(out, eligible);
          const information = pool.map((item) => itemInformation(item, theta));
          const better = information.filter((value, i) => eligible[i] === 1 && value > (information[chosen] ?? 0));
          expect(eligible[chosen]).toBe(1);
          expect(better.length).toBeLessThan(count);
        },
      ),
    );
  });

  it("selects each of those items, reproducibly with a seed", () => {
    const pool = POOL_3PL.slice(0, 10);
    const eligible = new Uint8Array(pool.length).fill(1);
    const rule = randomesque(maximumInformation, 3);
    const draws = (seed: number): number[] => {
      const random = createRandom(seed);
      const context = { pool, administered: [], responses: [], estimate: { theta: 0, standardError: 1 }, eligible };
      return Array.from({ length: 300 }, () => {
        const out = new Float64Array(pool.length);
        rule.scores({ ...context, random }, out);
        return highestScore(out, eligible);
      });
    };
    const information = pool.map((item) => itemInformation(item, 0));
    const top = Array.from(pool.keys())
      .sort((i, j) => (information[j] ?? 0) - (information[i] ?? 0))
      .slice(0, 3);
    expect([...new Set(draws(5))].sort()).toEqual(top.sort());
    expect(draws(5)).toEqual(draws(5));
  });

  it("rejects a count that is not a positive integer", () => {
    expect(() => randomesque(maximumInformation, 0)).toThrow(
      "The number of items of randomesque selection must be a positive integer, found 0.",
    );
    expect(() => randomesque(maximumInformation, 1.5)).toThrow(DataError);
    expect(() => randomesque(JSON.parse("{}"), 2)).toThrow(
      "Randomesque selection needs a selection rule, such as maximumInformation, found an object.",
    );
  });
});

describe("maximumInformation", () => {
  for (const { source, pool, runs } of FIXTURES) {
    for (const run of runs.filter(({ selection }) => selection === "MFI")) {
      it(`matches catR's item information at each step's estimate: ${source}`, () => {
        expectScores(pool, run, maximumInformation);
      });
    }
  }
});

describe("simulation", () => {
  const random = createRandom(20261029);
  const thetas = Array.from({ length: 300 }, () => random.normal());
  const stages = [
    [{ id: "routing", items: POOL_GPCM.slice(0, 4), cuts: [-0.5, 0.5] }],
    [
      { id: "low", items: POOL_GPCM.slice(4, 8) },
      { id: "middle", items: POOL_GPCM.slice(8, 12) },
      { id: "high", items: POOL_GPCM.slice(12, 16) },
    ],
  ];

  it("reports the bias, root mean square error, mean length and exposure rates of its persons", () => {
    const test = adaptiveTest({ pool: POOL_GPCM, stopping: anyOf(testLength(12), standardErrorThreshold(0.35)) });
    const result = simulateAdaptiveTest(test, thetas, createRandom(3));
    expect(result.persons.map(({ theta }) => theta)).toEqual(thetas);
    const differences = result.persons.map(({ theta, state }) => state.estimate.theta - theta);
    const lengths = result.persons.map(({ state }) => state.items.length);
    expectClose(result.bias, mean(differences), 1e-12);
    expectClose(result.rootMeanSquareError, Math.sqrt(mean(differences.map((d) => d * d))), 1e-12);
    expectClose(result.meanLength, mean(lengths), 1e-12);
    expect(result.exposureRates.map(({ id }) => id)).toEqual(POOL_GPCM.map(({ id }) => id));
    for (const { id, rate } of result.exposureRates) {
      expect(rate).toBe(result.persons.filter(({ state }) => state.items.includes(id)).length / thetas.length);
    }
    // Responses come from the true thetas: twelve informative items estimate them closely.
    expect(Math.abs(result.bias)).toBeLessThan(0.1);
    expect(result.rootMeanSquareError).toBeLessThan(0.45);
  });

  it("simulates a multistage test: everyone takes the routing module and one module of the next stage", () => {
    const test = multistageTest({ stages, routing: "theta" });
    const result = simulateMultistageTest(test, thetas, createRandom(3));
    expect(result.meanLength).toBe(8);
    const rates = new Map(result.exposureRates.map(({ id, rate }) => [id, rate]));
    for (const item of POOL_GPCM.slice(0, 4)) expect(rates.get(item.id)).toBe(1);
    const moduleRates = ["low", "middle", "high"].map((id) => result.persons.filter((p) => p.state.modules[1] === id));
    expect(moduleRates.reduce((sum, persons) => sum + persons.length, 0)).toBe(thetas.length);
    for (const persons of moduleRates) expect(persons.length).toBeGreaterThan(0);
    expect(result.rootMeanSquareError).toBeLessThan(0.6);
  });

  it("reproduces a seeded simulation exactly", () => {
    const pool = POOL_CONTENT.map((item, i) => ({ ...item, exposure: i % 4 === 0 ? 0.4 : 1 }));
    const test = adaptiveTest({
      pool,
      selection: randomesque(aStratified([3, 3, 4]), 2),
      estimator: untilMixed(posteriorMeanEstimate, weightedLikelihoodEstimate),
      stopping: testLength(10),
      contentTargets: catContent.targets,
    });
    expect(simulateAdaptiveTest(test, thetas, createRandom(9))).toEqual(
      simulateAdaptiveTest(test, thetas, createRandom(9)),
    );
    const panel = multistageTest({ stages, routing: "number-correct" });
    expect(simulateMultistageTest(panel, thetas, createRandom(9))).toEqual(
      simulateMultistageTest(panel, thetas, createRandom(9)),
    );
  });

  it("rejects missing and non-finite thetas", () => {
    const test = adaptiveTest({ pool: POOL_3PL, stopping: testLength(2) });
    expect(() => simulateAdaptiveTest(test, [], createRandom(1))).toThrow("A simulation needs at least one theta.");
    expect(() => simulateAdaptiveTest(test, [0, Number.NaN], createRandom(1))).toThrow(
      "Theta 2 must be a finite number, found NaN.",
    );
    const panel = multistageTest({ stages, routing: "theta" });
    expect(() => simulateMultistageTest(panel, [Number.POSITIVE_INFINITY], createRandom(1))).toThrow(DataError);
  });

  it("rejects a test that is not the kind of the function", () => {
    expect(() => simulateAdaptiveTest(JSON.parse("null"), [0], createRandom(1))).toThrow(
      "The test must be an adaptive test from adaptiveTest, found null.",
    );
    expect(() => simulateMultistageTest(JSON.parse("{}"), [0], createRandom(1))).toThrow(
      "The test must be a multistage test from multistageTest, found an object.",
    );
    const adaptive = adaptiveTest({ pool: [item2pl("q1", 1, 0)], stopping: testLength(1) });
    expect(() => simulateAdaptiveTest(adaptive, [0], JSON.parse("null"))).toThrow(
      "A random number generator must be one from createRandom, found null.",
    );
    expect(() => simulateAdaptiveTest(adaptive, JSON.parse("null"), createRandom(1))).toThrow(
      "A simulation needs an array of thetas, found null.",
    );
  });

  it("rejects a test whose pool or stages were replaced with invalid ones", () => {
    const adaptive = adaptiveTest({ pool: [item2pl("q1", 1, 0)], stopping: testLength(1) });
    expect(() => simulateAdaptiveTest({ ...adaptive, pool: JSON.parse("[null]") }, [0], createRandom(1))).toThrow(
      "Item 1 must be an item with an id, a model and parameters, found null.",
    );
    const panel = multistageTest({ stages, routing: "theta" });
    expect(() => simulateMultistageTest({ ...panel, stages: JSON.parse("[[null]]") }, [0], createRandom(1))).toThrow(
      "The modules of stage 1 must be objects { id, items, cuts }, found null.",
    );
    const noItems = JSON.parse('[[{ "id": "routing", "items": [{}] }]]');
    expect(() => simulateMultistageTest({ ...panel, stages: noItems }, [0], createRandom(1))).toThrow(DataError);
  });
});

describe("stopping rules", () => {
  it("match catR's checkStopRule decisions along a test, alone and any of them", () => {
    const run = defined(cat3pl.runs[0]);
    const rules = cat3pl.stoppingRules.map(({ rule, threshold }) => defined(STOPPING_RULES.get(rule))(threshold));
    for (const { administered, theta, standardError, decisions, any } of cat3pl.stopping) {
      const context = contextAfter(POOL_3PL, run.steps, administered, { theta, standardError });
      expect(
        rules.map((rule) => rule.stops(context)),
        `${administered} items`,
      ).toEqual(decisions);
      expect(anyOf(...rules).stops(context)).toBe(any);
    }
  });

  it("combine into a minimum and a maximum length", () => {
    const rule = anyOf(testLength(6), allOf(testLength(3), standardErrorThreshold(0.8)));
    expect(rule.stops(contextOf(POOL_3PL, 2, { theta: 0, standardError: 0.5 }))).toBe(false);
    expect(rule.stops(contextOf(POOL_3PL, 3, { theta: 0, standardError: 0.5 }))).toBe(true);
    expect(rule.stops(contextOf(POOL_3PL, 5, { theta: 0, standardError: 0.9 }))).toBe(false);
    expect(rule.stops(contextOf(POOL_3PL, 6, { theta: 0, standardError: 0.9 }))).toBe(true);
  });

  it("stop on information when no item is eligible, and never on an infinite standard error", () => {
    const pool = POOL_3PL.slice(0, 2);
    expect(informationThreshold(0.01).stops(contextOf(pool, 2, { theta: 0, standardError: 1 }))).toBe(true);
    const infinite = { theta: 6, standardError: Number.POSITIVE_INFINITY };
    expect(standardErrorThreshold(10).stops(contextOf(pool, 1, infinite))).toBe(false);
  });

  it("reject invalid thresholds and combinations without rules", () => {
    expect(() => testLength(0)).toThrow("A test length must be a positive integer, found 0.");
    expect(() => testLength(2.5)).toThrow(DataError);
    expect(() => standardErrorThreshold(0)).toThrow("A standard error threshold must be a positive number, found 0.");
    expect(() => informationThreshold(Number.NaN)).toThrow(
      "An information threshold must be a positive number, found NaN.",
    );
    expect(() => allOf()).toThrow("allOf needs at least one stopping rule.");
    expect(() => anyOf()).toThrow("anyOf needs at least one stopping rule.");
    const missing: StoppingRule = JSON.parse("null");
    expect(() => anyOf(testLength(3), missing)).toThrow(
      "anyOf needs stopping rules, such as testLength(20), found null.",
    );
    expect(() => allOf(JSON.parse("{}"))).toThrow(
      "allOf needs stopping rules, such as testLength(20), found an object.",
    );
    expect(() => standardErrorThreshold(JSON.parse('"0.3"'))).toThrow(
      'A standard error threshold must be a positive number, found "0.3".',
    );
    expect(() => informationThreshold(JSON.parse("null"))).toThrow(
      "An information threshold must be a positive number, found null.",
    );
  });
});

describe("Sympson-Hetter exposure control", () => {
  it("administers the item every test selects first at the rate of its exposure parameter", () => {
    // At θ = 0, where every test starts, "best" is by far the most informative item.
    const pool = [{ ...item2pl("best", 2, 0), exposure: 0.3 }, ...POOL_3PL.slice(0, 9)];
    const random = createRandom(11);
    const thetas = Array.from({ length: 4000 }, () => random.normal());
    const result = simulateAdaptiveTest(adaptiveTest({ pool, stopping: testLength(3) }), thetas, random);
    const rate = defined(result.exposureRates[0]).rate;
    // Within four binomial standard errors.
    expect(Math.abs(rate - 0.3)).toBeLessThan(4 * Math.sqrt((0.3 * 0.7) / thetas.length));
    for (const { state } of result.persons) {
      if (state.withheld.includes("best")) expect(state.items).not.toContain("best");
      else expect(state.items[0]).toBe("best");
    }
  });

  it("never withholds the last eligible item", () => {
    const pool = [
      { ...item2pl("first", 2, 0), exposure: 0 },
      { ...item2pl("second", 1, 0), exposure: 0 },
    ];
    const state = administer(adaptiveTest({ pool, stopping: testLength(2) }), () => 1);
    expect(state).toMatchObject({ items: ["second"], withheld: ["first"], next: null });
  });

  it("draws no random number for items without exposure parameters below 1", () => {
    let draws = 0;
    const counting: Random = {
      uniform: () => {
        draws++;
        return 0.5;
      },
      normal: () => 0,
    };
    const pool = POOL_3PL.map((item, i) => (i % 2 === 0 ? { ...item, exposure: 1 } : item));
    administer(adaptiveTest({ pool, stopping: testLength(10) }), () => 1, counting);
    expect(draws).toBe(0);
  });
});

/** The posterior and the attribute mastery of a profile estimate against the written-out replay in base R. */
const POSTERIOR_TOLERANCE = 1e-12;

type CdmReference = (typeof catCdm.cases)[number];
type DiagnosticRule = ItemSelectionRule<DiagnosticTestItem, ProfileEstimate>;

// By the names of the replays.
const DIAGNOSTIC_RULES = new Map<string, DiagnosticRule>([
  ["GDI", gdinaDiscriminationIndex],
  ["MPWKL", modifiedPosteriorWeightedKullbackLeibler],
  ["PWKL", posteriorWeightedKullbackLeibler],
  ["SHE", shannonEntropy],
  ["KL", diagnosticKullbackLeibler],
]);

const SKILL_POOL: readonly DiagnosticPoolItem[] = [
  { id: "q1", attributes: ["add"], probabilities: [0.2, 0.9] },
  { id: "q2", attributes: ["subtract"], probabilities: [0.15, 0.85] },
  { id: "q3", attributes: ["add", "subtract"], probabilities: [0.1, 0.4, 0.4, 0.9] },
];
const SKILLS = ["add", "subtract"];

function diagnosticPool(reference: CdmReference): DiagnosticPoolItem[] {
  return reference.items.map(({ id, attributes, probabilities }) => ({ id, attributes, probabilities }));
}

function referenceTest(
  reference: CdmReference,
  selection: DiagnosticRule,
  stopping: StoppingRule<DiagnosticTestItem, ProfileEstimate> = testLength(catCdm.length),
): DiagnosticAdaptiveTest {
  return diagnosticAdaptiveTest({
    pool: diagnosticPool(reference),
    attributes: reference.attributes,
    prior: reference.profiles,
    selection,
    stopping,
  });
}

/** The id of an item of a reference by its number in the replay, which counts from 1. */
function itemId(reference: CdmReference, item: number): string {
  return defined(reference.items[item - 1]).id;
}

/** The response of a reference's person, counting from 1, to an item by id. */
function respondOf(reference: CdmReference, person: number): (id: string) => number {
  const row = defined(reference.responses[person - 1]);
  return (id) => defined(row[reference.items.findIndex((item) => item.id === id)]);
}

/** Administers a diagnostic test to its end with the response `respond` gives each item, and returns every state. */
function administerDiagnostic(
  test: DiagnosticAdaptiveTest,
  respond: (id: string) => number,
  random = createRandom(1),
): DiagnosticAdaptiveTestState[] {
  const states = [test.start(random)];
  for (let state = defined(states[0]); state.next !== null; ) {
    state = test.answer(state, respond(state.next), random);
    states.push(state);
  }
  return states;
}

function expectProfileEstimate(
  actual: ProfileEstimate,
  expected: {
    readonly posterior: readonly number[];
    readonly mastery: readonly number[];
    readonly map: readonly number[];
  },
  label: string,
): void {
  expected.posterior.forEach((p, c) => {
    expectClose(defined(actual.posterior[c]), p, POSTERIOR_TOLERANCE, `${label}, profile ${c}`);
  });
  expected.mastery.forEach((m, k) => {
    expectClose(defined(actual.mastery[k]), m, POSTERIOR_TOLERANCE, `${label}, attribute ${k}`);
  });
  expect(actual.map, label).toEqual(expected.map);
  expect(actual.eap, label).toEqual(expected.mastery.map((m) => (m > 0.5 ? 1 : 0)));
}

describe("diagnosticAdaptiveTest", () => {
  for (const reference of catCdm.cases) {
    for (const [name, rule] of DIAGNOSTIC_RULES) {
      it(`selects the items and estimates of the written-out ${name} replay on ${reference.name}`, () => {
        const test = referenceTest(reference, rule);
        for (const run of reference.runs.filter((candidate) => candidate.rule === name)) {
          const label = `person ${run.person}`;
          const states = administerDiagnostic(test, respondOf(reference, run.person));
          expect(states.map((state) => state.next).slice(0, run.items.length), label).toEqual(
            run.items.map((item) => itemId(reference, item)),
          );
          run.steps.forEach((step, k) => {
            expectProfileEstimate(defined(states[k]).estimate, step, `${label}, step ${k}`);
          });
          const last = defined(states.at(-1));
          expectProfileEstimate(last.estimate, run.final, `${label}, final`);
          expect(last.items, label).toEqual(run.items.map((item) => itemId(reference, item)));
          expect(last.next).toBeNull();
          // cdcatR's fixed-length test administers the same items where it has the rule.
          if (run.cdcat !== undefined)
            expect(last.items, label).toEqual(run.cdcat.map((item) => itemId(reference, item)));
        }
      });
    }

    it(`ends at the length of cdcatR's fixed-precision test on ${reference.name}`, () => {
      const test = referenceTest(
        reference,
        gdinaDiscriminationIndex,
        anyOf(testLength(catCdm.length), posteriorThreshold(0.8)),
      );
      for (const { person, items } of reference.precision) {
        const last = defined(administerDiagnostic(test, respondOf(reference, person)).at(-1));
        expect(last.items, `person ${person}`).toEqual(items.map((item) => itemId(reference, item)));
      }
    });
  }

  it("starts from the prior and updates it by the probability of each response in the item's latent group", () => {
    const pool = [{ id: "q", attributes: ["a"], probabilities: [0.2, 0.9] }];
    const test = diagnosticAdaptiveTest({ pool, attributes: ["a"], stopping: testLength(1) });
    const random = createRandom(1);
    const start = test.start(random);
    expect(start).toEqual({
      items: [],
      responses: [],
      withheld: [],
      next: "q",
      estimate: { posterior: [0.5, 0.5], mastery: [0.5], map: [0], eap: [0] },
    });
    const correct = test.answer(start, 1, random);
    expectClose(defined(correct.estimate.posterior[1]), 0.9 / 1.1, 1e-15);
    expectClose(defined(correct.estimate.mastery[0]), 0.9 / 1.1, 1e-15);
    expect(correct).toMatchObject({ items: ["q"], responses: [1], next: null, estimate: { map: [1], eap: [1] } });
    const wrong = test.answer(start, 0, random);
    expectClose(defined(wrong.estimate.posterior[1]), 0.1 / 0.9, 1e-15);
    expect(wrong.estimate.map).toEqual([0]);
  });

  it("matches the prior to the profiles by pattern, in any order, and normalizes it", () => {
    const pool = [...SKILL_POOL];
    const prior = [
      { profile: [1, 1], proportion: 0.4 },
      { profile: [0, 1], proportion: 0.2 },
      { profile: [1, 0], proportion: 0.6 },
      { profile: [0, 0], proportion: 0.8 },
    ];
    const test = diagnosticAdaptiveTest({ pool, attributes: SKILLS, prior, stopping: testLength(1) });
    // The order of the profiles: 00, 10, 01, 11.
    const posterior = test.start(createRandom(1)).estimate.posterior;
    [0.4, 0.3, 0.1, 0.2].forEach((p, c) => {
      expectClose(defined(posterior[c]), p, 1e-15, `profile ${c}`);
    });
  });

  it("follows the order of an item's own attributes for its latent groups", () => {
    // The item's second group is the profile over (b, a) that masters b only, whatever the test's attribute order.
    const pool = [{ id: "q", attributes: ["b", "a"], probabilities: [0.1, 0.9, 0.1, 0.1] }];
    const test = diagnosticAdaptiveTest({ pool, attributes: ["a", "b"], stopping: testLength(1) });
    const random = createRandom(1);
    const { mastery, map } = test.answer(test.start(random), 1, random).estimate;
    expectClose(defined(mastery[0]), 1 / 6, 1e-15);
    expectClose(defined(mastery[1]), 5 / 6, 1e-15);
    expect(map).toEqual([0, 1]);
  });

  it("gives no posterior to a profile the prior excludes", () => {
    const prior = [
      { profile: [0, 0], proportion: 1 },
      { profile: [1, 0], proportion: 1 },
      { profile: [0, 1], proportion: 0 },
      { profile: [1, 1], proportion: 1 },
    ];
    const test = diagnosticAdaptiveTest({ pool: SKILL_POOL, attributes: SKILLS, prior, stopping: testLength(3) });
    for (const state of administerDiagnostic(test, (id) => (id === "q2" ? 1 : 0))) {
      expect(state.estimate.posterior[2]).toBe(0);
    }
  });

  it("keeps the posterior defined when a response is impossible in every latent group", () => {
    const pool = [{ id: "q", attributes: ["a"], probabilities: [0, 1] }];
    const test = diagnosticAdaptiveTest({ pool, attributes: ["a"], stopping: testLength(1) });
    const random = createRandom(1);
    const { posterior } = test.answer(test.start(random), 0, random).estimate;
    expect(posterior).toEqual([1, Number.MIN_VALUE]);
  });

  it("uses the G-DINA discrimination index unless given a selection rule", () => {
    const reference = defined(catCdm.cases[0]);
    const run = (test: DiagnosticAdaptiveTest) => administerDiagnostic(test, respondOf(reference, 1)).at(-1);
    const stopping = testLength(catCdm.length);
    const standard = diagnosticAdaptiveTest({
      pool: diagnosticPool(reference),
      attributes: reference.attributes,
      prior: reference.profiles,
      stopping,
    });
    expect(run(standard)).toEqual(run(referenceTest(reference, gdinaDiscriminationIndex)));
    expect(run(standard)).not.toEqual(run(referenceTest(reference, shannonEntropy)));
  });

  it("selects by a custom rule and ends when the pool does", () => {
    const first: DiagnosticRule = {
      scores({ eligible }, out) {
        eligible.forEach((flag, i) => {
          if (flag === 1) out[i] = -i;
        });
      },
    };
    const test = diagnosticAdaptiveTest({
      pool: SKILL_POOL,
      attributes: SKILLS,
      selection: first,
      stopping: testLength(10),
    });
    expect(defined(administerDiagnostic(test, () => 1).at(-1))).toMatchObject({
      items: ["q1", "q2", "q3"],
      next: null,
    });
  });

  it("selects among the items of the content area furthest below its target", () => {
    const pool = SKILL_POOL.map((item, i) => ({ ...item, contentAreas: [i === 2 ? "both" : "single"] }));
    const test = diagnosticAdaptiveTest({
      pool,
      attributes: SKILLS,
      contentTargets: { single: 0.5, both: 0.5 },
      stopping: testLength(2),
    });
    for (let seed = 0; seed < 10; seed++) {
      const { items } = defined(administerDiagnostic(test, () => 1, createRandom(seed)).at(-1));
      expect(new Set(items.map((id) => pool.find((item) => item.id === id)?.contentAreas?.[0]))).toEqual(
        new Set(["single", "both"]),
      );
    }
  });

  it("never administers enemies together", () => {
    const test = diagnosticAdaptiveTest({
      pool: SKILL_POOL,
      attributes: SKILLS,
      enemies: [["q1", "q2"]],
      stopping: testLength(3),
    });
    for (let seed = 0; seed < 10; seed++) {
      const { items } = defined(administerDiagnostic(test, () => 0, createRandom(seed)).at(-1));
      expect(items).toHaveLength(2);
      expect(items.includes("q3")).toBe(true);
    }
  });

  it("withholds the item it selects with probability one minus its exposure parameter", () => {
    const pool = SKILL_POOL.map((item) => (item.id === "q3" ? { ...item, exposure: 0 } : item));
    const lastItemFirst: DiagnosticRule = {
      scores({ eligible }, out) {
        eligible.forEach((flag, i) => {
          if (flag === 1) out[i] = i;
        });
      },
    };
    const test = diagnosticAdaptiveTest({
      pool,
      attributes: SKILLS,
      selection: lastItemFirst,
      stopping: testLength(2),
    });
    const state = test.start(createRandom(1));
    expect(state).toMatchObject({ withheld: ["q3"], next: "q2" });
    expect(defined(administerDiagnostic(test, () => 1).at(-1))).toMatchObject({
      items: ["q2", "q1"],
      withheld: ["q3"],
    });
  });

  it("ends at once without items, at the prior", () => {
    const test = diagnosticAdaptiveTest({ pool: [], attributes: ["a"], stopping: testLength(1) });
    expect(test.start(createRandom(1))).toMatchObject({ next: null, estimate: { posterior: [0.5, 0.5] } });
  });

  it("continues identically from a state stored as JSON", () => {
    const reference = defined(catCdm.cases[1]);
    const test = referenceTest(reference, posteriorWeightedKullbackLeibler);
    const respond = respondOf(reference, 2);
    const states = administerDiagnostic(test, respond, createRandom(3));
    const middle = defined(states[4]);
    const stored: DiagnosticAdaptiveTestState = JSON.parse(JSON.stringify(middle));
    expect(stored).toEqual(middle);
    let resumed = stored;
    while (resumed.next !== null) resumed = test.answer(resumed, respond(resumed.next), createRandom(3));
    expect(resumed).toEqual(states.at(-1));
  });

  it("leaves the state it is given unchanged", () => {
    const test = diagnosticAdaptiveTest({ pool: SKILL_POOL, attributes: SKILLS, stopping: testLength(2) });
    const random = createRandom(1);
    const state = test.start(random);
    const copy: DiagnosticAdaptiveTestState = JSON.parse(JSON.stringify(state));
    test.answer(state, 1, random);
    expect(state).toEqual(copy);
  });

  it("rejects options and pools of other types and invalid ones", () => {
    const options = { pool: SKILL_POOL, attributes: SKILLS, stopping: testLength(2) };
    const parse = (json: string) => JSON.parse(json);
    expect(() => diagnosticAdaptiveTest(parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => diagnosticAdaptiveTest({ ...options, stopping: parse("null") })).toThrow(
      "The option stopping must be a stopping rule, such as testLength(20)",
    );
    expect(() => diagnosticAdaptiveTest({ ...options, selection: parse('"gdi"') })).toThrow(
      "The option selection must be a selection rule, such as gdinaDiscriminationIndex",
    );
    expect(() => diagnosticAdaptiveTest({ ...options, pool: parse('"q1"') })).toThrow(
      'The option pool must be an array of items, found "q1".',
    );
    expect(() => diagnosticAdaptiveTest({ ...options, pool: parse("[null]") })).toThrow(
      "Item 1 must be an object { id, attributes, probabilities }, found null.",
    );
    expect(() => diagnosticAdaptiveTest({ ...options, pool: parse('[{"id":3}]') })).toThrow(
      "Item 1 needs a string id, found 3.",
    );
    expect(() => diagnosticAdaptiveTest({ ...options, pool: [...SKILL_POOL, defined(SKILL_POOL[0])] })).toThrow(
      'Item ids must be unique; "q1" appears twice.',
    );
    expect(() =>
      diagnosticAdaptiveTest({ ...options, pool: [{ id: "x", attributes: [], probabilities: [] }] }),
    ).toThrow('Item "x" needs at least one attribute.');
    expect(() =>
      diagnosticAdaptiveTest({ ...options, pool: [{ id: "x", attributes: ["divide"], probabilities: [0.2, 0.9] }] }),
    ).toThrow('Item "x" has attribute "divide", which is not an attribute of the test.');
    expect(() =>
      diagnosticAdaptiveTest({ ...options, pool: [{ id: "x", attributes: ["add"], probabilities: [0.2, 0.9, 0.5] }] }),
    ).toThrow('Item "x" needs 2 probabilities from 0 to 1, one per latent group.');
    expect(() =>
      diagnosticAdaptiveTest({ ...options, pool: [{ id: "x", attributes: ["add"], probabilities: [0.2, 1.5] }] }),
    ).toThrow('Item "x" needs 2 probabilities from 0 to 1, one per latent group.');
    expect(() => diagnosticAdaptiveTest({ ...options, pool: [{ ...defined(SKILL_POOL[0]), exposure: 2 }] })).toThrow(
      'The exposure parameter of item "q1" must be a number from 0 to 1, found 2.',
    );
    expect(() => diagnosticAdaptiveTest({ ...options, contentTargets: { easy: 1 } })).toThrow(
      'Item "q1" needs exactly one content area among the targets (easy), found undefined.',
    );
    expect(() => diagnosticAdaptiveTest({ ...options, enemies: [["q1", "zz"]] })).toThrow(
      'Enemy set 1 names "zz", which is not a pool item.',
    );
  });

  it("rejects attributes and priors of other types and invalid ones", () => {
    const options = { pool: SKILL_POOL, stopping: testLength(2) };
    const parse = (json: string) => JSON.parse(json);
    const attributes = (value: unknown) => () =>
      diagnosticAdaptiveTest({ ...options, attributes: parse(JSON.stringify(value)) });
    expect(attributes("add")).toThrow('The option attributes must be an array of attribute names, found "add".');
    expect(attributes([])).toThrow("The option attributes must list from 1 to 20 distinct non-empty names");
    expect(attributes(["add", "add"])).toThrow("distinct non-empty names");
    expect(attributes(["add", ""])).toThrow("distinct non-empty names");
    expect(attributes(["add", 2])).toThrow("distinct non-empty names");
    expect(attributes(Array.from({ length: 21 }, (_, k) => `a${k}`))).toThrow("from 1 to 20 distinct");
    const prior = (value: unknown) => () =>
      diagnosticAdaptiveTest({ ...options, attributes: SKILLS, prior: parse(JSON.stringify(value)) });
    const entries = [
      { profile: [0, 0], proportion: 1 },
      { profile: [1, 0], proportion: 1 },
      { profile: [0, 1], proportion: 1 },
      { profile: [1, 1], proportion: 1 },
    ];
    expect(prior("uniform")).toThrow('The option prior must be an array of { profile, proportion }, found "uniform".');
    expect(prior([null, ...entries])).toThrow("Prior entry 1 must be an object { profile, proportion }, found null.");
    expect(prior(entries.slice(1))).toThrow("The option prior must list each of the 4 profiles once, found 3 entries.");
    expect(prior([...entries, entries[0]])).toThrow("Prior entry 5 repeats the profile [0, 0].");
    expect(prior([{ profile: [0], proportion: 1 }, ...entries.slice(1)])).toThrow(
      "Prior entry 1 needs a 0 or 1 for each of the 2 attributes, found [0].",
    );
    expect(prior([{ profile: [0, 2], proportion: 1 }, ...entries.slice(1)])).toThrow("Prior entry 1 needs a 0 or 1");
    expect(prior([{ profile: [0, 0], proportion: 1.5 }, ...entries.slice(1)])).toThrow(
      "The proportion of prior entry 1 must be a number from 0 to 1, found 1.5.",
    );
    expect(prior(entries.map((entry) => ({ ...entry, proportion: 0 })))).toThrow(
      "The option prior needs profile proportions that are not all 0.",
    );
  });

  it("rejects states and responses of other types and invalid ones", () => {
    const test = diagnosticAdaptiveTest({ pool: SKILL_POOL, attributes: SKILLS, stopping: testLength(2) });
    const random = createRandom(1);
    const state = test.start(random);
    const parse = (json: string) => JSON.parse(json);
    expect(() => test.start(parse("null"))).toThrow(DataError);
    expect(() => test.answer(state, 1, parse("null"))).toThrow(DataError);
    expect(() => test.answer(parse("null"), 1, random)).toThrow("A test state needs the arrays items and responses.");
    expect(() => test.answer(state, 2, random)).toThrow(
      `The response to item "${state.next}" must be an integer from 0 to 1, found 2.`,
    );
    expect(() => test.answer(state, parse('"1"'), random)).toThrow("must be an integer from 0 to 1");
    expect(() => test.answer(state, 0.5, random)).toThrow("must be an integer from 0 to 1");
    expect(() => test.answer({ ...state, items: ["zz"], responses: [1] }, 1, random)).toThrow(
      'The test has no item "zz".',
    );
    expect(() => test.answer({ ...state, items: ["q1"], responses: [] }, 1, random)).toThrow(
      "A test state needs one response per item, found 0 responses to 1 items.",
    );
    expect(() => test.answer({ ...state, items: ["q1"], responses: [3] }, 1, random)).toThrow(
      'The response to item "q1" must be an integer from 0 to 1, found 3.',
    );
    expect(() => test.answer({ ...state, withheld: parse("null") }, 1, random)).toThrow(
      "An adaptive test state needs the array withheld.",
    );
    expect(() => test.answer({ ...state, withheld: ["q1"], next: "q1" }, 1, random)).toThrow(
      'A test state names item "q1" twice; an item is administered or withheld once.',
    );
    const ended = test.answer(test.answer(state, 1, random), 1, random);
    expect(() => test.answer(ended, 1, random)).toThrow("The test has ended; it takes no more responses.");
  });
});

describe("diagnostic selection rules", () => {
  for (const reference of catCdm.cases) {
    for (const [name, rule] of DIAGNOSTIC_RULES) {
      it(`scores the items as the written-out ${name} does on ${reference.name}`, () => {
        const space = profileSpace(reference.attributes, reference.profiles);
        const { items } = bindPool(diagnosticPool(reference), space);
        const profiles = reference.profiles.map(({ profile }) => profile);
        for (const run of reference.runs.filter((candidate) => candidate.rule === name)) {
          run.steps.forEach((step, k) => {
            const administered = run.items.slice(0, k).map((item) => item - 1);
            const eligible = Uint8Array.from(items, (_, i) => (administered.includes(i) ? 0 : 1));
            const context: SelectionContext<DiagnosticTestItem, ProfileEstimate> = {
              pool: items,
              administered,
              responses: run.steps.slice(0, k).map((earlier) => earlier.response),
              estimate: profileEstimate(step.posterior, profiles),
              eligible,
              random: createRandom(0),
            };
            const out = new Float64Array(items.length);
            rule.scores(context, out);
            step.scores.forEach((expected, i) => {
              if (expected === null) return;
              expectClose(
                defined(out[i]),
                expected,
                SCORE_TOLERANCE * Math.max(1, Math.abs(expected)),
                `person ${run.person}, step ${k}, item ${i + 1}`,
              );
            });
          });
        }
      });
    }
  }

  it("scores the items of one latent group by the probability of its profiles alone, as plain KL sums them", () => {
    // One attribute, one item: KL sums the divergence of the distribution at the MAP profile from each profile.
    const pool = [{ id: "q", attributes: ["a"], probabilities: [0.2, 0.9] }];
    const space = profileSpace(
      ["a"],
      [
        { profile: [0, 0], proportion: 0.3 },
        { profile: [1, 1], proportion: 0.7 },
      ]
        .slice(0, 0)
        .concat([
          { profile: [0], proportion: 0.3 },
          { profile: [1], proportion: 0.7 },
        ]),
    );
    const { items } = bindPool(pool, space);
    const estimate = profileEstimate([0.3, 0.7], [[0], [1]]);
    const out = new Float64Array(1);
    diagnosticKullbackLeibler.scores(
      { pool: items, administered: [], responses: [], estimate, eligible: Uint8Array.of(1), random: createRandom(0) },
      out,
    );
    const divergence = (p: number, q: number): number => p * Math.log(p / q) + (1 - p) * Math.log((1 - p) / (1 - q));
    // The MAP profile masters the attribute: P = 0.9, against 0.2 and 0.9.
    expectClose(defined(out[0]), divergence(0.9, 0.2) + divergence(0.9, 0.9), 1e-15);
  });

  it("scores an item that tells the profiles apart above one that does not", () => {
    const pool: DiagnosticPoolItem[] = [
      { id: "flat", attributes: ["a"], probabilities: [0.5, 0.5] },
      { id: "sharp", attributes: ["a"], probabilities: [0.05, 0.95] },
    ];
    const test = diagnosticAdaptiveTest({ pool, attributes: ["a"], stopping: testLength(1) });
    for (const [name, rule] of DIAGNOSTIC_RULES) {
      const selecting = diagnosticAdaptiveTest({ pool, attributes: ["a"], selection: rule, stopping: testLength(1) });
      expect(selecting.start(createRandom(1)).next, name).toBe("sharp");
    }
    expect(test.start(createRandom(1)).next).toBe("sharp");
  });

  it("scores every item 0 when no item tells the profiles apart, for the rules that are zero there", () => {
    const pool: DiagnosticPoolItem[] = [{ id: "flat", attributes: ["a"], probabilities: [0.5, 0.5] }];
    const { items } = bindPool(pool, profileSpace(["a"], undefined));
    const estimate = profileEstimate([0.5, 0.5], [[0], [1]]);
    for (const rule of [
      diagnosticKullbackLeibler,
      posteriorWeightedKullbackLeibler,
      modifiedPosteriorWeightedKullbackLeibler,
      gdinaDiscriminationIndex,
    ]) {
      const out = new Float64Array(1);
      rule.scores(
        { pool: items, administered: [], responses: [], estimate, eligible: Uint8Array.of(1), random: createRandom(0) },
        out,
      );
      expect(out[0]).toBe(0);
    }
  });

  it("scores the Shannon entropy rule by minus the expected entropy after the response", () => {
    // Two profiles, probabilities 0.2 and 0.9, posterior 0.5 and 0.5: P(correct) = 0.55; the posteriors after a
    // correct response are (0.1, 0.45) / 0.55 and after an incorrect one (0.4, 0.05) / 0.45.
    const pool: DiagnosticPoolItem[] = [{ id: "q", attributes: ["a"], probabilities: [0.2, 0.9] }];
    const { items } = bindPool(pool, profileSpace(["a"], undefined));
    const entropy = (p: number): number => -(p * Math.log(p) + (1 - p) * Math.log(1 - p));
    const expected = -(0.55 * entropy(0.45 / 0.55) + 0.45 * entropy(0.4 / 0.45));
    const out = new Float64Array(1);
    shannonEntropy.scores(
      {
        pool: items,
        administered: [],
        responses: [],
        estimate: profileEstimate([0.5, 0.5], [[0], [1]]),
        eligible: Uint8Array.of(1),
        random: createRandom(0),
      },
      out,
    );
    expectClose(defined(out[0]), expected, 1e-15);
  });

  it("keeps every score finite where success probabilities are 0 or 1 and a latent group has no posterior", () => {
    const pool: DiagnosticPoolItem[] = [
      { id: "sure", attributes: ["c"], probabilities: [1, 0] },
      { id: "never", attributes: ["c"], probabilities: [0, 0] },
      { id: "faint", attributes: ["a", "b", "c"], probabilities: [1.7763568394002505e-15, 0, 0, 0, 0, 0, 0, 0] },
    ];
    const space = profileSpace(["a", "b", "c"], undefined);
    const { items } = bindPool(pool, space);
    // The posterior after a correct response to "sure": profiles that master c have none, and "faint" has groups of 0.
    const posterior = space.profiles.map((profile) => (profile[2] === 1 ? 0 : 0.25));
    const estimate = profileEstimate(posterior, space.profiles);
    for (const rule of DIAGNOSTIC_RULES.values()) {
      const out = new Float64Array(items.length);
      rule.scores(
        {
          pool: items,
          administered: [0],
          responses: [1],
          estimate,
          eligible: Uint8Array.of(0, 1, 1),
          random: createRandom(0),
        },
        out,
      );
      for (const score of [out[1], out[2]]) expect(Number.isFinite(score)).toBe(true);
    }
  });

  it("leaves the scores of items that are not eligible as they are", () => {
    const { items } = bindPool([...SKILL_POOL], profileSpace(SKILLS, undefined));
    const estimate = profileEstimate(
      [0.25, 0.25, 0.25, 0.25],
      [
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
      ],
    );
    for (const rule of DIAGNOSTIC_RULES.values()) {
      const out = new Float64Array(3).fill(-7);
      rule.scores(
        {
          pool: items,
          administered: [0],
          responses: [1],
          estimate,
          eligible: Uint8Array.of(0, 1, 1),
          random: createRandom(0),
        },
        out,
      );
      expect(out[0]).toBe(-7);
      expect(out[1]).not.toBe(-7);
    }
  });
});

describe("diagnostic stopping rules", () => {
  for (const reference of catCdm.cases) {
    it(`stops as the written-out posterior and mastery thresholds do on ${reference.name}`, () => {
      const { items } = bindPool(diagnosticPool(reference), profileSpace(reference.attributes, reference.profiles));
      const profiles = reference.profiles.map(({ profile }) => profile);
      const gdi = reference.runs.filter((run) => run.rule === "GDI");
      gdi.forEach((run, person) => {
        for (const decision of defined(reference.stopping[person])) {
          const state = decision.administered < run.steps.length ? run.steps[decision.administered] : run.final;
          const administered = run.items.slice(0, decision.administered).map((item) => item - 1);
          const context = {
            pool: items,
            administered,
            responses: administered.map(() => 1),
            estimate: profileEstimate(defined(state).posterior, profiles),
            eligible: new Uint8Array(items.length).fill(1),
          };
          catCdm.thresholds.forEach((threshold, t) => {
            const label = `person ${run.person}, ${decision.administered} items, threshold ${threshold}`;
            expect(posteriorThreshold(threshold).stops(context), label).toBe(defined(decision.posterior[t]));
            expect(masteryThreshold(threshold).stops(context), label).toBe(defined(decision.mastery[t]));
          });
        }
      });
    });
  }

  it("stops a test once the posterior or the mastery reaches the threshold", () => {
    const pool: DiagnosticPoolItem[] = [
      { id: "q1", attributes: ["a"], probabilities: [0.05, 0.95] },
      { id: "q2", attributes: ["b"], probabilities: [0.05, 0.95] },
      { id: "q3", attributes: ["a"], probabilities: [0.05, 0.95] },
      { id: "q4", attributes: ["b"], probabilities: [0.05, 0.95] },
    ];
    const attributes = ["a", "b"];
    const lengths = (stopping: StoppingRule<DiagnosticTestItem, ProfileEstimate>) =>
      defined(administerDiagnostic(diagnosticAdaptiveTest({ pool, attributes, stopping }), () => 1).at(-1)).items
        .length;
    // After q1 the posterior of the profiles that master a is 0.95 / 1.0 = 0.95, split over two profiles.
    expect(lengths(masteryThreshold(0.9))).toBe(2);
    expect(lengths(posteriorThreshold(0.9))).toBe(2);
    expect(lengths(anyOf(posteriorThreshold(0.99), testLength(3)))).toBe(3);
    expect(lengths(allOf(testLength(3), masteryThreshold(0.6)))).toBe(3);
  });

  it("rejects thresholds of other types and outside their ranges", () => {
    const parse = (json: string) => JSON.parse(json);
    expect(() => posteriorThreshold(parse('"0.8"'))).toThrow(
      'A posterior threshold must be above 0 and at most 1, found "0.8".',
    );
    expect(() => posteriorThreshold(0)).toThrow("A posterior threshold must be above 0 and at most 1, found 0.");
    expect(() => posteriorThreshold(1.5)).toThrow("at most 1, found 1.5");
    expect(() => posteriorThreshold(parse("null"))).toThrow("found null");
    expect(() => masteryThreshold(0.5)).toThrow("A mastery threshold must be above 0.5 and at most 1, found 0.5.");
    expect(() => masteryThreshold(parse('"0.9"'))).toThrow('found "0.9"');
    expect(() => masteryThreshold(2)).toThrow("found 2");
    expect(posteriorThreshold(1)).toHaveProperty("stops");
    expect(masteryThreshold(1)).toHaveProperty("stops");
  });
});

describe("simulateDiagnosticTest", () => {
  const SingleAttributePool: DiagnosticPoolItem[] = ["a", "b", "c"].flatMap((attribute) =>
    [1, 2, 3].map((j) => ({ id: `${attribute}${j}`, attributes: [attribute], probabilities: [0.01, 0.99] })),
  );
  const Profiles = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
  ];

  it("classifies nearly every person with near-deterministic items", () => {
    const test = diagnosticAdaptiveTest({
      pool: SingleAttributePool,
      attributes: ["a", "b", "c"],
      stopping: testLength(9),
    });
    const random = createRandom(5);
    const profiles = Array.from({ length: 400 }, (_, i) => defined(Profiles[i % 8]));
    const result = simulateDiagnosticTest(test, profiles, random);
    expect(result.patternAccuracy).toBeGreaterThan(0.99);
    expect(Math.min(...result.attributeAccuracy)).toBeGreaterThan(0.995);
    expect(result.meanLength).toBe(9);
    expect(result.persons).toHaveLength(400);
  });

  it("measures the share of persons whose MAP profile is the true profile, and of each attribute", () => {
    const test = diagnosticAdaptiveTest({
      pool: SingleAttributePool.map((item) => ({ ...item, probabilities: [0.3, 0.7] })),
      attributes: ["a", "b", "c"],
      stopping: testLength(4),
    });
    const result = simulateDiagnosticTest(test, Profiles.concat(Profiles, Profiles), createRandom(8));
    const { persons } = result;
    const hits = persons.filter(({ profile, state }) => profile.every((value, k) => state.estimate.map[k] === value));
    expect(result.patternAccuracy).toBe(hits.length / persons.length);
    expect(result.patternAccuracy).toBeGreaterThan(0);
    expect(result.patternAccuracy).toBeLessThan(1);
    result.attributeAccuracy.forEach((accuracy, k) => {
      expect(accuracy).toBe(
        persons.filter(({ profile, state }) => state.estimate.map[k] === profile[k]).length / persons.length,
      );
      expect(accuracy).toBeGreaterThanOrEqual(result.patternAccuracy);
    });
  });

  it("reports the exposure rate of every item and the mean length", () => {
    const test = diagnosticAdaptiveTest({
      pool: SingleAttributePool,
      attributes: ["a", "b", "c"],
      stopping: anyOf(testLength(9), masteryThreshold(0.95)),
    });
    const result = simulateDiagnosticTest(test, Profiles, createRandom(2));
    expect(result.exposureRates.map(({ id }) => id)).toEqual(SingleAttributePool.map(({ id }) => id));
    const total = result.exposureRates.reduce((sum, { rate }) => sum + rate, 0);
    expectClose(total, result.meanLength, 1e-12);
    expect(result.meanLength).toBeLessThan(9);
    for (const { rate } of result.exposureRates) expect(rate).toBeGreaterThanOrEqual(0);
  });

  it("draws each person's responses to the pool from the true profile as simulateCdmResponses does", () => {
    const test = diagnosticAdaptiveTest({
      pool: SKILL_POOL,
      attributes: SKILLS,
      stopping: testLength(3),
    });
    const profiles = [
      [1, 0],
      [0, 1],
      [1, 1],
      [0, 0],
    ];
    const result = simulateDiagnosticTest(test, profiles, createRandom(21));
    const matrix = simulateCdmResponses({ attributes: SKILLS, items: SKILL_POOL }, profiles, createRandom(21));
    result.persons.forEach(({ state }, person) => {
      state.items.forEach((id, j) => {
        expect(state.responses[j]).toBe(defined(defined(matrix.responses[person])[matrix.itemIds.indexOf(id)]));
      });
    });
  });

  it("reproduces a simulation from its seed", () => {
    const test = diagnosticAdaptiveTest({ pool: SKILL_POOL, attributes: SKILLS, stopping: testLength(2) });
    const run = () =>
      simulateDiagnosticTest(
        test,
        Profiles.map((profile) => profile.slice(0, 2)),
        createRandom(77),
      );
    expect(run()).toEqual(run());
  });

  it("rejects tests, profiles and generators of other types and invalid ones", () => {
    const test = diagnosticAdaptiveTest({ pool: SKILL_POOL, attributes: SKILLS, stopping: testLength(2) });
    const parse = (json: string) => JSON.parse(json);
    const random = createRandom(1);
    expect(() => simulateDiagnosticTest(parse("null"), [[1, 0]], random)).toThrow(
      "The test must be a diagnostic adaptive test from diagnosticAdaptiveTest, found null.",
    );
    expect(() => simulateDiagnosticTest(parse('{"pool":[],"start":1,"answer":2}'), [[1, 0]], random)).toThrow(
      "The test must be a diagnostic adaptive test",
    );
    expect(() => simulateDiagnosticTest({ ...test, attributes: parse('"ab"') }, [[1, 0]], random)).toThrow(
      "The test must be a diagnostic adaptive test",
    );
    expect(() => simulateDiagnosticTest(test, [[1, 0]], parse("null"))).toThrow(DataError);
    expect(() => simulateDiagnosticTest(test, parse('"10"'), random)).toThrow("The profiles must be an array");
    expect(() => simulateDiagnosticTest(test, [], random)).toThrow(DataError);
    expect(() => simulateDiagnosticTest(test, [[1]], random)).toThrow(
      "Profile 1 needs a 0 or 1 for each of the 2 attributes.",
    );
    expect(() => simulateDiagnosticTest(test, [[1, 2]], random)).toThrow("Profile 1 needs a 0 or 1");
  });
});

/** The estimate and the item scores of a multidimensional test against the written-out replay in base R. */
const MULTIDIMENSIONAL_TOLERANCE = 1e-8;

type MirtReference = (typeof catMirt.cases)[number];

/** The priors of a case: its covariance, `undefined` for the identity that the test takes by default. */
function priorOf(reference: MirtReference): MultivariateNormalPrior | undefined {
  const identity = reference.covariance.every((row, i) => row.every((value, j) => value === (i === j ? 1 : 0)));
  return identity
    ? undefined
    : { mean: reference.covariance.map(() => 0), covariance: reference.covariance.map((row) => [...row]) };
}

/** An item of a multidimensional pool: the M3PL when it has guessing, the M2PL otherwise. */
function multidimensionalItem(id: string, a: readonly number[], intercept: number, c = 0): MultidimensionalPoolItem {
  return {
    id,
    model: c > 0 ? multidimensionalThreeParameterLogistic : multidimensionalTwoParameterLogistic,
    parameters: { a, intercept, c, d: 1 },
  };
}

function multidimensionalPool(reference: MirtReference): MultidimensionalPoolItem[] {
  return reference.items.map(({ id, a, intercept, c }) => multidimensionalItem(id, a, intercept, c));
}

function multidimensionalTest(
  reference: MirtReference,
  stopping: StoppingRule<MultidimensionalPoolItem, MultidimensionalEstimate> = testLength(catMirt.length),
): MultidimensionalAdaptiveTest {
  const prior = priorOf(reference);
  return multidimensionalAdaptiveTest({
    pool: multidimensionalPool(reference),
    stopping,
    ...(prior !== undefined && {
      selection: maximumDeterminant(prior),
      estimator: (items, answers) => multidimensionalPosteriorModeEstimate(items, answers, { prior }),
    }),
  });
}

/** Administers a multidimensional test to its end with the response `respond` gives each item; returns every state. */
function administerMultidimensional(
  test: MultidimensionalAdaptiveTest,
  respond: (id: string) => number,
  random = createRandom(1),
): MultidimensionalAdaptiveTestState[] {
  const states = [test.start(random)];
  for (let state = defined(states[0]); state.next !== null; ) {
    state = test.answer(state, respond(state.next), random);
    states.push(state);
  }
  return states;
}

function expectMultidimensionalEstimate(
  actual: MultidimensionalEstimate,
  expected: {
    readonly theta: readonly number[];
    readonly standardErrors: readonly number[];
    readonly covariance?: readonly (readonly number[])[];
  },
  label: string,
): void {
  expected.theta.forEach((value, d) => {
    expectClose(defined(actual.theta[d]), value, MULTIDIMENSIONAL_TOLERANCE, `${label}, theta ${d + 1}`);
  });
  expected.standardErrors.forEach((value, d) => {
    expectClose(
      defined(actual.standardErrors[d]),
      value,
      MULTIDIMENSIONAL_TOLERANCE,
      `${label}, standard error ${d + 1}`,
    );
  });
  expected.covariance?.forEach((row, i) => {
    row.forEach((value, j) => {
      expectClose(
        defined(actual.covariance[i])[j] ?? null,
        value,
        MULTIDIMENSIONAL_TOLERANCE,
        `${label}, covariance ${i + 1}, ${j + 1}`,
      );
    });
  });
}

/** Returns the determinant of a small square matrix by cofactor expansion along its first row. */
function determinantOf(matrix: readonly (readonly number[])[]): number {
  if (matrix.length === 1) return defined(defined(matrix[0])[0]);
  return defined(matrix[0]).reduce((sum, value, column) => {
    const minor = matrix.slice(1).map((row) => row.filter((_, c) => c !== column));
    return sum + (column % 2 === 0 ? 1 : -1) * value * determinantOf(minor);
  }, 0);
}

/** A multidimensional estimate at `theta` with standard errors of 1 and an identity covariance. */
function estimateAt(...theta: number[]): MultidimensionalEstimate {
  return {
    theta,
    standardErrors: theta.map(() => 1),
    covariance: theta.map((_, i) => theta.map((__, j) => (i === j ? 1 : 0))),
  };
}

/** A context of a multidimensional test with the first `count` items administered. */
function multidimensionalContext(
  pool: readonly MultidimensionalPoolItem[],
  count: number,
  estimate: MultidimensionalEstimate,
): SelectionContext<MultidimensionalPoolItem, MultidimensionalEstimate> {
  const administered = Array.from({ length: count }, (_, i) => i);
  return {
    pool,
    administered,
    responses: administered.map(() => 1),
    estimate,
    eligible: Uint8Array.from(pool, (_, i) => (i < count ? 0 : 1)),
    random: createRandom(0),
  };
}

describe("multidimensionalAdaptiveTest", () => {
  for (const reference of catMirt.cases) {
    it(`selects the items, estimates and stops of the written-out replay and mirtCAT on ${reference.name}`, () => {
      const test = multidimensionalTest(reference);
      for (const run of reference.runs) {
        const label = `person ${run.person}`;
        const row = defined(reference.responses[run.person - 1]);
        const states = administerMultidimensional(test, (id) =>
          defined(row[reference.items.findIndex((item) => item.id === id)]),
        );
        const ids = run.items.map((item) => defined(reference.items[item - 1]).id);
        expect(states.map((state) => state.next).slice(0, ids.length), label).toEqual(ids);
        expect(run.mirtCAT, label).toEqual(run.items);
        run.steps.forEach((step, k) => {
          expectMultidimensionalEstimate(defined(states[k]).estimate, step, `${label}, step ${k}`);
        });
        const last = defined(states.at(-1));
        expectMultidimensionalEstimate(last.estimate, run.final, `${label}, final`);
        expect(last.items, label).toEqual(ids);
        expect(last.next).toBeNull();
        // The standard error rule on the estimates of the replay, at every state.
        run.stopping.forEach((decision) => {
          const state = defined(states[decision.administered]);
          catMirt.thresholds.forEach((threshold, t) => {
            const context = {
              pool: test.pool,
              administered: [],
              responses: [],
              estimate: state.estimate,
              eligible: new Uint8Array(test.pool.length),
            };
            expect(
              standardErrorThreshold<MultidimensionalPoolItem, MultidimensionalEstimate>(threshold).stops(context),
              `${label}, ${decision.administered} items, threshold ${threshold}`,
            ).toBe(defined(decision.stops[t]));
          });
        });
      }
    });

    it(`scores the items as the written-out D-rule does on ${reference.name}`, () => {
      const pool = multidimensionalPool(reference);
      const rule = maximumDeterminant(priorOf(reference));
      for (const run of reference.runs) {
        run.steps.forEach((step, k) => {
          const administered = run.items.slice(0, k).map((item) => item - 1);
          const context: SelectionContext<MultidimensionalPoolItem, MultidimensionalEstimate> = {
            pool,
            administered,
            responses: administered.map(() => 1),
            estimate: step,
            eligible: Uint8Array.from(pool, (_, i) => (administered.includes(i) ? 0 : 1)),
            random: createRandom(0),
          };
          const out = new Float64Array(pool.length);
          rule.scores(context, out);
          step.scores.forEach((expected, i) => {
            if (expected === null) return;
            expectClose(
              defined(out[i]),
              expected,
              SCORE_TOLERANCE * Math.max(1, Math.abs(expected)),
              `person ${run.person}, step ${k}, item ${i + 1}`,
            );
          });
        });
      }
    });
  }

  it("starts at the prior, from the estimator's answer without items or responses", () => {
    const pool = [multidimensionalItem("q1", [1, 0], 0), multidimensionalItem("q2", [0, 1], 0)];
    const start = multidimensionalAdaptiveTest({ pool, stopping: testLength(2) }).start(createRandom(1));
    expect(start).toMatchObject({ items: [], responses: [], withheld: [] });
    expect(start.estimate.theta).toEqual([0, 0]);
    expect(start.estimate.standardErrors).toEqual([1, 1]);
    expect(start.estimate.covariance).toEqual([
      [1, 0],
      [0, 1],
    ]);
    const prior = {
      mean: [0.5, -1],
      covariance: [
        [4, 1],
        [1, 2],
      ],
    };
    const centered = multidimensionalAdaptiveTest({
      pool,
      stopping: testLength(2),
      selection: maximumDeterminant(prior),
      estimator: (items, answers) => multidimensionalPosteriorModeEstimate(items, answers, { prior }),
    }).start(createRandom(1));
    expect(centered.estimate.theta).toEqual([0.5, -1]);
    prior.covariance.forEach((row, i) => {
      row.forEach((value, j) => {
        expectClose(defined(centered.estimate.covariance[i])[j] ?? null, value, 1e-14);
      });
    });
  });

  it("calls the estimator with the administered items, and with no items at the start", () => {
    const pool = [
      multidimensionalItem("q1", [1, 0], 0),
      multidimensionalItem("q2", [0, 1], 0),
      multidimensionalItem("q3", [1, 1], 0),
    ];
    const prior = {
      mean: [0, 0],
      covariance: [
        [1, 0],
        [0, 1],
      ],
    };
    const calls: { ids: string[]; answers: (number | null)[] }[] = [];
    const test = multidimensionalAdaptiveTest({
      pool,
      stopping: testLength(2),
      estimator: (items, answers) => {
        calls.push({ ids: items.map((item) => (item as MultidimensionalPoolItem).id), answers: [...answers] });
        return multidimensionalPosteriorModeEstimate(items, answers, { prior });
      },
    });
    const states = administerMultidimensional(test, () => 1);
    expect(calls[0]).toEqual({ ids: [], answers: [] });
    expect(calls[1]).toEqual({ ids: [defined(states[0]?.next)], answers: [1] });
    expect(calls[2]?.answers).toEqual([1, 1]);
  });

  it("starts with the estimate of a posterior mean estimator under a prior, and fails without a prior", () => {
    const pool = [multidimensionalItem("q1", [1, 0], 0), multidimensionalItem("q2", [0, 1], 0)];
    const prior = {
      mean: [0.2, -0.4],
      covariance: [
        [1.5, 0.2],
        [0.2, 1],
      ],
    };
    const mean = multidimensionalAdaptiveTest({
      pool,
      stopping: testLength(1),
      selection: maximumDeterminant(prior),
      estimator: (items, answers) => multidimensionalPosteriorMeanEstimate(items, answers, { prior }),
    }).start(createRandom(1));
    expectClose(defined(mean.estimate.theta[0]), 0.2, 1e-12);
    expectClose(defined(mean.estimate.theta[1]), -0.4, 1e-12);
    const withoutPrior = multidimensionalAdaptiveTest({
      pool,
      stopping: testLength(1),
      estimator: multidimensionalPosteriorModeEstimate,
    });
    expect(() => withoutPrior.start(createRandom(1))).toThrow(
      "Scoring needs at least one item, or a prior, which gives the dimensions.",
    );
  });

  it("selects the item with the largest determinant, a within-dimension item first at the prior", () => {
    const pool = [
      multidimensionalItem("a", [1, 0], 0),
      multidimensionalItem("b", [0, 1], 0),
      multidimensionalItem("c", [1, 1], 0),
    ];
    const state = multidimensionalAdaptiveTest({ pool, stopping: testLength(1) }).start(createRandom(1));
    expect(state.next).toBe("c");
  });

  it("stops when every standard error is at most the threshold, and at once when they already are", () => {
    const random = createRandom(4);
    const pool = Array.from({ length: 30 }, (_, i) =>
      multidimensionalItem(`q${i}`, [i % 2 === 0 ? 1.4 : 0.3, i % 2 === 1 ? 1.4 : 0.3], random.normal() * 0.5),
    );
    const run = (threshold: number): MultidimensionalAdaptiveTestState[] =>
      administerMultidimensional(
        multidimensionalAdaptiveTest({
          pool,
          stopping: anyOf(
            testLength(30),
            standardErrorThreshold<MultidimensionalPoolItem, MultidimensionalEstimate>(threshold),
          ),
        }),
        (id) => Number(id.slice(1)) % 2,
      );
    expect(run(1.5)).toHaveLength(1);
    const states = run(0.7);
    const last = defined(states.at(-1));
    expect(last.items.length).toBeGreaterThan(0);
    expect(last.items.length).toBeLessThan(30);
    expect(Math.max(...last.estimate.standardErrors)).toBeLessThanOrEqual(0.7);
    // The test did not stop before: the state one item earlier had a standard error above the threshold.
    expect(Math.max(...defined(states.at(-2)).estimate.standardErrors)).toBeGreaterThan(0.7);
  });

  it("ends when the pool does, and selects by a custom rule", () => {
    const pool = [
      multidimensionalItem("q1", [1, 0], 0),
      multidimensionalItem("q2", [0, 1], 0),
      multidimensionalItem("q3", [1, 1], 0),
    ];
    const lastFirst: ItemSelectionRule<MultidimensionalPoolItem, MultidimensionalEstimate> = {
      scores({ eligible }, out) {
        eligible.forEach((flag, i) => {
          if (flag === 1) out[i] = i;
        });
      },
    };
    const test = multidimensionalAdaptiveTest({ pool, selection: lastFirst, stopping: testLength(10) });
    expect(defined(administerMultidimensional(test, () => 0).at(-1))).toMatchObject({
      items: ["q3", "q2", "q1"],
      next: null,
    });
  });

  it("balances content areas, keeps enemies apart and withholds items at their exposure parameters", () => {
    const base = [
      multidimensionalItem("q1", [1, 0], 0),
      multidimensionalItem("q2", [0, 1], 0),
      multidimensionalItem("q3", [1, 1], 0),
    ];
    const balanced = multidimensionalAdaptiveTest({
      pool: base.map((item, i) => ({ ...item, contentAreas: [i === 2 ? "both" : "single"] })),
      contentTargets: { single: 0.5, both: 0.5 },
      stopping: testLength(2),
    });
    for (let seed = 0; seed < 8; seed++) {
      const { items } = defined(administerMultidimensional(balanced, () => 1, createRandom(seed)).at(-1));
      expect(items.includes("q3")).toBe(true);
    }
    const apart = multidimensionalAdaptiveTest({ pool: base, enemies: [["q1", "q3"]], stopping: testLength(3) });
    for (let seed = 0; seed < 8; seed++) {
      const { items } = defined(administerMultidimensional(apart, () => 0, createRandom(seed)).at(-1));
      expect(items.includes("q1") && items.includes("q3")).toBe(false);
    }
    const guarded = multidimensionalAdaptiveTest({
      pool: base.map((item) => (item.id === "q3" ? { ...item, exposure: 0 } : item)),
      stopping: testLength(2),
    });
    expect(guarded.start(createRandom(1))).toMatchObject({ withheld: ["q3"] });
  });

  it("continues identically from a state stored as JSON, and leaves the state it is given unchanged", () => {
    const reference = defined(catMirt.cases[2]);
    const test = multidimensionalTest(reference);
    const row = defined(reference.responses[1]);
    const respond = (id: string) => defined(row[reference.items.findIndex((item) => item.id === id)]);
    const states = administerMultidimensional(test, respond, createRandom(3));
    const stored: MultidimensionalAdaptiveTestState = JSON.parse(JSON.stringify(defined(states[4])));
    expect(stored).toEqual(states[4]);
    let resumed = stored;
    while (resumed.next !== null) resumed = test.answer(resumed, respond(resumed.next), createRandom(3));
    expect(resumed).toEqual(states.at(-1));
    const copy: MultidimensionalAdaptiveTestState = JSON.parse(JSON.stringify(stored));
    test.answer(stored, respond(defined(stored.next)), createRandom(3));
    expect(stored).toEqual(copy);
  });

  it("rejects options, pools and rules of other types and invalid ones", () => {
    const pool = [multidimensionalItem("q1", [1, 0], 0), multidimensionalItem("q2", [0, 1], 0)];
    const options = { pool, stopping: testLength(2) };
    const parse = (json: string) => JSON.parse(json);
    expect(() => multidimensionalAdaptiveTest(parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => multidimensionalAdaptiveTest({ ...options, stopping: parse("null") })).toThrow(
      "The option stopping must be a stopping rule, such as testLength(20)",
    );
    expect(() => multidimensionalAdaptiveTest({ ...options, selection: parse('"d"') })).toThrow(
      "The option selection must be a selection rule, such as maximumDeterminant()",
    );
    expect(() => multidimensionalAdaptiveTest({ ...options, estimator: parse('"map"') })).toThrow(
      "The option estimator must be a function of items and responses",
    );
    expect(() => multidimensionalAdaptiveTest({ ...options, estimator: parse("null") })).toThrow(
      "The option estimator must be a function of items and responses",
    );
    expect(() => multidimensionalAdaptiveTest({ ...options, pool: parse('"q1"') })).toThrow(
      "A multidimensional adaptive test needs an array of items",
    );
    expect(() => multidimensionalAdaptiveTest({ ...options, pool: [] })).toThrow(
      "A multidimensional adaptive test needs at least one item.",
    );
    expect(() =>
      multidimensionalAdaptiveTest({
        ...options,
        pool: [defined(pool[0]), multidimensionalItem("q2", [0, 1, 0], 0)],
      }),
    ).toThrow("Item 2 has 3 dimensions, but item 1 has 2.");
    expect(() => multidimensionalAdaptiveTest({ ...options, pool: [...pool, defined(pool[0])] })).toThrow(
      'Item ids must be unique; "q1" appears twice.',
    );
    expect(() => multidimensionalAdaptiveTest({ ...options, pool: parse('[{"id":"x"}]') })).toThrow(DataError);
    expect(() =>
      multidimensionalAdaptiveTest({ ...options, pool: [{ ...defined(pool[0]), exposure: 2 }, ...pool.slice(1)] }),
    ).toThrow('The exposure parameter of item "q1" must be a number from 0 to 1, found 2.');
    expect(() => multidimensionalAdaptiveTest({ ...options, contentTargets: { easy: 1 } })).toThrow(
      'Item "q1" needs exactly one content area among the targets (easy), found undefined.',
    );
    expect(() => multidimensionalAdaptiveTest({ ...options, enemies: [["q1", "zz"]] })).toThrow(
      'Enemy set 1 names "zz", which is not a pool item.',
    );
  });

  it("rejects states and responses of other types and invalid ones", () => {
    const pool = [multidimensionalItem("q1", [1, 0], 0), multidimensionalItem("q2", [0, 1], 0)];
    const test = multidimensionalAdaptiveTest({ pool, stopping: testLength(2) });
    const random = createRandom(1);
    const state = test.start(random);
    const parse = (json: string) => JSON.parse(json);
    expect(() => test.start(parse("null"))).toThrow(DataError);
    expect(() => test.answer(state, 1, parse("null"))).toThrow(DataError);
    expect(() => test.answer(parse("null"), 1, random)).toThrow("A test state needs the arrays items and responses.");
    expect(() => test.answer(state, 2, random)).toThrow(
      `The response to item "${state.next}" must be an integer from 0 to 1, found 2.`,
    );
    expect(() => test.answer(state, parse('"1"'), random)).toThrow("must be an integer from 0 to 1");
    expect(() => test.answer({ ...state, items: ["zz"], responses: [1] }, 1, random)).toThrow(
      'The test has no item "zz".',
    );
    expect(() => test.answer({ ...state, withheld: parse("null") }, 1, random)).toThrow(
      "An adaptive test state needs the array withheld.",
    );
    const ended = test.answer(test.answer(state, 1, random), 1, random);
    expect(() => test.answer(ended, 1, random)).toThrow("The test has ended; it takes no more responses.");
  });
});

describe("maximumDeterminant", () => {
  it("scores the items by hand at the prior, where an item's information is a quarter of its slopes' outer product", () => {
    // At theta 0, P = 1/2 and the information of an item is a aᵀ / 4. With the prior precision I: a = (1, 0) and (0, 1)
    // give det diag(5/4, 1) = 5/4, and a = (1, 1) gives det [[5/4, 1/4], [1/4, 5/4]] = 25/16 - 1/16 = 3/2.
    const pool = [
      multidimensionalItem("a", [1, 0], 0),
      multidimensionalItem("b", [0, 1], 0),
      multidimensionalItem("c", [1, 1], 0),
    ];
    const out = new Float64Array(3);
    maximumDeterminant().scores(multidimensionalContext(pool, 0, estimateAt(0, 0)), out);
    expectClose(defined(out[0]), 1.25, 1e-15);
    expectClose(defined(out[1]), 1.25, 1e-15);
    expectClose(defined(out[2]), 1.5, 1e-15);
  });

  it("adds the information of the administered items and the inverse of the prior covariance", () => {
    const pool = [
      multidimensionalItem("a", [1, 0], 0),
      multidimensionalItem("b", [0, 1], 0),
      multidimensionalItem("c", [1, 1], 0),
    ];
    const prior = {
      mean: [0, 0],
      covariance: [
        [4, 0],
        [0, 4],
      ],
    };
    const out = new Float64Array(3);
    maximumDeterminant(prior).scores(multidimensionalContext(pool, 1, estimateAt(0, 0)), out);
    // The precision is I / 4 and a adds diag(1/4, 0), giving diag(1/2, 1/4);
    // b adds diag(0, 1/4) and c adds [[1, 1], [1, 1]] / 4.
    expectClose(defined(out[1]), 0.5 * 0.5, 1e-15);
    expectClose(defined(out[2]), (0.5 + 0.25) * (0.25 + 0.25) - 0.25 * 0.25, 1e-15);
  });

  it("agrees with the determinant of the public information matrix of the test", () => {
    const pool = Array.from({ length: 8 }, (_, i) =>
      multidimensionalItem(`q${i}`, [0.5 + i * 0.15, 1.6 - i * 0.12, 0.3 * (i % 3)], i * 0.2 - 0.7),
    );
    const theta = [0.4, -0.3, 0.8];
    const out = new Float64Array(pool.length);
    maximumDeterminant().scores(multidimensionalContext(pool, 3, estimateAt(...theta)), out);
    for (let k = 3; k < pool.length; k++) {
      const information = multidimensionalInformation([...pool.slice(0, 3), defined(pool[k])], theta);
      const precisionPlus = information.map((row, r) => row.map((value, col) => value + (r === col ? 1 : 0)));
      const determinant = determinantOf(precisionPlus);
      expectClose(defined(out[k]), determinant, 1e-12 * Math.max(1, determinant), `item ${k + 1}`);
    }
  });

  it("leaves the scores of the administered items as they are, and scores no item below the determinant", () => {
    const pool = [multidimensionalItem("a", [1, 0], 0), multidimensionalItem("b", [0, 1], 0)];
    const out = new Float64Array(2).fill(-7);
    maximumDeterminant().scores(multidimensionalContext(pool, 1, estimateAt(0, 0)), out);
    expect(out[0]).toBe(-7);
    expect(defined(out[1])).toBeGreaterThanOrEqual(1);
  });

  it("is defined where an item's information has underflowed", () => {
    const pool = [multidimensionalItem("far", [3, 3], 0)];
    const out = new Float64Array(1);
    maximumDeterminant().scores(multidimensionalContext(pool, 0, estimateAt(-300, -300)), out);
    expect(out[0]).toBe(1);
  });

  it("rejects priors of other types and invalid ones, and priors of other dimensions in a test", () => {
    const parse = (json: string) => JSON.parse(json);
    expect(() => maximumDeterminant(parse('"prior"'))).toThrow(
      "The option prior must be an object { mean, covariance }",
    );
    expect(() => maximumDeterminant(parse("null"))).toThrow(DataError);
    expect(() => maximumDeterminant(parse('{"mean":[0,0],"covariance":"x"}'))).toThrow(
      "The covariance of the prior must be an array of rows",
    );
    expect(() =>
      maximumDeterminant({
        mean: [0, 0],
        covariance: [
          [1, 2],
          [2, 1],
        ],
      }),
    ).toThrow("The covariance of the prior must be a symmetric positive definite matrix.");
    const three = maximumDeterminant({
      mean: [0, 0, 0],
      covariance: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    });
    const pool = [multidimensionalItem("a", [1, 0], 0)];
    expect(() => three.scores(multidimensionalContext(pool, 0, estimateAt(0, 0)), new Float64Array(1))).toThrow(
      "The prior needs a mean of 2 values and a covariance of 2 rows.",
    );
  });
});

describe("simulateMultidimensionalAdaptiveTest", () => {
  /** A pool of 80 items on two dimensions: half load on one, half on both, with intercepts around 0. */
  function largePool(): MultidimensionalPoolItem[] {
    const random = createRandom(17);
    return Array.from({ length: 80 }, (_, i) => {
      const slopes =
        i % 4 === 0 ? [1.4, 0] : i % 4 === 1 ? [0, 1.4] : [0.9 + 0.3 * random.uniform(), 0.9 + 0.3 * random.uniform()];
      return multidimensionalItem(`q${i}`, slopes, random.normal() * 0.7);
    });
  }
  const randomThetas = (count: number, seed: number): number[][] => {
    const random = createRandom(seed);
    return Array.from({ length: count }, () => [random.normal(), random.normal()]);
  };

  it("recovers thetas on both dimensions with little bias, as the posterior mode shrinks toward the prior mean", () => {
    // Thresholds fixed before the first run: an error of 0.5 and a bias of 0.1 for 20 items of slopes near 1.
    const test = multidimensionalAdaptiveTest({ pool: largePool(), stopping: testLength(20) });
    const result = simulateMultidimensionalAdaptiveTest(test, randomThetas(300, 3), createRandom(5));
    for (const error of result.rootMeanSquareError) expect(error).toBeLessThan(0.5);
    for (const bias of result.bias) expect(Math.abs(bias)).toBeLessThan(0.1);
    expect(result.meanLength).toBe(20);
    expect(result.persons).toHaveLength(300);
  });

  it("estimates more precisely by the D-rule than by items in random order", () => {
    const pool = largePool();
    const randomOrder: ItemSelectionRule<MultidimensionalPoolItem, MultidimensionalEstimate> = {
      scores({ eligible, random }, out) {
        eligible.forEach((flag, i) => {
          if (flag === 1) out[i] = random.uniform();
        });
      },
    };
    const thetas = randomThetas(200, 8);
    const precision = (selection?: typeof randomOrder): number => {
      const test = multidimensionalAdaptiveTest({ pool, stopping: testLength(12), ...(selection && { selection }) });
      const { persons } = simulateMultidimensionalAdaptiveTest(test, thetas, createRandom(9));
      return mean(persons.map(({ state }) => mean(state.estimate.standardErrors)));
    };
    expect(precision()).toBeLessThan(precision(randomOrder));
  });

  it("reports the mean length and the exposure rate of every item", () => {
    const pool = largePool();
    const test = multidimensionalAdaptiveTest({
      pool,
      stopping: anyOf(testLength(25), standardErrorThreshold<MultidimensionalPoolItem, MultidimensionalEstimate>(0.5)),
    });
    const result = simulateMultidimensionalAdaptiveTest(test, randomThetas(60, 2), createRandom(4));
    expect(result.exposureRates.map(({ id }) => id)).toEqual(pool.map(({ id }) => id));
    expectClose(
      result.exposureRates.reduce((sum, { rate }) => sum + rate, 0),
      result.meanLength,
      1e-12,
    );
    expect(result.meanLength).toBeLessThan(25);
    expect(result.meanLength).toBeGreaterThan(0);
  });

  it("draws each person's responses from the true thetas and reproduces from its seed", () => {
    const pool = largePool().slice(0, 12);
    const test = multidimensionalAdaptiveTest({ pool, stopping: testLength(6) });
    const run = () => simulateMultidimensionalAdaptiveTest(test, randomThetas(15, 6), createRandom(31));
    expect(run()).toEqual(run());
    // Persons far above and below on both dimensions answer the easiest items alike.
    const extremes = simulateMultidimensionalAdaptiveTest(
      test,
      [
        [4, 4],
        [-4, -4],
      ],
      createRandom(2),
    );
    const [high, low] = extremes.persons;
    expect(mean(defined(high).state.responses)).toBeGreaterThan(mean(defined(low).state.responses));
  });

  it("rejects tests, thetas and generators of other types and invalid ones", () => {
    const pool = largePool().slice(0, 6);
    const test = multidimensionalAdaptiveTest({ pool, stopping: testLength(2) });
    const parse = (json: string) => JSON.parse(json);
    const random = createRandom(1);
    expect(() => simulateMultidimensionalAdaptiveTest(parse("null"), [[0, 0]], random)).toThrow(
      "The test must be a multidimensional adaptive test from multidimensionalAdaptiveTest, found null.",
    );
    expect(() => simulateMultidimensionalAdaptiveTest(test, [[0, 0]], parse("null"))).toThrow(DataError);
    expect(() => simulateMultidimensionalAdaptiveTest(test, parse('"00"'), random)).toThrow(
      "A simulation needs an array of thetas by person",
    );
    expect(() => simulateMultidimensionalAdaptiveTest(test, [], random)).toThrow(
      "A simulation needs at least one person's thetas.",
    );
    expect(() => simulateMultidimensionalAdaptiveTest(test, [[0]], random)).toThrow(
      "Person 1 needs 2 thetas for items on 2 dimensions, found 1.",
    );
    expect(() => simulateMultidimensionalAdaptiveTest(test, [[0, Number.NaN]], random)).toThrow(DataError);
  });
});

describe("rules of tests of other items and estimates", () => {
  interface Precision {
    readonly standardErrors: readonly number[];
  }
  const context = <TEstimate>(estimate: TEstimate): AdaptiveTestContext<PoolItem, TEstimate> => ({
    pool: [],
    administered: [0, 1, 2],
    responses: [1, 0, 1],
    estimate,
    eligible: new Uint8Array(0),
  });

  it("stops at a standard error threshold once every standard error of a multidimensional estimate is below it", () => {
    const rule = standardErrorThreshold<PoolItem, Precision>(0.3);
    expect(rule.stops(context({ standardErrors: [0.3, 0.1] }))).toBe(true);
    expect(rule.stops(context({ standardErrors: [0.2, 0.31] }))).toBe(false);
    expect(rule.stops(context({ standardErrors: [Number.POSITIVE_INFINITY, 0.1] }))).toBe(false);
    expect(standardErrorThreshold(0.3).stops(context({ theta: 0, standardError: 0.29 }))).toBe(true);
    expect(standardErrorThreshold(0.3).stops(context({ theta: 0, standardError: 0.31 }))).toBe(false);
    expect(standardErrorThreshold(0.3).stops(context(JSON.parse('{"theta":0}')))).toBe(false);
  });

  it("combines and cuts off the stopping rules of any estimate", () => {
    const either = anyOf<PoolItem, Precision>(testLength(10), standardErrorThreshold(0.3));
    expect(either.stops(context({ standardErrors: [0.2, 0.2] }))).toBe(true);
    expect(either.stops(context({ standardErrors: [0.2, 0.4] }))).toBe(false);
    const both = allOf<PoolItem, Precision>(testLength(3), standardErrorThreshold(0.3));
    expect(both.stops(context({ standardErrors: [0.2, 0.2] }))).toBe(true);
    expect(
      allOf<PoolItem, Precision>(testLength(4), standardErrorThreshold(0.3)).stops(context({ standardErrors: [0.2] })),
    ).toBe(false);
  });

  it("selects at random among the best items of any rule", () => {
    const pool: DiagnosticPoolItem[] = Array.from({ length: 5 }, (_, j) => ({
      id: `q${j}`,
      attributes: ["x"],
      probabilities: [0.5 - 0.09 * j, 0.5 + 0.09 * j],
    }));
    const rule = randomesque(gdinaDiscriminationIndex, 2);
    const firsts = new Set<string | null>();
    for (let seed = 0; seed < 40; seed++) {
      const test = diagnosticAdaptiveTest({ pool, attributes: ["x"], selection: rule, stopping: testLength(1) });
      firsts.add(test.start(createRandom(seed)).next);
    }
    // GDI ranks q4 above q3 above the others.
    expect(firsts).toEqual(new Set(["q4", "q3"]));
  });
});

describe("properties", () => {
  it("an item's information is largest at its peak", () => {
    fc.assert(
      fc.property(arbitraryItem, (item) => {
        const peak = informationPeak(item);
        const largest = item.model.information(item.parameters, peak);
        for (let theta = -6; theta <= 6; theta += 0.05) {
          expect(item.model.information(item.parameters, theta)).toBeLessThanOrEqual(largest * (1 + 1e-9));
        }
      }),
    );
  });

  it("adaptive tests administer eligible items once each, never withheld ones, until a rule or the pool ends them", () => {
    const pools = fc.array(
      fc.tuple(arbitraryItem, fc.option(fc.double({ min: 0, max: 1, noNaN: true }), { nil: undefined })),
      { minLength: 1, maxLength: 8 },
    );
    const selections = fc.constantFrom<ItemSelectionRule>(
      maximumInformation,
      kullbackLeibler,
      aStratified([2, 2]),
      randomesque(maximumInformation, 2),
    );
    fc.assert(
      fc.property(
        pools,
        selections,
        fc.integer({ min: 1, max: 10 }),
        fc.double({ min: -3, max: 3, noNaN: true }),
        fc.integer(),
        (items, selection, length, theta, seed) => {
          const pool = items.map(([item, exposure], i) => ({ ...item, id: `q${i}`, exposure }));
          const test = adaptiveTest({ pool, selection, stopping: testLength(length) });
          const { persons } = simulateAdaptiveTest(test, [theta], createRandom(seed));
          const { state } = defined(persons[0]);
          expect(new Set(state.items).size).toBe(state.items.length);
          expect(state.items.filter((id) => state.withheld.includes(id))).toEqual([]);
          expect(state.items.length === length || state.items.length + state.withheld.length === pool.length).toBe(
            true,
          );
          expect(Number.isFinite(state.estimate.theta)).toBe(true);
          expect(state.next).toBeNull();
        },
      ),
      { numRuns: 60 },
    );
  });
  it("testlets are administered whole in one block, and the test ends only between them", () => {
    const assignments = fc.array(fc.constantFrom<string | undefined>("a", "b", "c", undefined), {
      minLength: POOL_3PL.length,
      maxLength: POOL_3PL.length,
    });
    fc.assert(
      fc.property(
        assignments,
        fc.constantFrom<WithinTestlet>("in-order", "adaptive"),
        fc.integer({ min: 1, max: 30 }),
        fc.integer(),
        (assignment, withinTestlet, length, seed) => {
          const pool = POOL_3PL.map((item, i) => {
            const testlet = assignment[i];
            return testlet === undefined ? item : { ...item, testlet };
          });
          const random = createRandom(seed);
          const test = adaptiveTest({ pool, stopping: testLength(length), withinTestlet });
          const state = administer(test, () => (random.uniform() < 0.5 ? 1 : 0), random);
          expectTestletBlocks(state.items, pool, withinTestlet === "in-order");
          // The length is reached, and passed by less than the largest testlet.
          const largest = Math.max(1, ...["a", "b", "c"].map((id) => assignment.filter((t) => t === id).length));
          expect(state.items.length).toBeGreaterThanOrEqual(length);
          expect(state.items.length).toBeLessThan(length + largest);
        },
      ),
      { numRuns: 30 },
    );
  });

  /** A test on a blueprint built from a hidden feasible form of `length` items, whose bounds are its counts less or plus up to `margin`. */
  function hiddenFormTest(domains: number, types: number, size: number, length: number, margin: number, seed: number) {
    const random = createRandom(seed);
    const pool: PoolItem[] = Array.from({ length: size }, (_, i) => ({
      ...item2pl(`q${i}`, 0.5 + 1.5 * random.uniform(), random.normal()),
      contentAreas: [`d${Math.floor(random.uniform() * domains)}`, `t${Math.floor(random.uniform() * types)}`],
    }));
    const order = Int32Array.from(pool.keys());
    shuffle(order, random);
    const hidden = Array.from(order.slice(0, length), (index) => defined(pool[index]));
    const areas = [...new Set(pool.flatMap((item) => item.contentAreas ?? []))];
    const counts = (items: readonly PoolItem[]) =>
      areas.map((area) => items.filter((item) => item.contentAreas?.includes(area)).length);
    const constraints = counts(hidden).map((count, k) => ({
      area: defined(areas[k]),
      minimum: Math.max(0, count - Math.floor(random.uniform() * (margin + 1))),
      maximum: count + Math.floor(random.uniform() * (margin + 1)),
    }));
    const test = adaptiveTest({
      pool,
      selection: contentBounds(maximumPriorityIndex(maximumInformation, constraints, length), constraints, length),
      stopping: testLength(length),
    });
    const state = administer(test, () => (random.uniform() < 0.5 ? 1 : 0), random);
    const given = counts(state.items.map((id) => defined(pool.find((item) => item.id === id))));
    return { constraints, state, given };
  }

  it("tests never give more items than the length or more of an area than its maximum", () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 2, max: 4 }),
        fc.integer({ min: 2, max: 3 }),
        fc.integer({ min: 40, max: 120 }),
        fc.integer({ min: 4, max: 14 }),
        fc.integer({ min: 0, max: 2 }),
        fc.integer(),
        (domains, types, size, length, margin, seed) => {
          const { constraints, state, given } = hiddenFormTest(domains, types, size, length, margin, seed);
          expect(state.items.length).toBeLessThanOrEqual(length);
          constraints.forEach(({ area, maximum }, k) => {
            expect(given[k], area).toBeLessThanOrEqual(maximum);
          });
        },
      ),
      { numRuns: 100 },
    );
  });

  it("tests on blueprints built from a hidden feasible form meet every bound, in the cases of a fixed seed", () => {
    // The greedy cover of the minimums is a heuristic that can miss a feasible completion, in about 1 of 3,000 blueprints
    // as tight as 60 items for 24 cells; the seed fixes the cases, so that a change of the cover shows as a change here.
    fc.assert(
      fc.property(
        fc.integer({ min: 2, max: 4 }),
        fc.integer({ min: 2, max: 3 }),
        fc.integer({ min: 60, max: 120 }),
        fc.integer({ min: 6, max: 14 }),
        fc.integer({ min: 0, max: 1 }),
        fc.integer(),
        (domains, types, size, length, margin, seed) => {
          const { constraints, state, given } = hiddenFormTest(domains, types, size, length, margin, seed);
          expect(state.items).toHaveLength(length);
          constraints.forEach(({ area, minimum, maximum }, k) => {
            expect(given[k], area).toBeGreaterThanOrEqual(minimum);
            expect(given[k], area).toBeLessThanOrEqual(maximum);
          });
        },
      ),
      { numRuns: 200, seed: 20261029 },
    );
  });

  it("forms hold at most their length, their enemies apart, their testlets whole and at most the overlap in common", () => {
    fc.assert(
      fc.property(
        fc.integer({ min: 8, max: 30 }),
        fc.integer({ min: 1, max: 8 }),
        fc.integer({ min: 1, max: 4 }),
        fc.integer({ min: 0, max: 3 }),
        fc.integer(),
        (size, length, forms, overlap, seed) => {
          const random = createRandom(seed);
          const sets = ["a", "b", "c", undefined];
          const pool: PoolItem[] = Array.from({ length: size }, (_, i) => {
            const testlet = sets[Math.floor(random.uniform() * sets.length)];
            return {
              ...item2pl(`q${i}`, 0.5 + random.uniform(), random.normal()),
              ...(testlet !== undefined && { testlet }),
            };
          });
          const enemies = Array.from({ length: 3 }, () => {
            const first = Math.floor(random.uniform() * size);
            return [`q${first}`, `q${(first + 1 + Math.floor(random.uniform() * (size - 1))) % size}`];
          });
          const assembled = assembleForms({
            pool,
            length: Math.min(length, size),
            enemies,
            forms,
            overlap,
            target: [{ theta: random.normal(), information: 2 }],
            random,
          });
          const ids = assembled.map((form) => form.items.map((item) => item.id));
          assembled.forEach((_, f) => {
            const own = defined(ids[f]);
            expect(own.length).toBeLessThanOrEqual(Math.min(length, size));
            expect(new Set(own).size).toBe(own.length);
            for (const set of enemies) expect(set.filter((id) => own.includes(id)).length).toBeLessThanOrEqual(1);
            for (const testlet of ["a", "b", "c"]) {
              const members = pool.filter((item) => item.testlet === testlet).map((item) => item.id);
              const held = members.filter((id) => own.includes(id)).length;
              expect([0, members.length]).toContain(held);
            }
            ids.slice(f + 1).forEach((other) => {
              expect(own.filter((id) => other.includes(id)).length).toBeLessThanOrEqual(overlap);
            });
          });
        },
      ),
      { numRuns: 100 },
    );
  });

  describe("diagnostic tests", () => {
    const attributeNames = ["a", "b", "c"];
    /** A pool over the first `count` attributes: items of one to three of them, with any success probabilities. */
    const arbitraryDiagnosticPool = fc.integer({ min: 1, max: 3 }).chain((count) => {
      const names = attributeNames.slice(0, count);
      const item = fc.subarray(names, { minLength: 1 }).chain((attributes) =>
        fc
          .array(fc.double({ min: 0, max: 1, noNaN: true }), {
            minLength: 2 ** attributes.length,
            maxLength: 2 ** attributes.length,
          })
          .map((probabilities) => ({ attributes, probabilities })),
      );
      return fc.record({
        names: fc.constant(names),
        pool: fc
          .array(item, { minLength: 1, maxLength: 6 })
          .map((items) => items.map((entry, i) => ({ id: `q${i}`, ...entry }))),
        prior: fc.array(fc.double({ min: 0, max: 1, noNaN: true }), { minLength: 2 ** count, maxLength: 2 ** count }),
      });
    });
    /** The profiles of `count` attributes in the order of the posterior. */
    const profilesFor = (count: number) => profileSpace(attributeNames.slice(0, count), undefined).profiles;

    it("the posterior sums to 1, the estimates follow from it, and prior 0 stays 0", () => {
      fc.assert(
        fc.property(
          arbitraryDiagnosticPool,
          fc.integer(),
          fc.integer({ min: 1, max: 6 }),
          ({ names, pool, prior }, seed, length) => {
            const entries = profilesFor(names.length).map((profile) => ({ profile, proportion: 0 }));
            const proportions = prior.some((value) => value > 0) ? prior : prior.map(() => 1);
            const shaped = entries.map((entry, c) => ({ ...entry, proportion: defined(proportions[c]) }));
            const test = diagnosticAdaptiveTest({
              pool,
              attributes: names,
              prior: shaped,
              stopping: testLength(length),
            });
            const random = createRandom(seed);
            let state = test.start(random);
            for (;;) {
              const { posterior, mastery, map, eap } = state.estimate;
              expect(Math.abs(posterior.reduce((sum, p) => sum + p, 0) - 1)).toBeLessThan(1e-12);
              posterior.forEach((p, c) => {
                expect(p).toBeGreaterThanOrEqual(0);
                if (defined(proportions[c]) === 0) expect(p).toBe(0);
              });
              mastery.forEach((m, k) => {
                expect(m).toBeGreaterThanOrEqual(0);
                expect(m).toBeLessThanOrEqual(1 + 1e-12);
                expect(eap[k]).toBe(m > 0.5 ? 1 : 0);
              });
              expect(
                posterior[profilesFor(names.length).findIndex((profile) => profile.every((v, k) => v === map[k]))],
              ).toBe(Math.max(...posterior));
              if (state.next === null) break;
              state = test.answer(state, random.uniform() < 0.5 ? 1 : 0, random);
            }
          },
        ),
        { numRuns: 80 },
      );
    });

    it("the posterior does not depend on the order of the responses", () => {
      fc.assert(
        fc.property(arbitraryDiagnosticPool, fc.integer(), ({ names, pool }, seed) => {
          const random = createRandom(seed);
          const responses = pool.map(() => (random.uniform() < 0.5 ? 1 : 0));
          const posterior = (order: readonly number[]): readonly number[] => {
            const test = diagnosticAdaptiveTest({
              pool: order.map((i) => defined(pool[i])),
              attributes: names,
              selection: {
                scores: ({ eligible }, out) =>
                  eligible.forEach((flag, i) => {
                    if (flag === 1) out[i] = -i;
                  }),
              },
              stopping: testLength(pool.length),
            });
            let state = test.start(random);
            order.forEach((i) => {
              state = test.answer(state, defined(responses[i]), random);
            });
            return state.estimate.posterior;
          };
          const forward = pool.map((_, i) => i);
          const first = posterior(forward);
          const second = posterior([...forward].reverse());
          first.forEach((p, c) => {
            expect(Math.abs(p - defined(second[c]))).toBeLessThan(1e-12);
          });
        }),
        { numRuns: 60 },
      );
    });

    it("tests administer each item at most once, never enemies together, and reproduce from their seed", () => {
      const selections = fc.constantFrom<DiagnosticRule>(
        gdinaDiscriminationIndex,
        diagnosticKullbackLeibler,
        posteriorWeightedKullbackLeibler,
        modifiedPosteriorWeightedKullbackLeibler,
        shannonEntropy,
        randomesque(gdinaDiscriminationIndex, 2),
      );
      fc.assert(
        fc.property(
          arbitraryDiagnosticPool,
          selections,
          fc.integer({ min: 1, max: 6 }),
          fc.integer(),
          ({ names, pool }, selection, length, seed) => {
            const enemies = pool.length > 1 ? [[defined(pool[0]).id, defined(pool[1]).id]] : [];
            const test = diagnosticAdaptiveTest({
              pool,
              attributes: names,
              selection,
              enemies,
              stopping: testLength(length),
            });
            const profiles = [profilesFor(names.length).at(-1) ?? []];
            const { persons } = simulateDiagnosticTest(test, profiles, createRandom(seed));
            const { state } = defined(persons[0]);
            expect(new Set(state.items).size).toBe(state.items.length);
            expect(state.items.length).toBeLessThanOrEqual(length);
            for (const set of enemies) expect(set.every((id) => state.items.includes(id))).toBe(false);
            expect(state.next).toBeNull();
            expect(simulateDiagnosticTest(test, profiles, createRandom(seed))).toEqual(
              simulateDiagnosticTest(test, profiles, createRandom(seed)),
            );
          },
        ),
        { numRuns: 60 },
      );
    });
  });
  it("multidimensional tests administer each item once, keep enemies apart, and have a symmetric covariance", () => {
    const arbitraryPool = fc.array(
      fc.record({
        a: fc.tuple(fc.double({ min: 0, max: 2, noNaN: true }), fc.double({ min: 0, max: 2, noNaN: true })),
        intercept: fc.double({ min: -2, max: 2, noNaN: true }),
        c: fc.double({ min: 0, max: 0.3, noNaN: true }),
        exposure: fc.option(fc.double({ min: 0, max: 1, noNaN: true }), { nil: undefined }),
      }),
      { minLength: 2, maxLength: 7 },
    );
    fc.assert(
      fc.property(arbitraryPool, fc.integer({ min: 1, max: 6 }), fc.integer(), (items, length, seed) => {
        const pool = items.map(({ a, intercept, c, exposure }, i) => ({
          ...multidimensionalItem(`q${i}`, [a[0] + 0.01, a[1]], intercept, c),
          exposure,
        }));
        const test = multidimensionalAdaptiveTest({ pool, enemies: [["q0", "q1"]], stopping: testLength(length) });
        const { persons } = simulateMultidimensionalAdaptiveTest(test, [[0.3, -0.5]], createRandom(seed));
        const { state } = defined(persons[0]);
        expect(new Set(state.items).size).toBe(state.items.length);
        expect(state.items.filter((id) => state.withheld.includes(id))).toEqual([]);
        expect(state.items.includes("q0") && state.items.includes("q1")).toBe(false);
        expect(state.items.length).toBeLessThanOrEqual(length);
        expect(state.next).toBeNull();
        const { covariance, standardErrors, theta } = state.estimate;
        theta.forEach((value) => {
          expect(Number.isFinite(value)).toBe(true);
        });
        covariance.forEach((row, i) => {
          expect(defined(standardErrors[i])).toBeCloseTo(Math.sqrt(defined(row[i])), 12);
          row.forEach((value, j) => {
            expect(value).toBeCloseTo(defined(covariance[j])[i] ?? Number.NaN, 12);
          });
        });
      }),
      { numRuns: 50 },
    );
  });

  it("the D-rule never scores an item below the determinant of the precision it adds to", () => {
    const pool = Array.from({ length: 6 }, (_, i) =>
      multidimensionalItem(`q${i}`, [0.4 + 0.2 * i, 1.8 - 0.25 * i], 0.3 * i - 0.8),
    );
    const coordinate = fc.double({ min: -3, max: 3, noNaN: true });
    fc.assert(
      fc.property(coordinate, coordinate, fc.integer({ min: 0, max: 4 }), (x, y, count) => {
        const out = new Float64Array(pool.length);
        maximumDeterminant().scores(multidimensionalContext(pool, count, estimateAt(x, y)), out);
        const information =
          count === 0
            ? [
                [0, 0],
                [0, 0],
              ]
            : multidimensionalInformation(pool.slice(0, count), [x, y]);
        const baseline = determinantOf(information.map((row, r) => row.map((value, c) => value + (r === c ? 1 : 0))));
        for (let k = count; k < pool.length; k++) {
          expect(defined(out[k])).toBeGreaterThanOrEqual(baseline * (1 - 1e-12));
        }
      }),
      { numRuns: 100 },
    );
  });

  it("routing cuts between modules that translate each other by any shift are the midpoints of their centers", () => {
    const slope = fc.double({ min: 0.8, max: 1.5, noNaN: true });
    const first = fc.double({ min: -3, max: 0, noNaN: true });
    const gap = fc.double({ min: 0.8, max: 2.5, noNaN: true });
    fc.assert(
      fc.property(slope, first, gap, gap, (a, center, gap1, gap2) => {
        const centers = [center, center + gap1, center + gap1 + gap2];
        const stages = [
          [module("routing", [0])],
          centers.map((c, j) => ({
            id: `m${j}`,
            items: [-0.3, 0, 0.3].map((offset, k) => item2pl(`m${j}-${k}`, a, c + offset)),
          })),
        ];
        const [cuts] = multistageCuts(stages);
        expectClose(defined(cuts)[0] ?? null, center + gap1 / 2, 1e-8);
        expectClose(defined(cuts)[1] ?? null, center + gap1 + gap2 / 2, 1e-8);
      }),
      { numRuns: 40 },
    );
  });

  it("routing cuts by shares ascend, are the same for the modules of a stage, and are whole scores on the scale of scores", () => {
    const shares = (count: number) =>
      fc
        .array(fc.double({ min: 0.05, max: 1, noNaN: true }), { minLength: count, maxLength: count })
        .map((values) => values.map((value) => value / values.reduce((sum, other) => sum + other, 0)));
    const widths = fc.array(fc.integer({ min: 1, max: 4 }), { minLength: 1, maxLength: 2 });
    const populations = fc.option(
      fc
        .uniqueArray(fc.double({ min: -4, max: 4, noNaN: true }), {
          minLength: 1,
          maxLength: 8,
          comparator: "SameValueZero",
        })
        .chain((nodes) => shares(nodes.length).map((weights) => ({ nodes: nodes.toSorted((x, y) => x - y), weights }))),
      { nil: undefined },
    );
    fc.assert(
      fc.property(
        widths.chain((counts) => fc.tuple(fc.constant(counts), fc.tuple(...counts.map(shares)), populations)),
        fc.constantFrom<MultistageRouting>("theta", "number-correct"),
        ([counts, perStage, population], scale) => {
          const stages = [
            [module("start", [0, 0.5])],
            ...counts.map((count, k) =>
              Array.from({ length: count }, (_, j) => module(`s${k}m${j}`, [j * 0.7 - 1, j * 0.7 - 0.5])),
            ),
          ];
          const cuts = multistageCuts(stages, { method: "routing-shares", scale, shares: perStage, population });
          expect(cuts).toHaveLength(stages.slice(0, -1).flat().length);
          let next = 0;
          stages.slice(0, -1).forEach((modules, stage) => {
            const following = defined(stages[stage + 1]).length;
            const lists = modules.map(() => defined(cuts[next++]));
            if (scale === "theta") for (const list of lists) expect(list).toEqual(lists[0]);
            for (const list of lists) {
              expect(list).toHaveLength(following - 1);
              list.forEach((cut, j) => {
                expect(Number.isFinite(cut)).toBe(true);
                expect(cut).toBeGreaterThanOrEqual(list[j - 1] ?? Number.NEGATIVE_INFINITY);
                if (scale === "number-correct") {
                  expect(Number.isInteger(cut)).toBe(true);
                  expect(cut).toBeGreaterThanOrEqual(0);
                }
              });
            }
          });
        },
      ),
      { numRuns: 60 },
    );
  });
});

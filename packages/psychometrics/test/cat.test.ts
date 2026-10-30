import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { informationPeak } from "../src/cat/a-stratified.js";
import { highestScore } from "../src/cat/selection.js";
import {
  type AbilityEstimate,
  type AbilityEstimator,
  type AdaptiveTest,
  type AdaptiveTestContext,
  type AdaptiveTestState,
  adaptiveTest,
  allOf,
  anyOf,
  aStratified,
  assembleForm,
  calibrateExposure,
  createRandom,
  DataError,
  generalizedPartialCredit,
  type ItemSelectionRule,
  informationThreshold,
  itemInformation,
  kullbackLeibler,
  type MultistageModule,
  type MultistageTest,
  maximumInformation,
  maximumLikelihoodEstimate,
  multistageTest,
  type PoolItem,
  posteriorMeanEstimate,
  posteriorModeEstimate,
  type Random,
  randomesque,
  type SelectionContext,
  type StoppingRule,
  simulateAdaptiveTest,
  simulateItems,
  simulateMultistageTest,
  standardErrorThreshold,
  testInformation,
  testLength,
  threeParameterLogistic,
  twoParameterLogistic,
  untilMixed,
  weightedLikelihoodEstimate,
} from "../src/index.js";
import cat2pl from "./fixtures/cat-2pl.json" with { type: "json" };
import cat3pl from "./fixtures/cat-3pl.json" with { type: "json" };
import catContent from "./fixtures/cat-content.json" with { type: "json" };
import catGpcm from "./fixtures/cat-gpcm.json" with { type: "json" };
import { arbitraryItem, defined, expectClose, item2pl, mean } from "./helpers.js";

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
/** The theta of an item's largest information against catR's closed form in thOpt. */
const PEAK_TOLERANCE = 1e-10;

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
  contentArea,
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

  it("throws when the selection rule scores every eligible item -Infinity", () => {
    const none: ItemSelectionRule = { scores: (_, out) => out.fill(Number.NEGATIVE_INFINITY) };
    const test = adaptiveTest({ pool: POOL_3PL, selection: none, stopping: testLength(1) });
    expect(() => test.start(createRandom(1))).toThrow("The selection rule scored every eligible item -Infinity.");
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
      expect(candidates, `step ${k}`).toContain(next.contentArea);
      if (candidates.length === 1) expect(next.id, `step ${k}`).toBe(step.nextItem);
      const remaining = POOL_CONTENT.filter(
        ({ id, contentArea }) => contentArea === next.contentArea && !state.items.includes(id),
      );
      const information = remaining.map((item) => itemInformation(item, state.estimate.theta));
      expect(itemInformation(next, state.estimate.theta), `step ${k}`).toBe(Math.max(...information));
    });
  });

  it("takes each area first, then the area furthest below its target", () => {
    const pool = ["x", "y"].flatMap((area) =>
      Array.from({ length: 10 }, (_, i) => ({ ...item2pl(`${area}${i}`, 1 + i / 10, i / 5 - 1), contentArea: area })),
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
      { ...item2pl("x1", 1, 0), contentArea: "x" },
      { ...item2pl("y1", 1, 0.5), contentArea: "y" },
      { ...item2pl("y2", 1, -0.5), contentArea: "y" },
    ];
    const balanced = adaptiveTest({ pool, stopping: testLength(3), contentTargets: { x: 0.9, y: 0.1 } });
    expect(administer(balanced, () => 0).items).toHaveLength(3);
  });

  it("rejects missing and invalid targets", () => {
    const stopping = testLength(1);
    expect(() => adaptiveTest({ pool: POOL_CONTENT, stopping, contentTargets: {} })).toThrow(DataError);
    expect(() =>
      adaptiveTest({ pool: POOL_CONTENT, stopping, contentTargets: { ...catContent.targets, algebra: 0 } }),
    ).toThrow('The target of content area "algebra" must be a positive number, found 0.');
    expect(() => adaptiveTest({ pool: POOL_CONTENT, stopping, contentTargets: { algebra: 1 } })).toThrow(
      'Item "item001" needs a content area among the targets (algebra), found "statistics".',
    );
    expect(() => adaptiveTest({ pool: POOL_3PL, stopping, contentTargets: { algebra: 1 } })).toThrow(DataError);
    expect(() => adaptiveTest({ pool: POOL_3PL, stopping, contentTargets: JSON.parse("null") })).toThrow(
      "The option contentTargets must be an object of target proportions by area, found null.",
    );
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

describe("assembleForm", () => {
  const random = createRandom(20261029);
  const areas = ["algebra", "geometry", "number"];
  const pool: PoolItem[] = simulateItems(twoParameterLogistic, 300, random).map((item, i) => ({
    ...item,
    id: `q${i + 1}`,
    contentArea: defined(areas[i % 3]),
  }));

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
    const form = assembleForm({ pool, length: 30, content, target });
    expect(new Set(form.items.map((item) => item.id)).size).toBe(30);
    for (const { area, minimum, maximum } of content) {
      const count = form.items.filter((item) => item.contentArea === area).length;
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
    const form = assembleForm({ pool, length: 10, target: [{ theta: 0.5, information: 1000 }] });
    const ranked = [...pool].sort((x, y) => itemInformation(y, 0.5) - itemInformation(x, 0.5));
    expect(new Set(form.items.map((item) => item.id))).toEqual(new Set(ranked.slice(0, 10).map((item) => item.id)));
  });

  it("assembles the whole pool and is deterministic", () => {
    const small = pool.slice(0, 6);
    const form = assembleForm({ pool: small, length: 6, content: [{ area: "algebra", minimum: 2, maximum: 2 }] });
    expect(new Set(form.items)).toEqual(new Set(small));
    const options = { pool, length: 12, target: [{ theta: 0, information: 3 }] };
    expect(assembleForm(options)).toEqual(assembleForm(options));
  });

  it("rejects an invalid length, constraint or target", () => {
    expect(() => assembleForm({ pool, length: 301 })).toThrow(
      "The option length must be an integer from 1 to the pool size 300, found 301.",
    );
    expect(() => assembleForm({ pool, length: 5, content: [{ area: "algebra", minimum: 3, maximum: 2 }] })).toThrow(
      'The constraint on "algebra" needs the minimum at most the maximum, found 3 and 2.',
    );
    expect(() =>
      assembleForm({ pool, length: 5, content: [{ area: "algebra", minimum: 1, maximum: 2, weight: 0 }] }),
    ).toThrow('The weight of the constraint on "algebra" must be a positive number, found 0.');
    expect(() => assembleForm({ pool, length: 5, target: [{ theta: 0, information: Number.NaN }] })).toThrow(
      "The information of the target at theta 0 must be a finite number, found NaN.",
    );
    expect(() => assembleForm({ pool, length: 5, target: [{ theta: Infinity, information: 1 }] })).toThrow(DataError);
    expect(() => assembleForm(JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => assembleForm({ pool, length: 5, content: JSON.parse("{}") })).toThrow(
      "The option content must be an array of content constraints, found an object.",
    );
    expect(() => assembleForm({ pool, length: 5, target: JSON.parse("[null]") })).toThrow(
      "An information target must be an object { theta, information }, found null.",
    );
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

describe("multistageTest", () => {
  const module = (id: string, difficulties: readonly number[], cuts?: readonly number[]): MultistageModule => ({
    id,
    items: difficulties.map((b, j) => item2pl(`${id}-${j + 1}`, 1, b)),
    ...(cuts && { cuts }),
  });
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
});

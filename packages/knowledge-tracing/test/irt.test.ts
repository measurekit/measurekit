import {
  gradedProbabilities,
  logisticProbabilities,
  nominalProbabilities,
  partialCreditProbabilities,
  type ThetaDistribution,
} from "@measurekit/core";
import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { buildEventLog, type Event, type EventLog, splitByLearner } from "../src/data/index.js";
import { compareModels, crossValidate } from "../src/evaluate/index.js";
import { createRandom, DataError } from "../src/index.js";
import {
  type IrtDrift,
  type IrtItem,
  type IrtModel,
  type IrtParameters,
  type IrtPrior,
  type IrtState,
  irt,
  type NominalItem,
} from "../src/irt/index.js";
import { categoryCount, categoryTerms, differences, expectedOutcome } from "../src/irt/internal.js";
import { reflectedWalk } from "../src/math/internal.js";
import { allPredicted, defined, expectAllClose, expectClose } from "./assertions.js";
import { readFixture, type StoredStateFixture } from "./fixtures.js";
import { expectUnaffectedByLessons } from "./logs.js";

interface IrtAbilitiesFixture {
  items: { i1: { a: number; b: number }; g1: { model: string; a: number; b: number[] } };
  events: { learner: string; item: string; skills: never[]; time: number; outcome: number }[];
  standard: {
    prior: { mean: number; sd: number };
    points: number;
    trace: {
      prediction: number[];
      logProbability: number[];
      eap: number[];
      eapStandardError: number[];
      map: number[];
      mapStandardError: number[];
    };
  };
  shifted: {
    prior: { mean: number; sd: number };
    points: number;
    trace: {
      prediction: number[];
      logProbability: number[];
      eap: number[];
      eapStandardError: number[];
      map: number[];
      mapStandardError: number[];
    };
  };
  mirt: { eap: number[]; eapStandardError: number[]; map: number[]; mapStandardError: number[] };
}
const reference = readFixture<IrtAbilitiesFixture>("irt-abilities.json");

/** An ability trace of R: before each event the prediction and its log-probability, after it EAP and MAP. */
type AbilityTrace = IrtAbilitiesFixture["standard"]["trace"];

interface IrtNominalFixture {
  items: Record<string, IrtItem>;
  events: { learner: string; item: string; skills: never[]; time: number; outcome: number }[];
  standard: { prior: { mean: number; sd: number }; points: number; trace: AbilityTrace };
  mirt: { eap: number[]; eapStandardError: number[]; map: number[]; mapStandardError: number[] };
}
const nominal = readFixture<IrtNominalFixture>("irt-nominal.json");

interface IrtDriftFixture {
  drift: { variance: number; per: number };
  points: number;
  cases: {
    name: string;
    kind: string;
    events: { time: number[]; a: number[]; b: number[]; outcome: number[] };
    checkpoints: number[];
    sameGrid: { mean: number[]; sd: number[]; logLikelihood: number; depth: number };
    fine: { mean: number[]; sd: number[]; mode: number[] };
  }[];
  particleFilter: { case: string; mean: number; sd: number; meanError: number };
}
const drifting = readFixture<IrtDriftFixture>("irt-drift.json");

const stored = readFixture<StoredStateFixture>("stored-state.json");

/** Absolute tolerance of EAP estimates, predictions and log-probabilities against the written-out grid posterior. */
const GRID_TOLERANCE = 1e-10;
/**
 * Absolute tolerance of the MAP estimate against the exact posterior mode at the default grid spacing, 0.2: the
 * error of the quintic interpolation between grid points, which grows with the fifth power of the spacing.
 */
const MODE_TOLERANCE = 1e-5;
/** Absolute tolerance of the MAP standard error at the default spacing, growing with its fourth power. */
const MODE_ERROR_TOLERANCE = 1e-4;
/** Absolute tolerance of the log-probabilities of an item against the probabilities of core: rounding only. */
const CORE_TOLERANCE = 1e-12;
/**
 * Absolute tolerance of a slope against a five-point difference of the probabilities of core with the step below: the
 * truncation error, h⁴ / 30 times the fifth derivative, reaches about 1e-13 for a slope of 3 and 5 categories, and the
 * rounding error about 1e-12.
 */
const DIFFERENCE_TOLERANCE = 1e-9;
/** The step of that difference. */
const DIFFERENCE_STEP = 1e-4;

/** Returns the mean and the standard deviation of a distribution of θ. */
function momentsOf({ nodes, weights }: ThetaDistribution): [number, number] {
  const mean = weights.reduce((sum, weight, q) => sum + weight * (nodes[q] ?? 0), 0);
  const variance = weights.reduce((sum, weight, q) => sum + weight * ((nodes[q] ?? 0) - mean) ** 2, 0);
  return [mean, Math.sqrt(variance)];
}

const items = reference.items as Record<string, IrtItem>;
const log = buildEventLog(reference.events);

/**
 * Expects a model of the items to follow R's written-out posterior event by event: the predictions, EAP and the
 * posterior's moments, MAP and its standard error after each learner's last event, and the log-likelihood of a fit.
 */
function expectTrace(
  parameters: IrtParameters,
  events: EventLog,
  { prior, points, trace }: { prior: IrtPrior; points: number; trace: AbilityTrace },
): void {
  const spacing = (12 * prior.sd) / (points - 1) / 0.2;
  const model = irt(parameters, { prior, points });
  const fit = model.fit(events);
  const eap: number[] = [];
  const moments: number[] = [];
  const map: number[] = [];
  const states = new Map<string, IrtState>();
  for (const event of events) {
    const state = model.update(states.get(event.learner) ?? model.initialState(event.learner), event);
    states.set(event.learner, state);
    eap.push(model.ability(state).theta, model.ability(state).standardError);
    moments.push(...momentsOf(model.posterior(state)));
    map.push(model.ability(state, { estimator: "map" }).theta);
  }
  expectAllClose(allPredicted(fit.predictions), trace.prediction, GRID_TOLERANCE, "prediction");
  expectAllClose(
    eap,
    trace.eap.flatMap((theta, i) => [theta, trace.eapStandardError[i] ?? 0]),
    GRID_TOLERANCE,
    "EAP",
  );
  expectAllClose(moments, eap, GRID_TOLERANCE, "posterior mean and standard deviation");
  expectAllClose(map, trace.map, MODE_TOLERANCE * spacing ** 5, "MAP");
  const errors = [...states.values()].map((state) => model.ability(state, { estimator: "map" }).standardError);
  const lastOfEach = [...states.keys()].map((learner) => events.findLastIndex((event) => event.learner === learner));
  expectAllClose(
    errors,
    lastOfEach.map((i) => trace.mapStandardError[i] ?? 0),
    MODE_ERROR_TOLERANCE * spacing ** 4,
    "MAP standard error",
  );
  expect(Math.abs(fit.logLikelihood - trace.logProbability.reduce((sum, value) => sum + value, 0))).toBeLessThan(
    GRID_TOLERANCE,
  );
  expect(fit.states).toEqual(Object.fromEntries(states));
}

describe("IRT ability estimates against R", () => {
  for (const name of ["standard", "shifted"] as const) {
    it(`follows the written-out posterior event by event, with the ${name} prior`, () => {
      expectTrace({ items }, log, reference[name]);
    });
  }

  it("matches mirt's fscores for EAP, as the written-out posterior does", () => {
    expectAllClose(reference.mirt.eap, reference.standard.trace.eap, GRID_TOLERANCE, "mirt EAP");
  });

  it("follows the written-out posterior with nominal response items, whose EAP matches mirt's", () => {
    expectTrace({ items: nominal.items }, buildEventLog(nominal.events), nominal.standard);
    expectAllClose(nominal.mirt.eap, nominal.standard.trace.eap, GRID_TOLERANCE, "mirt EAP");
    expectAllClose(nominal.mirt.eapStandardError, nominal.standard.trace.eapStandardError, GRID_TOLERANCE, "mirt SD");
  });
});

describe("IRT model", () => {
  const model = irt({ items });

  it("returns itself from a fit, and works with cross-validation and model comparison unchanged", () => {
    expect(model.fit(log).model).toBe(model);
    expect(model.fit(log).aic).toBeNull();
    const folds = splitByLearner(log, 2, createRandom(1));
    expect(crossValidate(model, folds).metrics.count).toBe(log.length);
    expect(compareModels({ irt: model }, folds)[0]?.name).toBe("irt");
  });

  const dichotomous: fc.Arbitrary<IrtItem> = fc.record({
    a: fc.double({ min: 0.2, max: 3, noNaN: true }),
    b: fc.double({ min: -3, max: 3, noNaN: true }),
    c: fc.double({ min: 0, max: 0.3, noNaN: true }),
    d: fc.double({ min: 0.8, max: 1, noNaN: true }),
  });
  const answers = fc.array(fc.tuple(fc.constantFrom(...Object.keys(items)), fc.nat()), { maxLength: 20 });
  const lastCategory = (id: string): number => categoryCount(defined(items[id])) - 1;
  const eventsOf = (pairs: readonly (readonly [string, number])[]): Event[] =>
    pairs.map(([item, k], time) => {
      const last = lastCategory(item);
      return { learner: "ana", item, skills: [], time, outcome: (k % (last + 1)) / last };
    });

  it("raises the estimate after a correct answer and lowers it after a wrong one", () => {
    fc.assert(
      fc.property(dichotomous, answers, (item, pairs) => {
        const tested = irt({ items: { ...items, x: item } });
        let state = tested.initialState("ana");
        for (const event of eventsOf(pairs)) state = tested.update(state, event);
        const before = tested.ability(state).theta;
        const step = { learner: "ana", item: "x", skills: [], time: 100 };
        expect(tested.ability(tested.update(state, { ...step, outcome: 1 })).theta).toBeGreaterThan(before);
        expect(tested.ability(tested.update(state, { ...step, outcome: 0 })).theta).toBeLessThan(before);
      }),
    );
  });

  it("predicts outcomes from 0 to 1 and never changes the state it is given", () => {
    fc.assert(
      fc.property(answers, (pairs) => {
        let state = model.initialState("ana");
        for (const event of eventsOf(pairs)) {
          const prediction = model.predict(state, event);
          expect(prediction >= 0 && prediction <= 1).toBe(true);
          const copy = JSON.stringify(state);
          const next = model.update(Object.freeze(state), event);
          expect(JSON.stringify(state)).toBe(copy);
          state = next;
        }
      }),
    );
  });

  it("continues identically from a state stored as JSON", () => {
    fc.assert(
      fc.property(answers, fc.nat(), (pairs, cut) => {
        const events = eventsOf(pairs);
        const split = cut % (events.length + 1);
        let direct = model.initialState("ana");
        for (const event of events) direct = model.update(direct, event);
        let first = model.initialState("ana");
        for (const event of events.slice(0, split)) first = model.update(first, event);
        let state: IrtState = JSON.parse(JSON.stringify(first));
        for (const event of events.slice(split)) state = model.update(state, event);
        expect(state).toEqual(direct);
      }),
    );
  });

  it("gives the posterior of θ on the grid, a distribution that sums to 1", () => {
    fc.assert(
      fc.property(answers, (pairs) => {
        let state = model.initialState("ana");
        for (const event of eventsOf(pairs)) state = model.update(state, event);
        const { nodes, weights } = model.posterior(state);
        expect(nodes).toHaveLength(61);
        expect(weights).toHaveLength(61);
        expect(nodes[0]).toBe(-6);
        expect(nodes[60]).toBe(6);
        expect(nodes.every((node, q) => q === 0 || node > (nodes[q - 1] ?? 0))).toBe(true);
        expect(weights.every((weight) => weight >= 0)).toBe(true);
        expectClose(
          weights.reduce((sum, weight) => sum + weight, 0),
          1,
          1e-12,
        );
        const [mean, sd] = momentsOf({ nodes, weights });
        expectClose(mean, model.ability(state).theta, 1e-12);
        expectClose(sd, model.ability(state).standardError, 1e-12);
      }),
    );
  });

  it("starts from the prior, on the grid of the prior", () => {
    const shifted = irt({ items }, { prior: { mean: 0.5, sd: 2 }, points: 41 });
    const { nodes, weights } = shifted.posterior(shifted.initialState("ana"));
    expect(nodes[0]).toBe(0.5 - 12);
    expect(nodes[40]).toBe(0.5 + 12);
    const density = nodes.map((node) => Math.exp(-(((node - 0.5) / 2) ** 2) / 2));
    const total = density.reduce((sum, value) => sum + value, 0);
    expectAllClose(
      weights,
      density.map((value) => value / total),
      1e-15,
      "prior weights",
    );
  });

  it("returns a posterior that does not change the state, and is the same from a state stored as JSON", () => {
    const state = model.update(model.initialState("ana"), {
      learner: "ana",
      item: "g1",
      skills: [],
      time: 0,
      outcome: 1,
    });
    const frozen = Object.freeze(state);
    const posterior = model.posterior(frozen);
    expect(model.posterior(JSON.parse(JSON.stringify(frozen)))).toEqual(posterior);
    (posterior.nodes as number[]).fill(0);
    expect(model.posterior(frozen).nodes[0]).toBe(-6);
  });

  it("updates and estimates on a grid of more points than a spread into Math.max takes", () => {
    const fine = irt({ items }, { points: 200_001 });
    const state = fine.update(fine.initialState("ana"), {
      learner: "ana",
      item: "i1",
      skills: [],
      time: 0,
      outcome: 1,
    });
    expect(Number.isFinite(fine.ability(state, { estimator: "map" }).theta)).toBe(true);
  });

  it("reports a mode beyond the grid at its end", () => {
    const hard = irt({ items: { x: { a: 1, b: 10 } } });
    let state = hard.initialState("ana");
    for (let time = 0; time < 10; time++) {
      state = hard.update(state, { learner: "ana", item: "x", skills: [], time, outcome: 1 });
    }
    expect(hard.ability(state, { estimator: "map" }).theta).toBe(6);
  });

  it("gives the error of a mode as 1 / √(-ℓ''), and Infinity where ℓ is not curved down", () => {
    const model = irt({ items });
    const falling = (curvature: number): IrtState => ({
      learner: "ana",
      logDensity: Array.from({ length: 61 }, (_, q) => -q),
      slope: Array.from({ length: 61 }, () => -1),
      curvature: Array.from({ length: 61 }, () => curvature),
    });
    expect(model.ability(falling(-4), { estimator: "map" }).standardError).toBeCloseTo(0.5, 12);
    expect(model.ability(falling(0), { estimator: "map" }).standardError).toBe(Number.POSITIVE_INFINITY);
  });
});

describe("IRT with nominal items", () => {
  /** A nominal item of 2 to 5 categories, its slopes and intercepts from -3 to 3. */
  const nominalItem: fc.Arbitrary<NominalItem> = fc.integer({ min: 2, max: 5 }).chain((count) => {
    const values = fc.array(fc.double({ min: -3, max: 3, noNaN: true }), { minLength: count, maxLength: count });
    return fc.record({ model: fc.constant("nominal" as const), a: values, c: values });
  });
  const answers = fc.array(fc.tuple(fc.constantFrom(...Object.keys(items)), fc.nat()), { maxLength: 12 });
  /** Returns the state of a learner after an answer to each item of `pairs`, in a category taken modulo its count. */
  const stateAfter = (model: IrtModel, pairs: readonly (readonly [string, number])[]): IrtState => {
    let state = model.initialState("ana");
    pairs.forEach(([item, k], time) => {
      const last = categoryCount(defined(items[item])) - 1;
      state = model.update(state, { learner: "ana", item, skills: [], time, outcome: (k % (last + 1)) / last });
    });
    return state;
  };
  const answer = (outcome: number): Event => ({ learner: "ana", item: "x", skills: [], time: 100, outcome });

  it("raises the estimate after the category of the largest slope and lowers it after that of the smallest", () => {
    // P(k) of the largest slope rises with θ, as its log-slope a_k - E[a] is never below 0, and that of the smallest
    // falls: the posterior moves with it. A category between them has a P(k) that rises and then falls.
    fc.assert(
      fc.property(nominalItem, answers, (drawn, pairs) => {
        // Slopes a tenth apart at least, so that the direction is not lost to rounding.
        const item = { ...drawn, a: drawn.a.map((slope) => Math.round(10 * slope) / 10) };
        fc.pre(new Set(item.a).size === item.a.length);
        const tested = irt({ items: { ...items, x: item } });
        const state = stateAfter(tested, pairs);
        const before = tested.ability(state).theta;
        const last = item.a.length - 1;
        const highest = item.a.indexOf(Math.max(...item.a));
        const lowest = item.a.indexOf(Math.min(...item.a));
        expect(tested.ability(tested.update(state, answer(highest / last))).theta).toBeGreaterThan(before);
        expect(tested.ability(tested.update(state, answer(lowest / last))).theta).toBeLessThan(before);
      }),
    );
  });

  it("is unchanged by a constant added to every slope or to every intercept", () => {
    fc.assert(
      fc.property(nominalItem, fc.double({ min: -2, max: 2, noNaN: true }), fc.nat(), (item, shift, k) => {
        const moved: NominalItem = {
          ...item,
          a: item.a.map((slope) => slope + shift),
          c: item.c.map((c) => c - shift),
        };
        const [one, other] = [item, moved].map((x) => irt({ items: { x } }));
        const outcome = (k % item.a.length) / (item.a.length - 1);
        const [first, second] = [one, other].map((model) =>
          defined(model).update(defined(model).initialState("ana"), answer(outcome)),
        );
        expectAllClose(defined(second).logDensity, defined(first).logDensity, 1e-12, "log-density");
        expectAllClose(defined(second).slope ?? [], defined(first).slope ?? [], 1e-12, "slope");
        expectAllClose(defined(second).curvature ?? [], defined(first).curvature ?? [], 1e-12, "curvature");
        expectClose(
          defined(other).predict(defined(second), answer(0)),
          defined(one).predict(defined(first), answer(0)),
          1e-12,
        );
      }),
    );
  });

  it("gives the states of the 2PL item that a nominal item of two categories is, and of a GPCM item with slopes k a", () => {
    const run = (x: IrtItem, outcomes: readonly number[]): IrtState => {
      const model = irt({ items: { x } });
      return outcomes.reduce(
        (state, outcome, time) => model.update(state, { ...answer(outcome), time }),
        model.initialState("ana"),
      );
    };
    const expectSameStates = (found: IrtState, expected: IrtState, label: string): void => {
      expectAllClose(found.logDensity, expected.logDensity, 1e-12, `${label} log-density`);
      expectAllClose(found.slope ?? [], expected.slope ?? [], 1e-12, `${label} slope`);
      expectAllClose(found.curvature ?? [], expected.curvature ?? [], 1e-11, `${label} curvature`);
    };
    // P(1) = exp(a θ + c) / (1 + exp(a θ + c)) is the 2PL with b = -c / a.
    const binary = [1, 0, 1, 1, 0];
    expectSameStates(
      run({ model: "nominal", a: [0, 1.4], c: [0, -0.7] }, binary),
      run({ a: 1.4, b: 0.5 }, binary),
      "two categories",
    );
    // The GPCM's logits are Σ_{j ≤ k} a (θ - b_j) = k a θ - a Σ_{j ≤ k} b_j.
    const partial = [2 / 3, 0, 1, 1 / 3, 1, 2 / 3];
    expectSameStates(
      run({ model: "nominal", a: [0, 1.1, 2.2, 3.3], c: [0, 0.77, 0.33, -0.99] }, partial),
      run({ model: "gpcm", a: 1.1, b: [-0.7, 0.4, 1.2] }, partial),
      "slopes k a",
    );
  });
});

describe("IRT edge cases", () => {
  const model = irt({ items });
  const event: Event = { learner: "ana", item: "i1", skills: [], time: 1, outcome: 1 };

  it("rejects invalid items, priors and grids", () => {
    const build = (item: unknown) => () => irt({ items: { x: item as IrtItem } });
    expect(build({ a: 1, b: 0, c: 0.5, d: 0.4 })).toThrow('Item "x" needs 0 ≤ c < d ≤ 1, found c = 0.5 and d = 0.4.');
    expect(build({ a: 1, b: 0, c: "0.2" })).toThrow('Item "x" needs 0 ≤ c < d ≤ 1, found c = "0.2" and d = 1.');
    expect(build({ a: 1, b: 0, d: null })).toThrow('Item "x" needs 0 ≤ c < d ≤ 1, found c = 0 and d = null.');
    expect(build({ a: Number.NaN, b: 0 })).toThrow('Item "x" needs a, a finite number, found NaN.');
    expect(build({ a: 1, b: "0" })).toThrow('Item "x" needs b, a finite number, found "0".');
    expect(build({ model: "graded", a: 1, b: [1, 0] })).toThrow(
      'Item "x" is graded, so needs thresholds b strictly increasing when a > 0 and decreasing when a < 0, found a = 1, b = 1, 0.',
    );
    expect(() => build({ model: "graded", a: -1, b: [1, 0] })()).not.toThrow();
    expect(build({ model: "ordinal", a: 1, b: [0] })).toThrow('polytomous items are "graded", "gpcm" or "nominal"');
    expect(build({ model: "gpcm", a: 1, b: [] })).toThrow("non-empty array of finite thresholds");
    expect(build({ model: "gpcm", a: "1", b: [0] })).toThrow('Item "x" needs a, a finite number, found "1".');
    expect(build({ model: "nominal", a: [0, 1], c: [0] })).toThrow(
      'Item "x" is nominal, so needs a and c, arrays of finite numbers, a slope and an intercept for each of at least two categories, found a = [0, 1] and c = [0].',
    );
    expect(build({ model: "nominal", a: [1], c: [0] })).toThrow("for each of at least two categories");
    expect(build({ model: "nominal", a: [0, Number.NaN], c: [0, 0] })).toThrow("found a = [0, NaN]");
    expect(build(JSON.parse('{"model":"nominal","a":[0,1],"c":"0,0"}'))).toThrow('and c = "0,0".');
    expect(build({ model: "nominal", a: 1, b: [0] })).toThrow("found a = 1 and c = undefined.");
    // A hole, an index that holds no element, which every() would skip and a read would take as 0.
    const holed = (values: readonly number[], length: number): number[] =>
      Object.assign(new Array<number>(length), values);
    expect(build({ model: "nominal", a: holed([0, 1], 3), c: [0, 0, 0] })).toThrow("arrays of finite numbers");
    expect(build({ model: "gpcm", a: 1, b: holed([0], 2) })).toThrow("non-empty array of finite thresholds");
    expect(() => irt({ items }, { prior: { mean: 0, sd: 0 } })).toThrow(
      "The sd of the prior must be a positive number, found 0.",
    );
    expect(() => irt({ items }, { points: 2 })).toThrow("The option points must be an integer from 3");
    expect(() => irt({ items }, { prior: { mean: 0, sd: 1e-300 } })).toThrow(
      "The prior needs a mean and an sd whose grid has distinct, finite points, with finite squares of their spacing and of 1 / sd, found mean 0 and sd 1e-300.",
    );
    expect(() => irt({ items }, { prior: { mean: 0, sd: 1e300 } })).toThrow("finite squares of their spacing");
    expect(() => irt({ items }, { prior: { mean: 1e308, sd: 1 } })).toThrow("distinct, finite points");
  });

  it("keeps the log-probability of a wrong answer to an easy item where P(correct) rounds to 1", () => {
    // At θ = 6, a (θ - b) = 40 and 1 - logistic(40) is 0 in doubles; ln P(wrong) is ln logistic(-40).
    const easy = irt({ items: { x: { a: 4, b: -4 } } });
    const prior = easy.initialState("ana");
    const state = easy.update(prior, { ...event, item: "x", outcome: 0 });
    const gain = (q: number) => (state.logDensity[q] ?? 0) - (prior.logDensity[q] ?? 0);
    const logWrong = (theta: number) => -Math.log1p(Math.exp(4 * (theta + 4)));
    expect(gain(60) - gain(59)).toBeCloseTo(logWrong(6) - logWrong(5.8), 12);
    expect(easy.predict(state, { ...event, item: "x" })).toBeGreaterThan(0);
  });

  it("keeps states finite for any item and outcomes, however extreme", () => {
    const extreme = fc.record({
      a: fc.double({ min: 0.1, max: 60, noNaN: true }),
      b: fc.double({ min: -6, max: 6, noNaN: true }),
      c: fc.constantFrom(0, 0.2),
      d: fc.constantFrom(1, 0.95),
    });
    fc.assert(
      fc.property(extreme, fc.array(fc.constantFrom(0, 1), { minLength: 1, maxLength: 30 }), (item, outcomes) => {
        const tested = irt({
          items: {
            x: item,
            g: { model: "graded", a: item.a, b: [item.b, item.b + 1] },
            n: { model: "nominal", a: [-item.a, 0, item.a], c: [item.b, 0, -item.b] },
          },
        });
        let state = tested.initialState("ana");
        outcomes.forEach((outcome, time) => {
          state = tested.update(JSON.parse(JSON.stringify(state)), {
            ...event,
            item: ["x", "g", "n"][time % 3] ?? "x",
            time,
            outcome,
          });
        });
        const prediction = tested.predict(state, { ...event, item: "x" });
        expect(prediction >= 0 && prediction <= 1).toBe(true);
        expect(Number.isNaN(tested.ability(state, { estimator: "map" }).standardError)).toBe(false);
      }),
    );
  });

  it("rejects unknown items, outcomes that are not categories, other learners' and malformed states", () => {
    const state = model.initialState("ana");
    expect(() => model.fit([])).toThrow("A log needs at least one event.");
    expect(() => model.update(state, { ...event, item: "zz" })).toThrow('The model has no parameters for item "zz".');
    expect(() => model.update(state, { ...event, outcome: 0.5 })).toThrow(
      'The outcome 0.5 of item "i1" is not one of its 2 categories, k / 1.',
    );
    expect(model.update(state, { ...event, item: "g1", outcome: 1 / 3 }).learner).toBe("ana");
    expect(() => model.predict(state, { ...event, learner: "budi" })).toThrow('given the state of learner "ana"');
    expect(() => model.predict({ ...state, slope: [] }, event)).toThrow("needs slope, an array of 61 finite numbers");
    expect(() => model.predict({ ...state, slope: state.logDensity.map(() => Number.NaN) }, event)).toThrow(
      "needs slope, an array of 61 finite numbers",
    );
    expect(() => model.predict({ ...state, logDensity: state.logDensity.map((value) => value + 800) }, event)).toThrow(
      "needs logDensity less its maximum, so that its largest value is 0",
    );
    expect(() => model.ability(state, { estimator: "mle" as "eap" })).toThrow('"eap" or "map"');
    expect(() => model.posterior({ ...state, curvature: [] })).toThrow(
      "needs curvature, an array of 61 finite numbers",
    );
    expect(() => model.posterior(JSON.parse("null"))).toThrow(DataError);
    expect(() => model.initialState("")).toThrow(DataError);
  });
});

describe("IRT options from JavaScript", () => {
  it("rejects options, priors and ability options that are not objects", () => {
    const missing = JSON.parse("null");
    expect(() => irt({ items }, missing)).toThrow("The options must be an object, found null.");
    expect(() => irt({ items }, JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
    expect(() => irt(missing)).toThrow("IRT parameters must be an object, found null.");
    expect(() => irt({ items }, { prior: missing })).toThrow("The prior must be an object { mean, sd }, found null.");
    expect(() => irt({ items }, { prior: JSON.parse('{"mean":0,"sd":1,"df":3}') })).toThrow(
      'The prior may hold only mean and sd, found "df".',
    );
    const model = irt({ items });
    expect(() => model.ability(model.initialState("ana"), missing)).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => model.ability(model.initialState("ana"), JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
  });
});

describe("IRT with events that have no outcome", () => {
  const model = irt({ items });

  it("is unaffected by them: a lesson is no item, and it does not move the posterior", () => {
    expectUnaffectedByLessons(model, log);
  });

  it("returns the state as it was, as a copy, for an event without an outcome", () => {
    const state = model.update(model.initialState("ana"), { ...defined(log[0]), learner: "ana" });
    const next = model.update(state, { learner: "ana", item: "lesson", skills: [], time: 99, outcome: null });
    expect(next).toEqual(state);
    expect(next.logDensity).not.toBe(state.logDensity);
  });
});

/** Mean, SD and MAP of the posterior at the end of a pattern, against the grid of 1921 points, by number of points. */
const FINE_TOLERANCES = {
  61: { mean: 0.03, sd: 0.01, mode: 0.06, steady: 0.17 },
  121: { mean: 0.004, sd: 0.003, mode: 0.009, steady: 0.1 },
  241: { mean: 0.001, sd: 0.0007, mode: 0.003, steady: 0.04 },
} as const;
/** The mean through the reversal at a = 1.2 against the grid of 1921 points, by number of points. */
const REVERSAL_TOLERANCES = { 121: 0.07, 241: 0.006 } as const;
/** Mean, SD and log-likelihood against the same chain computed in logs, while no grid point the mass reaches underflowed. */
const SAME_GRID_TOLERANCE = 1e-9;
/** Mean and SD against the same chain in logs once the final mode sank below where exp underflows, at -744.4. */
const DEEP_TOLERANCE = 1e-3;
const UNDERFLOW_DEPTH = -Math.log(Number.MIN_VALUE);
/** The variance added by pure drift, relative to the due variance. */
const VARIANCE_TOLERANCE = 1e-6;
/** The mass after a step of the walk, relative to the mass before, after as many as 942 parts at the cap. */
const MASS_TOLERANCE = 1e-11;
/** The posterior after a gap past the cap, relative to the uniform distribution on the grid. */
const UNIFORM_TOLERANCE = 1e-11;
/** The posterior after two gaps against the posterior after one gap of their sum. */
const COMPOSITION_TOLERANCE = 1e-12;

const DRIFT = drifting.drift as IrtDrift;
const MINUTE = 60_000;
const DAY = 86_400_000;
const WEEK = 7 * DAY;
const START = 1_767_225_600_000;

type DriftCase = (typeof drifting.cases)[number];

/** Returns a case of the drift fixture, by name. */
function caseNamed(name: string): DriftCase {
  return defined(drifting.cases.find((driftCase) => driftCase.name === name));
}

/** Returns the items and the log of a case of the drift fixture, with an item of its own for each event. */
function caseLog({ events }: DriftCase): { parameters: IrtParameters; log: EventLog } {
  const items = Object.fromEntries(events.a.map((a, i) => [`item${i}`, { a, b: defined(events.b[i]) }]));
  const log = buildEventLog(
    events.time.map((time, i) => ({
      learner: "ana",
      item: `item${i}`,
      skills: [],
      time,
      outcome: defined(events.outcome[i]),
    })),
  );
  return { parameters: { items }, log };
}

/** Runs a case through a model with drift on `points` points: EAP, SD and MAP at its checkpoints, and the fit. */
function runCase(driftCase: DriftCase, points: number) {
  const { parameters, log } = caseLog(driftCase);
  const model = irt(parameters, { points, drift: DRIFT });
  const checkpoints = new Set(driftCase.checkpoints);
  const trace = { mean: [] as number[], sd: [] as number[], mode: [] as number[] };
  let state = model.initialState("ana");
  log.forEach((event, i) => {
    state = model.update(state, event);
    if (!checkpoints.has(i)) return;
    const eap = model.ability(state);
    trace.mean.push(eap.theta);
    trace.sd.push(eap.standardError);
    trace.mode.push(model.ability(state, { estimator: "map" }).theta);
  });
  return { ...trace, logLikelihood: model.fit(log).logLikelihood };
}

/** Returns the absolute difference of the last entries of two lists. */
function lastGap(actual: readonly number[], expected: readonly number[]): number {
  return Math.abs(defined(actual.at(-1)) - defined(expected.at(-1)));
}

describe("IRT with drift against R", () => {
  it("follows the same chain computed in logs, event by event, on the same grid", () => {
    for (const driftCase of drifting.cases) {
      const run = runCase(driftCase, drifting.points);
      const { mean, sd, logLikelihood, depth } = driftCase.sameGrid;
      const deep = depth > UNDERFLOW_DEPTH;
      const tolerance = deep ? DEEP_TOLERANCE : SAME_GRID_TOLERANCE;
      expectAllClose(run.mean, mean, tolerance, `${driftCase.name} mean`);
      expectAllClose(run.sd, sd, tolerance, `${driftCase.name} SD`);
      if (!deep) expectClose(run.logLikelihood, logLikelihood, SAME_GRID_TOLERANCE, `${driftCase.name} likelihood`);
    }
  }, 30_000);

  it("meets the tolerances of each number of points against a grid of 1921 points", () => {
    for (const [points, tolerance] of Object.entries(FINE_TOLERANCES)) {
      for (const driftCase of drifting.cases.filter(({ kind }) => kind !== "reversal")) {
        const run = runCase(driftCase, Number(points));
        const label = `${driftCase.name} on ${points} points`;
        if (driftCase.kind === "steady") {
          expect(lastGap(run.mean, driftCase.fine.mean), label).toBeLessThanOrEqual(tolerance.steady);
          continue;
        }
        expect(lastGap(run.mean, driftCase.fine.mean), `${label}, mean`).toBeLessThanOrEqual(tolerance.mean);
        expect(lastGap(run.sd, driftCase.fine.sd), `${label}, SD`).toBeLessThanOrEqual(tolerance.sd);
        expect(lastGap(run.mode, driftCase.fine.mode), `${label}, MAP`).toBeLessThanOrEqual(tolerance.mode);
      }
    }
    const reversal = caseNamed("reversal-60-120");
    for (const [points, tolerance] of Object.entries(REVERSAL_TOLERANCES)) {
      expectAllClose(runCase(reversal, Number(points)).mean, reversal.fine.mean, tolerance, `reversal on ${points}`);
    }
  }, 40_000);

  it("needs 481 points for a reversal at a = 2, and 961 at a = 3, where the posterior is narrower than 241 resolve", () => {
    for (const [name, points, tolerance] of [
      ["reversal-90-200", 481, 0.03],
      ["reversal-150-300", 961, 0.07],
    ] as const) {
      const driftCase = caseNamed(name);
      const coarse = runCase(driftCase, drifting.points);
      expect(defined(coarse.sd.at(-1)), name).toBeLessThan(3 * (12 / (drifting.points - 1)));
      expect(lastGap(coarse.mean, driftCase.fine.mean), name).toBeGreaterThan(REVERSAL_TOLERANCES[241]);
      expect(lastGap(runCase(driftCase, points).mean, driftCase.fine.mean), name).toBeLessThanOrEqual(tolerance);
    }
  });

  it("matches a particle filter of the continuous model within three Monte Carlo errors", () => {
    const { case: name, mean, meanError } = drifting.particleFilter;
    const run = runCase(caseNamed(name), drifting.points);
    expect(Math.abs(defined(run.mean.at(-1)) - mean)).toBeLessThanOrEqual(3 * meanError);
  });
});

describe("IRT with drift", () => {
  const items: Record<string, IrtItem> = { easy: { a: 1.5, b: -1 }, hard: { a: 1.5, b: 1 } };
  const model = irt({ items }, { points: 241, drift: DRIFT });
  const attempt = { learner: "ana", item: "easy", skills: [], time: START };
  /** A state at START whose posterior has an SD of about 0.4, after 10 outcomes at one time. */
  const settled = Array.from({ length: 10 }, (_, k) => k).reduce(
    (state, k) => model.update(state, { ...attempt, item: k % 2 ? "easy" : "hard", outcome: k % 3 ? 1 : 0 }),
    model.initialState("ana"),
  );
  const varianceOf = (posterior: ThetaDistribution): number => momentsOf(posterior)[1] ** 2;

  it("adds exactly the due variance, and keeps the mean, at gaps from a second to four weeks", () => {
    const before = model.posterior(settled);
    for (const gap of [1000, MINUTE, 60 * MINUTE, DAY, WEEK, 4 * WEEK]) {
      const after = model.posterior(settled, { at: START + gap });
      const due = (DRIFT.variance * gap) / DRIFT.per;
      expect(Math.abs((varianceOf(after) - varianceOf(before)) / due - 1), `gap ${gap}`).toBeLessThan(
        VARIANCE_TOLERANCE,
      );
      expect(momentsOf(after)[0]).toBeCloseTo(momentsOf(before)[0], 12);
    }
  });

  it("conserves mass and keeps every density finite and above its lower bound, however long the gap", () => {
    const walk = reflectedWalk(241);
    const logDensity = settled.logDensity;
    const mass = (values: readonly number[]): number => values.reduce((sum, value) => sum + Math.exp(value), 0);
    for (const jumps of [1e-9, 1e-3, 0.5, 8, 499, 501, 2e4, 1e12]) {
      const next = walk(logDensity, jumps);
      expect(Math.abs(mass(next) / mass(logDensity) - 1), `${jumps} jumps`).toBeLessThan(MASS_TOLERANCE);
      expect(next.every((value, q) => Number.isFinite(value) && value >= (logDensity[q] ?? 0) - jumps)).toBe(true);
    }
  });

  it("composes: the posterior after two gaps is the posterior after one gap of their sum", () => {
    const lesson = { ...attempt, item: "lesson", outcome: null };
    for (const [first, second] of [
      [1000, 59_000],
      [MINUTE, DAY],
      [3 * DAY, 4 * DAY],
      [WEEK, 3 * WEEK],
    ] as const) {
      const once = model.posterior(settled, { at: START + first + second });
      const halfway = model.update(settled, { ...lesson, time: START + first });
      const twice = model.posterior(halfway, { at: START + first + second });
      expectAllClose(twice.weights, once.weights, COMPOSITION_TOLERANCE, `${first} then ${second}`);
    }
    let minutes = settled;
    for (let k = 1; k <= 60; k++) minutes = model.update(minutes, { ...lesson, time: START + k * MINUTE });
    expectAllClose(
      model.posterior(minutes).weights,
      model.posterior(settled, { at: START + 60 * MINUTE }).weights,
      1e-12,
    );
  });

  it("predicts and estimates from the posterior advanced to the attempt's time or to `at`", () => {
    const later = START + 2 * WEEK;
    const { nodes, weights } = model.posterior(settled, { at: later });
    const expected = weights.reduce((sum, weight, q) => sum + weight / (1 + Math.exp(-1.5 * ((nodes[q] ?? 0) + 1))), 0);
    expect(model.predict(settled, { ...attempt, time: later })).toBeCloseTo(expected, 12);
    expect(model.predict(settled, { ...attempt, time: later })).not.toBeCloseTo(model.predict(settled, attempt), 4);
    const [mean, sd] = momentsOf(model.posterior(settled, { at: later }));
    expect(model.ability(settled, { at: later }).theta).toBeCloseTo(mean, 12);
    expect(model.ability(settled, { at: later }).standardError).toBeCloseTo(sd, 12);
    expect(model.ability(settled).standardError).toBeLessThan(sd);
  });

  it("takes no drift for an event, attempt or time before the state's time, and keeps that time", () => {
    const early = START - DAY;
    const fixed = irt({ items }, { points: 241 });
    const static_ = fixed.update(fixed.initialState("ana"), { ...attempt, outcome: 1 });
    const next = model.update(model.initialState("ana"), { ...attempt, outcome: 1 });
    const back = model.update(next, { ...attempt, item: "hard", time: early, outcome: 0 });
    expect(back.time).toBe(START);
    expect(back.logDensity).toEqual(
      fixed.update(static_, { ...attempt, item: "hard", time: early, outcome: 0 }).logDensity,
    );
    expect(model.predict(settled, { ...attempt, time: early })).toBe(model.predict(settled, attempt));
    expect(model.ability(settled, { at: early })).toEqual(model.ability(settled));
    expect(model.posterior(settled, { at: early })).toEqual(model.posterior(settled));
  });

  it("starts without a time, takes its first event's time with no drift, and keeps no derivatives", () => {
    const fixed = irt({ items }, { points: 241 });
    expect(model.initialState("ana")).toEqual({ learner: "ana", logDensity: fixed.initialState("ana").logDensity });
    const first = model.update(model.initialState("ana"), { ...attempt, outcome: 1 });
    expect(first).toEqual({
      learner: "ana",
      logDensity: fixed.update(fixed.initialState("ana"), { ...attempt, outcome: 1 }).logDensity,
      time: START,
    });
    expect(model.predict(model.initialState("ana"), attempt)).toBe(fixed.predict(fixed.initialState("ana"), attempt));
  });

  it("advances the state to the time of an event without an outcome", () => {
    const lesson = { ...attempt, item: "video", time: START + DAY, outcome: null };
    const next = model.update(settled, lesson);
    expect(next.time).toBe(START + DAY);
    expect(model.posterior(next)).toEqual(model.posterior(settled, { at: START + DAY }));
  });

  it("fits with the forward likelihood, and gives the states and predictions that updates give, lessons included", () => {
    const events: Event[] = [
      { ...attempt, outcome: 1 },
      { ...attempt, item: "hard", time: START + DAY, outcome: 0 },
      { ...attempt, item: "lesson", time: START + 3 * DAY, outcome: null },
      { ...attempt, item: "hard", time: START + WEEK, outcome: 1 },
      { ...attempt, item: "lesson", time: START + 2 * WEEK, outcome: null },
    ];
    const fit = model.fit(buildEventLog(events));
    let state = model.initialState("ana");
    let likelihood = 0;
    const predictions = events.map((event) => {
      const prediction = event.outcome === null ? null : model.predict(state, event);
      if (prediction !== null) likelihood += Math.log(event.outcome ? prediction : 1 - prediction);
      state = model.update(state, event);
      return prediction;
    });
    expect(fit.states).toEqual({ ana: state });
    expect(state.time).toBe(START + 2 * WEEK);
    expect(fit.predictions).toEqual(predictions);
    expect(fit.logLikelihood).toBeCloseTo(likelihood, 12);
  });

  it("goes flat after a gap past its cap, and returns at once for a wrong unit of time", () => {
    const wrongUnit = irt({ items }, { points: 241, drift: { variance: 0.02, per: 7 } });
    const state = wrongUnit.update(wrongUnit.initialState("ana"), { ...attempt, outcome: 1 });
    const { weights } = wrongUnit.posterior(state, { at: START + 30 * DAY });
    for (const weight of weights) expect(Math.abs(241 * weight - 1)).toBeLessThan(UNIFORM_TOLERANCE);
  });

  it("derives the derivatives of the log-density by differences exact for a polynomial of degree 4", () => {
    const h = 0.05;
    const f = (x: number) => 1 - 2 * x + 0.5 * x ** 2 - 0.3 * x ** 3 + 0.1 * x ** 4;
    const slope = (x: number) => -2 + x - 0.9 * x ** 2 + 0.4 * x ** 3;
    const curvature = (x: number) => 1 - 1.8 * x + 1.2 * x ** 2;
    const grid = Array.from({ length: 9 }, (_, q) => -0.2 + q * h);
    const derived = differences(grid.map(f), h);
    grid.forEach((x, q) => {
      expect(derived(q)[0], `slope at ${q}`).toBeCloseTo(slope(x), 9);
      expect(derived(q)[1], `curvature at ${q}`).toBeCloseTo(curvature(x), 7);
    });
  });

  it("reads a stored state of a model without drift, and drops its derivatives at the first event", () => {
    const found = JSON.parse(JSON.stringify(stored.models.irt));
    const fixed = irt(found.parameters);
    const reading = irt(found.parameters, { points: 61, drift: DRIFT });
    const state: IrtState = found.states[found.attempt.learner];
    expect(reading.predict(state, found.attempt)).toBe(found.prediction);
    expect(reading.ability(state)).toEqual(fixed.ability(state));
    expect(reading.ability(state, { estimator: "map" }).theta).toBeCloseTo(
      fixed.ability(state, { estimator: "map" }).theta,
      3,
    );
    expect(reading.update(state, found.answered.event)).toEqual({
      learner: found.attempt.learner,
      logDensity: found.answered.state.logDensity,
      time: found.answered.event.time,
    });
    expect(reading.update(state, found.lesson.event)).toEqual({
      learner: found.attempt.learner,
      logDensity: state.logDensity,
      time: found.lesson.event.time,
    });
  });

  it("continues identically from a state stored as JSON, and never changes the state it is given", () => {
    fc.assert(
      fc.property(fc.array(fc.tuple(fc.boolean(), fc.nat({ max: 10 * DAY })), { maxLength: 8 }), (steps) => {
        let state = model.initialState("ana");
        let time = START;
        for (const [correct, gap] of steps) {
          time += gap;
          const event = { ...attempt, time, outcome: correct ? 1 : 0 };
          const copy = JSON.parse(JSON.stringify(state));
          const next = model.update(state, event);
          expect(state).toEqual(copy);
          expect(model.update(copy, event)).toEqual(next);
          state = next;
        }
        expect(model.predict(state, { ...attempt, time: time + DAY })).toBe(
          model.predict(JSON.parse(JSON.stringify(state)), { ...attempt, time: time + DAY }),
        );
      }),
      { numRuns: 30 },
    );
  });

  it("rejects drift without points, invalid drift and points, and states and times it cannot read", () => {
    const parameters = { items };
    expect(() => irt(parameters, { drift: DRIFT })).toThrow(
      "The option drift needs the option points: 241 is recommended, or the number of points of stored states.",
    );
    expect(() => irt(parameters, { points: 5, drift: DRIFT })).toThrow(
      "The option points must be an integer from 6, found 5.",
    );
    expect(() => irt(parameters, { points: 241, drift: JSON.parse("null") })).toThrow(
      "The option drift must be an object { variance, per }, found null.",
    );
    expect(() => irt(parameters, { points: 241, drift: JSON.parse('{"variance":0.1,"per":1,"rate":1}') })).toThrow(
      'The option drift may hold only variance and per, found "rate".',
    );
    expect(() => irt(parameters, { points: 241, drift: { variance: 0, per: 1 } })).toThrow(
      "The variance of the drift must be a positive number, found 0.",
    );
    expect(() => irt(parameters, { points: 241, drift: { variance: 1, per: JSON.parse('"7"') } })).toThrow(
      'The per of the drift must be a positive number, found "7".',
    );
    const fixed = irt(parameters, { points: 241 });
    expect(() => fixed.predict(settled, attempt)).toThrow(
      'The IRT state of learner "ana" has no slope, which a model without the option drift needs: a state written by a model with drift is read by a model with it.',
    );
    expect(() => model.predict({ ...settled, time: JSON.parse("null") }, attempt)).toThrow(
      'The IRT state of learner "ana" needs time, a finite number, or none before its first event, found null.',
    );
    expect(() => model.ability(settled, { at: Number.NaN })).toThrow(
      "The option at must be a finite number, found NaN.",
    );
    expect(() => model.posterior(settled, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => model.posterior(settled, JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
    expect(() => fixed.posterior(fixed.initialState("ana"), { at: JSON.parse('"1"') })).toThrow(
      'The option at must be a finite number, found "1".',
    );
  });
});

describe("IRT items against the probabilities of core", () => {
  /** Returns P(k) of an item at θ for each category, from core. */
  function probabilitiesOf(item: IrtItem, theta: number): Float64Array {
    const out = new Float64Array(categoryCount(item));
    if (!("model" in item)) logisticProbabilities(item.a * (theta - item.b), item.c ?? 0, item.d ?? 1, out);
    else if (item.model === "graded") gradedProbabilities(item.a, item.b, theta, out);
    else if (item.model === "gpcm") partialCreditProbabilities(item.a, 0, item.b, theta, out);
    else nominalProbabilities(item.a, item.c, theta, out);
    return out;
  }

  const slopes = fc.double({ min: 0.2, max: 3, noNaN: true });
  const dichotomous: fc.Arbitrary<IrtItem> = fc.record({
    a: slopes,
    b: fc.double({ min: -3, max: 3, noNaN: true }),
    c: fc.constantFrom(0, 0.2),
    d: fc.constantFrom(1, 0.95),
  });
  const steps = fc.array(fc.double({ min: 0.2, max: 1.5, noNaN: true }), { minLength: 1, maxLength: 4 });
  const graded: fc.Arbitrary<IrtItem> = fc
    .tuple(slopes, fc.double({ min: -3, max: 0, noNaN: true }), steps)
    .map(([a, start, gaps]) => {
      let location = start;
      const b = gaps.map((gap) => {
        location += gap;
        return location;
      });
      return { model: "graded", a, b };
    });
  const partialCredit: fc.Arbitrary<IrtItem> = fc
    .tuple(slopes, fc.array(fc.double({ min: -3, max: 3, noNaN: true }), { minLength: 1, maxLength: 4 }))
    .map(([a, b]) => ({ model: "gpcm", a, b }));
  const nominal: fc.Arbitrary<IrtItem> = fc.integer({ min: 2, max: 5 }).chain((count) => {
    const values = fc.array(fc.double({ min: -3, max: 3, noNaN: true }), { minLength: count, maxLength: count });
    return fc.record({ model: fc.constant("nominal" as const), a: values, c: values });
  });
  const anyItem = fc.oneof(dichotomous, graded, partialCredit, nominal);
  const theta = fc.double({ min: -6, max: 6, noNaN: true });

  it("the probabilities of an item's categories sum to 1", () => {
    fc.assert(
      fc.property(anyItem, theta, (item, point) => {
        const terms = new Float64Array(3);
        let total = 0;
        for (let k = 0; k < categoryCount(item); k++) {
          categoryTerms(item, k, point, terms);
          total += Math.exp(terms[0] ?? 0);
        }
        expect(Math.abs(total - 1)).toBeLessThan(CORE_TOLERANCE);
      }),
    );
  });

  it("the log-probability of each category is the logarithm of its probability", () => {
    fc.assert(
      fc.property(anyItem, theta, (item, point) => {
        const probabilities = probabilitiesOf(item, point);
        const terms = new Float64Array(3);
        for (let k = 0; k < probabilities.length; k++) {
          categoryTerms(item, k, point, terms);
          expect(Math.abs(Math.exp(terms[0] ?? 0) - (probabilities[k] ?? 0))).toBeLessThan(CORE_TOLERANCE);
        }
      }),
    );
  });

  it("the slope of the log-probability is the slope of the probability over the probability", () => {
    const step = DIFFERENCE_STEP;
    fc.assert(
      fc.property(anyItem, theta, (item, point) => {
        const [two, one, back, backTwo] = [2 * step, step, -step, -2 * step].map((shift) =>
          probabilitiesOf(item, point + shift),
        );
        const probabilities = probabilitiesOf(item, point);
        const terms = new Float64Array(3);
        for (let k = 0; k < probabilities.length; k++) {
          const derivative =
            (8 * ((one?.[k] ?? 0) - (back?.[k] ?? 0)) - ((two?.[k] ?? 0) - (backTwo?.[k] ?? 0))) / (12 * step);
          categoryTerms(item, k, point, terms);
          expect(Math.abs((terms[1] ?? 0) * (probabilities[k] ?? 0) - derivative)).toBeLessThan(DIFFERENCE_TOLERANCE);
        }
      }),
    );
  });

  it("the expected outcome is the expected category over K - 1 of the probabilities", () => {
    fc.assert(
      fc.property(anyItem, theta, (item, point) => {
        const probabilities = probabilitiesOf(item, point);
        let expected = 0;
        probabilities.forEach((probability, k) => {
          expected += (k * probability) / (probabilities.length - 1);
        });
        expect(Math.abs(expectedOutcome(item, point, new Float64Array(3)) - expected)).toBeLessThan(CORE_TOLERANCE);
      }),
    );
  });

  it("evaluates items of more categories than the scratch space of a caller holds", () => {
    const item: IrtItem = { model: "gpcm", a: 1.1, b: [-1, -0.3, 0.4, 1.2, 2] };
    const probabilities = probabilitiesOf(item, 0.3);
    const expected = probabilities.reduce((sum, probability, k) => sum + (k * probability) / 5, 0);
    expect(expectedOutcome(item, 0.3, new Float64Array(3))).toBeCloseTo(expected, 14);
    const terms = new Float64Array(3);
    categoryTerms(item, 5, 0.3, terms);
    expect(Math.exp(terms[0] ?? 0)).toBeCloseTo(probabilities[5] ?? 0, 14);
  });
});

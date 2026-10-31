import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  compareModels,
  createRandom,
  crossValidate,
  DataError,
  type Event,
  eventLog,
  type IrtItem,
  type IrtPrior,
  type IrtState,
  irt,
  splitByLearner,
  type ThetaDistribution,
} from "../src/index.js";
import reference from "./fixtures/irt-abilities.json" with { type: "json" };
import { expectAllClose, expectClose } from "./helpers.js";

/** Absolute tolerance of EAP estimates, predictions and log-probabilities against the written-out grid posterior. */
const GRID_TOLERANCE = 1e-10;
/**
 * Absolute tolerance of the MAP estimate against the exact posterior mode at the default grid spacing, 0.2: the
 * error of the quintic interpolation between grid points, which grows with the fifth power of the spacing.
 */
const MODE_TOLERANCE = 1e-5;
/** Absolute tolerance of the MAP standard error at the default spacing, growing with its fourth power. */
const MODE_ERROR_TOLERANCE = 1e-4;

/** Returns the mean and the standard deviation of a distribution of θ. */
function momentsOf({ nodes, weights }: ThetaDistribution): [number, number] {
  const mean = weights.reduce((sum, weight, q) => sum + weight * (nodes[q] ?? 0), 0);
  const variance = weights.reduce((sum, weight, q) => sum + weight * ((nodes[q] ?? 0) - mean) ** 2, 0);
  return [mean, Math.sqrt(variance)];
}

const items = reference.items as Record<string, IrtItem>;
const log = eventLog(reference.events);

describe("IRT ability estimates against R", () => {
  for (const name of ["standard", "shifted"] as const) {
    const { prior, points, trace } = reference[name];
    it(`follows the written-out posterior event by event, with the ${name} prior`, () => {
      const spacing = (12 * prior.sd) / (points - 1) / 0.2;
      const model = irt({ parameters: { items }, prior: prior as IrtPrior, points });
      const fit = model.fit(log);
      const eap: number[] = [];
      const moments: number[] = [];
      const map: number[] = [];
      const states = new Map<string, IrtState>();
      for (const event of log) {
        const state = model.update(states.get(event.learner) ?? model.initialState(event.learner), event);
        states.set(event.learner, state);
        eap.push(model.ability(state).theta, model.ability(state).standardError);
        moments.push(...momentsOf(model.posterior(state)));
        map.push(model.ability(state, { estimator: "map" }).theta);
      }
      expectAllClose(fit.predictions, trace.prediction, GRID_TOLERANCE, "prediction");
      expectAllClose(
        eap,
        trace.eap.flatMap((theta, i) => [theta, trace.eapStandardError[i] ?? 0]),
        GRID_TOLERANCE,
        "EAP",
      );
      expectAllClose(moments, eap, GRID_TOLERANCE, "posterior mean and standard deviation");
      expectAllClose(map, trace.map, MODE_TOLERANCE * spacing ** 5, "MAP");
      const errors = [...states.values()].map((state) => model.ability(state, { estimator: "map" }).standardError);
      const lastOfEach = [...states.keys()].map((learner) => log.findLastIndex((event) => event.learner === learner));
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
    });
  }

  it("matches mirt's fscores for EAP, as the written-out posterior does", () => {
    expectAllClose(reference.mirt.eap, reference.standard.trace.eap, GRID_TOLERANCE, "mirt EAP");
  });
});

describe("IRT model", () => {
  const model = irt({ parameters: { items } });

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
  const lastCategory = (id: string): number => {
    const item = items[id];
    return item !== undefined && "model" in item ? item.b.length : 1;
  };
  const eventsOf = (pairs: readonly (readonly [string, number])[]): Event[] =>
    pairs.map(([item, k], time) => {
      const last = lastCategory(item);
      return { learner: "ana", item, skills: [], time, outcome: (k % (last + 1)) / last };
    });

  it("raises the estimate after a correct answer and lowers it after a wrong one", () => {
    fc.assert(
      fc.property(dichotomous, answers, (item, pairs) => {
        const tested = irt({ parameters: { items: { ...items, x: item } } });
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
    const shifted = irt({ parameters: { items }, prior: { mean: 0.5, sd: 2 }, points: 41 });
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
    const fine = irt({ parameters: { items }, points: 200_001 });
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
    const hard = irt({ parameters: { items: { x: { a: 1, b: 10 } } } });
    let state = hard.initialState("ana");
    for (let time = 0; time < 10; time++) {
      state = hard.update(state, { learner: "ana", item: "x", skills: [], time, outcome: 1 });
    }
    expect(hard.ability(state, { estimator: "map" }).theta).toBe(6);
  });

  it("gives the error of a mode as 1 / √(-ℓ''), and Infinity where ℓ is not curved down", () => {
    const model = irt({ parameters: { items } });
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

describe("IRT edge cases", () => {
  const model = irt({ parameters: { items } });
  const event: Event = { learner: "ana", item: "i1", skills: [], time: 1, outcome: 1 };

  it("rejects invalid items, priors and grids", () => {
    const build = (item: unknown) => () => irt({ parameters: { items: { x: item as IrtItem } } });
    expect(build({ a: 1, b: 0, c: 0.5, d: 0.4 })).toThrow('Item "x" needs 0 ≤ c < d ≤ 1, found c = 0.5 and d = 0.4.');
    expect(build({ a: 1, b: 0, c: "0.2" })).toThrow('Item "x" needs 0 ≤ c < d ≤ 1, found c = "0.2" and d = 1.');
    expect(build({ a: 1, b: 0, d: null })).toThrow('Item "x" needs 0 ≤ c < d ≤ 1, found c = 0 and d = null.');
    expect(build({ a: Number.NaN, b: 0 })).toThrow('Item "x" needs a, a finite number, found NaN.');
    expect(build({ model: "graded", a: 1, b: [1, 0] })).toThrow(
      'Item "x" is graded, so needs thresholds b strictly increasing when a > 0 and decreasing when a < 0, found a = 1, b = 1, 0.',
    );
    expect(() => build({ model: "graded", a: -1, b: [1, 0] })()).not.toThrow();
    expect(build({ model: "nominal", a: 1, b: [0] })).toThrow('polytomous items are "graded" or "gpcm"');
    expect(build({ model: "gpcm", a: 1, b: [] })).toThrow("non-empty array of finite thresholds");
    expect(() => irt({ parameters: { items }, prior: { mean: 0, sd: 0 } })).toThrow(
      "The sd of the prior must be a positive number, found 0.",
    );
    expect(() => irt({ parameters: { items }, points: 2 })).toThrow("integer number of points from 3");
    expect(() => irt({ parameters: { items }, prior: { mean: 0, sd: 1e-300 } })).toThrow(
      "The prior needs a mean and an sd whose grid has distinct, finite points, with finite squares of their spacing and of 1 / sd, found mean 0 and sd 1e-300.",
    );
    expect(() => irt({ parameters: { items }, prior: { mean: 0, sd: 1e300 } })).toThrow(
      "finite squares of their spacing",
    );
    expect(() => irt({ parameters: { items }, prior: { mean: 1e308, sd: 1 } })).toThrow("distinct, finite points");
  });

  it("keeps the log-probability of a wrong answer to an easy item where P(correct) rounds to 1", () => {
    // At θ = 6, a (θ - b) = 40 and 1 - logistic(40) is 0 in doubles; ln P(wrong) is ln logistic(-40).
    const easy = irt({ parameters: { items: { x: { a: 4, b: -4 } } } });
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
          parameters: { items: { x: item, g: { model: "graded", a: item.a, b: [item.b, item.b + 1] } } },
        });
        let state = tested.initialState("ana");
        outcomes.forEach((outcome, time) => {
          state = tested.update(JSON.parse(JSON.stringify(state)), {
            ...event,
            item: time % 2 ? "g" : "x",
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
    expect(() => irt(missing)).toThrow("The options must be an object, found null.");
    expect(() => irt({ parameters: { items }, prior: missing })).toThrow(
      "The prior must be an object { mean, sd }, found null.",
    );
    const model = irt({ parameters: { items } });
    expect(() => model.ability(model.initialState("ana"), missing)).toThrow(
      "The options must be an object, found null.",
    );
  });
});

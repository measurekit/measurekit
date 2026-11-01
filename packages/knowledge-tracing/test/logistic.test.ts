import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  afm,
  compareModels,
  createRandom,
  crossValidate,
  DataError,
  type Event,
  type EventLog,
  eventLog,
  type LogisticFeature,
  type LogisticOptions,
  type LogisticParameters,
  type LogisticState,
  learningCurve,
  logistic,
  type Proficiency,
  pfa,
  splitByLearner,
} from "../src/index.js";
import { checkFeatures, featureValue, needsOf } from "../src/logistic/features.js";
import { nextHistory, type SkillHistory } from "../src/logistic/state.js";
import exposureFixture from "./fixtures/logistic-exposures.json" with { type: "json" };
import glm from "./fixtures/logistic-glm.json" with { type: "json" };
import lkt from "./fixtures/logistic-lkt.json" with { type: "json" };
import lktFeaturesFixture from "./fixtures/logistic-lkt-features.json" with { type: "json" };
import { allPredicted, defined, expectClose, expectUnaffectedByLessons, online } from "./helpers.js";

/** Absolute tolerance of features against LKT's. */
const FEATURE_TOLERANCE = 1e-12;
/** Absolute tolerance of the features of LKT's other names, which its columns give to 1e-10 of their size. */
const LKT_FEATURES_TOLERANCE = 1e-10;
/** Absolute tolerance of the time-window counts against base R. */
const WINDOW_TOLERANCE = 1e-12;
/** Absolute tolerance of coefficients, proficiencies and log-likelihoods against glm. */
const GLM_TOLERANCE = 1e-8;
/** Absolute tolerance of ridge fits against LiblineaR. */
const RIDGE_TOLERANCE = 1e-6;
/** Absolute tolerance of random-intercept estimates and modes against the Laplace maximum of lme4's approximation. */
const RANDOM_TOLERANCE = 1e-5;
/** Absolute tolerance of the random-intercept log-likelihood. */
const RANDOM_LOG_LIKELIHOOD_TOLERANCE = 1e-6;

/** A fit of `validation/knowledge-tracing/logistic.R`: the model's options and the reference's estimates. */
interface Reference {
  readonly name: string;
  readonly features: readonly LogisticFeature[];
  readonly proficiency: Proficiency;
  readonly singleSkill?: boolean;
  readonly expected: LogisticParameters & { readonly logLikelihood: number };
}

// Parsed rather than imported as typed: TypeScript merges the fits' coefficient records into one type with
// optional keys.
const references: readonly Reference[] = JSON.parse(JSON.stringify(glm.fits));
const simulated = eventLog(glm.events);
const singleSkill = glm.singleSkill.map((index) => defined(simulated[index]));
const sample = eventLog(lkt.events);

/** Expects parameters to equal the expected ones within `tolerance`, skill by skill and learner by learner. */
function expectParameters(actual: LogisticParameters, expected: LogisticParameters, tolerance: number): void {
  expect(Object.keys(actual.skills).toSorted()).toEqual(Object.keys(expected.skills).toSorted());
  for (const [skill, coefficients] of Object.entries(expected.skills)) {
    const found = defined(actual.skills[skill]);
    expectClose(found.intercept, coefficients.intercept, tolerance, `${skill} intercept`);
    for (const [label, value] of Object.entries(coefficients.coefficients)) {
      expectClose(found.coefficients[label] ?? null, value, tolerance, `${skill} ${label}`);
    }
  }
  for (const [learner, theta] of Object.entries(expected.learners ?? {})) {
    expectClose(actual.learners?.[learner] ?? null, theta, tolerance, learner);
  }
}

describe("logistic features against LKT", () => {
  it("computes LKT's features of the sample before each attempt", () => {
    const decays = lkt.decays;
    const features: Record<string, LogisticFeature> = {
      opportunities: "opportunities",
      successes: "successes",
      failures: "failures",
      logOpportunities: "logOpportunities",
      logSuccesses: "logSuccesses",
      logFailures: "logFailures",
      decayedProportion: { decayedProportion: decays.proportion },
      decayedLogit: { decayedLogit: decays.logit },
      recency: { recency: decays.recency },
      decayedOpportunities: { decayedOpportunities: decays.counts },
      decayedSuccesses: { decayedSuccesses: decays.counts },
      decayedFailures: { decayedFailures: decays.counts },
    };
    const specs = checkFeatures(Object.values(features));
    const needs = needsOf(specs);
    const histories = new Map<string, SkillHistory>();
    // LKT's rows are in learner order, each learner's in time order.
    lkt.events.forEach((event, index) => {
      const skill = defined(event.skills[0]);
      const key = `${event.learner}\u0000${skill}`;
      const history = histories.get(key);
      specs.forEach((spec, f) => {
        const name = defined(Object.keys(features)[f]) as keyof typeof lkt.features;
        const expected = defined(lkt.features[name][index]);
        expectClose(
          featureValue(spec, history, undefined, event.time, 0),
          expected,
          FEATURE_TOLERANCE,
          `${name} at ${index + 1}`,
        );
      });
      histories.set(key, nextHistory(history, event.outcome, event.time, 0, needs));
    });
  });
});

/** A feature of `validation/knowledge-tracing/lkt-features.R`: the model's feature and the reference's value at each event. */
interface FeatureReference {
  readonly name: string;
  readonly feature: LogisticFeature;
  readonly lkt: string | null;
  readonly values: readonly number[];
}
const featureReferences: readonly FeatureReference[] = JSON.parse(JSON.stringify(lktFeaturesFixture.features));
const featureEvents: readonly Event[] = JSON.parse(JSON.stringify(lktFeaturesFixture.events));

/** Returns the values of a feature at each event of the fixture, from the state that a model's updates make. */
function replayedValues(feature: LogisticFeature, events: readonly Event[] = featureEvents): number[] {
  const model = logistic({ features: [feature] });
  const [spec] = checkFeatures([feature]);
  const states = new Map<string, LogisticState>();
  return events.map((event) => {
    const state = states.get(event.learner) ?? model.initialState(event.learner);
    const skill = defined(event.skills[0]);
    const value = featureValue(defined(spec), state.skills[skill], undefined, event.time, state.practice ?? 0);
    states.set(event.learner, model.update(state, event));
    return value;
  });
}

describe("logistic features of the other names of LKT, and of DAS3H", () => {
  it.each(featureReferences)("matches $lkt of LKT, or base R, for $name", (reference) => {
    const tolerance = reference.lkt === null ? WINDOW_TOLERANCE : LKT_FEATURES_TOLERANCE;
    const values = replayedValues(reference.feature);
    reference.values.forEach((expected, index) => {
      expectClose(defined(values[index]), expected, tolerance, `${reference.name} at event ${index + 1}`);
    });
  });
});

/** An event of learner ana at skill k. */
const event = (time: number, outcome: number | null, extra: Partial<Event> = {}): Event => ({
  learner: "ana",
  item: "q",
  skills: ["k"],
  time,
  outcome,
  ...extra,
});

/** The value of a feature before an attempt at `time`, after the events. */
function valueAfter(feature: LogisticFeature, events: readonly Event[], time: number): number {
  const model = logistic({ features: [feature] });
  const state = events.reduce((current, next) => model.update(current, next), model.initialState("ana"));
  const [spec] = checkFeatures([feature]);
  return featureValue(defined(spec), state.skills.k, undefined, time, state.practice ?? 0);
}

describe("logistic history features", () => {
  it("track in a state only what the features of the model read", () => {
    const keys = (feature: LogisticFeature | null) => {
      const model = logistic({ features: feature === null ? [] : [feature] });
      return Object.keys(
        defined(model.update(model.initialState("ana"), event(0, 1, { duration: 5 })).skills.k),
      ).sort();
    };
    const plain = ["decayed", "failures", "last", "successes"];
    const withFields = (...fields: string[]) => [...plain, ...fields].sort();
    expect(keys(null)).toEqual(plain);
    expect(keys("proportion")).toEqual(plain);
    expect(keys({ recencySuccess: 1 })).toEqual(withFields("previous"));
    expect(keys({ base: 1 })).toEqual(withFields("first"));
    expect(keys({ basePractice: { decay: 1, scale: 1 } })).toEqual(withFields("first", "firstPractice"));
    expect(keys({ dash: 10 })).toEqual(withFields("leaky"));
    expect(keys({ ppe: { power: 1, decay: 1, spacing: 1, weight: 1 } })).toEqual(withFields("outcomes", "times"));
    expect(keys({ windowOpportunities: 10 })).toEqual(withFields("outcomes", "times"));
    const model = logistic({ features: [{ basePractice: { decay: 1, scale: 1 } }] });
    expect(model.update(model.initialState("ana"), event(0, 1, { duration: 5 })).practice).toBe(5);
    expect(logistic().update(logistic().initialState("ana"), event(0, 1, { duration: 5 })).practice).toBeUndefined();
  });

  it("gives the proportion 0.5 before any attempt, and the share of successes after", () => {
    expect(valueAfter("proportion", [], 0)).toBe(0.5);
    expect(valueAfter("proportion", [event(0, 1), event(1, 0), event(2, 1)], 3)).toBeCloseTo(2 / 3, 15);
    expect(valueAfter("netSuccesses", [event(0, 1), event(1, 0), event(2, 1)], 3)).toBe(1);
  });

  it("gives the leaky counts of dash as the closed form: each attempt adds 1, and the count decays in time", () => {
    const events = [event(0, 1), event(100, 0)];
    const afterSecond = (Math.exp(-1) + 1) * Math.exp(-0.5);
    expect(valueAfter({ dash: 100 }, events, 150)).toBeCloseTo(Math.log(1 + afterSecond), 14);
    expect(valueAfter({ dashSuccess: 100 }, events, 150)).toBeCloseTo(Math.log(1 + Math.exp(-1.5)), 14);
    expect(valueAfter({ dash: 100 }, [], 150)).toBe(0);
    expect(valueAfter({ dash: 100 }, [event(7, 1)], 7)).toBeCloseTo(Math.log(2), 15);
  });

  it("gives base 0 at an age of 0, and ln(1 + n) A^-d after it", () => {
    expect(valueAfter({ base: 0.5 }, [event(0, 1), event(0, 0)], 0)).toBe(0);
    expect(valueAfter({ base: 0.5 }, [event(0, 1), event(0, 0)], 100)).toBeCloseTo(Math.log(3) / 10, 14);
    expect(valueAfter({ baseSuccess: 0.5 }, [event(0, 1), event(5, 0)], 100)).toBeCloseTo(Math.log(2) / 10, 14);
    expect(valueAfter({ baseFailure: 0.5 }, [event(0, 1), event(5, 0)], 100)).toBeCloseTo(Math.log(2) / 10, 14);
    expect(valueAfter({ base: 0.5 }, [], 100)).toBe(0);
  });

  it("gives basePractice the age of the practice time plus the scaled time away, and base at a scale of 1", () => {
    const events = [event(0, 1, { duration: 10 }), event(1000, 0, { duration: 20 })];
    const age = (2000 - 30) * 0.5 + 30;
    expect(valueAfter({ basePractice: { decay: 0.5, scale: 0.5 } }, events, 2000)).toBeCloseTo(
      Math.log(3) / Math.sqrt(age),
      14,
    );
    expect(valueAfter({ basePractice: { decay: 0.5, scale: 1 } }, events, 2000)).toBeCloseTo(
      valueAfter({ base: 0.5 }, events, 2000),
      14,
    );
    expect(valueAfter({ basePractice: { decay: 0.5, scale: 0 } }, events, 2000)).toBeCloseTo(
      Math.log(3) / Math.sqrt(30),
      14,
    );
  });

  it("counts the duration of an event without an outcome in the practice time, and needs one for every event", () => {
    const model = logistic({ features: [{ basePractice: { decay: 0.5, scale: 0 } }] });
    const lesson = model.update(model.initialState("ana"), event(0, null, { duration: 50 }));
    expect(lesson.practice).toBe(50);
    expect(() => model.update(lesson, event(10, 1))).toThrow(
      "The event needs a duration, which the features basePractice read for the practice time.",
    );
    expect(() => model.fit(eventLog([event(0, 1, { duration: 5 }), event(1, 0)]))).toThrow(
      "Event 2 needs a duration, which the features basePractice read for the practice time.",
    );
    expect(() => logistic().update(logistic().initialState("ana"), event(0, 1))).not.toThrow();
  });

  it("splits recency by the outcome of the last attempt, the two summing to recency", () => {
    const events = featureEvents.slice(0, 600);
    const both = replayedValues({ recency: 0.7 }, events);
    const success = replayedValues({ recencySuccess: 0.7 }, events);
    const failure = replayedValues({ recencyFailure: 0.7 }, events);
    both.forEach((value, index) => {
      expectClose(defined(success[index]) + defined(failure[index]), value, 1e-12, `event ${index + 1}`);
    });
  });

  it("counts the attempts within a window, keeps only those the largest window can reach, and all with Infinity", () => {
    const events = [event(0, 1), event(5, 0), event(100, 1)];
    expect(valueAfter({ windowOpportunities: 10 }, events, 105)).toBeCloseTo(Math.log(2), 15);
    expect(valueAfter({ windowSuccesses: 10 }, events, 105)).toBeCloseTo(Math.log(2), 15);
    expect(valueAfter({ windowFailures: 10 }, events, 105)).toBe(0);
    expect(valueAfter({ windowOpportunities: Number.POSITIVE_INFINITY }, events, 105)).toBeCloseTo(Math.log(4), 15);
    const keep = (feature: LogisticFeature) => {
      const model = logistic({ features: [feature] });
      return events.reduce((state, next) => model.update(state, next), model.initialState("ana")).skills.k?.times;
    };
    expect(keep({ windowOpportunities: 10 })).toEqual([100]);
    expect(keep({ windowOpportunities: Number.POSITIVE_INFINITY })).toEqual([0, 5, 100]);
    expect(keep({ ppe: { power: 1, decay: 1, spacing: 0, weight: 1 } })).toEqual([0, 5, 100]);
  });

  it("gives ppe n to the power with no earlier attempt, and 1 for the weighted age where an attempt is at the same time", () => {
    const ppe = { power: 0.5, decay: 0.3, spacing: 0.4, weight: 0.6 };
    expect(valueAfter({ ppe }, [], 10)).toBe(0);
    expect(valueAfter({ ppe }, [event(5, 1), event(5, 0)], 5)).toBeCloseTo(Math.sqrt(2), 15);
    expect(valueAfter({ ppe: { ...ppe, power: 0 } }, [], 10)).toBe(1);
    // One earlier attempt, 4 before: the weighted age is 4 and there is no gap to space.
    expect(valueAfter({ ppe }, [event(6, 1)], 10)).toBeCloseTo(4 ** -0.3, 14);
  });

  it("rejects a state without what a feature reads, and keeps what other models stored", () => {
    const plain = logistic();
    const stored = plain.update(plain.initialState("ana"), event(0, 1, { duration: 5 }));
    const needs: [LogisticFeature, string][] = [
      [{ recencySuccess: 1 }, "lacks previous"],
      [{ base: 1 }, "lacks first"],
      [{ basePractice: { decay: 1, scale: 1 } }, "lacks first"],
      [{ dash: 10 }, "lacks the leaky counts of scale 10"],
      [{ windowOpportunities: 10 }, "lacks times"],
    ];
    for (const [feature, message] of needs) {
      expect(() => logistic({ features: [feature] }).update(stored, event(1, 1, { duration: 5 }))).toThrow(message);
    }
    const rich = logistic({ features: [{ base: 1 }, { dash: 10 }, { recencySuccess: 1 }] });
    const state = rich.update(rich.initialState("ana"), event(0, 1));
    expect(() => plain.update(state, event(1, 1))).not.toThrow();
  });

  it("continues a state through JSON as if it had not been stored", () => {
    const features: LogisticFeature[] = [
      "proportion",
      { base: 0.5 },
      { basePractice: { decay: 0.5, scale: 0.2 } },
      { dash: 50 },
      { dashSuccess: 20 },
      { recencyFailure: 0.5 },
      { windowSuccesses: 30 },
      { ppe: { power: 0.3, decay: 0.4, spacing: 0.2, weight: 0.5 } },
    ];
    const model = logistic({ features });
    const events = fc.array(
      fc.record({
        gap: fc.integer({ min: 1, max: 40 }),
        outcome: fc.constantFrom(0, 1, null),
        duration: fc.integer({ min: 1, max: 30 }),
      }),
      {
        minLength: 2,
        maxLength: 14,
      },
    );
    fc.assert(
      fc.property(events, (draws) => {
        const log = draws.map(({ outcome, duration }, index) =>
          event(
            draws.slice(0, index + 1).reduce((sum, draw) => sum + draw.gap, 0),
            outcome,
            { duration },
          ),
        );
        const split = Math.floor(log.length / 2);
        const direct = log.reduce((state, next) => model.update(state, next), model.initialState("ana"));
        const halfway = log
          .slice(0, split)
          .reduce((state, next) => model.update(state, next), model.initialState("ana"));
        const stored: LogisticState = JSON.parse(JSON.stringify(halfway));
        const resumed = log.slice(split).reduce((state, next) => model.update(state, next), stored);
        expect(resumed).toEqual(direct);
        expect(JSON.parse(JSON.stringify(direct))).toEqual(direct);
      }),
    );
  });

  it("fits a model of the new features, with predictions from 0 to 1", () => {
    const model = logistic({
      features: ["proportion", { base: 0.5 }, { dash: 86_400 }, { windowOpportunities: 86_400 }],
      ridge: 1,
    });
    const fit = model.fit(eventLog(featureEvents));
    expect(fit.converged).toBe(true);
    expect(fit.predictions.every((p) => p !== null && p >= 0 && p <= 1)).toBe(true);
  });

  it("rejects parameters that are out of range or of another type", () => {
    const rejected = (feature: unknown) => () => logistic({ features: [JSON.parse(JSON.stringify(feature))] });
    expect(rejected({ powerOpportunities: 0 })).toThrow(
      "The parameter of powerOpportunities must be a positive number, found 0.",
    );
    expect(rejected({ dash: -5 })).toThrow("The parameter of dash must be a positive number, found -5.");
    expect(rejected({ base: -1 })).toThrow("The decay of base must be a non-negative number, found -1.");
    expect(rejected({ recencySuccess: "fast" })).toThrow(
      'The exponent of recencySuccess must be a non-negative number, found "fast".',
    );
    expect(rejected({ windowSuccesses: 0 })).toThrow(
      "The window of windowSuccesses must be a number above 0, or Infinity, found 0.",
    );
    expect(rejected({ decayedProportionFailed: 1.5 })).toThrow(
      "The decay of decayedProportionFailed must be a number above 0 and at most 1, found 1.5.",
    );
    expect(rejected({ basePractice: 2 })).toThrow(
      "The feature basePractice needs an object with decay, scale, found 2.",
    );
    expect(rejected({ basePractice: null })).toThrow(
      "The feature basePractice needs an object with decay, scale, found null.",
    );
    expect(rejected({ basePractice: { decay: 1 } })).toThrow(
      "The scale of basePractice must be a non-negative number, found undefined.",
    );
    expect(rejected({ ppe: { power: 1, decay: 1, spacing: 1, weight: -1 } })).toThrow(
      "The weight of ppe must be a non-negative number, found -1.",
    );
    expect(rejected({ ppe: { power: 1, decay: 1, spacing: 1, weight: 1, extra: 1 } })).toThrow(
      'The feature ppe has no field "extra"; its fields are power, decay, spacing, weight.',
    );
    expect(rejected("netSuccess")).toThrow('Unknown feature "netSuccess"');
    expect(rejected({ unknown: 1 })).toThrow('Unknown feature {"unknown":1}');
    expect(() => logistic({ features: [{ dash: 5 }, { dash: 5 }] })).toThrow("The feature dash(5) is listed twice.");
    expect(
      () => logistic({ features: [{ ppe: { power: 1, decay: 1, spacing: 1, weight: 1 } }] }).parameters,
    ).not.toThrow();
  });
});

describe("logistic fits against R", () => {
  it.each(references)("fits as glm or lme4 do: $name", (reference) => {
    const log = reference.singleSkill === true ? singleSkill : simulated;
    const fit = logistic({ features: reference.features, proficiency: reference.proficiency }).fit(log);
    expect(fit.converged).toBe(true);
    const random = reference.proficiency === "random";
    expectParameters(fit.model.parameters, reference.expected, random ? RANDOM_TOLERANCE : GLM_TOLERANCE);
    const tolerance = random ? RANDOM_LOG_LIKELIHOOD_TOLERANCE : GLM_TOLERANCE;
    expectClose(fit.logLikelihood, reference.expected.logLikelihood, tolerance, "log-likelihood");
    if (random) {
      expectClose(
        fit.model.parameters.learnerSd ?? null,
        defined(reference.expected.learnerSd),
        RANDOM_TOLERANCE,
        "sd",
      );
    }
  });

  it("fits a ridge as LiblineaR does at LKT's cost", () => {
    const ridged = afm({ ridge: lkt.ridge }).fit(sample);
    expectParameters(ridged.model.parameters, lkt.afm, RIDGE_TOLERANCE);
    expectParameters(pfa({ ridge: lkt.ridge }).fit(sample).model.parameters, lkt.pfa, RIDGE_TOLERANCE);
    expect([ridged.aic, ridged.bic]).toEqual([null, null]);
  });

  it("counts the estimated parameters in the information criteria", () => {
    // 5 skills with an intercept and a slope: 10 coefficients; 80 learners, or one fewer with every event of one skill.
    const counts: [LogisticOptions, EventLog, number][] = [
      [{ features: ["opportunities"], proficiency: "fixed" }, simulated, 90],
      [
        { features: ["opportunities"], proficiency: "fixed" },
        singleSkill,
        10 + new Set(singleSkill.map((e) => e.learner)).size - 1,
      ],
      [{ features: ["opportunities"], proficiency: "random" }, simulated, 11],
      [{ features: ["successes", "failures"] }, simulated, 15],
    ];
    for (const [options, log, k] of counts) {
      const fit = logistic(options).fit(log);
      expectClose(fit.aic, 2 * k - 2 * fit.logLikelihood, 1e-9);
      expectClose(fit.bic, k * Math.log(log.length) - 2 * fit.logLikelihood, 1e-9);
    }
  });
});

describe("logistic models", () => {
  it("predict from a fit as online inference does, and give an absent learner the mean or the population proficiency", () => {
    for (const proficiency of ["fixed", "random"] as const) {
      const fit = afm({ proficiency }).fit(simulated);
      const run = online(fit.model, simulated);
      expect(run.predictions).toEqual(fit.predictions);
      expect(Object.fromEntries(run.states)).toEqual(fit.states);
      const thetas = Object.values(fit.model.parameters.learners ?? {});
      const absent = proficiency === "fixed" ? thetas.reduce((sum, theta) => sum + theta, 0) / thetas.length : 0;
      const { intercept } = defined(fit.model.parameters.skills.s1);
      const attempt = { learner: "new", item: "i1", skills: ["s1"], time: 0 };
      expectClose(
        fit.model.predict(fit.model.initialState("new"), attempt),
        1 / (1 + Math.exp(-(absent + intercept))),
        1e-15,
      );
    }
  });

  it("stop at maxIterations without converging, and report it", () => {
    for (const proficiency of ["fixed", "random"] as const) {
      const stopped = afm({ proficiency, maxIterations: 1 }).fit(simulated);
      expect(stopped.converged).toBe(false);
      expect(stopped.iterations).toBe(1);
    }
  });

  it("center fixed proficiencies on 0 when every event has one skill", () => {
    const thetas = Object.values(afm().fit(singleSkill).model.parameters.learners ?? {});
    expectClose(thetas.reduce((sum, theta) => sum + theta, 0) / thetas.length, 0, 1e-12);
  });

  it("give AFM fixed proficiency by default, also for an option left undefined, and PFA none", () => {
    const fixed = afm().fit(singleSkill).model.parameters;
    expect(afm({ proficiency: undefined }).fit(singleSkill).model.parameters).toEqual(fixed);
    expect(fixed.learners).toBeDefined();
    expect(pfa().fit(singleSkill).model.parameters.learners).toBeUndefined();
  });

  it("work with cross-validation, model comparison and learning curves unchanged", () => {
    const folds = splitByLearner(simulated, 4, createRandom(1));
    expect(crossValidate(afm({ proficiency: "random" }), folds).metrics.count).toBe(simulated.length);
    const comparison = compareModels(
      { afm: afm(), pfa: pfa(), lkt: logistic({ features: [{ decayedLogit: 0.8 }] }) },
      folds,
    );
    expect(comparison.map((row) => row.name)).toEqual(["afm", "pfa", "lkt"]);
    expect(learningCurve(simulated, pfa().fit(simulated).predictions).overall[0]?.predictedErrorRate).not.toBeNull();
  });

  const parameters: LogisticParameters = {
    skills: {
      s: { intercept: -0.5, coefficients: { opportunities: 0.2, "decayedLogit(0.8)": 0.3, "recency(0.5)": 0.4 } },
      t: { intercept: 0.1, coefficients: { opportunities: 0.1, "decayedLogit(0.8)": -0.2, "recency(0.5)": 0.1 } },
    },
    learners: { ana: 0.4 },
  };
  const model = logistic({
    features: ["opportunities", { decayedLogit: 0.8 }, { recency: 0.5 }],
    proficiency: "fixed",
    parameters,
  });
  const arbitraryEvents: fc.Arbitrary<Event[]> = fc
    .array(
      fc.record({
        skills: fc.subarray(["s", "t"], { minLength: 1 }),
        gap: fc.double({ min: 0, max: 100, noNaN: true }),
        outcome: fc.oneof(fc.constantFrom(0, 1), fc.double({ min: 0, max: 1, noNaN: true })),
      }),
      { minLength: 1, maxLength: 25 },
    )
    .map((records) => {
      let time = 0;
      return records.map(({ gap, ...record }) => {
        time += gap;
        return { learner: "ana", item: "q", time, ...record };
      });
    });

  it("predict probabilities from 0 to 1 and never change the state they are given", () => {
    fc.assert(
      fc.property(arbitraryEvents, (events) => {
        let state = model.initialState("ana");
        for (const event of events) {
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

  it("continue identically from a state stored as JSON", () => {
    fc.assert(
      fc.property(arbitraryEvents, fc.nat(), (events, cut) => {
        const split = cut % (events.length + 1);
        let direct = model.initialState("ana");
        for (const event of events) direct = model.update(direct, event);
        let first = model.initialState("ana");
        for (const event of events.slice(0, split)) first = model.update(first, event);
        let state: LogisticState = JSON.parse(JSON.stringify(first));
        for (const event of events.slice(split)) state = model.update(state, event);
        expect(state).toEqual(direct);
      }),
    );
  });
});

describe("logistic edge cases", () => {
  const event: Event = { learner: "ana", item: "q", skills: ["s"], time: 10, outcome: 1 };

  it("fit one learner, and logs all correct or all wrong, without NaN", () => {
    const one = eventLog(Array.from({ length: 8 }, (_, time) => ({ ...event, time, outcome: time % 3 === 0 ? 0 : 1 })));
    for (const log of [
      one,
      eventLog([{ ...event }, { ...event, time: 11 }]),
      eventLog([
        { ...event, outcome: 0 },
        { ...event, time: 11, outcome: 0 },
      ]),
    ]) {
      for (const options of [{}, { proficiency: "random" as const }, { ridge: 0.1, proficiency: "fixed" as const }]) {
        const fit = pfa(options).fit(log);
        expect(allPredicted(fit.predictions).every((p) => p >= 0 && p <= 1)).toBe(true);
        expect(Number.isNaN(fit.logLikelihood)).toBe(false);
      }
    }
  });

  it("keep the decayed logit finite after a run of one outcome long enough to underflow its prior", () => {
    // 0.5¹²⁰⁰ underflows to 0, so the prior does not keep the decayed successes or failures above 0.
    const run = (outcome: number) =>
      eventLog([
        ...Array.from({ length: 1200 }, (_, time) => ({ ...event, time, outcome })),
        { ...event, learner: "budi", time: 1200, outcome: 1 - outcome },
      ]);
    for (const outcome of [0, 1]) {
      const fit = logistic({ features: [{ decayedLogit: 0.5 }] }).fit(run(outcome));
      expect(fit.converged).toBe(true);
      expect(Number.isFinite(fit.logLikelihood)).toBe(true);
      expect(allPredicted(fit.predictions).every((p) => p >= 0 && p <= 1)).toBe(true);
    }
    const history = { successes: 0, failures: 1200, last: 0, decayed: { "0.5": { successes: 0, failures: 2 } } };
    const [spec] = checkFeatures([{ decayedLogit: 0.5 }]);
    expect(featureValue(defined(spec), history, undefined, 1, 0)).toBe(1200 * Math.log(0.5) - Math.log(2));
  });

  it("stop without converging when a feature's square overflows", () => {
    // Recency at a gap of 1e-310 is 1e155, whose square overflows the Newton system's diagonal to Infinity.
    const log = eventLog([
      { ...event, time: 0, outcome: 1 },
      { ...event, time: 1e-310, outcome: 0 },
      { ...event, learner: "budi", time: 1, outcome: 0 },
    ]);
    const fit = logistic({ features: [{ recency: 0.5 }] }).fit(log);
    expect(fit.converged).toBe(false);
    expect(allPredicted(fit.predictions).every((p) => p >= 0 && p <= 1)).toBe(true);
  });

  it("keep the coefficient of a feature that is 0 on every event at 0, and converge", () => {
    // No event follows a success, so the feature of successes is 0 throughout.
    const log = eventLog([
      { ...event, time: 1, outcome: 0 },
      { ...event, time: 2, outcome: 1 },
      { ...event, learner: "budi", time: 3, outcome: 1 },
    ]);
    const fit = logistic({ features: ["successes"] }).fit(log);
    expect(fit.converged).toBe(true);
    expect(fit.model.parameters.skills.s?.coefficients.successes).toBe(0);
  });

  // Two cohorts that share no skill, as two versions of a course: each has its own flat direction of θ.
  const random = createRandom(7);
  const cohort = (prefix: string, skills: readonly string[]) =>
    Array.from({ length: 25 }, (_, l) =>
      Array.from({ length: 12 }, (_, k) => ({
        learner: `${prefix}${l}`,
        item: "q",
        skills: [skills[k % skills.length] ?? "s"],
        time: k * 100 + l,
        outcome: random.uniform() < 0.3 + 0.04 * k ? 1 : 0,
      })),
    ).flat();
  const first = cohort("a", ["s1", "s2"]);
  const second = cohort("b", ["t1", "t2", "t3"]);

  it("fit cohorts that share no skill as if fitted apart, with θ of mean 0 in each", () => {
    const together = afm().fit(eventLog([...first, ...second]));
    const apart = [afm().fit(eventLog(first)), afm().fit(eventLog(second))];
    expect(together.converged).toBe(true);
    expectClose(together.logLikelihood, (apart[0]?.logLikelihood ?? 0) + (apart[1]?.logLikelihood ?? 0), 1e-8);
    const learners = defined(together.model.parameters.learners);
    for (const prefix of ["a", "b"]) {
      const thetas = Object.entries(learners).filter(([learner]) => learner.startsWith(prefix));
      expectClose(thetas.reduce((sum, [, theta]) => sum + theta, 0) / thetas.length, 0, 1e-12);
    }
    // k: two coefficients for each of 5 skills, and 50 learners less a flat direction per cohort.
    expectClose(together.aic, 2 * (10 + 48) - 2 * together.logLikelihood, 1e-9);
  });

  it.each([
    [
      ["opportunities", "successes", "failures"],
      ["successes", "failures"],
    ],
    // With decay 1, the decayed opportunities equal the opportunities: two equal columns.
    [["opportunities", { decayedOpportunities: 1 }], ["opportunities"]],
  ] as const)("converge with features %j, linear in others, at the likelihood without them", (features, reduced) => {
    const log = eventLog(first);
    const redundant = logistic({ features: [...features] }).fit(log);
    expect(redundant.converged).toBe(true);
    expectClose(redundant.logLikelihood, logistic({ features: [...reduced] }).fit(log).logLikelihood, 1e-8);
  });

  it("reject an empty log, events without skills, unknown skills and events out of time order", () => {
    const model = afm({ parameters: { skills: { s: { intercept: 0, coefficients: { opportunities: 0.1 } } } } });
    expect(() => model.fit([])).toThrow("A log needs at least one event.");
    expect(() => model.fit(eventLog([{ ...event, skills: [] }]))).toThrow(
      "Event 1 has no skills; a logistic model needs at least one.",
    );
    const state = model.update(model.initialState("ana"), event);
    expect(() => model.predict(state, { ...event, skills: ["other"] })).toThrow(
      'The model has no parameters for skill "other".',
    );
    expect(() => model.update(state, { ...event, time: 5 })).toThrow(
      `The event at time 5 is earlier than the learner's last attempt at skill "s", at time 10.`,
    );
    const recency = logistic({
      features: [{ recency: 0.5 }],
      parameters: { skills: { s: { intercept: 0, coefficients: { "recency(0.5)": 1 } } } },
    });
    expect(() => recency.predict(recency.update(recency.initialState("ana"), event), { ...event, time: 5 })).toThrow(
      `The attempt at time 5 is earlier than the learner's last attempt at skill "s", at time 10.`,
    );
  });

  it.each([
    ['{"features": ["nothing"]}', 'Unknown feature "nothing"'],
    [
      '{"features": [{"decayedLogit": 1.5}]}',
      "The decay of decayedLogit must be a number above 0 and at most 1, found 1.5.",
    ],
    ['{"features": [{"recency": -1}]}', "The exponent of recency must be a non-negative number, found -1."],
    ['{"features": ["successes", "successes"]}', "The feature successes is listed twice."],
    ['{"proficiency": "mixed"}', 'The option proficiency must be "none", "fixed" or "random", found "mixed".'],
    ['{"ridge": -1}', "The option ridge must be a non-negative number, found -1."],
    [
      '{"features": ["successes"], "parameters": {"skills": {"s": {"intercept": 0, "coefficients": {}}}}}',
      'The coefficient of successes in the logistic parameters of skill "s" must be a finite number, found undefined.',
    ],
  ])("reject the options %s", (json, message) => {
    const options: LogisticOptions = JSON.parse(json);
    expect(() => logistic(options)).toThrow(DataError);
    expect(() => logistic(options)).toThrow(message);
  });

  it("reject malformed states, as from storage", () => {
    const model = pfa({
      parameters: { skills: { s: { intercept: 0, coefficients: { successes: 0.1, failures: 0 } } } },
    });
    const parse = (json: string): LogisticState => JSON.parse(json);
    expect(() =>
      model.predict(
        parse('{"learner": "ana", "skills": {"s": {"successes": -1, "failures": 0, "last": 0, "decayed": {}}}}'),
        event,
      ),
    ).toThrow('The logistic state of learner "ana", skill "s", needs successes, a non-negative number, found -1.');
    const decayed = logistic({
      features: [{ decayedLogit: 0.8 }],
      parameters: { skills: { s: { intercept: 0, coefficients: { "decayedLogit(0.8)": 1 } } } },
    });
    const withoutCounts = parse(
      '{"learner": "ana", "skills": {"s": {"successes": 1, "failures": 0, "last": 0, "decayed": {}}}}',
    );
    for (const use of [() => decayed.update(withoutCounts, event), () => decayed.predict(withoutCounts, event)]) {
      expect(use).toThrow('skill "s", lacks the decayed counts of decay 0.8, which the model\'s features need.');
    }
    expect(() => model.update(model.initialState("budi"), event)).toThrow(
      'An attempt of learner "ana" was given the state of learner "budi".',
    );
  });
});

describe("logistic options from JavaScript", () => {
  it("rejects options that are not an object", () => {
    for (const factory of [logistic, afm, pfa]) {
      expect(() => factory(JSON.parse("null"))).toThrow("The options must be an object, found null.");
    }
  });

  it("rejects parameters and a proficiency of null, which do not stand for the defaults", () => {
    expect(() => logistic(JSON.parse('{"parameters": null}'))).toThrow(
      "Logistic parameters must be an object, found null.",
    );
    expect(() => afm(JSON.parse('{"proficiency": null}'))).toThrow(
      'The option proficiency must be "none", "fixed" or "random", found null.',
    );
  });
});

describe("logistic exposure features against R", () => {
  const log = eventLog(exposureFixture.events);
  const features: readonly LogisticFeature[] = [
    "exposures",
    { decayedExposures: exposureFixture.decay },
    { recencyExposure: exposureFixture.exponent },
  ];

  it("computes each feature from the state as base R does", () => {
    const specs = checkFeatures(features);
    const model = logistic({ features });
    const states = new Map<string, LogisticState>();
    const columns = [
      exposureFixture.features.exposures,
      exposureFixture.features.decayedExposures,
      exposureFixture.features.recencyExposure,
    ];
    log.forEach((event, index) => {
      const state = states.get(event.learner) ?? model.initialState(event.learner);
      const skill = defined(event.skills[0]);
      const history = state.skills[skill];
      const exposure = state.exposures?.[skill];
      specs.forEach((spec, f) => {
        const expected = defined(columns[f]?.[index]);
        expectClose(
          featureValue(spec, history, exposure, event.time, 0),
          expected,
          FEATURE_TOLERANCE,
          `${spec.label} at ${index}`,
        );
      });
      states.set(event.learner, model.update(state, event));
    });
  });

  it("fits as glm does on the events with an outcome", () => {
    const fit = logistic({ features, tolerance: 1e-12, maxIterations: 200 }).fit(log);
    expectParameters(fit.model.parameters, { skills: exposureFixture.skills }, GLM_TOLERANCE);
    expectClose(fit.logLikelihood, exposureFixture.logLikelihood, GLM_TOLERANCE, "log-likelihood");
    const observed = log.filter((event) => event.outcome !== null).length;
    expect(fit.predictions.filter((prediction) => prediction !== null)).toHaveLength(observed);
    // An intercept and 3 coefficients for each of 2 skills.
    expectClose(fit.aic, 2 * 8 - 2 * fit.logLikelihood, 1e-9);
    expectClose(fit.bic, 8 * Math.log(observed) - 2 * fit.logLikelihood, 1e-9);
  });
});

describe("logistic with events that have no outcome", () => {
  const lesson: Event = { learner: "ana", item: "video", skills: ["fractions"], time: 5, outcome: null };
  const outcomeFeatures: readonly LogisticFeature[] = [
    "opportunities",
    "successes",
    "failures",
    { decayedLogit: 0.8 },
    { recency: 0.5 },
  ];

  it.each([
    ["without a proficiency", logistic({ features: outcomeFeatures })],
    ["with a fixed proficiency", afm()],
    ["with performance factors", pfa({ proficiency: "random" })],
  ])("is unaffected by them when no feature reads exposures: %s", (_, model) => {
    expectUnaffectedByLessons(model, simulated);
  });

  it("keeps the history of outcomes as it was, and records the exposures apart from it", () => {
    const model = logistic({ features: ["successes", "exposures", { decayedExposures: 0.5 }] });
    let state = model.initialState("ana");
    state = model.update(state, { ...lesson, outcome: 1, time: 1 });
    const history = state.skills.fractions;
    state = model.update(state, { ...lesson, time: 2 });
    state = model.update(state, { ...lesson, skills: ["fractions", "decimals"], time: 4 });
    expect(state.skills.fractions).toEqual(history);
    expect(state.skills.decimals).toBeUndefined();
    expect(state.exposures).toEqual({
      fractions: { count: 2, last: 4, decayed: { "0.5": 1.5 } },
      decimals: { count: 1, last: 4, decayed: { "0.5": 1 } },
    });
  });

  it("leaves a model without exposure features alone, and an event without skills changes nothing", () => {
    const plain = logistic({ features: ["opportunities"] });
    const state = plain.update(plain.initialState("ana"), { ...lesson, outcome: 1, time: 1 });
    expect(plain.update(state, lesson)).toEqual(state);
    const exposed = logistic({ features: ["exposures"] });
    expect(exposed.update(exposed.initialState("ana"), { ...lesson, skills: [] })).toEqual(exposed.initialState("ana"));
    expect(() => exposed.update(exposed.initialState("ana"), { ...lesson, skills: [], outcome: 1 })).toThrow(
      "The event has no skills; a logistic model needs at least one.",
    );
  });

  it("predicts from exposures, reads a state stored without them, and refuses an exposure in the future", () => {
    const parameters: LogisticParameters = {
      skills: { fractions: { intercept: -1, coefficients: { exposures: 0.5, "recencyExposure(1)": 2 } } },
    };
    const model = logistic({ features: ["exposures", { recencyExposure: 1 }], parameters });
    const attempt = { learner: "ana", item: "q", skills: ["fractions"], time: 9 };
    expectClose(model.predict(model.initialState("ana"), attempt), 1 / (1 + Math.exp(1)), 1e-15, "before any exposure");
    const state = model.update(model.update(model.initialState("ana"), lesson), { ...lesson, time: 6 });
    // Two exposures, the latest 3 before the attempt: -1 + 0.5 × 2 + 2 / 3.
    expectClose(model.predict(state, attempt), 1 / (1 + Math.exp(-(-1 + 1 + 2 / 3))), 1e-15, "after two exposures");
    expect(() => model.predict(state, { ...attempt, time: 5 })).toThrow(
      'The attempt at time 5 is earlier than the learner\'s last exposure to skill "fractions", at time 6.',
    );
  });

  it("rejects a malformed state of exposures", () => {
    const model = logistic({
      features: [{ decayedExposures: 0.5 }],
      parameters: { skills: { fractions: { intercept: 0, coefficients: { "decayedExposures(0.5)": 1 } } } },
    });
    const attempt = { learner: "ana", item: "q", skills: ["fractions"], time: 9 };
    const state = (exposures: unknown): LogisticState =>
      JSON.parse(JSON.stringify({ learner: "ana", skills: {}, exposures }));
    expect(() => model.predict(state([]), attempt)).toThrow(
      'The logistic state of learner "ana" needs exposures, an object of exposures by skill id, found an array.',
    );
    expect(() => model.predict(state({ fractions: { count: -1, last: 1, decayed: { "0.5": 1 } } }), attempt)).toThrow(
      "needs count, a non-negative number, found -1.",
    );
    expect(() => model.predict(state({ fractions: { count: 1, last: "1", decayed: { "0.5": 1 } } }), attempt)).toThrow(
      'needs last, the time of the last exposure, found "1".',
    );
    expect(() => model.predict(state({ fractions: { count: 1, last: 1, decayed: {} } }), attempt)).toThrow(
      "lacks the decayed count of decay 0.5, which the model's features need.",
    );
  });

  it("estimates no coefficients for a skill that only lessons practice", () => {
    const events = eventLog([
      { learner: "a", item: "q", skills: ["s"], time: 1, outcome: 1 },
      { learner: "b", item: "q", skills: ["s"], time: 1, outcome: 0 },
      { learner: "a", item: "video", skills: ["s", "taught"], time: 2, outcome: null },
    ]);
    const fit = logistic({ features: ["exposures"], ridge: 0.1 }).fit(events);
    expect(Object.keys(fit.model.parameters.skills)).toEqual(["s"]);
    expect(fit.predictions).toEqual([expect.any(Number), expect.any(Number), null]);
    expect(Object.keys(defined(fit.states.a?.exposures))).toEqual(["s", "taught"]);
  });

  it("refuses to fit a log without an event with an outcome", () => {
    expect(() => logistic().fit(eventLog([lesson]))).toThrow(
      "The log has no event with an outcome; a logistic model needs at least one to fit.",
    );
  });
});

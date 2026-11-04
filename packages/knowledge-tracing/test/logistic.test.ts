import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  afm,
  buildEventLog,
  compareModels,
  createRandom,
  crossValidate,
  DataError,
  type Event,
  type EventInput,
  type EventLog,
  type LogisticFeature,
  type LogisticOptions,
  type LogisticParameters,
  type LogisticState,
  learningCurve,
  logistic,
  type Proficiency,
  pfa,
  searchRates,
  simulate,
  splitByLearner,
} from "../src/index.js";
import {
  checkTerms,
  type FeatureSpec,
  featureValue,
  needsOf,
  nextHistory,
  type SkillHistory,
} from "../src/logistic/internal.js";
import { allPredicted, defined, expectAllClose, expectClose } from "./assertions.js";
import { type LogisticLktFeaturesFixture, type LogisticRatesFixture, readFixture } from "./fixtures.js";
import { expectUnaffectedByLessons, online } from "./logs.js";

interface LogisticExposuresFixture {
  decay: number;
  exponent: number;
  events: { learner: string; item: string; skills: string[]; time: number; outcome: number | null }[];
  features: { exposures: number[]; decayedExposures: number[]; recencyExposure: number[] };
  skills: {
    s1: { intercept: number; coefficients: { exposures: number } };
    s2: { intercept: number; coefficients: { exposures: number } };
  };
  logLikelihood: number;
}
const exposureFixture = readFixture<LogisticExposuresFixture>("logistic-exposures.json");

// biome-ignore-start lint/style/useNamingConvention: the keys are those of the fixture file.
interface LogisticGlmFixture {
  decays: { proportion: number; logit: number; recency: number; counts: number };
  events: { learner: string; item: string; skills: string[]; time: number; outcome: number }[];
  singleSkill: number[];
  fits: {
    name: string;
    features: (string | { decayedLogit?: number; recency?: number })[];
    proficiency: string;
    expected: {
      skills: {
        s1: {
          intercept: number;
          coefficients: {
            opportunities?: number;
            successes?: number;
            failures?: number;
            "decayedLogit(0.8)"?: number;
            "recency(0.5)"?: number;
          };
        };
        s2: {
          intercept: number;
          coefficients: {
            opportunities?: number;
            successes?: number;
            failures?: number;
            "decayedLogit(0.8)"?: number;
            "recency(0.5)"?: number;
          };
        };
        s3: {
          intercept: number;
          coefficients: {
            opportunities?: number;
            successes?: number;
            failures?: number;
            "decayedLogit(0.8)"?: number;
            "recency(0.5)"?: number;
          };
        };
      };
      learners: {
        learner01: number;
        learner02: number;
        learner03: number;
        learner04: number;
        learner05: number;
        learner06: number;
        learner07: number;
        learner08: number;
        learner09: number;
        learner10: number;
        learner11: number;
        learner12: number;
        learner13: number;
        learner14: number;
        learner15: number;
        learner16: number;
        learner17: number;
        learner18: number;
        learner19: number;
        learner20: number;
        learner21: number;
        learner22: number;
        learner23: number;
        learner24: number;
        learner25: number;
        learner26: number;
        learner27: number;
        learner28: number;
        learner29: number;
        learner30: number;
        learner31: number;
        learner32: number;
        learner33: number;
        learner34: number;
        learner35: number;
        learner36: number;
        learner37: number;
        learner38: number;
        learner39: number;
        learner40: number;
        learner41: number;
        learner42: number;
        learner43: number;
        learner44: number;
        learner45: number;
        learner46: number;
        learner47?: number;
        learner48: number;
        learner49: number;
        learner50: number;
        learner51: number;
        learner52: number;
        learner53: number;
        learner54: number;
        learner55: number;
        learner56: number;
        learner57: number;
        learner58: number;
        learner59: number;
        learner60: number;
        learner61: number;
        learner62: number;
        learner63: number;
        learner64: number;
        learner65: number;
        learner66: number;
        learner67: number;
        learner68: number;
        learner69: number;
        learner70: number;
        learner71: number;
        learner72: number;
        learner73: number;
        learner74: number;
        learner75: number;
        learner76: number;
        learner77: number;
        learner78: number;
        learner79: number;
        learner80: number;
      } | null;
      logLikelihood: number;
      learnerSd?: number;
    };
    singleSkill?: boolean;
  }[];
}
// biome-ignore-end lint/style/useNamingConvention: the keys are those of the fixture file.
const glm = readFixture<LogisticGlmFixture>("logistic-glm.json");

interface LogisticLktFixture {
  decays: { proportion: number; logit: number; recency: number; counts: number };
  ridge: number;
  events: { learner: string; item: string; skills: string[]; time: number; outcome: number }[];
  features: {
    opportunities: number[];
    successes: number[];
    failures: number[];
    logOpportunities: number[];
    logSuccesses: number[];
    logFailures: number[];
    decayedProportion: number[];
    decayedLogit: number[];
    recency: number[];
    decayedOpportunities: number[];
    decayedSuccesses: number[];
    decayedFailures: number[];
  };
  afm: {
    skills: Record<string, { intercept: number; coefficients: { opportunities: number } }>;
    learners: Record<string, number>;
  };
  pfa: { skills: Record<string, { intercept: number; coefficients: { successes: number; failures: number } }> };
  lkt: { columns: number[]; logLikelihood: number };
}
const lkt = readFixture<LogisticLktFixture>("logistic-lkt.json");

const lktFeaturesFixture = readFixture<LogisticLktFeaturesFixture>("logistic-lkt-features.json");

const rates = readFixture<LogisticRatesFixture>("logistic-rates.json");

// biome-ignore-start lint/style/useNamingConvention: the keys are those of the fixture file.
interface LogisticTermsFixture {
  events: {
    learner: string;
    item: string;
    skills: string[];
    time: number;
    outcome: number;
    covariates: { x: number };
  }[];
  fits: {
    name: string;
    options: {
      features: { feature: string; over: string; coefficient: string }[];
      intercepts: string[];
      proficiency: string;
      covariates?: string[];
      ridge?: number;
    };
    expected: {
      skills: {
        s1: {
          intercept: number;
          coefficients: { opportunities?: number; logSuccesses?: number; logFailures?: number };
        };
        s2: {
          intercept: number;
          coefficients: { opportunities?: number; logSuccesses?: number; logFailures?: number };
        };
        s3: {
          intercept: number;
          coefficients: { opportunities?: number; logSuccesses?: number; logFailures?: number };
        };
      };
      items?: { i01: { intercept: number; coefficients: { opportunities?: number } } };
      logLikelihood: number;
      learners?: Record<string, number>;
      shared?: { "item:opportunities"?: number; "learner:logFailures"?: number; "covariate:x"?: number };
      learnerSd?: number;
    };
    standardErrors?: {
      skills: {
        s1: { intercept: number; coefficients: { opportunities?: number } };
        s2: { intercept: number; coefficients: { opportunities?: number } };
        s3: { intercept: number; coefficients: { opportunities?: number } };
      };
      items?: { i01: { intercept: number; coefficients: Record<string, never> } };
      learners?: Record<string, number>;
      shared?: { "item:opportunities"?: number; "learner:logFailures"?: number; "covariate:x"?: number };
    };
    subset?: number[];
  }[];
}
// biome-ignore-end lint/style/useNamingConvention: the keys are those of the fixture file.
const termsFixture = readFixture<LogisticTermsFixture>("logistic-terms.json");

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

/** Returns the checked features of the plain list. */
function specsOf(features: readonly LogisticFeature[]): FeatureSpec[] {
  return checkTerms(features).map(({ feature }) => feature);
}

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
const simulated = buildEventLog(glm.events);
const singleSkill = glm.singleSkill.map((index) => defined(simulated[index]));
const sample = buildEventLog(lkt.events);

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
  expect(Object.keys(actual.items ?? {}).toSorted()).toEqual(Object.keys(expected.items ?? {}).toSorted());
  for (const [item, coefficients] of Object.entries(expected.items ?? {})) {
    const found = defined(actual.items?.[item]);
    expectClose(found.intercept, coefficients.intercept, tolerance, `${item} intercept`);
    for (const [label, value] of Object.entries(coefficients.coefficients)) {
      expectClose(found.coefficients[label] ?? null, value, tolerance, `${item} ${label}`);
    }
  }
  expect(Object.keys(actual.shared ?? {}).toSorted()).toEqual(Object.keys(expected.shared ?? {}).toSorted());
  for (const [label, value] of Object.entries(expected.shared ?? {})) {
    expectClose(actual.shared?.[label] ?? null, value, tolerance, label);
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
    const specs = specsOf(Object.values(features));
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
  const [spec] = specsOf([feature]);
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
  const [spec] = specsOf([feature]);
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
    expect(() => model.fit(buildEventLog([event(0, 1, { duration: 5 }), event(1, 0)]))).toThrow(
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
    const fit = model.fit(buildEventLog(featureEvents));
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
    expect(
      learningCurve(simulated, { predictions: pfa().fit(simulated).predictions }).overall[0]?.predictedErrorRate,
    ).not.toBeNull();
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

  it("refuses a feature that overflows, as a power of a short time apart, rather than predict NaN", () => {
    const log = buildEventLog([
      { ...event, time: 0 },
      { ...event, time: 0.5, outcome: 0 },
    ]);
    const message =
      'The feature recency(1100) is Infinity at the attempt of learner "ana" at item "q", time 0.5: lower its exponent or decay, or give times in larger units.';
    expect(() => logistic({ features: [{ recency: 1100 }] }).fit(log)).toThrow(message);
    const model = logistic({
      features: [{ recency: 1100 }],
      parameters: { skills: { s: { intercept: 0, coefficients: { "recency(1100)": 0 } } } },
    });
    const state = model.update(model.initialState("ana"), defined(log[0]));
    expect(() => model.predict(state, { ...event, time: 0.5 })).toThrow(message);
    expect(() => logistic({ features: [{ over: "learner", feature: { recency: 1100 } }] }).fit(log)).toThrow(
      "The feature recency(1100) is Infinity",
    );
  });

  it("fit one learner, and logs all correct or all wrong, without NaN", () => {
    const one = buildEventLog(
      Array.from({ length: 8 }, (_, time) => ({ ...event, time, outcome: time % 3 === 0 ? 0 : 1 })),
    );
    for (const log of [
      one,
      buildEventLog([{ ...event }, { ...event, time: 11 }]),
      buildEventLog([
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
      buildEventLog([
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
    const [spec] = specsOf([{ decayedLogit: 0.5 }]);
    expect(featureValue(defined(spec), history, undefined, 1, 0)).toBe(1200 * Math.log(0.5) - Math.log(2));
  });

  it("stop without converging when a feature's square overflows", () => {
    // Recency at a gap of 1e-310 is 1e155, whose square overflows the Newton system's diagonal to Infinity.
    const log = buildEventLog([
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
    const log = buildEventLog([
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
    const together = afm().fit(buildEventLog([...first, ...second]));
    const apart = [afm().fit(buildEventLog(first)), afm().fit(buildEventLog(second))];
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
    const log = buildEventLog(first);
    const redundant = logistic({ features: [...features] }).fit(log);
    expect(redundant.converged).toBe(true);
    expectClose(redundant.logLikelihood, logistic({ features: [...reduced] }).fit(log).logLikelihood, 1e-8);
  });

  it("reject an empty log, events without skills, unknown skills and events out of time order", () => {
    const model = afm({ parameters: { skills: { s: { intercept: 0, coefficients: { opportunities: 0.1 } } } } });
    expect(() => model.fit([])).toThrow("A log needs at least one event.");
    expect(() => model.fit(buildEventLog([{ ...event, skills: [] }]))).toThrow(
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
      expect(() => factory(JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
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
  const log = buildEventLog(exposureFixture.events);
  const features: readonly LogisticFeature[] = [
    "exposures",
    { decayedExposures: exposureFixture.decay },
    { recencyExposure: exposureFixture.exponent },
  ];

  it("computes each feature from the state as base R does", () => {
    const specs = specsOf(features);
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
    const events = buildEventLog([
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
    expect(() => logistic().fit(buildEventLog([lesson]))).toThrow(
      "The log has no event with an outcome; a logistic model needs at least one to fit.",
    );
  });
});

/** Relative tolerance of standard errors against summary(glm) and the inverse Schur complement of lme4. */
const STANDARD_ERROR_TOLERANCE = 1e-6;

/** A fit of `validation/knowledge-tracing/logistic-terms.R`: the model's options, the reference's estimates and errors. */
interface TermsReference {
  readonly name: string;
  readonly options: LogisticOptions;
  /** The positions of the events the fit uses, 0-based; all of them without. */
  readonly subset?: readonly number[];
  readonly expected: LogisticParameters & { readonly logLikelihood: number };
  readonly standardErrors?: LogisticParameters;
}

const termsReferences: readonly TermsReference[] = JSON.parse(JSON.stringify(termsFixture.fits));
const termsLog = buildEventLog(JSON.parse(JSON.stringify(termsFixture.events)));

/** Returns every number of parameters by its path, such as `/items/i01/intercept`. */
function numbersOf(parameters: unknown, path = ""): Map<string, number> {
  if (typeof parameters === "number") return new Map([[path, parameters]]);
  if (typeof parameters !== "object" || parameters === null) return new Map();
  return new Map(Object.entries(parameters).flatMap(([key, value]) => [...numbersOf(value, `${path}/${key}`)]));
}

describe("logistic terms against R", () => {
  it.each(termsReferences)("fits as glm, lme4 or LiblineaR do: $name", (reference) => {
    const ridge = reference.options.ridge !== undefined;
    const random = reference.options.proficiency === "random";
    const log = reference.subset === undefined ? termsLog : reference.subset.map((index) => defined(termsLog[index]));
    const fit = logistic({ ...reference.options, standardErrors: !ridge }).fit(log);
    expect(fit.converged).toBe(true);
    expectParameters(
      fit.model.parameters,
      reference.expected,
      ridge ? RIDGE_TOLERANCE : random ? RANDOM_TOLERANCE : GLM_TOLERANCE,
    );
    const tolerance = random ? RANDOM_LOG_LIKELIHOOD_TOLERANCE : ridge ? RIDGE_TOLERANCE : GLM_TOLERANCE;
    expectClose(fit.logLikelihood, reference.expected.logLikelihood, tolerance, "log-likelihood");
    if (random) {
      expectClose(
        fit.model.parameters.learnerSd ?? null,
        defined(reference.expected.learnerSd),
        RANDOM_TOLERANCE,
        "sd",
      );
    }
    if (reference.standardErrors === undefined) {
      expect(fit.standardErrors).toBeNull();
      return;
    }
    const actual = numbersOf(fit.standardErrors);
    const expected = numbersOf(reference.standardErrors);
    expect([...actual.keys()].toSorted()).toEqual([...expected.keys()].toSorted());
    for (const [path, value] of expected) {
      // A standard error of 0 stands for a coefficient the model does not have.
      const error = value === 0 ? Math.abs(defined(actual.get(path))) : Math.abs(defined(actual.get(path)) / value - 1);
      expect(error, `${reference.name} ${path}`).toBeLessThanOrEqual(STANDARD_ERROR_TOLERANCE);
    }
  });

  it("counts the coefficients of both kinds of intercept and the flat direction in the information criteria", () => {
    // 24 items without skill intercepts; 60 learners less the mean fixed at 0.
    const fit = logistic({ intercepts: ["item"], proficiency: "fixed" }).fit(termsLog);
    const k = 24 + 60 - 1;
    expectClose(fit.aic, 2 * k - 2 * fit.logLikelihood, 1e-9);
    expectClose(fit.bic, k * Math.log(termsLog.length) - 2 * fit.logLikelihood, 1e-9);
  });
});

describe("logistic terms", () => {
  const first: Event = { learner: "ana", item: "q1", skills: ["a"], time: 1, outcome: 1 };
  const events: Event[] = [
    first,
    { ...first, time: 2 },
    { learner: "ana", item: "q2", skills: ["b"], time: 3, outcome: 0 },
  ];
  const plain = { intercept: 0, coefficients: {} };

  it("adds shared coefficients over the attempt's skills, the item's intercept and the covariates to the linear predictor", () => {
    const model = logistic({
      features: [{ feature: "successes", coefficient: "shared" }],
      intercepts: ["skill", "item"],
      covariates: ["x"],
      ridge: 1,
      parameters: {
        skills: { a: { intercept: 0.5, coefficients: {} }, b: { intercept: -0.25, coefficients: {} } },
        items: { q3: { intercept: 0.1, coefficients: {} } },
        shared: { "skill:successes": 0.3, "covariate:x": 2 },
      },
    });
    const state = events.reduce((current, next) => model.update(current, next), model.initialState("ana"));
    const attempt = { learner: "ana", item: "q3", skills: ["a", "b"], time: 4, covariates: { x: 0.5 } };
    // The successes before the attempt are 2 at a and 0 at b.
    const logit = 0.5 - 0.25 + 0.1 + 0.3 * (2 + 0) + 2 * 0.5;
    expectClose(model.predict(state, attempt), 1 / (1 + Math.exp(-logit)), 1e-14);
  });

  it("gives a coefficient to each item for a term over items, and to all of them for a term over the learner", () => {
    const model = logistic({
      features: [
        { feature: "opportunities", over: "item" },
        { feature: "opportunities", over: "learner" },
      ],
      parameters: {
        skills: { a: plain, b: plain },
        items: {
          q1: { intercept: 0, coefficients: { opportunities: 0.5 } },
          q2: { intercept: 0, coefficients: { opportunities: -1 } },
        },
        shared: { "learner:opportunities": 0.1 },
      },
    });
    const state = events.reduce((current, next) => model.update(current, next), model.initialState("ana"));
    const at = (item: string) => model.predict(state, { learner: "ana", item, skills: ["a"], time: 4 });
    // Three earlier attempts: two at q1, one at q2, and none at an item the fit did not see.
    expectClose(at("q1"), 1 / (1 + Math.exp(-(0.5 * 2 + 0.1 * 3))), 1e-14);
    expectClose(at("q2"), 1 / (1 + Math.exp(-(-1 * 1 + 0.1 * 3))), 1e-14);
    expectClose(at("q9"), 1 / (1 + Math.exp(-0.1 * 3)), 1e-14);
  });

  it("gives an item the fit did not see the intercept and coefficients 0, and still refuses a skill it did not see", () => {
    const fit = logistic({
      intercepts: ["skill", "item"],
      features: [{ feature: "opportunities", over: "item" }],
      ridge: 1,
    }).fit(termsLog);
    const state = fit.model.initialState("ana");
    const seen = defined(fit.model.parameters.skills.s1).intercept;
    const unseen = { learner: "ana", item: "new", skills: ["s1"], time: 0 };
    expectClose(fit.model.predict(state, unseen), 1 / (1 + Math.exp(-seen)), 1e-14);
    expect(() => fit.model.predict(state, { ...unseen, skills: ["new"] })).toThrow(
      'The model has no parameters for skill "new".',
    );
  });

  it("reads no skill parameters when no term uses a skill", () => {
    const fit = logistic({ intercepts: ["item"] }).fit(termsLog);
    const attempt = { learner: "ana", item: "i01", skills: ["unknown"], time: 0 };
    expect(fit.model.predict(fit.model.initialState("ana"), attempt)).toBeGreaterThan(0);
    expect(Object.values(fit.model.parameters.skills).every((skill) => skill.intercept === 0)).toBe(true);
  });

  it("keeps the histories of items and of the learner in a state only when a term reads them", () => {
    const state = (features: LogisticOptions["features"]) =>
      events.reduce((current, next) => {
        const model = logistic({ features });
        return model.update(current, next);
      }, logistic().initialState("ana"));
    expect(Object.keys(state([]))).toEqual(["learner", "skills"]);
    expect(Object.keys(state([{ feature: "successes", over: "item" }]))).toEqual(["learner", "skills", "items"]);
    expect(Object.keys(state([{ feature: "successes", over: "learner" }]))).toEqual(["learner", "skills", "overall"]);
    const both = state([
      { feature: "successes", over: "item" },
      { feature: "failures", over: "learner" },
    ]);
    expect(Object.keys(both.items ?? {})).toEqual(["q1", "q2"]);
    expect(both.items?.q1?.successes).toBe(2);
    expect(both.overall).toMatchObject({ successes: 2, failures: 1 });
  });

  it("tracks in an item's history only what the features over items read", () => {
    const model = logistic({
      features: [
        { decayedSuccesses: 0.5 },
        { feature: { recency: 1 }, over: "item" },
        { feature: "successes", over: "learner" },
      ],
    });
    const state = events.reduce((current, next) => model.update(current, next), model.initialState("ana"));
    expect(Object.keys(state.skills.a?.decayed ?? {})).toEqual(["0.5"]);
    expect(state.items?.q1?.decayed).toEqual({});
    expect(state.overall?.decayed).toEqual({});
  });

  it("refuses an attempt earlier than the learner's last attempt at the item or at any item, when a term reads them", () => {
    const skills = { a: plain, b: plain };
    const itemModel = logistic({ features: [{ feature: "successes", over: "item" }], parameters: { skills } });
    const learnerModel = logistic({
      features: [{ feature: "successes", over: "learner" }],
      parameters: { skills, shared: { "learner:successes": 0.1 } },
    });
    const plainModel = logistic({ parameters: { skills } });
    const after = (model: typeof itemModel) =>
      events.reduce((current, next) => model.update(current, next), model.initialState("ana"));
    const earlier = { learner: "ana", item: "q2", skills: ["a"], time: 2 };
    expect(() => itemModel.predict(after(itemModel), earlier)).toThrow(
      'The attempt at time 2 is earlier than the learner\'s last attempt at item "q2", at time 3.',
    );
    expect(() => learnerModel.predict(after(learnerModel), { ...earlier, item: "q9" })).toThrow(
      "The attempt at time 2 is earlier than the learner's last attempt, at time 3.",
    );
    expect(plainModel.predict(after(plainModel), { ...earlier, item: "q9" })).toBeGreaterThan(0);
  });

  it("continues identically from a state stored as JSON, whatever the terms", () => {
    const model = logistic({
      features: [
        "opportunities",
        { feature: "logSuccesses", over: "item" },
        { feature: "logFailures", over: "learner" },
        { feature: "opportunities", over: "item", coefficient: "shared" },
      ],
      intercepts: ["skill", "item"],
      covariates: ["x"],
      ridge: 1,
      parameters: {
        skills: {
          s1: { intercept: 0.2, coefficients: { opportunities: 0.1 } },
          s2: { intercept: -0.2, coefficients: { opportunities: -0.1 } },
        },
        items: {
          i1: { intercept: 0.3, coefficients: { logSuccesses: 0.2 } },
          i2: { intercept: -0.3, coefficients: { logSuccesses: -0.2 } },
        },
        shared: { "item:opportunities": 0.1, "learner:logFailures": -0.2, "covariate:x": 0.5 },
      },
    });
    const arbitraryEvents = fc
      .array(
        fc.record({
          item: fc.constantFrom("i1", "i2", "i3"),
          skill: fc.constantFrom("s1", "s2"),
          outcome: fc.constantFrom(0, 1, null),
          x: fc.double({ min: -2, max: 2, noNaN: true }),
        }),
        { minLength: 1, maxLength: 12 },
      )
      .map((list) =>
        list.map(
          ({ item, skill, outcome, x }, time): Event => ({
            learner: "ana",
            item,
            skills: [skill],
            time,
            outcome,
            covariates: { x },
          }),
        ),
      );
    fc.assert(
      fc.property(arbitraryEvents, fc.nat(), (list, cut) => {
        const split = cut % (list.length + 1);
        let direct = model.initialState("ana");
        let first = model.initialState("ana");
        for (const [index, event] of list.entries()) {
          if (event.outcome !== null) {
            const prediction = model.predict(direct, event);
            expect(prediction >= 0 && prediction <= 1).toBe(true);
          }
          const copy = JSON.stringify(direct);
          const next = model.update(Object.freeze(direct), event);
          expect(JSON.stringify(direct)).toBe(copy);
          direct = next;
          if (index < split) first = model.update(first, event);
        }
        let stored: LogisticState = JSON.parse(JSON.stringify(first));
        for (const event of list.slice(split)) stored = model.update(stored, event);
        expect(stored).toEqual(direct);
      }),
    );
  });

  it("needs each covariate on every event with an outcome and every prediction, but not on an update or a lesson", () => {
    const model = logistic({ covariates: ["x"] });
    const withX: Event = { ...first, covariates: { x: 1 } };
    const lesson: Event = { ...first, outcome: null };
    expect(() => model.fit(buildEventLog([withX, { ...first, time: 2 }]))).toThrow(
      'Event 2 needs the covariate "x", which the model reads.',
    );
    expect(model.fit(buildEventLog([withX, { ...lesson, time: 2 }])).converged).toBe(true);
    const fitted = model.fit(buildEventLog([withX, { ...withX, time: 2, outcome: 0 }])).model;
    expect(() => fitted.predict(fitted.initialState("ana"), first)).toThrow(
      'The attempt needs the covariate "x", which the model reads.',
    );
    expect(fitted.update(fitted.initialState("ana"), first).skills.a?.successes).toBe(1);
  });

  it("fits more than 5,000 coefficients with a ridge, and refuses to without one or with random proficiency", () => {
    const many = buildEventLog(
      Array.from({ length: 5_001 }, (_, i) => ({
        learner: `l${i % 50}`,
        item: `i${i}`,
        skills: [`s${i % 3}`],
        time: i,
        outcome: (i * 7) % 5 < 3 ? 1 : 0,
      })),
    );
    const ridge = logistic({ intercepts: ["item"], ridge: 1 }).fit(many);
    expect(ridge.converged).toBe(true);
    expect(Object.keys(ridge.model.parameters.items ?? {})).toHaveLength(5_001);
    expect(ridge.standardErrors).toBeNull();
    expect(() => logistic({ intercepts: ["item"] }).fit(many)).toThrow(
      "The fit has 5001 coefficients, more than the 5000 that a fit without a ridge solves. Add a ridge",
    );
    expect(() => logistic({ intercepts: ["item"], ridge: 1, proficiency: "random" }).fit(many)).toThrow(
      "more than the 5000 that a fit with random proficiency solves",
    );
  });
});

describe("logistic fits replayed", () => {
  // Ids that are names of Object.prototype, which a record of histories must hold as entries of its own.
  const learners = ["__proto__", "constructor", "ana"];
  const skills = ["__proto__", "toString", "fractions"];
  const replayLog = buildEventLog(
    Array.from({ length: 90 }, (_, i): EventInput => {
      const base = { learner: defined(learners[i % 3]), item: i % 9 === 0 ? "__proto__" : `q${i % 7}`, time: i + 1 };
      const covariates = { x: ((i * 7) % 5) / 4 };
      const multiple = i % 4 === 0 ? [defined(skills[(i + 1) % 3])] : [];
      return {
        ...base,
        skills: [defined(skills[i % 3]), ...multiple],
        outcome: i % 6 === 5 ? null : (i * i + i) % 5 < 3 ? 1 : 0,
        duration: 1 + (i % 4),
        covariates,
      };
    }),
  );
  const cases: [string, LogisticOptions][] = [
    [
      "terms over skills, items and the learner, exposures, practice and covariates",
      {
        features: [
          "successes",
          "exposures",
          { basePractice: { decay: 0.5, scale: 1 } },
          { feature: "failures", over: "item" },
          { feature: "logSuccesses", over: "learner" },
          { feature: "opportunities", coefficient: "shared" },
        ],
        intercepts: ["skill", "item"],
        covariates: ["x"],
        proficiency: "fixed",
        ridge: 0.5,
      },
    ],
    [
      "decayed features and no proficiency",
      { features: [{ decayedLogit: 0.8 }, { dash: 10 }], proficiency: "none", ridge: 0.1 },
    ],
    ["random proficiency", { features: ["opportunities"], proficiency: "random" }],
    [
      "item intercepts without a ridge penalty on skills",
      {
        features: ["successes", { feature: "opportunities", over: "item" }],
        intercepts: ["item"],
        proficiency: "fixed",
      },
    ],
  ];

  it.each(cases)("gives the predictions and states of an online replay through the fitted model: %s", (_, option) => {
    const fit = logistic(option).fit(replayLog);
    const run = online(fit.model, replayLog);
    expect(fit.predictions.filter((prediction) => prediction !== null)).toHaveLength(
      replayLog.filter((event) => event.outcome !== null).length,
    );
    fit.predictions.forEach((prediction, i) => {
      if (prediction !== null) expectClose(prediction, defined(run.predictions[i]), 1e-14);
    });
    expect(Object.keys(fit.states)).toEqual([...run.states.keys()]);
    // The states are those of the replay in every field and in the order of the fields, which a stored state keeps.
    expect(JSON.stringify(fit.states)).toBe(JSON.stringify(Object.fromEntries(run.states)));
  });
});

describe("logistic standard errors", () => {
  it("are null unless asked, and with a ridge", () => {
    expect(logistic().fit(termsLog).standardErrors).toBeNull();
    expect(logistic({ ridge: 1, standardErrors: true }).fit(termsLog).standardErrors).toBeNull();
    expect(logistic({ standardErrors: true }).fit(termsLog).standardErrors).not.toBeNull();
  });

  it("have a value for each coefficient and, with fixed proficiency, each learner, and none for a random intercept", () => {
    const errors = defined(logistic({ standardErrors: true, proficiency: "fixed" }).fit(termsLog).standardErrors);
    expect(Object.keys(errors.learners ?? {})).toHaveLength(60);
    expect(Object.values(errors.skills).every(({ intercept }) => intercept > 0)).toBe(true);
    const random = defined(logistic({ standardErrors: true, proficiency: "random" }).fit(termsLog).standardErrors);
    expect(random.learners).toBeUndefined();
    expect(random.learnerSd).toBeUndefined();
  });

  it("use the binomial variance of an intercept alone for outcomes between 0 and 1", () => {
    const outcomes = [0.2, 0.8, 0.5, 1, 0.3, 0.7];
    const log = buildEventLog(
      outcomes.map((outcome, time) => ({ learner: "ana", item: "q", skills: ["s"], time, outcome })),
    );
    const errors = defined(logistic({ standardErrors: true }).fit(log).standardErrors);
    const p = outcomes.reduce((sum, outcome) => sum + outcome, 0) / outcomes.length;
    expectClose(defined(errors.skills.s).intercept, 1 / Math.sqrt(outcomes.length * p * (1 - p)), 1e-9);
  });

  it("are null where the information is singular, with parameters still estimated", () => {
    // Opportunities are successes plus failures: the three features are linearly dependent.
    const collinear = logistic({ standardErrors: true, features: ["opportunities", "successes", "failures"] }).fit(
      termsLog,
    );
    expect(collinear.converged).toBe(true);
    expect(collinear.standardErrors).toBeNull();
    // Each learner attempts each skill once, so the recency of every attempt is 0 and its coefficient has no information.
    const once = buildEventLog(
      ["a", "b", "c", "d"].flatMap((learner, l) =>
        ["x", "y"].map((skill, k) => ({
          learner,
          item: `${skill}${l}`,
          skills: [skill],
          time: k,
          outcome: (l + k) % 2,
        })),
      ),
    );
    const dead = logistic({ standardErrors: true, features: [{ recency: 1 }] }).fit(once);
    expect(dead.standardErrors).toBeNull();
  });

  it("shrink as events are added", () => {
    const small = defined(logistic({ standardErrors: true }).fit(termsLog.slice(0, 700)).standardErrors);
    const large = defined(logistic({ standardErrors: true }).fit(termsLog).standardErrors);
    expect(defined(large.skills.s1).intercept).toBeLessThan(defined(small.skills.s1).intercept);
  });
});

describe("logistic options of terms from JavaScript", () => {
  it.each([
    [
      '{"features": [{"feature": "successes", "over": "planet"}]}',
      'The field over of a term must be "skill", "item" or "learner", found "planet".',
    ],
    [
      '{"features": [{"feature": "successes", "coefficient": null}]}',
      'The field coefficient of a term must be "each" or "shared", found null.',
    ],
    [
      '{"features": [{"feature": "successes", "weight": 1}]}',
      'A term may hold only feature, over and coefficient, found "weight".',
    ],
    [
      '{"features": [{"feature": "successes", "over": "learner", "coefficient": "each"}]}',
      "The term successes over the learner needs one shared coefficient: a coefficient for each learner is a random slope",
    ],
    [
      '{"features": [{"feature": "exposures", "over": "item"}]}',
      "The feature exposures counts the exposures to a skill, so its term can only be over skills.",
    ],
    [
      '{"features": [{"feature": "successes", "over": "item"}, {"feature": "successes", "over": "item"}]}',
      "The term successes over item with a each coefficient is listed twice.",
    ],
    ['{"features": [{"feature": "nothing"}]}', 'Unknown feature "nothing"'],
    ['{"intercepts": null}', "The option intercepts must be an array, found null."],
    ['{"intercepts": ["planet"]}', 'An intercept of the option intercepts must be "skill" or "item", found "planet".'],
    ['{"intercepts": ["item", "item"]}', 'The option intercepts lists "item" twice.'],
    [
      '{"intercepts": ["skill", "item"]}',
      'The option intercepts has both "skill" and "item", which needs a ridge above 0: items nest in skills, so without one the pair adds nothing to predictions.',
    ],
    ['{"covariates": null}', "The option covariates must be an array, found null."],
    ['{"covariates": [""]}', 'The option covariates needs names, non-empty strings, found "".'],
    ['{"covariates": ["x", "x"]}', 'The option covariates lists "x" twice.'],
    ['{"standardErrors": "yes"}', 'The option standardErrors must be a boolean, found "yes".'],
    [
      '{"features": [{"feature": "successes", "coefficient": "shared"}], "parameters": {"skills": {}, "shared": {}}}',
      "The shared coefficient skill:successes of the logistic parameters must be a finite number, found undefined.",
    ],
    [
      '{"intercepts": ["item"], "parameters": {"skills": {}, "items": {"q": {"intercept": 0}}}}',
      'The logistic parameters of item "q" need coefficients, an object of coefficients by feature, found undefined.',
    ],
    [
      '{"parameters": {"skills": {}, "shared": null}}',
      "Logistic parameters need shared, an object of coefficients by label, found null.",
    ],
  ])("reject the options %s", (json, message) => {
    const options: LogisticOptions = JSON.parse(json);
    expect(() => logistic(options)).toThrow(DataError);
    expect(() => logistic(options)).toThrow(message);
  });

  it("refuses to predict with shared coefficients that the model lacks", () => {
    const model = logistic({ covariates: ["x"] });
    const attempt = { learner: "ana", item: "q", skills: ["s"], time: 0, covariates: { x: 1 } };
    const state = model.update(model.initialState("ana"), { ...attempt, outcome: 1 });
    expect(() =>
      logistic({ covariates: ["x"], parameters: { skills: { s: { intercept: 0, coefficients: {} } } } }).predict(
        state,
        attempt,
      ),
    ).toThrow("The model has no parameters for its shared coefficients covariate:x.");
  });

  it("rejects a state with malformed histories of items and of the learner", () => {
    const model = logistic({
      features: [
        { feature: "successes", over: "item" },
        { feature: "successes", over: "learner" },
      ],
      parameters: { skills: { s: { intercept: 0, coefficients: {} } }, shared: { "learner:successes": 0.1 } },
    });
    const attempt = { learner: "ana", item: "q", skills: ["s"], time: 0 };
    const parse = (json: string): LogisticState => JSON.parse(json);
    const history = '{"successes": 1, "failures": 0, "last": 0, "decayed": {}}';
    const base = '"learner": "ana", "skills": {}';
    expect(() => model.predict(parse(`{${base}, "items": null}`), attempt)).toThrow(
      'The logistic state of learner "ana" needs items, an object of histories by item id, found null.',
    );
    expect(() =>
      model.predict(
        parse(`{${base}, "items": {"q": {"successes": -1, "failures": 0, "last": 0, "decayed": {}}}}`),
        attempt,
      ),
    ).toThrow('The logistic state of learner "ana", item "q", needs successes, a non-negative number, found -1.');
    expect(() => model.predict(parse(`{${base}, "overall": {"successes": 1}}`), attempt)).toThrow(
      'The logistic state of learner "ana", overall history, needs failures, a non-negative number, found undefined.',
    );
    const stored = parse(`{${base}, "items": {"q": ${history}}, "overall": ${history}}`);
    expect(model.predict(stored, { ...attempt, time: 5 })).toBeGreaterThan(0);
  });
});

describe("rate search", () => {
  const log = buildEventLog(JSON.parse(JSON.stringify(rates.events)));
  const bounds = rates.bounds;
  const build = (values: readonly number[]) =>
    logistic({ features: [{ decayedProportion: values[0] ?? 0 }, { recency: values[1] ?? 0 }] });

  it("finds the rates that maximize the log-likelihood, as optim over glm does", () => {
    const found = searchRates(build, log, bounds);
    expect(found.converged).toBe(true);
    expectAllClose(found.rates, rates.rates, 1e-3, "rates");
    expectClose(found.fit.logLikelihood, rates.logLikelihood, 1e-4, "log-likelihood");
    expect(found.fit.logLikelihood).toBeGreaterThan(build(bounds.start).fit(log).logLikelihood);
    expect(found.iterations).toBeGreaterThan(0);
  });

  it("fits the gradient at the start and at the lengths its line search accepts, not at those it rejects", () => {
    const fitted: (readonly number[])[] = [];
    const recorded = (values: readonly number[]) => {
      fitted.push(values);
      return build(values);
    };
    const found = searchRates(recorded, log, bounds);
    const { lower, upper } = bounds;
    const logits = fitted.map((values) =>
      values.map((value, i) => {
        const place = (value - defined(lower[i])) / (defined(upper[i]) - defined(lower[i]));
        return Math.log(place / (1 - place));
      }),
    );
    // A fit of the gradient moves one rate of the point the line search tried by 1e-4 on the logit scale.
    let tried = defined(logits[0]);
    let gradientFits = 0;
    for (const point of logits.slice(1)) {
      const moved = point.map((value, i) => Math.abs(value - defined(tried[i])));
      const steps = moved.filter((distance) => distance > 1e-9);
      if (steps.length === 1 && Math.abs(defined(steps[0]) - 1e-4) < 1e-7) gradientFits++;
      else tried = point;
    }
    expect(gradientFits % 4).toBe(0);
    expect(gradientFits / 4).toBeLessThanOrEqual(found.iterations + 1);
    expect(fitted.length - gradientFits).toBeGreaterThan(found.iterations + 1);
  });

  it("recovers the decay of simulated logs, and keeps the weakly identified exponent within its bounds", () => {
    // On 20 logs of this design in base R, optim over glm spreads the decay with an SD of 0.13 and the exponent with one
    // of 0.90 (the fixture's recovery), so 2,000 events identify the decay only: every search converges, and the mean of
    // the 8 estimates of the decay is within three standard errors of the truth.
    const truth = [0.6, 0.4];
    const parameters = {
      skills: Object.fromEntries(
        ["s1", "s2", "s3"].map((skill, k) => [
          skill,
          { intercept: -0.3 + 0.2 * k, coefficients: { "decayedProportion(0.6)": 2, "recency(0.4)": 1.2 } },
        ]),
      ),
    };
    const truthModel = logistic({ features: [{ decayedProportion: 0.6 }, { recency: 0.4 }], parameters });
    const estimates = Array.from({ length: 8 }, (_, seed) => {
      const random = createRandom(seed + 1);
      const attempts = Array.from({ length: 50 }, (_, l) => {
        let time = 0;
        return Array.from({ length: 40 }, (_, a) => {
          // Gaps of at least 5 units, as a learner who returns at once is not one whose recency can be told apart.
          time += 5 - 25 * Math.log(1 - random.uniform());
          return { learner: `l${l}`, item: `q${a}`, skills: [`s${1 + Math.floor(random.uniform() * 3)}`], time };
        });
      }).flat();
      const found = searchRates(build, simulate(truthModel, attempts, random).log, {
        start: [0.5, 0.5],
        lower: [0.05, 0.01],
        upper: [1, 3],
      });
      expect(found.converged).toBe(true);
      return found.rates;
    });
    const decays = estimates.map((found) => defined(found[0]));
    const mean = decays.reduce((sum, value) => sum + value, 0) / decays.length;
    expect(Math.abs(mean - defined(truth[0]))).toBeLessThanOrEqual((3 * defined(rates.recovery.sd[0])) / Math.sqrt(8));
    for (const found of estimates) {
      expect(defined(found[1])).toBeGreaterThan(0.01);
      expect(defined(found[1])).toBeLessThan(3);
    }
    // Eight searches of about 0.4 s: under the default timeout of 5 s on a quiet machine, over it on a loaded one.
  }, 60_000);

  it("is reproducible and keeps the rates inside their bounds", () => {
    const narrow = { start: [0.65, 0.35], lower: [0.6, 0.3], upper: [0.8, 0.5] };
    const found = searchRates(build, log, narrow);
    const again = searchRates(build, log, narrow);
    expect(again.rates).toEqual(found.rates);
    expect([again.fit.logLikelihood, again.iterations]).toEqual([found.fit.logLikelihood, found.iterations]);
    found.rates.forEach((rate, j) => {
      expect(rate).toBeGreaterThan(defined(narrow.lower[j]));
      expect(rate).toBeLessThan(defined(narrow.upper[j]));
    });
  });

  it("counts a fit that does not converge as a failed trial and steps back, or stops at the start", () => {
    // Fits at a decay above 0.9 are cut after one iteration and do not converge: the maximum, at 0.7, lies below.
    const cut = (values: readonly number[]) =>
      logistic({
        features: [{ decayedProportion: values[0] ?? 0 }, { recency: values[1] ?? 0 }],
        maxIterations: (values[0] ?? 0) > 0.9 ? 1 : 100,
      });
    const found = searchRates(cut, log, { start: [0.5, 0.5], lower: [0.05, 0.01], upper: [1, 3] });
    expect(found.converged).toBe(true);
    expectAllClose(found.rates, rates.rates, 1e-3, "rates");
    const stuck = searchRates(cut, log, { start: [0.95, 0.5], lower: [0.05, 0.01], upper: [1, 3] });
    expect(stuck).toMatchObject({ rates: [0.95, 0.5], converged: false, iterations: 0 });
    expect(stuck.fit.converged).toBe(false);
  });

  it("passes on the error that building or fitting a model throws", () => {
    const failing = () => {
      throw new DataError("No model for these rates.");
    };
    expect(() => searchRates(failing, log, bounds)).toThrow("No model for these rates.");
    expect(() => searchRates((values) => build(values), buildEventLog([{ ...first(), skills: [] }]), bounds)).toThrow(
      DataError,
    );
  });

  it("rejects options out of range or of another type", () => {
    const options = (patch: Record<string, unknown>) => ({ ...bounds, ...patch }) as typeof bounds;
    expect(() => searchRates(JSON.parse("null"), log, bounds)).toThrow(
      "The argument build must be a function from the rates to a logistic model, found null.",
    );
    expect(() => searchRates(build, log, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => searchRates(build, log, JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
    expect(() => searchRates(build, log, options({ start: JSON.parse("null") }))).toThrow(
      "The option start must be an array of rates, found null.",
    );
    expect(() => searchRates(build, log, options({ lower: [0.05] }))).toThrow(
      "The option lower must hold one number for each rate, as many as the 2 of start, found 1.",
    );
    expect(() => searchRates(build, log, options({ start: [], lower: [], upper: [] }))).toThrow(
      "The option start must hold one number for each rate",
    );
    expect(() => searchRates(build, log, options({ upper: ["a", 3] }))).toThrow(
      'The upper of rate 1 must be a finite number, found "a".',
    );
    expect(() => searchRates(build, log, options({ start: [1, 0.5] }))).toThrow(
      "The start of rate 1 must lie strictly between its bounds 0.05 and 1, found 1.",
    );
    expect(() => searchRates(build, log, options({ tolerance: -1 }))).toThrow("The option tolerance");
    expect(() => searchRates(build, log, options({ maxIterations: 1.5 }))).toThrow("The option maxIterations");
    expect(() => searchRates(() => JSON.parse("{}"), log, bounds)).toThrow(
      "The function build must return a logistic model, found an object.",
    );
  });
});

/** A valid event, for the cases that need a log of one event. */
function first(): Event {
  return { learner: "ana", item: "q1", skills: ["a"], time: 1, outcome: 1 };
}

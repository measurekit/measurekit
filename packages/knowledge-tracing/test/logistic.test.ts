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
import { checkFeatures, decaysOf, featureValue } from "../src/logistic/features.js";
import { nextHistory, type SkillHistory } from "../src/logistic/state.js";
import glm from "./fixtures/logistic-glm.json" with { type: "json" };
import lkt from "./fixtures/logistic-lkt.json" with { type: "json" };
import { defined, expectClose, online } from "./helpers.js";

/** Absolute tolerance of features against LKT's. */
const FEATURE_TOLERANCE = 1e-12;
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

// Parsed rather than imported as typed: TypeScript merges the fits' coefficient records into one type with optional keys.
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
    const rates = decaysOf(specs);
    const histories = new Map<string, SkillHistory>();
    // LKT's rows are in learner order, each learner's in time order.
    lkt.events.forEach((event, index) => {
      const skill = defined(event.skills[0]);
      const key = `${event.learner}\u0000${skill}`;
      const history = histories.get(key);
      specs.forEach((spec, f) => {
        const name = defined(Object.keys(features)[f]) as keyof typeof lkt.features;
        const expected = defined(lkt.features[name][index]);
        expectClose(featureValue(spec, history, event.time), expected, FEATURE_TOLERANCE, `${name} at ${index + 1}`);
      });
      histories.set(key, nextHistory(history, event.outcome, event.time, rates));
    });
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
        expect(fit.predictions.every((p) => p >= 0 && p <= 1)).toBe(true);
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
      expect(fit.predictions.every((p) => p >= 0 && p <= 1)).toBe(true);
    }
    const history = { successes: 0, failures: 1200, last: 0, decayed: { "0.5": { successes: 0, failures: 2 } } };
    const [spec] = checkFeatures([{ decayedLogit: 0.5 }]);
    expect(featureValue(defined(spec), history, 1)).toBe(1200 * Math.log(0.5) - Math.log(2));
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

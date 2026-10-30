import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  type BktOptions,
  type BktParameters,
  type BktSkill,
  type BktState,
  bkt,
  compareModels,
  createRandom,
  crossValidate,
  DataError,
  type Event,
  type EventLog,
  elo,
  eventLog,
  learningCurve,
  splitByLearner,
} from "../src/index.js";
import opportunities from "./fixtures/bkt-opportunities.json" with { type: "json" };
import practice from "./fixtures/bkt-practice.json" with { type: "json" };
import recovery from "./fixtures/bkt-recovery.json" with { type: "json" };
import { defined, expectClose, online } from "./helpers.js";

/** Absolute tolerance of P(correct) and P(L) against pyBKT at fixed parameters. */
const INFERENCE_TOLERANCE = 1e-13;
/** Absolute tolerance of parameters and log-likelihoods fitted from pyBKT's starts. */
const FIT_TOLERANCE = 1e-8;

/** A configuration of `validation/knowledge-tracing/bkt.py`: the options it sets and pyBKT's results. */
interface Configuration {
  readonly name: string;
  readonly options: Pick<BktOptions, "forgetting" | "itemGuessSlip" | "itemLearning"> & {
    readonly priorGroups?: boolean;
  };
}
interface Inference extends Configuration {
  readonly parameters: BktParameters;
  readonly predictions: readonly number[];
  readonly mastery: readonly (readonly number[])[];
}
interface Fit extends Configuration {
  readonly start: BktParameters;
  readonly parameters: BktParameters;
  readonly iterations: number;
  readonly logLikelihood: number;
}

// Parsed rather than imported as typed: TypeScript merges the configurations' parameters into one type with optional
// keys.
const inference: readonly Inference[] = JSON.parse(JSON.stringify(practice.inference));
const fits: readonly Fit[] = JSON.parse(JSON.stringify(practice.fits));
const log = eventLog(practice.events);

/** The options of a configuration, with the fixture's groups for prior groups. */
function optionsOf({ options }: Configuration): BktOptions {
  const { priorGroups, ...rest } = options;
  return priorGroups === true ? { ...rest, priorGroups: practice.priorGroups } : rest;
}

/** Returns the numbers of a value by their paths, such as `itemGuessSlip.q1.guess`. */
function flatten(value: unknown, path = ""): [string, number][] {
  if (typeof value === "number") return [[path, value]];
  if (typeof value !== "object" || value === null) return [];
  return Object.entries(value).flatMap(([key, inner]) => flatten(inner, path === "" ? key : `${path}.${key}`));
}

/** Expects two skills' parameters to have the same rates within `tolerance`, groups and items included. */
function expectSkill(actual: BktSkill | undefined, expected: BktSkill, tolerance: number, label: string): void {
  const found = new Map(flatten(defined(actual)));
  const wanted = flatten(expected);
  expect([...found.keys()].toSorted(), label).toEqual(wanted.map(([name]) => name).toSorted());
  for (const [name, value] of wanted) expectClose(found.get(name) ?? null, value, tolerance, `${label} ${name}`);
}

describe("bkt against pyBKT", () => {
  it.each(inference)("predicts at fixed parameters as pyBKT does: $name", (reference) => {
    const model = bkt({ ...optionsOf(reference), parameters: reference.parameters });
    const states = new Map<string, BktState>();
    log.forEach((event, index) => {
      const state = states.get(event.learner) ?? model.initialState(event.learner);
      event.skills.forEach((skill, position) => {
        const expected = defined(reference.mastery[index]?.[position]);
        expectClose(model.mastery(state, skill), expected, INFERENCE_TOLERANCE, `event ${index + 1} ${skill}`);
      });
      expectClose(
        model.predict(state, event),
        defined(reference.predictions[index]),
        INFERENCE_TOLERANCE,
        `event ${index + 1}`,
      );
      states.set(event.learner, model.update(state, event));
    });
  });

  it.each(fits)("fits from pyBKT's start to pyBKT's estimates: $name", (reference) => {
    const fit = bkt({
      ...optionsOf(reference),
      parameters: reference.start,
      tolerance: practice.tolerance,
      maxIterations: practice.maxIterations,
    }).fit(log);
    for (const [skill, expected] of Object.entries(reference.parameters.skills)) {
      expectSkill(fit.model.parameters.skills[skill], expected, FIT_TOLERANCE, skill);
    }
    expectClose(fit.logLikelihood, reference.logLikelihood, FIT_TOLERANCE, "log-likelihood");
    expect(fit.iterations).toBe(reference.iterations);
    expect(fit.converged).toBe(true);
  });

  it("finds an optimum at least as high as pyBKT's random starts for every skill", () => {
    for (const [skill, reference] of Object.entries(practice.restarts)) {
      const fit = bkt().fit(oneSkill(log, skill));
      expect(fit.logLikelihood, skill).toBeGreaterThanOrEqual(reference.logLikelihood - 1e-6);
    }
  });
});

/** The events of one skill, with that skill only: what an app queries to fit the skill alone. */
function oneSkill(events: EventLog, skill: string): EventLog {
  return events.filter((event) => event.skills.includes(skill)).map((event) => ({ ...event, skills: [skill] }));
}

describe("bkt fitting", () => {
  it("recovers simulated parameters, with a log-likelihood at least that of the true parameters", () => {
    const simulated = eventLog(
      recovery.outcomes.map((outcome, time) => {
        const skill = `s${defined(recovery.skills[time]) + 1}`;
        return { learner: `r${recovery.learners[time]}`, item: skill, skills: [skill], time, outcome };
      }),
    );
    const fit = bkt().fit(simulated);
    expect(fit.converged).toBe(true);
    for (const [skill, truth] of Object.entries(recovery.truth)) {
      const estimates = defined(fit.model.parameters.skills[skill]);
      expect(Math.abs(estimates.prior - truth.prior), `${skill} prior`).toBeLessThanOrEqual(recovery.thresholds.prior);
      for (const name of ["learn", "guess", "slip"] as const) {
        expect(Math.abs(estimates[name] - truth[name]), `${skill} ${name}`).toBeLessThanOrEqual(
          recovery.thresholds.rate,
        );
      }
    }
    expect(fit.logLikelihood).toBeGreaterThanOrEqual(recovery.trueLogLikelihood);
    const truth = bkt({ parameters: { skills: recovery.truth } });
    const atTruth = online(truth, simulated).predictions.reduce(
      (sum, p, index) => sum + Math.log(defined(simulated[index]).outcome === 1 ? p : 1 - p),
      0,
    );
    expectClose(atTruth, recovery.trueLogLikelihood, 1e-8, "log-likelihood at the true parameters");
  });

  it("fits skill by skill to the parameters of the whole log, merged by spreading", () => {
    const configurations: BktOptions[] = [
      {},
      { forgetting: true, itemGuessSlip: true, itemLearning: true },
      { priorGroups: practice.priorGroups },
    ];
    for (const options of configurations) {
      let merged: BktParameters = { skills: {} };
      for (const skill of ["c", "b", "a"]) {
        const fit = bkt(options).fit(oneSkill(log, skill));
        merged = { skills: { ...merged.skills, ...fit.model.parameters.skills } };
      }
      const whole = bkt(options).fit(log).model.parameters;
      expect(merged.skills).toEqual(whole.skills);
    }
  });

  it("stops at maxIterations without converging, and reports it", () => {
    const stopped = bkt({ maxIterations: 2 }).fit(log);
    expect(stopped.converged).toBe(false);
    expect(stopped.iterations).toBe(2);
  });

  it("reproduces a fit from the same seed and reports pyBKT's numbers of parameters", () => {
    const fit = bkt({ seed: 7 }).fit(log);
    const again = bkt({ seed: 7 }).fit(log);
    expect([again.model.parameters, again.states, again.predictions]).toEqual([
      fit.model.parameters,
      fit.states,
      fit.predictions,
    ]);
    // Each skill: P(L0), P(T), P(G), P(S).
    expectClose(fit.aic, 2 * 12 - 2 * fit.logLikelihood, 1e-9);
    expectClose(fit.bic, 12 * Math.log(log.length) - 2 * fit.logLikelihood, 1e-9);
    // Skill a: 2 group priors, 4 items with learn and forget rates, 4 with guess and slip: 2 + 8 + 8; b alike; c: 2 + 4 + 4.
    const extended = bkt({
      forgetting: true,
      itemGuessSlip: true,
      itemLearning: true,
      priorGroups: practice.priorGroups,
    });
    const result = extended.fit(log);
    expectClose(result.aic, 2 * (18 + 18 + 10) - 2 * result.logLikelihood, 1e-9);
  });

  it("predicts, from the fitted parameters, as online inference does", () => {
    const fit = bkt({ itemGuessSlip: true }).fit(log);
    const run = online(fit.model, log);
    expect(run.predictions).toEqual(fit.predictions);
    expect(Object.fromEntries(run.states)).toEqual(fit.states);
  });

  it("works with cross-validation, model comparison and learning curves unchanged", () => {
    const folds = splitByLearner(log, 4, createRandom(1));
    const evaluation = crossValidate(bkt(), folds);
    expect(evaluation.metrics.count).toBe(log.length);
    const comparison = compareModels({ bkt: bkt(), elo: elo({ skills: true }) }, folds);
    expect(comparison.map((row) => row.name)).toEqual(["bkt", "elo"]);
    const curve = learningCurve(log, bkt().fit(log).predictions);
    expect(curve.overall[0]?.predictedErrorRate).not.toBeNull();
  });
});

describe("bkt online", () => {
  const parameters: BktParameters = {
    skills: {
      fractions: { prior: 0.3, learn: 0.2, forget: 0, guess: 0.2, slip: 0.1, groupPriors: { strong: 0.7 } },
      decimals: {
        prior: 0.4,
        learn: 0.1,
        forget: 0.05,
        guess: 0.25,
        slip: 0.15,
        itemGuessSlip: { hard: { guess: 0.05, slip: 0.3 } },
      },
    },
  };
  const model = bkt({ parameters, priorGroups: { budi: "strong" } });

  it("gives a learner's prior for a skill not practiced, their group's when they have one", () => {
    expect(model.mastery(model.initialState("ana"), "fractions")).toBe(0.3);
    expect(model.mastery(model.initialState("budi"), "fractions")).toBe(0.7);
    expect(model.mastery(model.initialState("budi"), "decimals")).toBe(0.4);
  });

  it("predicts an item's own guess and slip, and its skill's for an item without them", () => {
    const state = model.initialState("ana");
    const attempt = { learner: "ana", item: "hard", skills: ["decimals"], time: 0 };
    expect(model.predict(state, attempt)).toBe(0.4 * 0.7 + 0.6 * 0.05);
    expect(model.predict(state, { ...attempt, item: "new" })).toBe(0.4 * 0.85 + 0.6 * 0.25);
  });

  it("predicts an item with two skills by the mean of their predictions and updates both", () => {
    const state = model.initialState("ana");
    const attempt = { learner: "ana", item: "mixed", skills: ["fractions", "decimals"], time: 0 };
    const single = (skill: string): number => model.predict(state, { ...attempt, skills: [skill] });
    expect(model.predict(state, attempt)).toBe((single("fractions") + single("decimals")) / 2);
    const next = model.update(state, { ...attempt, outcome: 1 });
    expect(Object.keys(next.skills)).toEqual(["fractions", "decimals"]);
    expect(next.skills.fractions).toBe(
      model.update(state, { ...attempt, skills: ["fractions"], outcome: 1 }).skills.fractions,
    );
  });

  it.each(opportunities.cases)(
    "expects $opportunities opportunities to mastery from P(L) $mastery with P(T) $learn, P(F) $forget, threshold $threshold",
    (reference) => {
      const skill: BktSkill = {
        prior: reference.mastery,
        learn: reference.learn,
        forget: reference.forget,
        guess: 0.2,
        slip: 0.1,
      };
      const fitted = bkt({ parameters: { skills: { s: skill } } });
      const result = fitted.opportunitiesToMastery(fitted.initialState("ana"), "s", { threshold: reference.threshold });
      expect(result).toBe(reference.opportunities ?? Number.POSITIVE_INFINITY);
    },
  );

  it("uses a mastery threshold of 0.95 by default", () => {
    const fitted = bkt({ parameters: { skills: { s: { prior: 0.2, learn: 0.1, forget: 0, guess: 0.2, slip: 0.1 } } } });
    expect(fitted.opportunitiesToMastery(fitted.initialState("ana"), "s")).toBe(27);
  });

  const arbitraryRate = fc.double({ min: 0, max: 1, noNaN: true });
  const arbitraryInterior = fc.double({ min: 0.01, max: 0.99, noNaN: true });
  const arbitrarySkill: fc.Arbitrary<BktSkill> = fc.record({
    prior: arbitraryRate,
    learn: arbitraryRate,
    forget: arbitraryRate,
    guess: arbitraryInterior,
    slip: arbitraryInterior,
  });
  const arbitraryEvents: fc.Arbitrary<Event[]> = fc
    .array(fc.record({ skills: fc.subarray(["s", "t"], { minLength: 1 }), outcome: fc.constantFrom(0, 1) }), {
      minLength: 1,
      maxLength: 25,
    })
    .map((records) => records.map((record, time) => ({ learner: "ana", item: "q", time, ...record })));

  it("keeps P(L) and predictions between 0 and 1, and never changes the state it is given", () => {
    fc.assert(
      fc.property(arbitrarySkill, arbitrarySkill, arbitraryEvents, (s, t, events) => {
        const random = bkt({ parameters: { skills: { s, t } } });
        let state = random.initialState("ana");
        for (const event of events) {
          const prediction = random.predict(state, event);
          expect(prediction >= 0 && prediction <= 1).toBe(true);
          const copy = JSON.stringify(state);
          const next = random.update(Object.freeze(state), event);
          expect(JSON.stringify(state)).toBe(copy);
          for (const skill of ["s", "t"]) {
            const mastery = random.mastery(next, skill);
            expect(mastery >= 0 && mastery <= 1).toBe(true);
          }
          state = next;
        }
      }),
    );
  });

  it("continues identically from a state stored as JSON", () => {
    fc.assert(
      fc.property(arbitrarySkill, arbitraryEvents, fc.nat(), (s, events, cut) => {
        const random = bkt({ parameters: { skills: { s, t: s } } });
        const split = cut % (events.length + 1);
        const direct = online(random, events).states.get("ana");
        const first = online(random, events.slice(0, split)).states.get("ana") ?? random.initialState("ana");
        let state: BktState = JSON.parse(JSON.stringify(first));
        for (const event of events.slice(split)) state = random.update(state, event);
        expect(state).toEqual(direct);
      }),
    );
  });
});

describe("bkt edge cases", () => {
  const single = eventLog([{ learner: "ana", item: "q", skills: ["s"], time: 0, outcome: 1 }]);
  const constant = (outcome: number): EventLog =>
    eventLog(
      Array.from({ length: 12 }, (_, time) => ({ learner: `l${time % 3}`, item: "q", skills: ["s"], time, outcome })),
    );

  it("fits a single event, and logs all correct or all wrong, without NaN, to models that take any outcome", () => {
    for (const events of [single, constant(1), constant(0)]) {
      const fit = bkt().fit(events);
      const skill = defined(fit.model.parameters.skills.s);
      expect(Object.values(skill).every((rate) => typeof rate === "number" && rate >= 0 && rate <= 1)).toBe(true);
      for (const rate of [skill.guess, skill.slip]) {
        expect(rate >= Number.EPSILON && rate <= 1 - Number.EPSILON).toBe(true);
      }
      expect(Number.isFinite(fit.logLikelihood)).toBe(true);
      expect(fit.predictions.every((p) => p >= 0 && p <= 1)).toBe(true);
      for (const outcome of [0, 1]) {
        const state = fit.model.update(fit.model.initialState("new"), {
          learner: "new",
          item: "q",
          skills: ["s"],
          time: 0,
          outcome,
        });
        expect(Number.isFinite(fit.model.mastery(state, "s"))).toBe(true);
      }
    }
  });

  it("expects the opportunities to mastery of a learn rate so small that 1 - P(T) rounds to 1", () => {
    const slow = bkt({ parameters: { skills: { s: { prior: 0.2, learn: 1e-17, forget: 0, guess: 0.2, slip: 0.1 } } } });
    // (1 - P(T))ⁿ ≤ (1 - 0.95) / (1 - 0.2) = 1 / 16 at n = ln 16 / P(T), to first order in P(T).
    const result = slow.opportunitiesToMastery(slow.initialState("ana"), "s");
    expect(result / (Math.log(16) / 1e-17)).toBeCloseTo(1, 12);
  });

  it("keeps P(L) at most 1 where rounding of a learn rate near 1 would pass it", () => {
    const events = eventLog([
      { learner: "ana", item: "q1", skills: ["fractions"], time: 1, outcome: 1 },
      { learner: "budi", item: "q1", skills: ["fractions"], time: 2, outcome: 0 },
      { learner: "ana", item: "q2", skills: ["fractions"], time: 3, outcome: 1 },
      { learner: "budi", item: "q3", skills: ["decimals"], time: 4, outcome: 1 },
      { learner: "ana", item: "q3", skills: ["decimals"], time: 5, outcome: 0 },
    ]);
    const fit = bkt({ forgetting: true }).fit(events);
    expect(
      Object.values(fit.states)
        .flatMap((state) => Object.values(state.skills))
        .every((p) => p <= 1),
    ).toBe(true);
    const model = bkt({
      parameters: { skills: { s: { prior: 0.5, learn: 0.9999999999999999, forget: 0, guess: 0.2, slip: 0.1 } } },
    });
    let state = model.initialState("ana");
    for (let time = 0; time < 50; time++) {
      state = model.update(state, { learner: "ana", item: "q", skills: ["s"], time, outcome: time % 2 });
    }
    expect(model.mastery(state, "s")).toBeLessThanOrEqual(1);
  });

  it("accepts parameters at their bounds, and rejects an outcome they make impossible", () => {
    const certain = bkt({ parameters: { skills: { s: { prior: 0, learn: 1, forget: 0, guess: 0, slip: 0 } } } });
    const state = certain.initialState("ana");
    const event: Event = { learner: "ana", item: "q", skills: ["s"], time: 0, outcome: 0 };
    expect(certain.predict(state, event)).toBe(0);
    expect(certain.update(state, event).skills.s).toBe(1);
    expect(() => certain.update(state, { ...event, outcome: 1 })).toThrow(
      `The event's outcome has probability 0 for skill "s" under the learner's P(L) and the item's guess and slip.`,
    );
    expect(() => certain.fit(single)).toThrow('The outcomes of learner "ana" have probability 0 under the parameters');
  });

  it("rejects a skill without parameters, events without skills and outcomes other than 0 or 1", () => {
    const model = bkt({ parameters: { skills: { s: { prior: 0.5, learn: 0.1, forget: 0, guess: 0.2, slip: 0.1 } } } });
    const state = model.initialState("ana");
    expect(() => model.mastery(state, "other")).toThrow('The model has no parameters for skill "other".');
    expect(() => model.predict(state, { learner: "ana", item: "q", skills: [], time: 0 })).toThrow(
      "The attempt has no skills; a BKT model needs at least one.",
    );
    expect(() => bkt().fit(eventLog([{ learner: "ana", item: "q", skills: ["s"], time: 0, outcome: 0.5 }]))).toThrow(
      "Event 1 has the outcome 0.5; BKT needs outcomes of 0 or 1.",
    );
    expect(() => bkt().fit([])).toThrow("A log needs at least one event.");
  });

  it.each([
    ['{"restarts": 0}', "The option restarts must be a positive integer, found 0."],
    ['{"forgetting": "yes"}', 'The option forgetting must be a boolean, found "yes".'],
    ['{"priorGroups": {"ana": ""}}', 'The prior group of learner "ana" must be a non-empty string, found "".'],
    ['{"tolerance": -1}', "The option tolerance must be a non-negative number, found -1."],
    [
      '{"parameters": {"skills": {"s": {"prior": 1.5, "learn": 0, "forget": 0, "guess": 0, "slip": 0}}}}',
      'The BKT parameters of skill "s" prior must be a probability from 0 to 1, found 1.5.',
    ],
    [
      '{"parameters": {"skills": {"s": {"prior": 0.5, "learn": 0, "forget": 0, "guess": 0, "slip": 0, "itemGuessSlip": {"q": {"guess": 0.1}}}}}}',
      'The BKT parameters of skill "s", item "q", slip must be a probability from 0 to 1, found undefined.',
    ],
  ])("rejects the options %s", (json, message) => {
    const options: BktOptions = JSON.parse(json);
    expect(() => bkt(options)).toThrow(DataError);
    expect(() => bkt(options)).toThrow(message);
  });

  it("rejects malformed states, as from storage, and the state of another learner", () => {
    const model = bkt({ parameters: { skills: { s: { prior: 0.5, learn: 0.1, forget: 0, guess: 0.2, slip: 0.1 } } } });
    const event: Event = { learner: "ana", item: "q", skills: ["s"], time: 0, outcome: 1 };
    const parse = (json: string): BktState => JSON.parse(json);
    expect(() => model.update(parse('{"learner": "ana", "skills": {"s": 2}}'), event)).toThrow(
      'The BKT state of learner "ana" needs a probability from 0 to 1 for skill "s", found 2.',
    );
    expect(() => model.update(parse('{"learner": "ana"}'), event)).toThrow(
      "needs skills, an object of probabilities by skill id",
    );
    expect(() => model.update(model.initialState("budi"), event)).toThrow(
      'An attempt of learner "ana" was given the state of learner "budi".',
    );
    expect(() => model.opportunitiesToMastery(model.initialState("ana"), "s", { threshold: 2 })).toThrow(
      "The threshold must be a number from 0 to 1, found 2.",
    );
  });
});

describe("bkt options from JavaScript", () => {
  it("rejects options that are not an object", () => {
    const missing = JSON.parse("null");
    expect(() => bkt(missing)).toThrow("The options must be an object, found null.");
    const model = bkt({ parameters: { skills: { s: { prior: 0.3, learn: 0.2, forget: 0, guess: 0.2, slip: 0.1 } } } });
    expect(() => model.opportunitiesToMastery(model.initialState("ana"), "s", missing)).toThrow(
      "The options must be an object, found null.",
    );
  });

  it("rejects parameters of null, which do not stand for the default", () => {
    expect(() => bkt(JSON.parse('{"parameters": null}'))).toThrow("BKT parameters must be an object, found null.");
  });
});

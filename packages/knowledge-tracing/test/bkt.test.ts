import fc from "fast-check";
import { describe, expect, it } from "vitest";
import { type BktOptions, type BktParameters, type BktSkill, type BktState, bkt } from "../src/bkt/index.js";
import { buildEventLog, type Event, type EventLog, splitByLearner } from "../src/data/index.js";
import { elo } from "../src/elo/index.js";
import { compareModels, crossValidate, learningCurve } from "../src/evaluate/index.js";
import { createRandom, DataError } from "../src/index.js";
import { simulate } from "../src/sim/index.js";
import { allPredicted, defined, expectClose } from "./assertions.js";
import { readFixture } from "./fixtures.js";
import { online } from "./logs.js";

interface BktBoundsFixture {
  tolerance: number;
  start: { prior: number; learn: number; forget: number; guess: number; slip: number };
  truth: { prior: number; learn: number; forget: number; guess: number; slip: number };
  events: { learner: string; item: string; skills: string[]; time: number; outcome: number }[];
  fits: {
    name: string;
    forgetting: boolean;
    bounds: { guess?: number[]; slip?: number[]; learn?: number[]; forget?: number[]; prior?: number[] };
    parameters: { prior: number; learn: number; forget: number; guess: number; slip: number };
    logLikelihood: number;
  }[];
}
const bounded = readFixture<BktBoundsFixture>("bkt-bounds.json");

interface BktFixedFixture {
  tolerance: number;
  maxIterations: number;
  fits: {
    name: string;
    options: { resourceLearning?: boolean; itemGuessSlip?: boolean };
    log: string;
    fixed: {
      a?: { learn?: number; guess?: number; slip?: number };
      b?: { learn?: number; guess?: number; slip?: number };
      c?: { learn?: number; guess?: number; slip?: number };
      s?: { resources: { lesson: { learn: number } } };
    };
    start: {
      skills: {
        a?: { prior: number; learn: number; forget: number; guess: number; slip: number };
        b?: { prior: number; learn: number; forget: number; guess: number; slip: number };
        c?: { prior: number; learn: number; forget: number; guess: number; slip: number };
        s?: {
          prior: number;
          learn: number;
          forget: number;
          guess: number;
          slip: number;
          resourceLearning: { lesson: { learn: number; forget: number }; practice: { learn: number; forget: number } };
          itemGuessSlip: {
            p1: { guess: number; slip: number };
            p2: { guess: number; slip: number };
            p3: { guess: number; slip: number };
          };
        };
      };
    };
    parameters: {
      skills: {
        a?: { prior: number; learn: number; forget: number; guess: number; slip: number };
        b?: { prior: number; learn: number; forget: number; guess: number; slip: number };
        c?: { prior: number; learn: number; forget: number; guess: number; slip: number };
        s?: {
          prior: number;
          resourceLearning: { lesson: { learn: number; forget: number }; practice: { learn: number; forget: number } };
          learn: number;
          forget: number;
          itemGuessSlip: {
            p1: { guess: number; slip: number };
            p2: { guess: number; slip: number };
            p3: { guess: number; slip: number };
          };
          guess: number;
          slip: number;
        };
      };
    };
    iterations: number;
    logLikelihood: number;
  }[];
}
const fixedFixture = readFixture<BktFixedFixture>("bkt-fixed.json");

interface BktLessonsFixture {
  tolerance: number;
  maxIterations: number;
  truth: {
    skills: {
      s: {
        prior: number;
        resourceLearning: { lesson: { learn: number; forget: number }; practice: { learn: number; forget: number } };
      };
    };
  };
  events: { learner: string; item: string; resource: string; skills: string[]; time: number; outcome: number | null }[];
  inference: {
    name: string;
    options: { resourceLearning?: boolean; itemGuessSlip?: boolean; forgetting?: boolean };
    parameters: {
      skills: {
        s: {
          prior: number;
          learn: number;
          forget: number;
          guess: number;
          slip: number;
          resourceLearning?: { lesson: { learn: number; forget: number }; practice: { learn: number; forget: number } };
          itemGuessSlip?: {
            p1: { guess: number; slip: number };
            p2: { guess: number; slip: number };
            p3: { guess: number; slip: number };
          };
        };
      };
    };
    predictions: (number | null)[];
    mastery: number[];
  }[];
  fits: {
    name: string;
    options: { resourceLearning?: boolean; itemGuessSlip?: boolean; forgetting?: boolean };
    start: {
      skills: {
        s: {
          prior: number;
          learn: number;
          forget: number;
          guess: number;
          slip: number;
          resourceLearning?: { lesson: { learn: number; forget: number }; practice: { learn: number; forget: number } };
          itemGuessSlip?: {
            p1: { guess: number; slip: number };
            p2: { guess: number; slip: number };
            p3: { guess: number; slip: number };
          };
        };
      };
    };
    parameters: {
      skills: {
        s: {
          prior: number;
          resourceLearning?: { lesson: { learn: number; forget: number }; practice: { learn: number; forget: number } };
          learn: number;
          forget: number;
          itemGuessSlip?: {
            p1: { guess: number; slip: number };
            p2: { guess: number; slip: number };
            p3: { guess: number; slip: number };
          };
          guess: number;
          slip: number;
        };
      };
    };
    iterations: number;
    logLikelihood: number;
  }[];
}
const lessons = readFixture<BktLessonsFixture>("bkt-lessons.json");

interface BktOpportunitiesFixture {
  cases: { mastery: number; learn: number; forget: number; threshold: number; opportunities: number | null }[];
}
const opportunities = readFixture<BktOpportunitiesFixture>("bkt-opportunities.json");

interface BktPairsFixture {
  tolerance: number;
  maxIterations: number;
  options: { pairLearning: boolean; itemGuessSlip: boolean; forgetting: boolean };
  events: { learner: string; item: string; skills: string[]; time: number; outcome: number }[];
  inference: {
    parameters: {
      skills: {
        p: {
          prior: number;
          learn: number;
          forget: number;
          guess: number;
          slip: number;
          pairLearning: {
            i1: {
              i1: { learn: number; forget: number };
              i2: { learn: number; forget: number };
              i3: { learn: number; forget: number };
              i4: { learn: number; forget: number };
            };
            i2: {
              i1: { learn: number; forget: number };
              i2: { learn: number; forget: number };
              i3: { learn: number; forget: number };
              i4: { learn: number; forget: number };
            };
            i3: {
              i1: { learn: number; forget: number };
              i2: { learn: number; forget: number };
              i3: { learn: number; forget: number };
              i4: { learn: number; forget: number };
            };
            i4: {
              i1: { learn: number; forget: number };
              i2: { learn: number; forget: number };
              i3: { learn: number; forget: number };
              i4: { learn: number; forget: number };
            };
          };
          itemGuessSlip: {
            i1: { guess: number; slip: number };
            i2: { guess: number; slip: number };
            i3: { guess: number; slip: number };
            i4: { guess: number; slip: number };
          };
        };
      };
    };
    predictions: number[];
    mastery: number[];
  };
  fit: {
    start: {
      skills: {
        p: {
          prior: number;
          learn: number;
          forget: number;
          guess: number;
          slip: number;
          pairLearning: {
            i1: {
              i1: { learn: number; forget: number };
              i2: { learn: number; forget: number };
              i3: { learn: number; forget: number };
              i4: { learn: number; forget: number };
            };
            i2: {
              i1: { learn: number; forget: number };
              i2: { learn: number; forget: number };
              i3: { learn: number; forget: number };
              i4: { learn: number; forget: number };
            };
            i3: {
              i1: { learn: number; forget: number };
              i2: { learn: number; forget: number };
              i3: { learn: number; forget: number };
              i4: { learn: number; forget: number };
            };
            i4: {
              i1: { learn: number; forget: number };
              i2: { learn: number; forget: number };
              i3: { learn: number; forget: number };
              i4: { learn: number; forget: number };
            };
          };
          itemGuessSlip: {
            i1: { guess: number; slip: number };
            i2: { guess: number; slip: number };
            i3: { guess: number; slip: number };
            i4: { guess: number; slip: number };
          };
        };
      };
    };
    parameters: {
      skills: {
        p: {
          prior: number;
          learn: number;
          forget: number;
          pairLearning: {
            i4: {
              i3: { learn: number; forget: number };
              i1: { learn: number; forget: number };
              i2: { learn: number; forget: number };
              i4: { learn: number; forget: number };
            };
            i3: {
              i1: { learn: number; forget: number };
              i3: { learn: number; forget: number };
              i2: { learn: number; forget: number };
              i4: { learn: number; forget: number };
            };
            i1: {
              i2: { learn: number; forget: number };
              i4: { learn: number; forget: number };
              i3: { learn: number; forget: number };
              i1: { learn: number; forget: number };
            };
            i2: {
              i2: { learn: number; forget: number };
              i4: { learn: number; forget: number };
              i1: { learn: number; forget: number };
              i3: { learn: number; forget: number };
            };
          };
          itemGuessSlip: {
            i1: { guess: number; slip: number };
            i2: { guess: number; slip: number };
            i3: { guess: number; slip: number };
            i4: { guess: number; slip: number };
          };
          guess: number;
          slip: number;
        };
      };
    };
    iterations: number;
    logLikelihood: number;
  };
}
const pairFixture = readFixture<BktPairsFixture>("bkt-pairs.json");

interface BktPracticeFixture {
  tolerance: number;
  maxIterations: number;
  events: { learner: string; item: string; skills: string[]; time: number; outcome: number }[];
  priorGroups: Record<string, string>;
  inference: {
    name: string;
    options: { forgetting?: boolean; itemGuessSlip?: boolean; resourceLearning?: boolean; priorGroups?: boolean };
    parameters: {
      skills: {
        a: {
          prior: number;
          learn: number;
          forget: number;
          guess: number;
          slip: number;
          itemGuessSlip?: {
            a1: { guess: number; slip: number };
            a2: { guess: number; slip: number };
            a3: { guess: number; slip: number };
            ab: { guess: number; slip: number };
          };
          resourceLearning?: {
            a1: { learn: number; forget: number };
            a2: { learn: number; forget: number };
            a3: { learn: number; forget: number };
            ab: { learn: number; forget: number };
          };
          groupPriors?: { high: number };
        };
        b: {
          prior: number;
          learn: number;
          forget: number;
          guess: number;
          slip: number;
          itemGuessSlip?: {
            ab: { guess: number; slip: number };
            b1: { guess: number; slip: number };
            b2: { guess: number; slip: number };
            b3: { guess: number; slip: number };
          };
          resourceLearning?: {
            ab: { learn: number; forget: number };
            b1: { learn: number; forget: number };
            b2: { learn: number; forget: number };
            b3: { learn: number; forget: number };
          };
          groupPriors?: { high: number };
        };
        c: {
          prior: number;
          learn: number;
          forget: number;
          guess: number;
          slip: number;
          itemGuessSlip?: { c1: { guess: number; slip: number }; c2: { guess: number; slip: number } };
          resourceLearning?: { c1: { learn: number; forget: number }; c2: { learn: number; forget: number } };
          groupPriors?: { high: number };
        };
      };
    };
    predictions: number[];
    mastery: number[][];
  }[];
  fits: {
    name: string;
    options: { forgetting?: boolean; itemGuessSlip?: boolean; resourceLearning?: boolean; priorGroups?: boolean };
    start: {
      skills: {
        a: {
          prior: number;
          learn: number;
          forget: number;
          guess: number;
          slip: number;
          itemGuessSlip?: {
            a1: { guess: number; slip: number };
            a2: { guess: number; slip: number };
            a3: { guess: number; slip: number };
            ab: { guess: number; slip: number };
          };
          resourceLearning?: {
            a1: { learn: number; forget: number };
            a2: { learn: number; forget: number };
            a3: { learn: number; forget: number };
            ab: { learn: number; forget: number };
          };
          groupPriors?: { high: number };
        };
        b: {
          prior: number;
          learn: number;
          forget: number;
          guess: number;
          slip: number;
          itemGuessSlip?: {
            ab: { guess: number; slip: number };
            b1: { guess: number; slip: number };
            b2: { guess: number; slip: number };
            b3: { guess: number; slip: number };
          };
          resourceLearning?: {
            ab: { learn: number; forget: number };
            b1: { learn: number; forget: number };
            b2: { learn: number; forget: number };
            b3: { learn: number; forget: number };
          };
          groupPriors?: { high: number };
        };
        c: {
          prior: number;
          learn: number;
          forget: number;
          guess: number;
          slip: number;
          itemGuessSlip?: { c1: { guess: number; slip: number }; c2: { guess: number; slip: number } };
          resourceLearning?: { c1: { learn: number; forget: number }; c2: { learn: number; forget: number } };
          groupPriors?: { high: number };
        };
      };
    };
    parameters: {
      skills: {
        a: {
          prior: number;
          learn: number;
          forget: number;
          guess: number;
          slip: number;
          itemGuessSlip?: {
            a1: { guess: number; slip: number };
            a2: { guess: number; slip: number };
            a3: { guess: number; slip: number };
            ab: { guess: number; slip: number };
          };
          resourceLearning?: {
            a1: { learn: number; forget: number };
            a2: { learn: number; forget: number };
            a3: { learn: number; forget: number };
            ab: { learn: number; forget: number };
          };
          groupPriors?: { high: number };
        };
        b: {
          prior: number;
          learn: number;
          forget: number;
          guess: number;
          slip: number;
          itemGuessSlip?: {
            ab: { guess: number; slip: number };
            b1: { guess: number; slip: number };
            b2: { guess: number; slip: number };
            b3: { guess: number; slip: number };
          };
          resourceLearning?: {
            ab: { learn: number; forget: number };
            b1: { learn: number; forget: number };
            b2: { learn: number; forget: number };
            b3: { learn: number; forget: number };
          };
          groupPriors?: { high: number };
        };
        c: {
          prior: number;
          learn: number;
          forget: number;
          guess: number;
          slip: number;
          itemGuessSlip?: { c1: { guess: number; slip: number }; c2: { guess: number; slip: number } };
          resourceLearning?: { c1: { learn: number; forget: number }; c2: { learn: number; forget: number } };
          groupPriors?: { high: number };
        };
      };
    };
    iterations: number;
    logLikelihood: number;
  }[];
  restarts: {
    a: {
      parameters: { prior: number; learn: number; forget: number; guess: number; slip: number };
      logLikelihood: number;
    };
    b: {
      parameters: { prior: number; learn: number; forget: number; guess: number; slip: number };
      logLikelihood: number;
    };
    c: {
      parameters: { prior: number; learn: number; forget: number; guess: number; slip: number };
      logLikelihood: number;
    };
  };
}
const practice = readFixture<BktPracticeFixture>("bkt-practice.json");

interface BktRecoveryFixture {
  thresholds: { prior: number; rate: number };
  truth: {
    s1: { prior: number; learn: number; forget: number; guess: number; slip: number };
    s2: { prior: number; learn: number; forget: number; guess: number; slip: number };
  };
  learners: number[];
  skills: number[];
  outcomes: number[];
  trueLogLikelihood: number;
}
const recovery = readFixture<BktRecoveryFixture>("bkt-recovery.json");

interface BktSkillsFixture {
  tolerance: number;
  maxIterations: number;
  fits: {
    name: string;
    skills: {
      a: { iterations: number; logLikelihood: number; events: number };
      b: { iterations: number; logLikelihood: number; events: number };
      c: { iterations: number; logLikelihood: number; events: number };
    };
  }[];
}
const skillFixture = readFixture<BktSkillsFixture>("bkt-skills.json");

/** Absolute tolerance of P(correct) and P(L) against pyBKT at fixed parameters. */
const INFERENCE_TOLERANCE = 1e-13;
/** Absolute tolerance of parameters and log-likelihoods fitted from pyBKT's starts. */
const FIT_TOLERANCE = 1e-8;

/** A configuration of `validation/knowledge-tracing/bkt.py`: the options it sets and pyBKT's results. */
interface Configuration {
  readonly name: string;
  readonly options: Pick<BktOptions, "forgetting" | "itemGuessSlip" | "resourceLearning"> & {
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
/** pyBKT's EM for each skill alone, in `bkt-skills.json`. */
interface SkillFits {
  readonly name: string;
  readonly skills: Readonly<Record<string, { iterations: number; logLikelihood: number; events: number }>>;
}

// Parsed rather than imported as typed: TypeScript merges the configurations' parameters into one type with optional
// keys.
const inference: readonly Inference[] = JSON.parse(JSON.stringify(practice.inference));
const fits: readonly Fit[] = JSON.parse(JSON.stringify(practice.fits));
const skillFits: readonly SkillFits[] = JSON.parse(JSON.stringify(skillFixture.fits));
const log = buildEventLog(practice.events);

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

  it.each(fits)("reports the fit of each skill as pyBKT's EM for the skill alone gives it: $name", (reference) => {
    const options: BktOptions = {
      ...optionsOf(reference),
      parameters: reference.start,
      tolerance: skillFixture.tolerance,
      maxIterations: skillFixture.maxIterations,
    };
    const fit = bkt(options).fit(log);
    const expected = defined(skillFits.find((each) => each.name === reference.name)).skills;
    expect(Object.keys(fit.skills).toSorted()).toEqual(Object.keys(expected).toSorted());
    for (const [skill, found] of Object.entries(fit.skills)) {
      const wanted = defined(expected[skill]);
      expect(found.iterations, `${skill} iterations`).toBe(wanted.iterations);
      expect(found.events, `${skill} events`).toBe(wanted.events);
      expect(found.converged, `${skill} converged`).toBe(true);
      expectClose(found.logLikelihood, wanted.logLikelihood, FIT_TOLERANCE, `${skill} log-likelihood`);
      // The skills fit apart: the log of the skill's events alone gives the same fit.
      expect(bkt(options).fit(oneSkill(log, skill)).skills[skill], skill).toEqual(found);
    }
    expect(fit.iterations).toBe(Math.max(...Object.values(fit.skills).map(({ iterations }) => iterations)));
  });

  it("finds an optimum at least as high as pyBKT's random starts for every skill", () => {
    for (const [skill, reference] of Object.entries(practice.restarts)) {
      const fit = bkt().fit(oneSkill(log, skill));
      expect(fit.logLikelihood, skill).toBeGreaterThanOrEqual(reference.logLikelihood - 1e-6);
    }
  });
});

// The lessons' log has events with `outcome: null`, which pyBKT records as correct = -1.
const lessonInference: readonly (Configuration & {
  readonly parameters: BktParameters;
  readonly predictions: readonly (number | null)[];
  readonly mastery: readonly number[];
})[] = JSON.parse(JSON.stringify(lessons.inference));
const lessonFits: readonly Fit[] = JSON.parse(JSON.stringify(lessons.fits));
const lessonLog = buildEventLog(lessons.events);

describe("bkt against pyBKT with events that have no outcome", () => {
  it.each(lessonInference)("tracks and predicts as pyBKT does with correct = -1: $name", (reference) => {
    const model = bkt({ ...optionsOf(reference), parameters: reference.parameters });
    const states = new Map<string, BktState>();
    lessonLog.forEach((event, index) => {
      const state = states.get(event.learner) ?? model.initialState(event.learner);
      expectClose(model.mastery(state, "s"), defined(reference.mastery[index]), INFERENCE_TOLERANCE, `P(L) ${index}`);
      if (event.outcome !== null) {
        expectClose(model.predict(state, event), reference.predictions[index] ?? null, INFERENCE_TOLERANCE, `${index}`);
      }
      states.set(event.learner, model.update(state, event));
    });
  });

  it.each(lessonFits)("fits from pyBKT's start to pyBKT's estimates with correct = -1: $name", (reference) => {
    const fit = bkt({
      ...optionsOf(reference),
      parameters: reference.start,
      tolerance: lessons.tolerance,
      maxIterations: lessons.maxIterations,
    }).fit(lessonLog);
    expectSkill(fit.model.parameters.skills.s, defined(reference.parameters.skills.s), FIT_TOLERANCE, "s");
    expectClose(fit.logLikelihood, reference.logLikelihood, FIT_TOLERANCE, "log-likelihood");
    expect(fit.iterations).toBe(reference.iterations);
    expect(fit.converged).toBe(true);
  });
});

describe("bkt with events that have no outcome", () => {
  const rates: BktSkill = { prior: 0.3, learn: 0.2, forget: 0.1, guess: 0.2, slip: 0.1 };
  const lesson = { learner: "ana", item: "video", skills: ["s"], time: 1, outcome: null };

  it("moves P(L) by the learn and forget rates alone: P(L) (1 - P(F)) + (1 - P(L)) P(T)", () => {
    const model = bkt({ parameters: { skills: { s: rates } } });
    const first = model.update(model.initialState("ana"), lesson);
    expectClose(defined(first.skills.s), 0.3 * 0.9 + 0.7 * 0.2, 1e-15);
    expectClose(
      defined(model.update(first, lesson).skills.s),
      defined(first.skills.s) * 0.9 + (1 - defined(first.skills.s)) * 0.2,
      1e-15,
    );
  });

  it("applies the rates of the event's resource, which is its item without one", () => {
    const skill: BktSkill = {
      ...rates,
      resourceLearning: { video: { learn: 0.5, forget: 0 }, reading: { learn: 0.05, forget: 0 } },
    };
    const model = bkt({ resourceLearning: true, parameters: { skills: { s: skill } } });
    const initial = model.initialState("ana");
    expectClose(defined(model.update(initial, lesson).skills.s), 0.3 + 0.7 * 0.5, 1e-15);
    expectClose(defined(model.update(initial, { ...lesson, resource: "reading" }).skills.s), 0.3 + 0.7 * 0.05, 1e-15);
    expectClose(defined(model.update(initial, { ...lesson, item: "other", resource: "video" }).skills.s), 0.65, 1e-15);
  });

  it("leaves a skill without parameters as it is, and moves the others of the event", () => {
    const model = bkt({ parameters: { skills: { s: rates } } });
    const state = model.update(model.initialState("ana"), { ...lesson, skills: ["s", "unknown"] });
    expect(Object.keys(state.skills)).toEqual(["s"]);
    expect(model.update(model.initialState("ana"), { ...lesson, skills: ["unknown"] })).toEqual(
      model.initialState("ana"),
    );
  });

  it("accepts an event without skills, which changes nothing, and still rejects an outcome with none", () => {
    const model = bkt({ parameters: { skills: { s: rates } } });
    expect(model.update(model.initialState("ana"), { ...lesson, skills: [] })).toEqual(model.initialState("ana"));
    expect(() => model.update(model.initialState("ana"), { ...lesson, skills: [], outcome: 1 })).toThrow(
      "The event has no skills; a BKT model needs at least one.",
    );
  });

  it("leaves the state it was given unchanged, and a state continues after a JSON round trip", () => {
    const model = bkt({ parameters: { skills: { s: rates } } });
    const state = model.initialState("ana");
    const frozen: BktState = JSON.parse(JSON.stringify(state));
    const next = model.update(state, lesson);
    expect(state).toEqual(frozen);
    expect(model.update(JSON.parse(JSON.stringify(next)), lesson)).toEqual(model.update(next, lesson));
  });

  it("gives each event without an outcome a null prediction and no likelihood term", () => {
    const fit = bkt({ resourceLearning: true, itemGuessSlip: true }).fit(lessonLog);
    expect(fit.predictions.map((p) => p === null)).toEqual(lessonLog.map((event) => event.outcome === null));
    const observed = lessonLog.filter((event) => event.outcome !== null);
    const run = online(fit.model, observed);
    expect(Object.keys(fit.states)).toHaveLength(new Set(lessonLog.map((event) => event.learner)).size);
    expect(Number.isFinite(fit.logLikelihood)).toBe(true);
    expect(fit.logLikelihood).toBeGreaterThan(-Number.MAX_VALUE);
    expect(run.predictions).toHaveLength(observed.length);
  });

  it("counts the events with an outcome and the rates they estimate in AIC and BIC", () => {
    const fit = bkt({ resourceLearning: true, itemGuessSlip: true }).fit(lessonLog);
    const n = lessonLog.filter((event) => event.outcome !== null).length;
    // P(L0), a learn rate for each of 2 resources, and a guess and a slip for each of 3 items: a lesson has no item rates.
    expectClose(fit.aic, 2 * 9 - 2 * fit.logLikelihood, 1e-9);
    expectClose(fit.bic, 9 * Math.log(n) - 2 * fit.logLikelihood, 1e-9);
    expect(Object.keys(defined(fit.model.parameters.skills.s?.itemGuessSlip)).toSorted()).toEqual(["p1", "p2", "p3"]);
  });

  it("estimates no rate for a resource that no outcome follows, and none for a skill without an outcome", () => {
    const events = [
      { learner: "a", item: "q", skills: ["s"], time: 1, outcome: 1 },
      { learner: "a", item: "q", skills: ["s"], time: 2, outcome: 0 },
      { learner: "a", item: "outro", skills: ["s", "taught"], time: 3, outcome: null },
      { learner: "b", item: "q", skills: ["s"], time: 1, outcome: 0 },
      { learner: "b", item: "q", skills: ["s"], time: 2, outcome: 1 },
      { learner: "b", item: "outro", skills: ["s", "taught"], time: 3, outcome: null },
    ];
    const fit = bkt({ resourceLearning: true }).fit(buildEventLog(events));
    expect(Object.keys(fit.model.parameters.skills)).toEqual(["s"]);
    expect(Object.keys(fit.skills)).toEqual(["s"]);
    expect(fit.skills.s?.events).toBe(4);
    expect(Object.keys(defined(fit.model.parameters.skills.s?.resourceLearning))).toEqual(["q"]);
    expect(fit.predictions.map((prediction) => prediction === null)).toEqual([false, false, false, false, true, true]);
    // The taught skill has no parameters, so its events leave learners' states with s alone.
    expect(Object.keys(defined(fit.states.a?.skills))).toEqual(["s"]);
  });

  it("refuses to fit a log without an event with an outcome", () => {
    expect(() => bkt().fit(buildEventLog([lesson]))).toThrow(
      "The log has no event with an outcome; a BKT model needs at least one to fit.",
    );
  });

  it("recovers the rates of a lesson and of practice from a seeded simulation", () => {
    const truth: BktParameters = {
      skills: {
        s: {
          prior: 0.2,
          learn: 0.1,
          forget: 0,
          guess: 0.2,
          slip: 0.1,
          resourceLearning: { lesson: { learn: 0.35, forget: 0 }, practice: { learn: 0.1, forget: 0 } },
        },
      },
    };
    const random = createRandom(11);
    const attempts = Array.from({ length: 1500 }, (_, learner) =>
      Array.from({ length: 10 }, (__, step) => {
        const isLesson = random.uniform() < 0.3;
        return {
          learner: `l${learner}`,
          item: isLesson ? "lesson" : "practice",
          skills: ["s"],
          time: step * 10_000 + learner,
          observe: !isLesson,
        };
      }),
    ).flat();
    const { log: simulated } = simulate(bkt({ resourceLearning: true, parameters: truth }), attempts, random);
    expect(simulated.filter((event) => event.outcome === null)).toHaveLength(attempts.filter((a) => !a.observe).length);
    const fit = bkt({ resourceLearning: true }).fit(simulated);
    const estimates = defined(fit.model.parameters.skills.s);
    expect(fit.converged).toBe(true);
    expect(Math.abs(estimates.prior - 0.2)).toBeLessThanOrEqual(0.1);
    for (const [resource, rate] of [
      ["lesson", 0.35],
      ["practice", 0.1],
    ] as const) {
      expect(Math.abs(defined(estimates.resourceLearning?.[resource]).learn - rate), resource).toBeLessThanOrEqual(
        0.05,
      );
    }
  });
});

interface FixedFit {
  readonly name: string;
  readonly options: BktOptions;
  readonly log: "practice" | "lessons";
  readonly fixed: NonNullable<BktOptions["fixed"]>;
  readonly start: BktParameters;
  readonly parameters: BktParameters;
  readonly iterations: number;
  readonly logLikelihood: number;
}
const fixedFits: readonly FixedFit[] = JSON.parse(JSON.stringify(fixedFixture.fits));
const pairOptions: BktOptions = JSON.parse(JSON.stringify(pairFixture.options));
const pairLog = buildEventLog(pairFixture.events);
const pairInference: { parameters: BktParameters; predictions: number[]; mastery: number[] } = JSON.parse(
  JSON.stringify(pairFixture.inference),
);
const pairFit: Omit<Fit, "name" | "options"> = JSON.parse(JSON.stringify(pairFixture.fit));

describe("bkt with fixed rates and pairs against pyBKT", () => {
  it.each(fixedFits)("holds rates as pyBKT's fixed does: $name", (reference) => {
    const fit = bkt({
      ...reference.options,
      fixed: reference.fixed,
      parameters: reference.start,
      tolerance: fixedFixture.tolerance,
      maxIterations: fixedFixture.maxIterations,
    }).fit(reference.log === "practice" ? log : lessonLog);
    for (const [skill, expected] of Object.entries(reference.parameters.skills)) {
      expectSkill(fit.model.parameters.skills[skill], expected, FIT_TOLERANCE, skill);
    }
    expectClose(fit.logLikelihood, reference.logLikelihood, FIT_TOLERANCE, "log-likelihood");
    expect(fit.iterations).toBe(reference.iterations);
    expect(fit.converged).toBe(true);
  });

  it("tracks and predicts with pairs as pyBKT's multipair does, at fixed parameters", () => {
    const model = bkt({ ...pairOptions, parameters: pairInference.parameters });
    const states = new Map<string, BktState>();
    pairLog.forEach((event, index) => {
      const state = states.get(event.learner) ?? model.initialState(event.learner);
      expectClose(
        model.mastery(state, "p"),
        defined(pairInference.mastery[index]),
        INFERENCE_TOLERANCE,
        `P(L) ${index}`,
      );
      expectClose(
        model.predict(state, event),
        defined(pairInference.predictions[index]),
        INFERENCE_TOLERANCE,
        `${index}`,
      );
      states.set(event.learner, model.update(state, event));
    });
  });

  it("fits pairs from pyBKT's start to pyBKT's estimates, the 17 resources of multipair", () => {
    const fit = bkt({
      ...pairOptions,
      parameters: pairFit.start,
      tolerance: pairFixture.tolerance,
      maxIterations: pairFixture.maxIterations,
    }).fit(pairLog);
    expectSkill(fit.model.parameters.skills.p, defined(pairFit.parameters.skills.p), FIT_TOLERANCE, "p");
    expectClose(fit.logLikelihood, pairFit.logLikelihood, FIT_TOLERANCE, "log-likelihood");
    expect(fit.iterations).toBe(pairFit.iterations);
    // The skill's own rates, 16 pairs with their forget rates, and 4 items' guess and slip, and a prior: 1 + 2 + 32 + 8.
    expectClose(fit.aic, 2 * 43 - 2 * fit.logLikelihood, 1e-9);
  });
});

describe("bkt with bounds against base R", () => {
  const boundedLog = buildEventLog(bounded.events);
  interface Bounded {
    readonly name: string;
    readonly forgetting: boolean;
    readonly bounds: NonNullable<BktOptions["bounds"]>;
    readonly parameters: BktSkill;
    readonly logLikelihood: number;
  }
  const cases: readonly Bounded[] = JSON.parse(JSON.stringify(bounded.fits));

  it.each(cases)("reaches the maximum of the forward likelihood within the bounds: $name", (reference) => {
    const start: BktParameters = { skills: { s: JSON.parse(JSON.stringify(bounded.start)) } };
    const fit = bkt({
      forgetting: reference.forgetting,
      bounds: reference.bounds,
      parameters: start,
      tolerance: bounded.tolerance,
      maxIterations: 20_000,
    }).fit(boundedLog);
    const estimates = defined(fit.model.parameters.skills.s);
    for (const rate of ["prior", "learn", "forget", "guess", "slip"] as const) {
      expectClose(estimates[rate], reference.parameters[rate], 1e-4, rate);
      const [lower, upper] = reference.bounds[rate] ?? [0, 1];
      expect(estimates[rate], rate).toBeGreaterThanOrEqual(lower);
      expect(estimates[rate], rate).toBeLessThanOrEqual(upper);
    }
    expectClose(fit.logLikelihood, reference.logLikelihood, 1e-6, "log-likelihood");
    expect(fit.converged).toBe(true);
  });
});

describe("bkt constraints and pairs", () => {
  const events = buildEventLog(
    Array.from({ length: 40 }, (_, learner) =>
      Array.from({ length: 6 }, (__, step) => ({
        learner: `l${learner}`,
        item: `q${1 + (step % 2)}`,
        skills: ["s"],
        time: step * 100 + learner,
        outcome: (learner * 7 + step * 3 + (learner % 3)) % 5 < 2 + Math.floor(step / 2) ? 1 : 0,
      })),
    ).flat(),
  );
  const rates: BktSkill = { prior: 0.3, learn: 0.2, forget: 0.05, guess: 0.2, slip: 0.1 };

  it("holds a fixed rate exactly, leaves it out of the rates that aic and bic count, and estimates the others", () => {
    const free = bkt({ forgetting: true }).fit(events);
    const held = bkt({ forgetting: true, fixed: { s: { learn: 0.17, slip: 0.05 } } }).fit(events);
    expect(defined(held.model.parameters.skills.s).learn).toBe(0.17);
    expect(defined(held.model.parameters.skills.s).slip).toBe(0.05);
    expect(defined(held.model.parameters.skills.s).guess).not.toBe(defined(free.model.parameters.skills.s).guess);
    // P(L0), P(F) and P(G) are estimated, of the 5 rates of the free fit.
    expectClose(held.aic, 2 * 3 - 2 * held.logLikelihood, 1e-9);
    expectClose(free.aic, 2 * 5 - 2 * free.logLikelihood, 1e-9);
    expect(held.logLikelihood).toBeLessThanOrEqual(free.logLikelihood + 1e-9);
  });

  it("holds a rate that a skill's fixed rates do not name for every other skill, and fixes rates by skill id", () => {
    const two = buildEventLog(events.map((event) => ({ ...event, skills: event.learner < "l2" ? ["t"] : ["s"] })));
    const fit = bkt({ fixed: { t: { prior: 0.4 } } }).fit(two);
    expect(defined(fit.model.parameters.skills.t).prior).toBe(0.4);
    expect(defined(fit.model.parameters.skills.s).prior).not.toBe(0.4);
  });

  it("estimates the exact maximum within bounds: a rate that wants more than its bound sits on it", () => {
    const free = defined(bkt().fit(events).model.parameters.skills.s);
    const cap = free.guess / 2;
    const capped = defined(bkt({ bounds: { guess: [0, cap] } }).fit(events).model.parameters.skills.s);
    expect(capped.guess).toBe(cap);
    const floor = (free.learn + 1) / 2;
    expect(defined(bkt({ bounds: { learn: [floor, 1] } }).fit(events).model.parameters.skills.s).learn).toBe(floor);
  });

  it("keeps a guess and a slip off 0 and 1 when a bound allows them, and starts random starts within the bounds", () => {
    const fit = bkt({ bounds: { guess: [0, 0.3], slip: [0, 0.1] }, restarts: 3 }).fit(
      buildEventLog([
        { learner: "a", item: "q", skills: ["s"], time: 1, outcome: 1 },
        { learner: "a", item: "q", skills: ["s"], time: 2, outcome: 1 },
        { learner: "b", item: "q", skills: ["s"], time: 1, outcome: 1 },
        { learner: "b", item: "q", skills: ["s"], time: 2, outcome: 1 },
      ]),
    );
    const { guess, slip } = defined(fit.model.parameters.skills.s);
    expect(slip).toBe(Number.EPSILON);
    expect(guess).toBeLessThanOrEqual(0.3);
    expect(guess).toBeGreaterThanOrEqual(Number.EPSILON);
  });

  it("holds the rates of single resources, and the others of the skill are estimated", () => {
    const fit = bkt({
      resourceLearning: true,
      fixed: { s: { resources: { q1: { learn: 0.3 } } } },
    }).fit(events);
    const learning = defined(fit.model.parameters.skills.s?.resourceLearning);
    expect(defined(learning.q1).learn).toBe(0.3);
    expect(defined(learning.q2).learn).not.toBe(0.3);
    // P(L0), P(T) of q2 and P(G), P(S).
    expectClose(fit.aic, 2 * 4 - 2 * fit.logLikelihood, 1e-9);
  });

  it("uses the skill's rates for a first attempt and a pair that the fit has not seen, and the pair's otherwise", () => {
    const skill: BktSkill = {
      ...rates,
      forget: 0,
      pairLearning: { q1: { q2: { learn: 0.6, forget: 0 } } },
    };
    const model = bkt({ pairLearning: true, parameters: { skills: { s: skill } } });
    const at = (item: string, time: number) => ({ learner: "ana", item, skills: ["s"], time, outcome: null });
    const first = model.update(model.initialState("ana"), at("q1", 1));
    expectClose(defined(first.skills.s), 0.3 + 0.7 * 0.2, 1e-15);
    expect(first.previous).toEqual({ s: "q1" });
    const paired = model.update(first, at("q2", 2));
    expectClose(defined(paired.skills.s), defined(first.skills.s) + (1 - defined(first.skills.s)) * 0.6, 1e-15);
    expect(paired.previous).toEqual({ s: "q2" });
    const unseen = model.update(paired, at("q1", 3));
    expectClose(defined(unseen.skills.s), defined(paired.skills.s) + (1 - defined(paired.skills.s)) * 0.2, 1e-15);
    expect(JSON.parse(JSON.stringify(paired))).toEqual(paired);
  });

  it("fits a rate for each pair of resources whatever characters their ids hold", () => {
    const events = Array.from({ length: 60 }, (_, k) => ({
      learner: `l${Math.floor(k / 6)}`,
      item: k % 2 === 0 ? "x" : "x\u0000y",
      skills: ["s"],
      time: k,
      outcome: k % 3 === 0 ? 0 : 1,
    }));
    const pairs = bkt({ pairLearning: true, restarts: 1 }).fit(buildEventLog(events)).model.parameters.skills.s
      ?.pairLearning;
    expect(Object.entries(pairs ?? {}).map(([before, rates]) => [before, Object.keys(rates)])).toEqual([
      ["x", ["x\u0000y"]],
      ["x\u0000y", ["x"]],
    ]);
  });

  it("fits a rate for each pair of resources named as members of an object, and holds them through storage", () => {
    const events = Array.from({ length: 60 }, (_, k) => ({
      learner: `l${Math.floor(k / 6)}`,
      item: k % 2 === 0 ? "__proto__" : "constructor",
      skills: ["s"],
      time: k,
      outcome: k % 3 === 0 ? 0 : 1,
    }));
    const { model } = bkt({ pairLearning: true, restarts: 1 }).fit(buildEventLog(events));
    const pairs = model.parameters.skills.s?.pairLearning;
    expect(Object.entries(pairs ?? {}).map(([before, rates]) => [before, Object.keys(rates)])).toEqual([
      ["__proto__", ["constructor"]],
      ["constructor", ["__proto__"]],
    ]);
    expect(Object.getPrototypeOf(pairs)).toBe(Object.prototype);
    const stored = bkt({ pairLearning: true, parameters: JSON.parse(JSON.stringify(model.parameters)) });
    const first = stored.update(stored.initialState("ana"), { ...defined(events[0]), learner: "ana" });
    const attempt = { learner: "ana", item: "constructor", skills: ["s"], time: 2 };
    expect(stored.predict(JSON.parse(JSON.stringify(first)), attempt)).toBe(model.predict(first, attempt));
  });

  it("reads a state stored without previous resources as one with none, and writes none without pairs", () => {
    const skill: BktSkill = { ...rates, pairLearning: { q1: { q2: { learn: 0.6, forget: 0 } } } };
    const model = bkt({ pairLearning: true, parameters: { skills: { s: skill } } });
    const stored: BktState = { learner: "ana", skills: { s: 0.4 } };
    const event = { learner: "ana", item: "q2", skills: ["s"], time: 1, outcome: null };
    expectClose(defined(model.update(stored, event).skills.s), 0.4 * 0.95 + 0.6 * 0.2, 1e-15);
    const plain = bkt({ parameters: { skills: { s: rates } } });
    expect(plain.update({ ...stored, previous: { s: "q1" } }, event).previous).toBeUndefined();
  });

  it("rejects fixed rates and bounds that are malformed or contradict the options", () => {
    const fail = (options: BktOptions, message: string): void => {
      expect(() => bkt(options), message).toThrow(message);
    };
    fail(
      { fixed: { s: { learn: 1.2 } } },
      'The fixed rates of skill "s" have learn that is not a probability, found 1.2.',
    );
    fail(
      { fixed: { s: JSON.parse('{"rate":0.2}') } },
      'The option fixed, for skill "s", may hold only prior, learn, forget, guess, slip and resources, found "rate".',
    );
    fail(
      { resourceLearning: true, fixed: { s: { resources: { r: JSON.parse('{"rate":0.2}') } } } },
      'The option fixed, for skill "s" and resource "r", may hold only learn and forget, found "rate".',
    );
    fail({ fixed: { s: { forget: 0.1 } } }, 'The fixed rates of skill "s" have forget, which needs forgetting: true.');
    fail({ bounds: { forget: [0, 0.1] } }, "The bounds of forget need forgetting: true.");
    fail(
      { fixed: { s: { resources: { q: { learn: 0.1 } } } } },
      'The fixed rates of skill "s" have resources, which need resourceLearning: true.',
    );
    fail({ bounds: { guess: [0.5, 0.2] } }, "The lower bound of guess, 0.5, is above its upper bound, 0.2.");
    fail(
      { bounds: { guess: [0, 1.5] } },
      "The bounds of guess must be a pair of probabilities [lower, upper], found an array.",
    );
    fail(
      { bounds: { learn: [0, 0.5] }, fixed: { s: { learn: 0.7 } } },
      'The fixed rates of skill "s" have learn 0.7, outside its bounds [0, 0.5].',
    );
    fail(
      { bounds: { guess: [0, 0.3] }, parameters: { skills: { s: { ...rates, guess: 0.5 } } } },
      'The BKT parameters of skill "s" have guess 0.5, outside the bounds [0, 0.3].',
    );
    fail(
      { pairLearning: true, resourceLearning: true },
      "The option pairLearning cannot be used with resourceLearning: a rate has one key.",
    );
    fail(
      { bounds: JSON.parse('{"speed":[0,1]}') },
      'The option bounds may hold only prior, learn, forget, guess and slip, found "speed".',
    );
    fail({ fixed: JSON.parse("[]") }, "The option fixed must be an object of fixed rates by skill id, found an array.");
  });
});

/** The events of one skill, with that skill only: what an app queries to fit the skill alone. */
function oneSkill(events: EventLog, skill: string): EventLog {
  return events.filter((event) => event.skills.includes(skill)).map((event) => ({ ...event, skills: [skill] }));
}

/** The seeded simulation of two skills, one skill in every event. */
function simulatedLog(): EventLog {
  return buildEventLog(
    recovery.outcomes.map((outcome, time) => {
      const skill = `s${defined(recovery.skills[time]) + 1}`;
      return { learner: `r${recovery.learners[time]}`, item: skill, skills: [skill], time, outcome };
    }),
  );
}

describe("bkt fitting", () => {
  it("adds the log-likelihoods of the skills up to that of the fit when every event has one skill", () => {
    const simulated = simulatedLog();
    const fit = bkt().fit(simulated);
    expect(Object.keys(fit.skills)).toEqual(["s1", "s2"]);
    const total = Object.values(fit.skills).reduce((sum, skill) => sum + skill.logLikelihood, 0);
    expectClose(total, fit.logLikelihood, 1e-9, "log-likelihood");
    expect(Object.values(fit.skills).reduce((sum, skill) => sum + skill.events, 0)).toBe(simulated.length);
  });

  it("recovers simulated parameters, with a log-likelihood at least that of the true parameters", () => {
    const simulated = simulatedLog();
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
      { forgetting: true, itemGuessSlip: true, resourceLearning: true },
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
    // Skill a: 2 group priors, 4 items with learn and forget rates, 4 with guess and slip: 2 + 8 + 8; b alike; c: 2 +
    // 4 + 4.
    const extended = bkt({
      forgetting: true,
      itemGuessSlip: true,
      resourceLearning: true,
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
    const curve = learningCurve(log, { predictions: bkt().fit(log).predictions });
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
  const single = buildEventLog([{ learner: "ana", item: "q", skills: ["s"], time: 0, outcome: 1 }]);
  const constant = (outcome: number): EventLog =>
    buildEventLog(
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
      expect(allPredicted(fit.predictions).every((p) => p >= 0 && p <= 1)).toBe(true);
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
    const events = buildEventLog([
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
    expect(() =>
      bkt().fit(buildEventLog([{ learner: "ana", item: "q", skills: ["s"], time: 0, outcome: 0.5 }])),
    ).toThrow("Event 1 has the outcome 0.5; BKT needs outcomes of 0 or 1.");
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

  it("checks the entries of the skills a call reads, so that its cost does not grow with the others", () => {
    const rates = { prior: 0.5, learn: 0.1, forget: 0, guess: 0.2, slip: 0.1 };
    const model = bkt({ pairLearning: true, parameters: { skills: { s: rates, t: rates } } });
    const state: BktState = JSON.parse('{"learner": "ana", "skills": {"t": 2}, "previous": {"t": ""}}');
    const next = model.update(state, { learner: "ana", item: "q", skills: ["s"], time: 0, outcome: 1 });
    expect(model.mastery(next, "s")).toBeCloseTo(0.8364, 4);
    expect(() => model.mastery(next, "t")).toThrow('needs a probability from 0 to 1 for skill "t", found 2.');
    const resource = JSON.parse('{"learner": "ana", "skills": {}, "previous": {"t": ""}}');
    expect(() => model.predict(resource, { learner: "ana", item: "q", skills: ["t"], time: 1 })).toThrow(
      'needs a resource id, a non-empty string, for skill "t", found "".',
    );
  });
});

describe("bkt options from JavaScript", () => {
  it("rejects options that are not an object", () => {
    const missing = JSON.parse("null");
    expect(() => bkt(missing)).toThrow("The options must be an object, found null.");
    expect(() => bkt(JSON.parse('{"verbose":1}'))).toThrow('There is no option "verbose"');
    const model = bkt({ parameters: { skills: { s: { prior: 0.3, learn: 0.2, forget: 0, guess: 0.2, slip: 0.1 } } } });
    expect(() => model.opportunitiesToMastery(model.initialState("ana"), "s", missing)).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => model.opportunitiesToMastery(model.initialState("ana"), "s", JSON.parse('{"verbose":1}'))).toThrow(
      'There is no option "verbose"',
    );
  });

  it("rejects parameters of null, which do not stand for the default", () => {
    expect(() => bkt(JSON.parse('{"parameters": null}'))).toThrow("BKT parameters must be an object, found null.");
  });
});

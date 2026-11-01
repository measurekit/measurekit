import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  type Attempt,
  compareModels,
  createRandom,
  crossValidate,
  delineate,
  type Event,
  type EventLog,
  eventLog,
  fringes,
  iita,
  isKnowledgeSpace,
  isLearningSpace,
  type KnowledgeStructure,
  type KstState,
  kst,
  splitByLearner,
  surmiseStructure,
} from "../src/index.js";
import assessmentFixture from "./fixtures/kst-assessment.json" with { type: "json" };
import blimFixture from "./fixtures/kst-blim.json" with { type: "json" };
import iitaFixture from "./fixtures/kst-iita.json" with { type: "json" };
import structureFixture from "./fixtures/kst-structures.json" with { type: "json" };
import { defined, expectAllClose, expectClose, expectUnaffectedByLessons, online, withLessons } from "./helpers.js";

/** Absolute tolerance of BLIM parameters fitted by EM against pks, which stops on a change of the parameters. */
const ML_TOLERANCE = 1e-6;
/** Absolute tolerance of the log-likelihood of a BLIM fitted by EM. */
const ML_LOG_LIKELIHOOD_TOLERANCE = 1e-8;
/** Absolute tolerance of quantities computed in one pass of the same arithmetic: MD estimates, the assessment. */
const EXACT_TOLERANCE = 1e-12;
/** Relative tolerance of IITA's diff against DAKS. */
const DIFF_TOLERANCE = 1e-10;

type Prerequisites = Record<string, string[]>;
interface Structures {
  readonly surmise: readonly { name: string; prerequisites: Prerequisites; structure: KnowledgeStructure }[];
  readonly spaces: readonly {
    name: string;
    structure: KnowledgeStructure;
    knowledgeSpace: boolean;
    learningSpace: boolean;
    inner: string[][];
    outer: string[][];
  }[];
  readonly delineation: { structure: KnowledgeStructure; qMatrix: Prerequisites; items: KnowledgeStructure };
}
interface BlimReference {
  readonly name: string;
  readonly method: "ml" | "md";
  readonly structure: KnowledgeStructure;
  readonly patterns: Record<string, number>;
  readonly probabilities: number[];
  readonly guess: Record<string, number>;
  readonly slip: Record<string, number>;
  readonly logLikelihood: number;
}
interface IitaVersionReference {
  readonly diff: number[];
  readonly selected: number;
  readonly errorRate: number;
  readonly prerequisites: Prerequisites;
}
interface IitaReference {
  readonly name: string;
  readonly items: string[];
  readonly patterns: Record<string, number>;
  readonly selection: Prerequisites[];
  readonly versions: Record<"minimized" | "corrected" | "original", IitaVersionReference>;
}
interface Assessment {
  readonly threshold: number;
  readonly assessments: readonly {
    truth: string[];
    questions: string[];
    outcomes: number[];
    probabilities: number[];
  }[];
  readonly deterministic: { patterns: string[][]; states: string[][] };
}

// Parsed rather than imported as typed: TypeScript would merge the cases into unions with optional keys.
const structures: Structures = JSON.parse(JSON.stringify(structureFixture));
const blims: readonly BlimReference[] = JSON.parse(JSON.stringify(blimFixture));
const iitas: readonly IitaReference[] = JSON.parse(JSON.stringify(iitaFixture));
const assessment: Assessment = JSON.parse(JSON.stringify(assessmentFixture));

/**
 * Returns the log of learners who each answer every item once, as many with each pattern of responses, a string of 0
 * and 1 over the items, as its frequency.
 */
function patternLog(items: readonly string[], patterns: Readonly<Record<string, number>>): EventLog {
  const events: Event[] = [];
  let learner = 0;
  for (const [pattern, frequency] of Object.entries(patterns)) {
    for (let copy = 0; copy < frequency; copy++, learner++) {
      items.forEach((item, i) => {
        events.push({
          learner: `l${learner}`,
          item,
          skills: [],
          time: learner * items.length + i,
          outcome: Number(pattern[i]),
        });
      });
    }
  }
  return eventLog(events);
}

/** A structure's states as sorted keys, for comparisons that ignore order. */
const keys = (structure: KnowledgeStructure): string[] =>
  structure.states.map((state) => [...state].sort().join(",")).sort();

describe("knowledge structures against R", () => {
  it.each(structures.surmise)("builds the quasi-ordinal space of a surmise relation as DAKS: $name", (reference) => {
    const built = surmiseStructure(reference.prerequisites);
    expect(built.domain).toEqual(reference.structure.domain);
    expect(keys(built)).toEqual(keys(reference.structure));
    expect(isKnowledgeSpace(built)).toBe(true);
  });

  it.each(structures.spaces)("finds spaces, learning spaces and fringes as kst: $name", (reference) => {
    expect(isKnowledgeSpace(reference.structure)).toBe(reference.knowledgeSpace);
    expect(isLearningSpace(reference.structure)).toBe(reference.learningSpace);
    reference.structure.states.forEach((state, s) => {
      const found = fringes(reference.structure, state);
      expect([...found.inner].sort(), `state ${s + 1}`).toEqual(reference.inner[s]);
      expect([...found.outer].sort(), `state ${s + 1}`).toEqual(reference.outer[s]);
    });
  });

  it("delineates an item structure from skills as pks", () => {
    const { structure, qMatrix, items } = structures.delineation;
    const delineated = delineate(structure, qMatrix);
    expect(delineated.domain).toEqual(items.domain);
    expect(keys(delineated)).toEqual(keys(items));
  });
});

describe("BLIM against pks", () => {
  it.each(blims)("fits $name by $method as pks's blim", (reference) => {
    const { method } = reference;
    const fit = kst({
      structure: reference.structure,
      over: "items",
      method,
      tolerance: 1e-13,
      maxIterations: 1e6,
    }).fit(patternLog(reference.structure.domain, reference.patterns));
    const tolerance = method === "ml" ? ML_TOLERANCE : EXACT_TOLERANCE;
    expectAllClose(fit.model.parameters.probabilities, reference.probabilities, tolerance, "P(K)");
    for (const [item, guess] of Object.entries(reference.guess)) {
      const fitted = defined(fit.model.parameters.items[item]);
      expectClose(fitted.guess, guess, tolerance, `${item} guess`);
      expectClose(fitted.slip, defined(reference.slip[item]), tolerance, `${item} slip`);
    }
    const logLikelihoodTolerance = method === "ml" ? ML_LOG_LIKELIHOOD_TOLERANCE : EXACT_TOLERANCE;
    expectClose(fit.logLikelihood, reference.logLikelihood, logLikelihoodTolerance, "log-likelihood");
    expect(fit.converged).toBe(true);
  });
});

describe("IITA against DAKS", () => {
  const versions = ["minimized", "corrected", "original"] as const;
  it.each(iitas)("generates the selection set and fits each quasi-order as DAKS: $name", (reference) => {
    const log = patternLog(reference.items, reference.patterns);
    for (const version of versions) {
      const result = iita(log, { version });
      const expected = reference.versions[version];
      expect(result.candidates.map((candidate) => candidate.prerequisites)).toEqual(reference.selection);
      result.candidates.forEach((candidate, k) => {
        const diff = defined(expected.diff[k]);
        expect(Math.abs(candidate.diff - diff), `${version} diff ${k + 1}`).toBeLessThanOrEqual(DIFF_TOLERANCE * diff);
      });
      expect(result.prerequisites, version).toEqual(expected.prerequisites);
      expect(result).toMatchObject(defined(result.candidates[expected.selected - 1]));
      expectClose(result.errorRate, expected.errorRate, EXACT_TOLERANCE, `${version} error rate`);
    }
  });
});

describe("adaptive assessment", () => {
  const [reference] = blims;
  const { structure } = defined(reference);
  const parameters = {
    probabilities: defined(reference).probabilities,
    items: Object.fromEntries(
      Object.entries(defined(reference).guess).map(([item, guess]) => [
        item,
        { guess, slip: defined(defined(reference).slip[item]) },
      ]),
    ),
  };
  const model = kst({ structure, over: "items", parameters });
  const candidates: Attempt[] = structure.domain.map((item, time) => ({ learner: "ana", item, skills: [], time }));

  it.each(assessment.assessments)("asks the written-out questions and reaches its posterior: $truth", (reference) => {
    let state = model.initialState("ana");
    reference.questions.forEach((item, step) => {
      const question = defined(model.nextQuestion(state, candidates, { threshold: assessment.threshold }));
      expect(question.item, `question ${step + 1}`).toBe(item);
      state = model.update(state, { ...question, outcome: defined(reference.outcomes[step]) });
    });
    expect(model.nextQuestion(state, candidates, { threshold: assessment.threshold })).toBeNull();
    expectAllClose(state.probabilities, reference.probabilities, EXACT_TOLERANCE, "posterior");
  });

  it("assigns every response pattern that is a state to that state without errors, as kst's kassess", () => {
    const exact = kst({ structure, over: "items" });
    const items = Object.fromEntries(structure.domain.map((item) => [item, { guess: 0, slip: 0 }]));
    const deterministic = kst({ structure, over: "items", parameters: { ...exact.parameters, items } });
    assessment.deterministic.patterns.forEach((pattern, p) => {
      let state = deterministic.initialState("ana");
      for (let question = deterministic.nextQuestion(state, candidates, { threshold: 1 }); question !== null; ) {
        state = deterministic.update(state, { ...question, outcome: pattern.includes(question.item) ? 1 : 0 });
        question = deterministic.nextQuestion(state, candidates, { threshold: 1 });
      }
      expect(deterministic.mostLikelyState(state)).toEqual({
        state: assessment.deterministic.states[p],
        probability: 1,
      });
    });
  });

  it("reaches the same posterior whatever the order of the answers", () => {
    fc.assert(
      fc.property(
        fc.array(fc.tuple(fc.constantFrom(...structure.domain), fc.constantFrom(0, 1)), {
          minLength: 1,
          maxLength: 12,
        }),
        fc.nat(),
        (answers, seed) => {
          const events = answers.map(
            ([item, outcome], time): Event => ({ learner: "ana", item, skills: [], time, outcome }),
          );
          const shuffled = [...events].sort(() => createRandom(seed).uniform() - 0.5);
          const run = (sequence: readonly Event[]): KstState =>
            sequence.reduce((state, event) => model.update(state, event), model.initialState("ana"));
          expectAllClose(run(events).probabilities, run(shuffled).probabilities, 1e-12);
        },
      ),
    );
  });
});

describe("KST model", () => {
  const structure = surmiseStructure({
    add: [],
    subtract: ["add"],
    multiply: ["add"],
    divide: ["subtract", "multiply"],
  });
  const qMatrix = { q1: ["add"], q2: ["subtract"], q3: ["add", "multiply"], q4: ["divide"], q5: ["multiply"] };
  const random = createRandom(20261029);
  const truth = { probabilities: [0.2, 0.2, 0.15, 0.15, 0.1, 0.2], guess: 0.15, slip: 0.1 };
  const events = Array.from({ length: 120 }, (_, learner) => {
    const s = random.uniform() < 0.5 ? Math.floor(random.uniform() * 3) : 3 + Math.floor(random.uniform() * 3);
    const state = defined(structure.states[s]);
    return Object.entries(qMatrix).map(([item, skills], i) => {
      const mastered = skills.every((skill) => state.includes(skill));
      const outcome = random.uniform() < (mastered ? 1 - truth.slip : truth.guess) ? 1 : 0;
      return { learner: `l${learner}`, item, time: learner * 10 + i, outcome };
    });
  }).flat();
  const log = eventLog(events, { qMatrix });
  const model = kst({ structure });

  it("predicts from a fit as online inference does, and works with cross-validation and comparison unchanged", () => {
    const fit = model.fit(log);
    const run = online(fit.model, log);
    expect(run.predictions).toEqual(fit.predictions);
    expect(Object.fromEntries(run.states)).toEqual(fit.states);
    expect(fit.aic).not.toBeNull();
    const folds = splitByLearner(log, 3, createRandom(1));
    expect(crossValidate(model, folds).metrics.count).toBe(log.length);
    expect(compareModels({ kst: model, md: kst({ structure, method: "md" }) }, folds).map((row) => row.name)).toEqual([
      "kst",
      "md",
    ]);
  });

  it("is unaffected by events without an outcome, which are no items, and IITA reads the others", () => {
    for (const options of [{}, { method: "md" as const }])
      expectUnaffectedByLessons(kst({ structure, ...options }), log);
    expect(iita(withLessons(log, 3))).toEqual(iita(log));
  });

  it("returns the state it was given, as a copy, for an event without an outcome", () => {
    const fittedModel = model.fit(log).model;
    const state = fittedModel.update(fittedModel.initialState("ana"), { ...defined(log[0]), learner: "ana" });
    const next = fittedModel.update(state, { learner: "ana", item: "lesson", skills: [], time: 99, outcome: null });
    expect(next).toEqual(state);
    expect(next.probabilities).not.toBe(state.probabilities);
  });

  const fitted = model.fit(log).model;
  const sequences = fc
    .array(fc.constantFrom(...log), { maxLength: 20 })
    .map((picked) => picked.map((event, time) => ({ ...event, learner: "ana", time })));

  it("predicts probabilities from 0 to 1 and never changes the state it is given", () => {
    fc.assert(
      fc.property(sequences, (sequence) => {
        let state = fitted.initialState("ana");
        for (const event of sequence) {
          const prediction = fitted.predict(state, event);
          expect(prediction >= 0 && prediction <= 1).toBe(true);
          const copy = JSON.stringify(state);
          const next = fitted.update(Object.freeze(state), event);
          expect(JSON.stringify(state)).toBe(copy);
          state = next;
        }
      }),
    );
  });

  it("continues identically from a state stored as JSON", () => {
    fc.assert(
      fc.property(sequences, fc.nat(), (sequence, cut) => {
        const split = cut % (sequence.length + 1);
        const run = (state: KstState, events: readonly Event[]) => events.reduce((s, e) => fitted.update(s, e), state);
        const direct = run(fitted.initialState("ana"), sequence);
        const stored: KstState = JSON.parse(JSON.stringify(run(fitted.initialState("ana"), sequence.slice(0, split))));
        expect(run(stored, sequence.slice(split))).toEqual(direct);
      }),
    );
  });

  it("stops at maxIterations without converging, and reports it", () => {
    const stopped = kst({ structure, maxIterations: 2 }).fit(log);
    expect(stopped.converged).toBe(false);
    expect(stopped.iterations).toBe(2);
  });

  it("gives skill mastery as the probability of the states that hold the skill", () => {
    const state = fitted.update(fitted.initialState("ana"), {
      learner: "ana",
      item: "q4",
      skills: ["divide"],
      time: 0,
      outcome: 1,
    });
    const holding = structure.states.flatMap((s, k) => (s.includes("subtract") ? [state.probabilities[k] ?? 0] : []));
    expectClose(
      fitted.mastery(state, "subtract"),
      holding.reduce((sum, p) => sum + p, 0),
      1e-15,
    );
    expect(fitted.mastery(state, "divide")).toBeLessThanOrEqual(fitted.mastery(state, "add"));
  });
});

describe("KST edge cases", () => {
  it("rejects malformed structures and too many states, suggesting a split by unit", () => {
    expect(() => isKnowledgeSpace({ domain: ["a", "b"], states: [[], ["a"]] })).toThrow(
      "A knowledge structure needs the state of the whole domain.",
    );
    expect(() => isKnowledgeSpace({ domain: ["a"], states: [["a"]] })).toThrow("needs the empty state");
    expect(() => isKnowledgeSpace({ domain: ["a"], states: [[], ["b"], ["a"]] })).toThrow(
      'State 2 has "b", which is not in the domain.',
    );
    expect(() => isKnowledgeSpace({ domain: ["a"], states: [[], ["a"], ["a"]] })).toThrow(
      "State 3 repeats an earlier state.",
    );
    const independent = Object.fromEntries(Array.from({ length: 14 }, (_, i) => [`e${i}`, []]));
    expect(() => surmiseStructure(independent)).toThrow(
      "The structure has more than 10000 knowledge states. Split the domain into units, such as chapters, and build a structure for each, or raise maxStates.",
    );
    expect(surmiseStructure(independent, { maxStates: 2 ** 14 }).states).toHaveLength(2 ** 14);
    expect(() => surmiseStructure({ a: ["b"] })).toThrow('"a" requires "b", which is not a key of the prerequisites.');
  });

  it("freezes the structures it builds, and checks any other structure in full at each call", () => {
    const built = surmiseStructure({ a: [], b: ["a"], c: ["a"] });
    const delineated = delineate(built, { q: ["b"], r: ["c"] });
    for (const structure of [built, delineated]) {
      expect([structure, structure.domain, structure.states, ...structure.states].every(Object.isFrozen)).toBe(true);
    }
    const copy = JSON.parse(JSON.stringify(built));
    expect(fringes(copy, ["a"])).toEqual(fringes(built, ["a"]));
    copy.states.pop();
    expect(() => fringes(copy, ["a"])).toThrow("A knowledge structure needs the state of the whole domain.");
    expect(() => surmiseStructure({ "": [] })).toThrow("needs a domain, a non-empty list of element ids");
  });

  it("rejects fringes of a set that is not a state, and events outside the structure", () => {
    const structure = surmiseStructure({ a: [], b: ["a"] });
    expect(() => fringes(structure, ["b"])).toThrow("The state [b] is not a state of the structure.");
    const model = kst({
      structure,
      parameters: { probabilities: [1 / 3, 1 / 3, 1 / 3], items: { q: { guess: 0, slip: 0 } } },
    });
    const event = { learner: "ana", item: "q", skills: ["b"], time: 0, outcome: 1 };
    expect(() => model.update(model.initialState("ana"), { ...event, skills: ["c"] })).toThrow(
      'The attempt at item "q" needs "c", which is not in the structure.',
    );
    expect(() => model.update(model.initialState("ana"), { ...event, skills: [] })).toThrow("has no skills");
    const known = model.update(model.initialState("ana"), event);
    expect(() => model.update(known, { ...event, outcome: 0 })).toThrow("has probability 0");
    expect(() => model.predict({ learner: "ana", probabilities: [1] }, event)).toThrow("3 numbers of at least 0");
    expect(() => model.fit([])).toThrow("A log needs at least one event.");
    expect(() => model.nextQuestion(model.initialState("ana"), [{ ...event, learner: "budi" }])).toThrow(
      'An attempt of learner "budi" was given the state of learner "ana".',
    );
    expect(() => model.nextQuestion(model.initialState("ana"), JSON.parse("3"))).toThrow(
      "The candidates must be an array of attempts, found 3.",
    );
  });

  it("rejects IITA on incomplete or non-binary answers and on items no one solves", () => {
    const answer = (learner: string, item: string, outcome: number, time: number) => ({
      learner,
      item,
      skills: [],
      time,
      outcome,
    });
    expect(() => iita(eventLog([answer("a", "x", 1, 0), answer("a", "y", 1, 1), answer("b", "x", 1, 2)]))).toThrow(
      'IITA needs every learner to answer every item; "b" has no answer to "y".',
    );
    expect(() => iita(eventLog([answer("a", "x", 0.5, 0), answer("a", "y", 1, 1)]))).toThrow("needs outcomes 0 or 1");
    expect(() => iita(eventLog([answer("a", "x", 0, 0), answer("a", "y", 1, 1)]))).toThrow(
      'Item "x" is solved by no learner, so IITA cannot weigh it.',
    );
    expect(() => iita(eventLog([answer("a", "x", 1, 0)]))).toThrow("IITA needs at least 2 items.");
  });
});

describe("KST options from JavaScript", () => {
  it("rejects options that are not an object", () => {
    const missing = JSON.parse("null");
    const structure = surmiseStructure({ a: [] });
    const model = kst({ structure });
    const log = eventLog([
      { learner: "ana", item: "x", skills: [], time: 0, outcome: 1 },
      { learner: "ana", item: "y", skills: [], time: 1, outcome: 0 },
    ]);
    for (const use of [
      () => kst(missing),
      () => surmiseStructure({ a: [] }, missing),
      () => iita(log, missing),
      () => model.nextQuestion(model.initialState("ana"), [], missing),
    ]) {
      expect(use).toThrow("The options must be an object, found null.");
    }
  });

  it("rejects structures and logs of other types", () => {
    const missing = JSON.parse("null");
    for (const use of [() => isKnowledgeSpace(missing), () => isLearningSpace(missing), () => fringes(missing, [])]) {
      expect(use).toThrow("A knowledge structure must be an object of domain and states, found null.");
    }
    expect(() => delineate(JSON.parse("[]"), {})).toThrow(
      "A knowledge structure must be an object of domain and states, found an array.",
    );
    expect(() => iita(missing)).toThrow("A log must be an array of events, found null.");
  });

  it("rejects parameters of null, which do not stand for the default", () => {
    const structure = surmiseStructure({ a: [] });
    expect(() => kst({ structure, parameters: JSON.parse("null") })).toThrow(
      "KST parameters must be an object, found null.",
    );
  });
});

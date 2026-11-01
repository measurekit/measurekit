import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  type Attempt,
  clauseStructure,
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
  isWellGraded,
  type KnowledgeStructure,
  type KstState,
  kst,
  learningPaths,
  neighbourhood,
  notions,
  splitByLearner,
  structureAtoms,
  structureBase,
  structureTrace,
  surmiseStructure,
  unionClosure,
} from "../src/index.js";
import assessmentFixture from "./fixtures/kst-assessment.json" with { type: "json" };
import blimFixture from "./fixtures/kst-blim.json" with { type: "json" };
import iitaFixture from "./fixtures/kst-iita.json" with { type: "json" };
import clauseFixture from "./fixtures/kst-structure-clauses.json" with { type: "json" };
import toolFixture from "./fixtures/kst-structure-tools.json" with { type: "json" };
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
type States = readonly (readonly string[])[];
interface ToolReference {
  readonly name: string;
  readonly structure: KnowledgeStructure;
  readonly knowledgeSpace: boolean;
  readonly base: States | null;
  readonly atoms: Readonly<Record<string, States>>;
  readonly wellGraded: boolean | null;
  readonly paths: States | null;
  readonly neighbourhoods: readonly {
    state: string[];
    distance1: States;
    distance2: States;
    distance3: States;
  }[];
  readonly traces: readonly { elements: string[]; structure: KnowledgeStructure }[];
  readonly notions: States;
}
interface ToolReferences {
  readonly structures: readonly ToolReference[];
  readonly closures: readonly { domain: string[]; generators: States; space: KnowledgeStructure }[];
}
interface ClauseReference {
  readonly name: string;
  readonly clauses: Record<string, string[][]>;
  readonly unreachable: string[];
  readonly structure: KnowledgeStructure | null;
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
const tools: ToolReferences = JSON.parse(JSON.stringify(toolFixture));
const clauseReferences: readonly ClauseReference[] = JSON.parse(JSON.stringify(clauseFixture));
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

/** A family of states as sorted keys, for comparisons that ignore the order of states and of elements. */
const family = (states: States): string[] => states.map((state) => [...state].sort().join(",")).sort();

describe("structure tools against kst", () => {
  it.each(tools.structures)("finds the base of a space and the atoms of each element as kst: $name", (reference) => {
    const { structure } = reference;
    if (reference.base === null) expect(() => structureBase(structure)).toThrow("not closed under union");
    else expect(family(structureBase(structure))).toEqual(family(reference.base));
    for (const element of structure.domain) {
      expect(family(structureAtoms(structure, element)), element).toEqual(family(defined(reference.atoms[element])));
    }
  });

  it.each(tools.structures)("finds the states near a state as kst's neighbourhoods: $name", (reference) => {
    for (const { state, distance1, distance2, distance3 } of reference.neighbourhoods) {
      const label = `[${state.join(", ")}]`;
      expect(family(neighbourhood(reference.structure, state)), label).toEqual(family(distance1));
      expect(family(neighbourhood(reference.structure, state, { distance: 1 })), label).toEqual(family(distance1));
      expect(family(neighbourhood(reference.structure, state, { distance: 2 })), label).toEqual(family(distance2));
      expect(family(neighbourhood(reference.structure, state, { distance: 3 })), label).toEqual(family(distance3));
    }
  });

  it.each(tools.structures)(
    "tells well-graded structures and lists their learning paths as kst: $name",
    (reference) => {
      if (reference.wellGraded === null || reference.paths === null) return;
      expect(isWellGraded(reference.structure)).toBe(reference.wellGraded);
      const paths = learningPaths(reference.structure).map((path) => path.join(","));
      expect(paths.sort()).toEqual(reference.paths.map((path) => path.join(",")).sort());
    },
  );

  it.each(tools.structures)("cuts a structure to some elements and groups its elements as kst: $name", (reference) => {
    for (const { elements, structure } of reference.traces) {
      const trace = structureTrace(reference.structure, elements);
      expect(trace.domain).toEqual(structure.domain);
      expect(family(trace.states)).toEqual(family(structure.states));
    }
    expect(family(notions(reference.structure))).toEqual(family(reference.notions));
  });

  it.each(tools.closures.map((closure, i) => ({ ...closure, label: `closure ${i + 1}` })))(
    "closes states under union as kst's kspace: $label",
    ({ domain, generators, space }) => {
      const closed = unionClosure(domain, generators);
      expect(closed.domain).toEqual(space.domain);
      expect(family(closed.states)).toEqual(family(space.states));
    },
  );
});

describe("clause structures against enumeration", () => {
  it.each(clauseReferences)(
    "builds the space of clauses by enumeration, or names what cannot be learned: $name",
    (reference) => {
      if (reference.structure === null) {
        expect(() => clauseStructure(reference.clauses)).toThrow(
          `${reference.unreachable.map((element) => `"${element}"`).join(", ")} cannot be learned`,
        );
        return;
      }
      const built = clauseStructure(reference.clauses);
      expect(built.domain).toEqual(reference.structure.domain);
      expect(family(built.states)).toEqual(family(reference.structure.states));
      expect(isLearningSpace(built)).toBe(true);
    },
  );
});

/** Prerequisites of five elements in which each element needs some of the ones before it: no two need each other. */
const acyclicPrerequisites = fc
  .tuple(...[0, 1, 2, 3, 4].map((j) => fc.subarray(["a", "b", "c", "d", "e"].slice(0, j))))
  .map((required) => Object.fromEntries(required.map((needs, j) => [String.fromCharCode(97 + j), needs])));

describe("structure tools", () => {
  const lattice = unionClosure(["a", "b", "c"], [["a"], ["b"], ["c"]]);

  it("builds a closure with the empty state and the whole domain, ordered by size and then by the domain's order", () => {
    expect(lattice.states).toEqual([[], ["a"], ["b"], ["c"], ["a", "b"], ["a", "c"], ["b", "c"], ["a", "b", "c"]]);
    expect(unionClosure(["a", "b", "c"], [["c"], ["a", "b"], ["c"]]).states).toEqual([
      [],
      ["c"],
      ["a", "b"],
      ["a", "b", "c"],
    ]);
    expect(unionClosure(["a", "b"], []).states).toEqual([[], ["a", "b"]]);
  });

  it("keeps the order of the domain and not the order the elements are given in", () => {
    const space = surmiseStructure({ x: [], y: ["x"], z: ["x"] });
    expect(structureTrace(space, ["z", "x"]).domain).toEqual(["x", "z"]);
    expect(structureBase(space).map((state) => state.join(""))).toEqual(["x", "xy", "xz"]);
    expect(notions(unionClosure(["p", "q", "r"], [["q", "p"], ["r"]]))).toEqual([["p", "q"], ["r"]]);
  });

  it("finds one atom per element in a quasi-ordinal space, the element with what it needs", () => {
    fc.assert(
      fc.property(acyclicPrerequisites, (prerequisites) => {
        const space = surmiseStructure(prerequisites);
        for (const element of space.domain) {
          const atoms = structureAtoms(space, element);
          expect(atoms).toHaveLength(1);
          expect(atoms[0]).toContain(element);
        }
      }),
    );
  });

  it("rebuilds a space from its base, and the closure of anything is a space whose base lies in the generators and the domain", () => {
    fc.assert(
      fc.property(acyclicPrerequisites, (prerequisites) => {
        const space = surmiseStructure(prerequisites);
        expect(keys(unionClosure(space.domain, structureBase(space)))).toEqual(keys(space));
      }),
    );
    fc.assert(
      fc.property(fc.array(fc.subarray(["a", "b", "c", "d", "e"]), { maxLength: 5 }), (generators) => {
        const closed = unionClosure(["a", "b", "c", "d", "e"], generators);
        expect(isKnowledgeSpace(closed)).toBe(true);
        expect(keys(unionClosure(closed.domain, closed.states))).toEqual(keys(closed));
        const given = new Set([...generators, closed.domain].map((state) => [...state].sort().join(",")));
        for (const state of structureBase(closed)) expect(given.has([...state].sort().join(","))).toBe(true);
      }),
    );
  });

  it("is symmetric in its neighbourhoods, and every state is within distance of the whole domain", () => {
    const structure = lattice;
    for (const state of structure.states) {
      for (const other of neighbourhood(structure, state)) {
        expect(neighbourhood(structure, other).map((s) => s.join(""))).toContain(state.join(""));
      }
      expect(neighbourhood(structure, state, { distance: structure.domain.length })).toHaveLength(
        structure.states.length - 1,
      );
    }
  });

  it("lists every order of independent elements, and one order of a chain", () => {
    const independent = unionClosure(
      ["a", "b", "c", "d"],
      ["a", "b", "c", "d"].map((element) => [element]),
    );
    const orders = learningPaths(independent).map((path) => path.join(""));
    expect(orders).toHaveLength(24);
    expect(new Set(orders).size).toBe(24);
    expect(orders[0]).toBe("abcd");
    expect(orders.at(-1)).toBe("dcba");
    expect(learningPaths(surmiseStructure({ a: [], b: ["a"], c: ["b"] }))).toEqual([["a", "b", "c"]]);
  });

  it("makes each path a way in which every prefix is a state, in a learning space of clauses", () => {
    const space = clauseStructure({ a: [], b: [], c: [["a"], ["b", "d"]], d: [["a"]] });
    const stateKeys = new Set(keys(space));
    const paths = learningPaths(space);
    expect(paths.length).toBeGreaterThan(0);
    for (const path of paths) {
      for (let length = 0; length <= path.length; length++) {
        expect(stateKeys.has(path.slice(0, length).sort().join(","))).toBe(true);
      }
    }
  });

  it("leaves out chains that gain more than one element, so a structure that is not well-graded may have no path", () => {
    const gap = unionClosure(["a", "b", "c"], [["a", "b"]]);
    expect(learningPaths(gap)).toEqual([]);
    expect(isWellGraded(gap)).toBe(false);
    expect(isWellGraded(lattice)).toBe(true);
    expect(isWellGraded({ domain: ["a"], states: [[], ["a"]] })).toBe(true);
    expect(isWellGraded({ domain: ["a", "b"], states: [[], ["a", "b"]] })).toBe(false);
  });

  it("calls every space of acyclic prerequisites well-graded and a learning space", () => {
    fc.assert(
      fc.property(acyclicPrerequisites, (prerequisites) => {
        const space = surmiseStructure(prerequisites);
        expect(isWellGraded(space)).toBe(true);
        expect(isLearningSpace(space)).toBe(true);
        expect(learningPaths(space).length).toBeGreaterThan(0);
      }),
    );
  });

  it("cuts a space to a space, and the whole domain leaves it as it is", () => {
    fc.assert(
      fc.property(
        acyclicPrerequisites,
        fc.subarray(["a", "b", "c", "d", "e"], { minLength: 1 }),
        (prerequisites, kept) => {
          const space = surmiseStructure(prerequisites);
          expect(isKnowledgeSpace(structureTrace(space, kept))).toBe(true);
          expect(structureTrace(space, [...space.domain])).toEqual(space);
        },
      ),
    );
  });

  it("splits the domain into notions", () => {
    fc.assert(
      fc.property(acyclicPrerequisites, (prerequisites) => {
        const space = surmiseStructure(prerequisites);
        const classes = notions(space);
        expect(classes.flat().sort()).toEqual([...space.domain].sort());
        for (const state of space.states) {
          for (const members of classes)
            expect(members.every((e) => state.includes(e)) || members.every((e) => !state.includes(e))).toBe(true);
        }
      }),
    );
  });

  it("builds the space of one clause for each element as the quasi-ordinal space of the prerequisites", () => {
    fc.assert(
      fc.property(acyclicPrerequisites, (prerequisites) => {
        const clauses = Object.fromEntries(
          Object.entries(prerequisites).map(([element, needs]) => [element, needs.length === 0 ? [] : [needs]]),
        );
        expect(clauseStructure(clauses)).toEqual(surmiseStructure(prerequisites));
      }),
    );
  });

  it("learns an element with an empty clause and with no clauses alike, and an element by any of its clauses", () => {
    expect(clauseStructure({ a: [[]], b: [] }).states).toEqual(clauseStructure({ a: [], b: [] }).states);
    expect(clauseStructure({ a: [], b: [], c: [["a"], ["b"]] }).states).toEqual([
      [],
      ["a"],
      ["b"],
      ["a", "b"],
      ["a", "c"],
      ["b", "c"],
      ["a", "b", "c"],
    ]);
  });

  it("names the elements that no order of clauses reaches, and the limit on states", () => {
    expect(() => clauseStructure({ a: [], b: [["c"]], c: [["b"]], d: [["a"]] })).toThrow(
      '"b", "c" cannot be learned: no clause can be completed from the empty state.',
    );
    expect(() => clauseStructure({ a: [["a"]] })).toThrow('"a" cannot be learned');
    const independent = Object.fromEntries(Array.from({ length: 14 }, (_, i) => [`e${i}`, []]));
    expect(() => clauseStructure(independent)).toThrow("The structure has more than 10000 knowledge states.");
    expect(clauseStructure(independent, { maxStates: 2 ** 14 }).states).toHaveLength(2 ** 14);
  });

  it("stops a closure at the limit on states, and suggests a split by unit", () => {
    const domain = Array.from({ length: 14 }, (_, i) => `e${i}`);
    const singles = domain.map((element) => [element]);
    expect(() => unionClosure(domain, singles)).toThrow("The structure has more than 10000 knowledge states.");
    expect(unionClosure(domain, singles, { maxStates: 2 ** 14 }).states).toHaveLength(2 ** 14);
  });

  it("stops at the limit on learning paths, which grow as a factorial", () => {
    const domain = Array.from({ length: 7 }, (_, i) => `e${i}`);
    const independent = unionClosure(
      domain,
      domain.map((element) => [element]),
    );
    expect(() => learningPaths(independent, { maxPaths: 5039 })).toThrow(
      "The structure has more than 5039 learning paths, which grow as a factorial.",
    );
    expect(learningPaths(independent, { maxPaths: 5040 })).toHaveLength(5040);
  });

  it("freezes the structures it builds, and leaves its arguments as they were", () => {
    const states = [["a"], ["b", "c"]];
    const snapshot = JSON.stringify(states);
    const closed = unionClosure(["a", "b", "c"], states);
    const trace = structureTrace(closed, ["a", "b"]);
    for (const structure of [closed, trace, clauseStructure({ a: [], b: [["a"]] })]) {
      expect([structure, structure.domain, structure.states, ...structure.states].every(Object.isFrozen)).toBe(true);
    }
    expect(JSON.stringify(states)).toBe(snapshot);
    const copy = JSON.parse(JSON.stringify(closed));
    expect(structureBase(copy)).toEqual(structureBase(closed));
  });
});

describe("structure tools, errors", () => {
  const space = surmiseStructure({ a: [], b: ["a"], c: [] });

  it("rejects a structure that is malformed, as every function of the module does", () => {
    const broken = { domain: ["a", "b"], states: [[], ["a"]] };
    const message = "A knowledge structure needs the state of the whole domain.";
    for (const use of [
      () => structureBase(broken),
      () => structureAtoms(broken, "a"),
      () => structureTrace(broken, ["a"]),
      () => notions(broken),
      () => neighbourhood(broken, []),
      () => learningPaths(broken),
      () => isWellGraded(broken),
    ]) {
      expect(use).toThrow(message);
    }
  });

  it("rejects a structure that is not a knowledge space where a base is asked", () => {
    expect(structureBase({ domain: ["a", "b"], states: [[], ["a"], ["b"], ["a", "b"]] })).toEqual([["a"], ["b"]]);
    expect(() =>
      structureBase({
        domain: ["a", "b", "c"],
        states: [[], ["a"], ["b"], ["a", "b", "c"]],
      }),
    ).toThrow("The base is defined for a knowledge space, and this structure is not closed under union.");
  });

  it("rejects elements outside the domain, repeats, and states that do not exist", () => {
    expect(() => structureAtoms(space, "z")).toThrow('The element "z" is not in the domain.');
    expect(() => structureTrace(space, ["a", "z"])).toThrow('The element "z" is not in the domain.');
    expect(() => structureTrace(space, ["a", "a"])).toThrow('The elements of a trace list "a" twice.');
    expect(() => structureTrace(space, [])).toThrow("A trace needs at least one element.");
    expect(() => neighbourhood(space, ["b"])).toThrow("The state [b] is not a state of the structure.");
    expect(() => unionClosure(["a", "b"], [["a", "z"]])).toThrow('State 1 has "z", which is not in the domain.');
    expect(() => unionClosure(["a", "b"], [["a", "a"]])).toThrow('State 1 lists "a" twice.');
    expect(() => unionClosure([], [])).toThrow("needs a domain, a non-empty list of element ids");
    expect(() => unionClosure(["a", "a"], [])).toThrow("lists an element twice");
    expect(() => clauseStructure({})).toThrow("Clauses need at least one element.");
    expect(() => clauseStructure({ a: [["z"]] })).toThrow(
      'A clause of "a" has "z", which is not a key of the clauses.',
    );
  });

  it("rejects options that are not what they should be", () => {
    for (const distance of [0, 1.5, JSON.parse('"2"'), JSON.parse("null")]) {
      expect(() => neighbourhood(space, [], { distance })).toThrow("The option distance must be an integer from 1");
    }
    for (const maxPaths of [0, 2.5, JSON.parse('"3"'), JSON.parse("null")]) {
      expect(() => learningPaths(space, { maxPaths })).toThrow("The option maxPaths must be an integer from 1");
    }
    for (const maxStates of [1, 2.5, JSON.parse('"3"'), JSON.parse("null")]) {
      expect(() => unionClosure(["a"], [], { maxStates })).toThrow("The option maxStates must be an integer from 2");
      expect(() => clauseStructure({ a: [] }, { maxStates })).toThrow("The option maxStates must be an integer from 2");
    }
  });

  it("rejects options and arguments of other types as JavaScript passes them", () => {
    const missing = JSON.parse("null");
    for (const use of [
      () => neighbourhood(space, [], missing),
      () => learningPaths(space, missing),
      () => unionClosure(["a"], [], missing),
      () => clauseStructure({ a: [] }, missing),
    ]) {
      expect(use).toThrow("The options must be an object, found null.");
    }
    for (const use of [
      () => structureBase(missing),
      () => structureAtoms(missing, "a"),
      () => structureTrace(missing, []),
      () => notions(missing),
      () => neighbourhood(missing, []),
      () => learningPaths(missing),
      () => isWellGraded(missing),
    ]) {
      expect(use).toThrow("A knowledge structure must be an object of domain and states, found null.");
    }
    expect(() => structureTrace(space, JSON.parse('"ab"'))).toThrow(
      'The elements of a trace must be a list of elements, found "ab".',
    );
    expect(() => neighbourhood(space, JSON.parse("{}"))).toThrow(
      "A knowledge state must be a list of elements, found an object.",
    );
    expect(() => unionClosure(["a"], JSON.parse("{}"))).toThrow(
      "The states must be a list of lists of elements, found an object.",
    );
    expect(() => unionClosure(["a"], JSON.parse('["a"]'))).toThrow('State 1 must be a list of elements, found "a".');
    expect(() => unionClosure(JSON.parse("null"), [])).toThrow("needs a domain, a non-empty list of element ids");
    expect(() => clauseStructure(JSON.parse("[]"))).toThrow(
      "Clauses must be an object of clause lists by element, found an array.",
    );
    expect(() => clauseStructure(JSON.parse('{"a": 3}'))).toThrow(
      'The clauses of "a" must be a list of lists of elements.',
    );
    expect(() => clauseStructure(JSON.parse('{"a": [3]}'))).toThrow('Each clause of "a" must be a list of elements.');
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

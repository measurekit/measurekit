import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  type Attempt,
  backwardGraded,
  clauseStructure,
  compareFits,
  compareModels,
  createRandom,
  crossValidate,
  delineate,
  diffTest,
  type Event,
  type EventLog,
  eventLog,
  forwardGraded,
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
  structureValidity,
  surmiseStructure,
  unionClosure,
} from "../src/index.js";
import { defined, expectAllClose, expectClose } from "./assertions.js";
import assessmentFixture from "./fixtures/kst-assessment.json" with { type: "json" };
import blimFixture from "./fixtures/kst-blim.json" with { type: "json" };
import constraintFixture from "./fixtures/kst-blim-constraints.json" with { type: "json" };
import diffTestFixture from "./fixtures/kst-diff-test.json" with { type: "json" };
import gradedFixture from "./fixtures/kst-graded.json" with { type: "json" };
import identifiabilityFixture from "./fixtures/kst-identifiability.json" with { type: "json" };
import iitaFixture from "./fixtures/kst-iita.json" with { type: "json" };
import clauseFixture from "./fixtures/kst-structure-clauses.json" with { type: "json" };
import toolFixture from "./fixtures/kst-structure-tools.json" with { type: "json" };
import structureFixture from "./fixtures/kst-structures.json" with { type: "json" };
import validityFixture from "./fixtures/kst-validity.json" with { type: "json" };
import { expectUnaffectedByLessons, online, withLessons } from "./logs.js";

/** Absolute tolerance of BLIM parameters fitted by EM against pks, which stops on a change of the parameters. */
const ML_TOLERANCE = 1e-6;
/** Absolute tolerance of the log-likelihood of a BLIM fitted by EM. */
const ML_LOG_LIKELIHOOD_TOLERANCE = 1e-8;
/** Absolute tolerance of quantities computed in one pass of the same arithmetic: MD estimates, the assessment. */
const EXACT_TOLERANCE = 1e-12;
/** Relative tolerance of IITA's diff against DAKS. */
const DIFF_TOLERANCE = 1e-10;
/** Absolute tolerance of G² and the log-likelihood of an MD fit, which sum 1,000 learners' logs. */
const MD_FIT_TOLERANCE = 1e-10;
/**
 * Absolute tolerance of G² and the log-likelihood of an MDML fit. Its estimates maximize the likelihood of the nearest
 * states only, so the whole model's log-likelihood at them changes at first order with their distance from pks's, 1e-6.
 */
const MDML_FIT_TOLERANCE = 1e-4;
/** Relative tolerance of the test of a quasi order against the delta method, which differentiates numerically. */
const DELTA_TOLERANCE = 1e-6;
/** How far a mean estimate of a seeded recovery may be from the rate that made the data. */
const RECOVERY_TOLERANCE = 0.01;

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
type Rates = Readonly<Record<string, number>>;
interface ConstraintReference {
  readonly name: string;
  readonly method: "ml" | "md" | "mdml";
  readonly structure: KnowledgeStructure;
  readonly patterns: Record<string, number>;
  readonly fixed: { readonly guess: Rates; readonly slip: Rates };
  readonly equal: { readonly guess: string[][]; readonly slip: string[][] };
  readonly probabilities: number[];
  readonly guess: Rates;
  readonly slip: Rates;
  readonly logLikelihood: number;
  readonly parameters: number;
  readonly g2: number;
  readonly degreesOfFreedom: number;
  readonly pValue: number | null;
  readonly discrepancy: number;
}
interface ConstraintReferences {
  readonly cases: readonly ConstraintReference[];
  readonly comparisons: readonly {
    name: string;
    restricted: ConstraintReference;
    general: ConstraintReference;
    g2: number;
    degreesOfFreedom: number;
    pValue: number;
  }[];
  readonly recovery: {
    structure: KnowledgeStructure;
    slip: number;
    guess: number;
    sets: { patterns: Record<string, number>; slip: number; guess: number; probabilities: number[] }[];
  };
}
interface IdentifiabilityReference {
  readonly name: string;
  readonly structure: KnowledgeStructure;
  readonly probabilities: number[];
  readonly guess: Rates;
  readonly slip: Rates;
  readonly fixed: { readonly guess: Rates; readonly slip: Rates };
  readonly parameters: number;
  readonly rank: number;
}
interface GradedReference {
  readonly name: string;
  readonly structure: KnowledgeStructure;
  readonly forward: Record<string, boolean>;
  readonly backward: Record<string, boolean>;
}
interface ValidityReference {
  readonly name: string;
  readonly structure: KnowledgeStructure;
  readonly patterns: Record<string, number>;
  readonly gamma: number | null;
  readonly concordant: number;
  readonly discordant: number;
  readonly violationalCoefficient: number | null;
  readonly discrepancyIndex: number;
  readonly distances: Record<string, number>;
  readonly distanceAgreement: number | null;
  readonly solutionRates: Record<string, number>;
}
interface DiffTestReference {
  readonly name: string;
  readonly items: string[];
  readonly patterns: Record<string, number>;
  readonly version: "minimized" | "corrected";
  readonly prerequisites: Prerequisites;
  readonly alternative: Prerequisites | null;
  readonly mean: number;
  readonly confidence: number;
  readonly estimate: number;
  readonly variance: number;
  readonly z: number;
  readonly pValue: Record<"two-sided" | "less" | "greater", number>;
  readonly interval: Record<"two-sided" | "less" | "greater", number | number[]>;
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
const constraints: ConstraintReferences = JSON.parse(JSON.stringify(constraintFixture));
const identifiabilityReferences: readonly IdentifiabilityReference[] = JSON.parse(
  JSON.stringify(identifiabilityFixture),
);
const gradedReferences: readonly GradedReference[] = JSON.parse(JSON.stringify(gradedFixture));
const validities: readonly ValidityReference[] = JSON.parse(JSON.stringify(validityFixture));
const diffTests: readonly DiffTestReference[] = JSON.parse(JSON.stringify(diffTestFixture));

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

/** Fits a case of the constraint fixture as pks did, with the same rates held or shared. */
function fitConstraint(reference: ConstraintReference) {
  const { structure, method, fixed, equal } = reference;
  return kst({ structure, over: "items", method, fixed, equal, tolerance: 1e-13, maxIterations: 1e6 }).fit(
    patternLog(structure.domain, reference.patterns),
  );
}

describe("BLIM with fixed and equal rates against pks", () => {
  it.each(constraints.cases)("fits $name by $method on $structure.domain.length items as pks's blim", (reference) => {
    const fit = fitConstraint(reference);
    const tolerance = reference.method === "md" ? EXACT_TOLERANCE : ML_TOLERANCE;
    expectAllClose(fit.model.parameters.probabilities, reference.probabilities, tolerance, "P(K)");
    for (const [item, guess] of Object.entries(reference.guess)) {
      const fitted = defined(fit.model.parameters.items[item]);
      expectClose(fitted.guess, guess, tolerance, `${item} guess`);
      expectClose(fitted.slip, defined(reference.slip[item]), tolerance, `${item} slip`);
    }
    expect(fit.converged).toBe(true);
    expectClose(fit.discrepancy, reference.discrepancy, EXACT_TOLERANCE, "discrepancy");
    const fitted = defined(fit.goodnessOfFit);
    expect(fitted.degreesOfFreedom).toBe(reference.degreesOfFreedom);
    const likelihoodTolerance = { md: MD_FIT_TOLERANCE, ml: ML_LOG_LIKELIHOOD_TOLERANCE, mdml: MDML_FIT_TOLERANCE }[
      reference.method
    ];
    expectClose(fit.logLikelihood, reference.logLikelihood, likelihoodTolerance, "log-likelihood");
    expectClose(fitted.g2, reference.g2, likelihoodTolerance, "G²");
    expectClose(fitted.pValue, reference.pValue, likelihoodTolerance, "p-value");
    if (reference.method === "ml") {
      expectClose(fit.aic, 2 * reference.parameters - 2 * reference.logLikelihood, ML_LOG_LIKELIHOOD_TOLERANCE, "AIC");
    } else {
      expect(fit.aic).toBeNull();
    }
  });

  it.each(constraints.cases)("holds fixed rates and shares equal rates exactly: $name by $method", (reference) => {
    const { items } = fitConstraint(reference).model.parameters;
    for (const [name, rates] of [
      ["guess", reference.fixed.guess],
      ["slip", reference.fixed.slip],
    ] as const) {
      for (const [item, rate] of Object.entries(rates))
        expect(defined(items[item])[name], `${item} ${name}`).toBe(rate);
    }
    for (const [name, groups] of [
      ["guess", reference.equal.guess],
      ["slip", reference.equal.slip],
    ] as const) {
      for (const group of groups) {
        expect(new Set(group.map((item) => defined(items[item])[name])).size, group.join()).toBe(1);
      }
    }
  });

  it.each(constraints.comparisons)("compares fits as pks's anova: $name", (reference) => {
    const comparison = defined(compareFits(fitConstraint(reference.restricted), fitConstraint(reference.general)));
    expect(comparison.degreesOfFreedom).toBe(reference.degreesOfFreedom);
    expectClose(comparison.g2, reference.g2, ML_LOG_LIKELIHOOD_TOLERANCE, "G²");
    expectClose(comparison.pValue, reference.pValue, ML_LOG_LIKELIHOOD_TOLERANCE, "p-value");
  });

  it("recovers the rates of a chain of 6 items from 20 seeded sets as pks does, one slip and one guess", () => {
    const { structure, slip, guess, sets } = constraints.recovery;
    const rates = sets.map((set) => {
      const fit = kst({
        structure,
        over: "items",
        equal: { guess: [structure.domain], slip: [structure.domain] },
        tolerance: 1e-13,
        maxIterations: 1e6,
      }).fit(patternLog(structure.domain, set.patterns));
      const first = defined(fit.model.parameters.items[defined(structure.domain[0])]);
      expectClose(first.slip, set.slip, ML_TOLERANCE, "slip");
      expectClose(first.guess, set.guess, ML_TOLERANCE, "guess");
      expectAllClose(fit.model.parameters.probabilities, set.probabilities, ML_TOLERANCE, "P(K)");
      return first;
    });
    const mean = (values: readonly number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
    expect(Math.abs(mean(rates.map((rate) => rate.slip)) - slip)).toBeLessThan(RECOVERY_TOLERANCE);
    expect(Math.abs(mean(rates.map((rate) => rate.guess)) - guess)).toBeLessThan(RECOVERY_TOLERANCE);
  });
});

describe("local identifiability against pks", () => {
  it.each(identifiabilityReferences)(
    "finds the rank of the Jacobian of the pattern probabilities: $name",
    (reference) => {
      const { structure } = reference;
      const model = kst({
        structure,
        over: "items",
        fixed: reference.fixed,
        parameters: {
          probabilities: reference.probabilities,
          items: Object.fromEntries(
            structure.domain.map((item) => [
              item,
              { guess: defined(reference.guess[item]), slip: defined(reference.slip[item]) },
            ]),
          ),
        },
      });
      expect(model.identifiability()).toEqual({
        parameters: reference.parameters,
        rank: reference.rank,
        identifiable: reference.rank === reference.parameters,
      });
    },
  );
});

describe("gradedness against pks", () => {
  it.each(gradedReferences)(
    "finds the elements at which a structure is forward and backward graded: $name",
    (reference) => {
      expect(forwardGraded(reference.structure)).toEqual(reference.forward);
      expect(backwardGraded(reference.structure)).toEqual(reference.backward);
    },
  );
});

describe("validity of a structure against kst", () => {
  it.each(validities)("measures gamma, the violational coefficient, discrepancy and agreement: $name", (reference) => {
    const validity = structureValidity(reference.structure, patternLog(reference.structure.domain, reference.patterns));
    expectClose(validity.gamma, reference.gamma, EXACT_TOLERANCE, "gamma");
    expect(validity.concordant).toBe(reference.concordant);
    expect(validity.discordant).toBe(reference.discordant);
    expectClose(validity.violationalCoefficient, reference.violationalCoefficient, EXACT_TOLERANCE, "VC");
    expectClose(validity.discrepancyIndex, reference.discrepancyIndex, EXACT_TOLERANCE, "DI");
    expectClose(validity.distanceAgreement, reference.distanceAgreement, EXACT_TOLERANCE, "DA");
    expect(validity.distances).toEqual(reference.distances);
    for (const [item, rate] of Object.entries(reference.solutionRates)) {
      expectClose(validity.solutionRates[item] ?? null, rate, EXACT_TOLERANCE, `${item} solution rate`);
    }
  });
});

describe("the test of a quasi order against the delta method", () => {
  const tails = ["two-sided", "less", "greater"] as const;
  it.each(diffTests)("tests $name by the $version version against a mean of $mean", (reference) => {
    const log = patternLog(reference.items, reference.patterns);
    for (const tail of tails) {
      const result = diffTest(log, reference.prerequisites, {
        ...(reference.alternative === null ? {} : { alternative: reference.alternative }),
        version: reference.version,
        mean: reference.mean,
        confidence: reference.confidence,
        tail,
      });
      const close = (actual: number, expected: number, label: string): void => {
        expect(Math.abs(actual - expected), `${tail} ${label}`).toBeLessThanOrEqual(
          DELTA_TOLERANCE * Math.abs(expected),
        );
      };
      close(result.estimate, reference.estimate, "estimate");
      close(result.z, reference.z, "z");
      close(result.pValue, reference.pValue[tail], "p-value");
      const expected = reference.interval[tail];
      const [low, high] = result.interval;
      if (tail === "two-sided") {
        close(low, defined(Array.isArray(expected) ? expected[0] : null), "lower end");
        close(high, defined(Array.isArray(expected) ? expected[1] : null), "upper end");
      } else if (tail === "less") {
        expect(low).toBe(Number.NEGATIVE_INFINITY);
        close(high, Number(expected), "upper end");
      } else {
        close(low, Number(expected), "lower end");
        expect(high).toBe(Number.POSITIVE_INFINITY);
      }
    }
  });

  it("has a standard error that matches the spread of the estimate over bootstrap samples", () => {
    const reference = defined(
      diffTests.find((candidate) => candidate.name === "pisa" && candidate.alternative === null),
    );
    const learners = Object.entries(reference.patterns).flatMap(([pattern, count]) =>
      Array.from({ length: count }, () => pattern),
    );
    const random = createRandom(20261029);
    const estimates = Array.from({ length: 400 }, () => {
      const resample = Array.from({ length: learners.length }, () =>
        defined(learners[Math.floor(random.uniform() * learners.length)]),
      );
      const counts: Record<string, number> = {};
      for (const pattern of resample) counts[pattern] = (counts[pattern] ?? 0) + 1;
      return diffTest(patternLog(reference.items, counts), reference.prerequisites, { version: reference.version })
        .estimate;
    });
    const mean = estimates.reduce((sum, value) => sum + value, 0) / estimates.length;
    const spread = Math.sqrt(estimates.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (estimates.length - 1));
    const standardError = Math.sqrt(reference.variance / learners.length);
    expect(Math.abs(spread / standardError - 1)).toBeLessThan(0.15);
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

describe("BLIM constraints and diagnostics", () => {
  const chain = surmiseStructure({ a: [], b: ["a"], c: ["b"] });
  const chainPatterns = { "000": 12, "100": 10, "110": 9, "111": 14, "010": 2, "101": 1 };
  const chainLog = patternLog(chain.domain, chainPatterns);

  it("is unaffected by events without an outcome when fitted by MDML or with constraints", () => {
    for (const options of [
      { method: "mdml" as const },
      { equal: { slip: [["a", "b", "c"]] }, fixed: { guess: { a: 0.1 } } },
    ]) {
      expectUnaffectedByLessons(kst({ structure: chain, over: "items", ...options }), chainLog);
    }
  });

  it("gives G² and the discrepancy only for a table of one answer to every item", () => {
    const model = kst({ structure: chain, over: "items" });
    const complete = model.fit(chainLog);
    expect(complete.goodnessOfFit).not.toBeNull();
    expect(complete.discrepancy).not.toBeNull();
    const answer = (learner: string, item: string, outcome: number, time: number) => ({
      learner,
      item,
      skills: [],
      time,
      outcome,
    });
    const events = [...chainLog, answer("l0", "a", 1, 1000)];
    for (const log of [
      eventLog(chainLog.slice(1)),
      eventLog(events),
      eventLog(chainLog.map((event, i) => ({ ...event, outcome: i === 0 ? 0.5 : event.outcome }))),
    ]) {
      const fit = model.fit(log);
      expect(fit.goodnessOfFit).toBeNull();
      expect(fit.discrepancy).toBeNull();
      expect(fit.logLikelihood).toBeLessThan(0);
    }
  });

  it("leaves the p-value out when the model has no degrees of freedom", () => {
    const fit = kst({ structure: chain, over: "items" }).fit(
      patternLog(chain.domain, { "000": 1, "111": 1, "100": 1 }),
    );
    const gof = defined(fit.goodnessOfFit);
    expect(gof.degreesOfFreedom).toBe(3 - (4 - 1 + 6));
    expect(gof.pValue).toBeNull();
    expect(gof.g2).toBeGreaterThanOrEqual(0);
  });

  it("counts a parameter for each state but one, and for each group of free rates", () => {
    const parameters = (options: object): number => {
      const fit = kst({ structure: chain, over: "items", ...options }).fit(chainLog);
      return (defined(fit.aic) + 2 * fit.logLikelihood) / 2;
    };
    expectClose(parameters({}), 3 + 6, 1e-12);
    expectClose(parameters({ fixed: { guess: { a: 0.1 }, slip: { c: 0.2 } } }), 3 + 4, 1e-12);
    expectClose(parameters({ equal: { guess: [["a", "b", "c"]], slip: [["a", "b"]] } }), 3 + 1 + 2, 1e-12);
    expectClose(parameters({ equal: { slip: [["a", "b"]] }, fixed: { slip: { c: 0.1 } } }), 3 + 1 + 3, 1e-12);
  });

  it("holds a fixed rate at 0 or 1, which an estimated rate cannot reach", () => {
    const fit = kst({ structure: chain, over: "items", fixed: { guess: { a: 0 }, slip: { c: 0 } } }).fit(chainLog);
    expect(defined(fit.model.parameters.items.a).guess).toBe(0);
    expect(defined(fit.model.parameters.items.c).slip).toBe(0);
    const free = kst({ structure: chain, over: "items" }).fit(chainLog);
    expect(
      Math.min(...Object.values(free.model.parameters.items).flatMap((item) => [item.guess, item.slip])),
    ).toBeGreaterThanOrEqual(1e-6);
  });

  it("holds a rate at 0 through an MDML fit, and rejects answers that no state gives without an error", () => {
    const fit = kst({ structure: chain, over: "items", method: "mdml", fixed: { slip: { a: 0, c: 0 } } }).fit(chainLog);
    expect(fit.converged).toBe(true);
    expect(defined(fit.model.parameters.items.a).slip).toBe(0);
    expect(defined(fit.model.parameters.items.c).slip).toBe(0);
    const none = { guess: { a: 0, b: 0, c: 0 }, slip: { a: 0, b: 0, c: 0 } };
    expect(() => kst({ structure: chain, over: "items", method: "mdml", fixed: none }).fit(chainLog)).toThrow(
      "probability 0 under the parameters",
    );
  });

  it("restricts MDML to the states of least discrepancy, so its estimates differ from those of ML", () => {
    const ml = kst({ structure: chain, over: "items" }).fit(chainLog);
    const mdml = kst({ structure: chain, over: "items", method: "mdml" }).fit(chainLog);
    expect(mdml.converged).toBe(true);
    expect(mdml.aic).toBeNull();
    expect(mdml.logLikelihood).toBeLessThanOrEqual(ml.logLikelihood + 1e-9);
    expect(mdml.model.parameters.probabilities).not.toEqual(ml.model.parameters.probabilities);
  });

  it("compares fits by their G², and leaves the comparison out when it has no meaning", () => {
    const free = kst({ structure: chain, over: "items" }).fit(chainLog);
    const equal = kst({ structure: chain, over: "items", equal: { guess: [["a", "b", "c"]] } }).fit(chainLog);
    const comparison = defined(compareFits(equal, free));
    expect(comparison.degreesOfFreedom).toBe(2);
    expectClose(comparison.g2, defined(equal.goodnessOfFit).g2 - defined(free.goodnessOfFit).g2, 1e-12);
    expect(comparison.g2).toBeGreaterThanOrEqual(0);
    expect(compareFits(free, free)).toBeNull();
    expect(compareFits(free, equal)).toBeNull();
    const incomplete = kst({ structure: chain, over: "items" }).fit(eventLog(chainLog.slice(1)));
    expect(compareFits(equal, incomplete)).toBeNull();
    expect(compareFits(incomplete, free)).toBeNull();
    const md = kst({ structure: chain, over: "items", method: "md" }).fit(chainLog);
    expect(compareFits(equal, md)).toBeNull();
  });

  it("finds a single item with both of its rates free not identifiable, by the rank of 1 in 3 parameters", () => {
    const single = surmiseStructure({ a: [] });
    const model = kst({
      structure: single,
      over: "items",
      parameters: { probabilities: [0.4, 0.6], items: { a: { guess: 0.1, slip: 0.2 } } },
    });
    expect(model.identifiability()).toEqual({ parameters: 3, rank: 1, identifiable: false });
  });

  it("finds a model identifiable once enough rates are fixed, and keeps the rank within the patterns", () => {
    const base = { structure: chain, over: "items" as const };
    const parameters = {
      probabilities: [0.1, 0.2, 0.3, 0.4],
      items: { a: { guess: 0.1, slip: 0.2 }, b: { guess: 0.2, slip: 0.1 }, c: { guess: 0.15, slip: 0.25 } },
    };
    const free = kst({ ...base, parameters }).identifiability();
    expect(free.rank).toBeLessThanOrEqual(2 ** 3 - 1);
    const fixed = kst({ ...base, parameters, fixed: { guess: { a: 0.1 }, slip: { c: 0.25 } } }).identifiability();
    expect(fixed.parameters).toBe(free.parameters - 2);
    expect(fixed.rank).toBeLessThanOrEqual(fixed.parameters);
    expect(fixed.identifiable).toBe(fixed.rank === fixed.parameters);
  });

  it("starts items without parameters at 0.1 when it checks identifiability", () => {
    const explicit = {
      probabilities: [0.25, 0.25, 0.25, 0.25],
      items: Object.fromEntries(chain.domain.map((item) => [item, { guess: 0.1, slip: 0.1 }])),
    };
    const given = kst({ structure: chain, over: "items", parameters: explicit }).identifiability();
    const absent = kst({ structure: chain, over: "items" }).identifiability();
    expect(absent).toEqual(given);
  });

  it("refuses identifiability for a model over skills, with equal rates, or too big for the limit", () => {
    expect(() => kst({ structure: chain }).identifiability()).toThrow('create the model with over: "items"');
    expect(() => kst({ structure: chain, over: "items", equal: { slip: [["a", "b"]] } }).identifiability()).toThrow(
      "with equal rates has no reference",
    );
    for (const length of [20, 40, 1100]) {
      const long = surmiseStructure(
        Object.fromEntries(Array.from({ length }, (_, i) => [`q${i}`, i === 0 ? [] : [`q${i - 1}`]])),
        { maxStates: 2000 },
      );
      expect(() => kst({ structure: long, over: "items" }).identifiability()).toThrow(
        `a Jacobian of 2 ^ ${length} patterns by ${length + 2 * length} parameters, more than the limit of 10000000 cells`,
      );
    }
    expect(() => kst({ structure: chain, over: "items", fixed: { slip: { z: 0.1 } } }).identifiability()).toThrow(
      'The option fixed.slip names item "z", which is not an element of the structure.',
    );
  });

  it("rejects rates and groups that are not what they should be", () => {
    const build = (options: object) => () => kst({ structure: chain, over: "items", ...options });
    expect(build({ fixed: { guess: { a: 1.5 } } })).toThrow(
      'The option fixed.guess needs a probability from 0 to 1 for item "a", found 1.5.',
    );
    expect(build({ equal: { slip: [["a"]] } })).toThrow(
      "Each group of the option equal.slip needs at least 2 distinct item ids",
    );
    expect(build({ equal: { slip: [["a", "a"]] } })).toThrow("needs at least 2 distinct item ids");
    expect(
      build({
        equal: {
          guess: [
            ["a", "b"],
            ["b", "c"],
          ],
        },
      }),
    ).toThrow('Item "b" is in two groups of the option equal.guess.');
    expect(build({ equal: { slip: [["a", "b"]] }, fixed: { slip: { b: 0.1 } } })).toThrow(
      'Item "b" has a fixed slip and is in a group of the option equal.slip.',
    );
    expect(build({ method: "em" })).toThrow('The option method must be "ml", "md" or "mdml", found "em".');
  });

  it("rejects fixed and equal rates of items that the log does not answer", () => {
    expect(() => kst({ structure: chain, over: "items", fixed: { guess: { z: 0.1 } } }).fit(chainLog)).toThrow(
      'The option fixed.guess names item "z", which has no event with an outcome in the log.',
    );
    expect(() => kst({ structure: chain, over: "items", equal: { slip: [["a", "z"]] } }).fit(chainLog)).toThrow(
      'The option equal.slip names item "z", which has no event with an outcome in the log.',
    );
  });

  it("rejects a fit that has no state able to give a learner's answers", () => {
    const model = kst({
      structure: chain,
      over: "items",
      method: "md",
      fixed: { slip: { a: 0, b: 0, c: 0 }, guess: { a: 0, b: 0, c: 0 } },
    });
    expect(() => model.fit(patternLog(chain.domain, { "010": 1, "101": 1 }))).toThrow(
      "A learner's outcomes have probability 0 under the parameters",
    );
  });
});

describe("gradedness", () => {
  it("is forward graded where adding an element to any state leaves a state, and backward graded where removing it does", () => {
    const structure = surmiseStructure({ a: [], b: ["a"] });
    expect(forwardGraded(structure)).toEqual({ a: true, b: false });
    expect(backwardGraded(structure)).toEqual({ a: false, b: true });
    const subsets = unionClosure(["a", "b"], [["a"], ["b"]]);
    expect(forwardGraded(subsets)).toEqual({ a: true, b: true });
    expect(backwardGraded(subsets)).toEqual({ a: true, b: true });
  });

  it("swaps the directions when every state is replaced by its complement", () => {
    const states = fc.uniqueArray(fc.subarray(["a", "b", "c", "d"]), {
      selector: (state) => state.join(),
      maxLength: 9,
    });
    fc.assert(
      fc.property(states, (chosen) => {
        const domain = ["a", "b", "c", "d"];
        const build = (family: readonly (readonly string[])[]): KnowledgeStructure => ({
          domain,
          states: [...family, [], domain].filter(
            (state, i, all) => all.findIndex((other) => other.join() === state.join()) === i,
          ),
        });
        const original = build(chosen);
        const complement = build(original.states.map((state) => domain.filter((element) => !state.includes(element))));
        expect(forwardGraded(complement)).toEqual(backwardGraded(original));
        expect(backwardGraded(complement)).toEqual(forwardGraded(original));
      }),
    );
  });
});

describe("validity of a structure", () => {
  const structure = surmiseStructure({ a: [], b: ["a"] });

  it("counts the pairs a structure orders, as a table whose answers can be added by hand", () => {
    const validity = structureValidity(structure, patternLog(["a", "b"], { "11": 1, "10": 1, "01": 1, "00": 1 }));
    expect(validity).toEqual({
      gamma: 0,
      concordant: 1,
      discordant: 1,
      violationalCoefficient: 0.25,
      discrepancyIndex: 0.25,
      distances: { "0": 3, "1": 1 },
      distanceAgreement: 1,
      solutionRates: { a: 0.5, b: 0.5 },
    });
  });

  it("calls a structure without ordered pairs valid in no measure that needs a pair", () => {
    const free = unionClosure(["a", "b"], [["a"], ["b"]]);
    const validity = structureValidity(free, patternLog(["a", "b"], { "11": 1, "10": 2, "01": 1 }));
    expect(validity.gamma).toBeNull();
    expect(validity.violationalCoefficient).toBeNull();
    expect(validity.concordant + validity.discordant).toBe(0);
    expect(validity.discrepancyIndex).toBe(0);
    expect(validity.distanceAgreement).toBeNull();
  });

  it("leaves the distance agreement out above 20 elements, and keeps the rest", () => {
    const long = surmiseStructure(
      Object.fromEntries(Array.from({ length: 21 }, (_, i) => [`q${i}`, i === 0 ? [] : [`q${i - 1}`]])),
    );
    const validity = structureValidity(
      long,
      patternLog(long.domain, { [`${"1".repeat(10)}${"0".repeat(11)}`]: 3, [`${"1".repeat(21)}`]: 1 }),
    );
    expect(validity.distanceAgreement).toBeNull();
    expect(validity.discrepancyIndex).toBe(0);
    expect(validity.gamma).toBe(1);
  });

  it("is invariant to the order of learners and bounded", () => {
    const patterns = fc.array(fc.array(fc.constantFrom(0, 1), { minLength: 3, maxLength: 3 }), {
      minLength: 2,
      maxLength: 25,
    });
    const chain3 = surmiseStructure({ a: [], b: ["a"], c: ["b"] });
    fc.assert(
      fc.property(patterns, fc.nat(), (rows, shift) => {
        const log = (order: readonly (readonly number[])[]) =>
          eventLog(
            order.flatMap((row, learner) =>
              row.map((outcome, i) => ({
                learner: `l${learner}`,
                item: defined(chain3.domain[i]),
                skills: [],
                time: learner * 3 + i,
                outcome,
              })),
            ),
          );
        const validity = structureValidity(chain3, log(rows));
        const turned = rows.slice(shift % rows.length).concat(rows.slice(0, shift % rows.length));
        expect(structureValidity(chain3, log(turned))).toEqual(validity);
        expect(validity.gamma === null || (validity.gamma >= -1 && validity.gamma <= 1)).toBe(true);
        expect(Object.values(validity.distances).reduce((sum, count) => sum + count, 0)).toBe(rows.length);
        const total = Object.entries(validity.distances).reduce(
          (sum, [distance, count]) => sum + Number(distance) * count,
          0,
        );
        expectClose(validity.discrepancyIndex, total / rows.length, 1e-12);
        for (const rate of Object.values(validity.solutionRates)) expect(rate >= 0 && rate <= 1).toBe(true);
      }),
    );
  });

  it("needs a complete table of 0 and 1 over the structure's elements, and names what is missing", () => {
    const answer = (learner: string, item: string, outcome: number, time: number) => ({
      learner,
      item,
      skills: [],
      time,
      outcome,
    });
    expect(() =>
      structureValidity(structure, eventLog([answer("x", "a", 1, 0), answer("x", "b", 0, 1), answer("y", "a", 1, 2)])),
    ).toThrow('The validity of a structure needs every learner to answer every item; "y" has no answer to "b".');
    expect(() => structureValidity(structure, eventLog([answer("x", "a", 0.5, 0), answer("x", "b", 1, 1)]))).toThrow(
      "The validity of a structure needs outcomes 0 or 1",
    );
    expect(() => structureValidity(structure, eventLog([answer("x", "a", 1, 0), answer("x", "z", 1, 1)]))).toThrow(
      'The log has item "z", which is not in the structure\'s domain.',
    );
    expect(() => structureValidity(structure, eventLog([answer("x", "a", 1, 0)]))).toThrow(
      'The structure\'s element "b" is not an item of the log',
    );
  });
});

describe("the test of a quasi order", () => {
  const pisa = defined(diffTests.find((candidate) => candidate.name === "pisa" && candidate.version === "minimized"));
  const log = patternLog(pisa.items, pisa.patterns);
  const learners = Object.values(pisa.patterns).reduce((sum, count) => sum + count, 0);

  it("estimates the fit measure of IITA over the squared number of learners", () => {
    for (const version of ["minimized", "corrected"] as const) {
      const result = iita(log, { version });
      const test = diffTest(log, result.prerequisites, { version });
      expectClose(test.estimate, result.diff / learners ** 2, 1e-15 * result.diff);
    }
  });

  it("tests a quasi order against another by the difference, so that swapping them negates the estimate and z", () => {
    const first = pisa.prerequisites;
    const second = defined(
      diffTests.find((candidate) => candidate.name === "pisa" && candidate.alternative !== null)?.alternative,
    );
    const forward = diffTest(log, first, { alternative: second });
    const backward = diffTest(log, second, { alternative: first });
    expectClose(forward.estimate, -backward.estimate, 1e-15);
    expectClose(forward.z, -backward.z, 1e-9);
    expectClose(forward.pValue, backward.pValue, 1e-12);
    expect(() => diffTest(log, first, { alternative: first })).toThrow("the two quasi orders are the same");
  });

  it("splits the probability between the tails, and shifts z with the mean", () => {
    const less = diffTest(log, pisa.prerequisites, { tail: "less" });
    const greater = diffTest(log, pisa.prerequisites, { tail: "greater" });
    const two = diffTest(log, pisa.prerequisites);
    expectClose(less.pValue + greater.pValue, 1, 1e-12);
    expectClose(two.pValue, 2 * Math.min(less.pValue, greater.pValue), 1e-12);
    expect(less.interval[0]).toBe(Number.NEGATIVE_INFINITY);
    expect(greater.interval[1]).toBe(Number.POSITIVE_INFINITY);
    expect(two.interval[0]).toBeLessThan(two.estimate);
    expect(two.interval[1]).toBeGreaterThan(two.estimate);
    const wide = diffTest(log, pisa.prerequisites, { confidence: 0.99 });
    expect(wide.interval[1] - wide.interval[0]).toBeGreaterThan(two.interval[1] - two.interval[0]);
    const shifted = diffTest(log, pisa.prerequisites, { mean: two.estimate });
    expectClose(shifted.z, 0, 1e-12);
    expectClose(shifted.pValue, 1, 1e-12);
  });

  it("closes the prerequisites under transitivity, as a quasi order is", () => {
    const direct = { a: [], b: ["a"], c: ["b"] };
    const closed = { a: [], b: ["a"], c: ["a", "b"] };
    const answers = patternLog(["a", "b", "c"], { "111": 5, "110": 4, "100": 4, "000": 3, "010": 1, "101": 1 });
    expect(diffTest(answers, direct)).toEqual(diffTest(answers, closed));
  });

  it("rejects what cannot be tested", () => {
    const names = ["a", "b", "c"];
    const answers = patternLog(names, { "111": 5, "110": 4, "100": 4, "000": 3, "010": 1 });
    const chain3 = { a: [], b: ["a"], c: ["b"] };
    expect(() => diffTest(patternLog(names, { "111": 5 }), chain3)).toThrow("variance 0");
    expect(() => diffTest(answers, { a: [], b: [], c: [] })).toThrow("need at least one implication between two items");
    expect(() => diffTest(answers, { a: [], b: ["a"] })).toThrow(
      'The prerequisites have no key for item "c" of the log.',
    );
    expect(() => diffTest(answers, { ...chain3, z: [] })).toThrow('have key "z", which is not an item of the log');
    expect(() => diffTest(answers, { a: [], b: ["z"], c: [] })).toThrow(
      '"b" requires "z", which is not a key of the prerequisites.',
    );
    expect(() => diffTest(answers, chain3, { alternative: { a: [], b: [] } })).toThrow(
      "The alternative prerequisites have no key for item",
    );
    expect(() => diffTest(patternLog(names, { "110": 3, "100": 1 }), chain3)).toThrow(
      'Item "c" is solved by no learner',
    );
    expect(() => diffTest(patternLog(["a"], { "1": 3 }), { a: [] })).toThrow("needs at least 2 items");
    const incomplete = eventLog([...answers.slice(1)]);
    expect(() => diffTest(incomplete, chain3)).toThrow("The difference test needs every learner to answer every item");
    expect(() => diffTest(answers, chain3, { tail: "both" as "less" })).toThrow(
      'The option tail must be "two-sided", "less" or "greater", found "both".',
    );
    expect(() => diffTest(answers, chain3, { version: "original" as "minimized" })).toThrow(
      'The option version must be "minimized" or "corrected", found "original".',
    );
    for (const confidence of [0, 1, 1.5]) {
      expect(() => diffTest(answers, chain3, { confidence })).toThrow(
        "The option confidence must be above 0 and below 1",
      );
    }
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

  it("rejects fixed and equal rates of other types, which do not stand for none", () => {
    const structure = surmiseStructure({ a: [], b: ["a"] });
    const build = (options: string) => () => kst({ structure, ...JSON.parse(options) });
    expect(build('{"fixed": null}')).toThrow(
      "The option fixed must be an object of fixed rates by guess and slip, found null.",
    );
    expect(build('{"equal": null}')).toThrow(
      "The option equal must be an object of groups of items by guess and slip, found null.",
    );
    expect(build('{"fixed": {"guess": null}}')).toThrow(
      "The option fixed.guess must be an object of rates by item, found null.",
    );
    expect(build('{"fixed": {"slip": {"a": "x"}}}')).toThrow(
      'The option fixed.slip needs a probability from 0 to 1 for item "a", found "x".',
    );
    expect(build('{"equal": {"slip": "ab"}}')).toThrow(
      'The option equal.slip must be a list of groups of items, found "ab".',
    );
    expect(build('{"equal": {"guess": [null]}}')).toThrow(
      "Each group of the option equal.guess needs at least 2 distinct item ids, found null.",
    );
    expect(build('{"method": null}')).toThrow("The option method must be");
  });

  it("rejects fits, relations and structures of other types in the diagnostics", () => {
    const structure = surmiseStructure({ a: [], b: ["a"] });
    const missing = JSON.parse("null");
    const log = patternLog(["a", "b"], { "11": 3, "10": 2, "00": 2, "01": 1 });
    const fit = kst({ structure, over: "items" }).fit(log);
    expect(() => compareFits(missing, fit)).toThrow("The restricted fit must be a fit of a KST model, found null.");
    expect(() => compareFits(fit, JSON.parse('{"goodnessOfFit": 3}'))).toThrow(
      "The general fit needs a goodnessOfFit, an object or null, found 3.",
    );
    expect(() => compareFits(fit, JSON.parse('{"goodnessOfFit": {"g2": "a", "degreesOfFreedom": 1}}'))).toThrow(
      "The general fit needs a goodnessOfFit of numbers, found an object.",
    );
    for (const use of [
      () => forwardGraded(missing),
      () => backwardGraded(missing),
      () => structureValidity(missing, log),
    ]) {
      expect(use).toThrow("A knowledge structure must be an object of domain and states, found null.");
    }
    expect(() => structureValidity(structure, missing)).toThrow("A log must be an array of events, found null.");
    const prerequisites = { a: [], b: ["a"] };
    expect(() => diffTest(missing, prerequisites)).toThrow("A log must be an array of events, found null.");
    expect(() => diffTest(log, missing)).toThrow(
      "The prerequisites must be an object of lists of items by item, found null.",
    );
    expect(() => diffTest(log, prerequisites, missing)).toThrow("The options must be an object, found null.");
    expect(() => diffTest(log, prerequisites, { alternative: missing })).toThrow(
      "The alternative prerequisites must be an object of lists of items by item, found null.",
    );
    expect(() => diffTest(log, prerequisites, JSON.parse('{"mean": "a"}'))).toThrow(
      'The option mean must be a finite number, found "a".',
    );
    expect(() => diffTest(log, prerequisites, JSON.parse('{"confidence": "0.9"}'))).toThrow(
      'The option confidence must be above 0 and below 1, found "0.9".',
    );
    expect(() => diffTest(log, prerequisites, JSON.parse('{"tail": null}'))).toThrow("The option tail must be");
  });

  it("rejects parameters of null, which do not stand for the default", () => {
    const structure = surmiseStructure({ a: [] });
    expect(() => kst({ structure, parameters: JSON.parse("null") })).toThrow(
      "KST parameters must be an object, found null.",
    );
  });
});

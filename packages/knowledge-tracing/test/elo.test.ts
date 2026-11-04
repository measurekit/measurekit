import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  buildEventLog,
  DataError,
  type EloItem,
  type EloOptions,
  type EloParameters,
  type EloState,
  type Event,
  type EventInput,
  type EventLog,
  elo,
} from "../src/index.js";
import { allPredicted, correlation, defined, expectAllClose, expectClose } from "./assertions.js";
import { readFixture } from "./fixtures.js";
import { expectUnaffectedByLessons } from "./logs.js";

interface EloRecoveryFixture {
  thresholds: { ability: number; difficulty: number };
  learners: number[];
  items: number[];
  outcomes: number[];
  trueAbility: number[];
  trueDifficulty: number[];
  ability: number[];
  difficulty: number[];
}
const recovery = readFixture<EloRecoveryFixture>("elo-recovery.json");

interface EloWorkedFixture {
  cases: {
    name: string;
    options: { uncertainty: { a: number; b: number }; skills: boolean; skillWeight?: number };
    parameters: { items: { q1?: { difficulty: number; count: number }; q3?: { difficulty: number; count: number } } };
    events: { learner: string; item: string; skills: string[]; time: number; outcome: number }[];
    predictions: number[];
    states: {
      ana: {
        learner: string;
        ability: number;
        count: number;
        skills: { fractions?: { ability: number; count: number }; decimals?: { ability: number; count: number } };
      };
      budi: {
        learner: string;
        ability: number;
        count: number;
        skills: { fractions?: { ability: number; count: number }; decimals?: { ability: number; count: number } };
      };
    };
    items: {
      q1: { difficulty: number; count: number };
      q2: { difficulty: number; count: number };
      q3: { difficulty: number; count: number };
    };
    logLikelihood: number;
  }[];
}
const worked = readFixture<EloWorkedFixture>("elo-worked.json");

/** Absolute tolerance of Elo estimates and predictions against the R replay. */
const ELO_TOLERANCE = 1e-12;

/** A worked log of `validation/knowledge-tracing/elo.R` with the R replay's results. */
interface WorkedCase {
  readonly name: string;
  readonly options: EloOptions;
  readonly parameters: EloParameters;
  readonly events: readonly EventInput[];
  readonly predictions: readonly number[];
  readonly states: Readonly<Record<string, EloState>>;
  readonly items: Readonly<Record<string, EloItem>>;
  readonly logLikelihood: number;
}

// Parsed rather than imported as typed: TypeScript merges the cases' item records into one type with optional keys.
const cases: readonly WorkedCase[] = JSON.parse(JSON.stringify(worked.cases));

/** Returns a deep copy of plain data. */
function copy<T>(value: T): T {
  return JSON.parse(JSON.stringify(value));
}

function expectState(actual: EloState | undefined, expected: EloState, label: string): void {
  const state = defined(actual);
  expect(state.learner, label).toBe(expected.learner);
  expect(state.count, label).toBe(expected.count);
  expectClose(state.ability, expected.ability, ELO_TOLERANCE, `${label} ability`);
  expect(Object.keys(state.skills), label).toEqual(Object.keys(expected.skills));
  for (const [skill, rating] of Object.entries(expected.skills)) {
    expect(state.skills[skill]?.count, `${label} ${skill}`).toBe(rating.count);
    expectClose(defined(state.skills[skill]).ability, rating.ability, ELO_TOLERANCE, `${label} ${skill}`);
  }
}

/** Item ratings and learner states, as an app stores them in rows. */
interface Rows {
  readonly items: Record<string, EloItem>;
  readonly states: Map<string, EloState>;
}

/**
 * Runs a log online as an app backed by a database does: for each event, a model built from the item's row predicts
 * it, then the learner's and the item's rows are updated.
 */
function online(
  options: EloOptions,
  log: EventLog,
  rows: Rows = { items: {}, states: new Map() },
): Rows & { predictions: number[] } {
  const items = { ...rows.items };
  const states = new Map(rows.states);
  const predictions = log.map((event) => {
    const row = items[event.item];
    const model = elo({ ...options, parameters: { items: row === undefined ? {} : { [event.item]: row } } });
    const state = states.get(event.learner) ?? model.initialState(event.learner);
    const prediction = model.predict(state, event);
    states.set(event.learner, model.update(state, event));
    items[event.item] = model.updateItem(state, event);
    return prediction;
  });
  return { items, states, predictions };
}

/** Freezes a value and everything it holds, so that a mutation throws. */
function deepFreeze<T>(value: T): T {
  if (typeof value === "object" && value !== null) {
    for (const child of Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

/** Events of one learner over three items and two skills, with partial credit. */
const arbitraryEvents: fc.Arbitrary<Event[]> = fc
  .array(
    fc.record({
      item: fc.constantFrom("q1", "q2", "q3"),
      skills: fc.subarray(["fractions", "decimals"]),
      outcome: fc.oneof(fc.constantFrom(0, 1), fc.double({ min: 0, max: 1, noNaN: true })),
    }),
    { minLength: 1, maxLength: 30 },
  )
  .map((records) => records.map((record, time) => ({ learner: "ana", time, ...record })));

const arbitraryOptions: fc.Arbitrary<EloOptions> = fc.oneof(
  fc.constant({}),
  fc.record({
    uncertainty: fc.record({
      a: fc.double({ min: 0.01, max: 4, noNaN: true }),
      b: fc.double({ min: 0, max: 1, noNaN: true }),
    }),
    skills: fc.constant(true),
    skillWeight: fc.double({ min: 0.1, max: 2, noNaN: true }),
  }),
);

describe("elo against R", () => {
  it.each(cases)("replays the worked log: $name", (reference) => {
    const fit = elo({ ...reference.options, parameters: reference.parameters }).fit(buildEventLog(reference.events));
    expectAllClose(allPredicted(fit.predictions), reference.predictions, ELO_TOLERANCE, "predictions");
    expect(Object.keys(fit.states)).toEqual(Object.keys(reference.states));
    for (const [learner, state] of Object.entries(reference.states)) expectState(fit.states[learner], state, learner);
    const { items } = fit.model.parameters;
    expect(Object.keys(items).toSorted()).toEqual(Object.keys(reference.items).toSorted());
    for (const [item, rating] of Object.entries(reference.items)) {
      expect(items[item]?.count, item).toBe(rating.count);
      expectClose(defined(items[item]).difficulty, rating.difficulty, ELO_TOLERANCE, item);
    }
    expectClose(fit.logLikelihood, reference.logLikelihood, ELO_TOLERANCE, "log-likelihood");
    expect(fit).toMatchObject({ aic: null, bic: null, converged: true, iterations: 1 });
  });

  it("makes the first update derived by hand: 0.5 up for the learner and down for the item", () => {
    const fit = elo().fit(buildEventLog([{ learner: "ana", item: "q1", time: 0, outcome: 1 }]));
    expect(fit.predictions).toEqual([0.5]);
    expect(fit.states.ana).toEqual({ learner: "ana", ability: 0.5, count: 1, skills: {} });
    expect(fit.model.parameters).toEqual({ items: { q1: { difficulty: -0.5, count: 1 } } });
  });

  it("recovers simulated abilities and difficulties, as the R replay does", () => {
    const log = buildEventLog(
      recovery.outcomes.map((outcome, time) => ({
        learner: `s${defined(recovery.learners[time]) + 1}`,
        item: `i${defined(recovery.items[time]) + 1}`,
        time,
        outcome,
      })),
    );
    const fit = elo().fit(log);
    const ability = recovery.trueAbility.map((_, s) => defined(fit.states[`s${s + 1}`]).ability);
    const difficulty = recovery.trueDifficulty.map(
      (_, i) => defined(fit.model.parameters.items[`i${i + 1}`]).difficulty,
    );
    expectAllClose(ability, recovery.ability, ELO_TOLERANCE, "ability");
    expectAllClose(difficulty, recovery.difficulty, ELO_TOLERANCE, "difficulty");
    expect(correlation(ability, recovery.trueAbility)).toBeGreaterThanOrEqual(recovery.thresholds.ability);
    expect(correlation(difficulty, recovery.trueDifficulty)).toBeGreaterThanOrEqual(recovery.thresholds.difficulty);
  });
});

describe("elo online", () => {
  it("predicts, updates learners and updates items online exactly as a replay does", () => {
    for (const reference of cases) {
      const log = buildEventLog(reference.events);
      const fit = elo({ ...reference.options, parameters: reference.parameters }).fit(log);
      const run = online(reference.options, log, { items: { ...reference.parameters.items }, states: new Map() });
      expect(run.predictions).toEqual(fit.predictions);
      expect(Object.fromEntries(run.states)).toEqual(fit.states);
      expect(run.items).toEqual(fit.model.parameters.items);
    }
  });

  it("predicts probabilities from 0 to 1, online exactly as in a replay", () => {
    fc.assert(
      fc.property(arbitraryOptions, arbitraryEvents, (options, events) => {
        const log = buildEventLog(events);
        const run = online(options, log);
        for (const prediction of run.predictions) expect(prediction >= 0 && prediction <= 1).toBe(true);
        const fit = elo(options).fit(log);
        expect(run.predictions).toEqual(fit.predictions);
        expect(run.items).toEqual(fit.model.parameters.items);
      }),
    );
  });

  it("never changes the state, the event or the model it is given", () => {
    fc.assert(
      fc.property(arbitraryOptions, arbitraryEvents, (options, events) => {
        const model = elo(options).fit(buildEventLog(events)).model;
        let state = model.initialState("ana");
        for (const event of events) {
          const frozen = deepFreeze(copy(state));
          const parameters = copy(model.parameters);
          const next = model.update(frozen, deepFreeze({ ...event }));
          model.updateItem(frozen, event);
          expect(frozen).toEqual(state);
          expect(model.parameters).toEqual(parameters);
          state = next;
        }
      }),
    );
  });

  it("continues identically from rows stored as JSON", () => {
    fc.assert(
      fc.property(arbitraryOptions, arbitraryEvents, fc.nat(), (options, events, cut) => {
        const log = buildEventLog(events);
        const split = cut % (log.length + 1);
        const direct = online(options, log);
        const first = online(options, log.slice(0, split));
        const stored = JSON.stringify({ items: first.items, states: Object.fromEntries(first.states) });
        const restored: { items: Record<string, EloItem>; states: Record<string, EloState> } = JSON.parse(stored);
        const resumed = online(options, log.slice(split), {
          items: restored.items,
          states: new Map(Object.entries(restored.states)),
        });
        expect(resumed.states).toEqual(direct.states);
        expect(resumed.items).toEqual(direct.items);
      }),
    );
  });

  it("starts every learner, item and skill at 0, so a new learner on a new item has a chance of 0.5", () => {
    const model = elo({ skills: true });
    const state = model.initialState("ana");
    expect(state).toEqual({ learner: "ana", ability: 0, count: 0, skills: {} });
    expect(model.predict(state, { learner: "ana", item: "new", skills: ["unseen"], time: 0 })).toBe(0.5);
  });

  it("fits a log of one learner", () => {
    const fit = elo().fit(
      buildEventLog([
        { learner: "ana", item: "q1", time: 0, outcome: 1 },
        { learner: "ana", item: "q1", time: 1, outcome: 1 },
      ]),
    );
    expect(Object.keys(fit.states)).toEqual(["ana"]);
    expect(fit.predictions[1]).toBeGreaterThan(0.5);
  });

  it("defines no mastery", () => {
    expect(elo().mastery).toBeUndefined();
  });
});

describe("elo input checks", () => {
  const event: Event = { learner: "ana", item: "q1", skills: [], time: 0, outcome: 1 };
  const parse = (json: string): EloState => JSON.parse(json);

  it("rejects an empty log and a log out of time order", () => {
    expect(() => elo().fit([])).toThrow("A log needs at least one event.");
    expect(() =>
      elo().fit([
        { ...event, time: 2 },
        { ...event, time: 1 },
      ]),
    ).toThrow("Event 2, at time 1, is earlier than event 1, at time 2; a log is in time order.");
  });

  it.each([
    ['{"uncertainty": {"a": 0, "b": 0.05}}', "The uncertainty function's a must be a positive number, found 0."],
    ['{"uncertainty": {"a": 1, "b": -1}}', "The uncertainty function's b must be a non-negative number, found -1."],
    ['{"skillWeight": 0.5}', "The option skillWeight needs skills: true."],
    ['{"skills": true, "skillWeight": 0}', "The option skillWeight must be a positive number, found 0."],
    ['{"skills": "yes"}', 'The option skills must be a boolean, found "yes".'],
    [
      '{"parameters": {"items": {"q1": {"difficulty": 0, "count": -1}}}}',
      'Item "q1" of the Elo parameters needs a count',
    ],
    ['{"parameters": {"items": []}}', "Elo parameters need items, an object of item ratings by item id"],
  ])("rejects the options %s", (json, message) => {
    const options: EloOptions = JSON.parse(json);
    expect(() => elo(options)).toThrow(DataError);
    expect(() => elo(options)).toThrow(message);
  });

  it("rejects malformed states, as from storage, and the state of another learner", () => {
    const model = elo();
    expect(() => model.update(parse('{"ability": 0, "count": 0, "skills": {}}'), event)).toThrow(
      "The Elo state needs a learner id, a non-empty string, found undefined.",
    );
    expect(() => model.update(parse('{"learner": "ana", "ability": null, "count": 0, "skills": {}}'), event)).toThrow(
      'The Elo state of learner "ana" needs an ability, a finite number, found null.',
    );
    expect(() => model.predict(parse('{"learner": "ana", "ability": 0, "count": 1.5, "skills": {}}'), event)).toThrow(
      "needs a count, a non-negative integer, found 1.5.",
    );
    expect(() =>
      model.update(parse('{"learner": "ana", "ability": 0, "count": 0, "skills": {"x": {"ability": 0}}}'), event),
    ).toThrow('The Elo state of learner "ana", skill "x", needs a count');
    expect(() => model.update(model.initialState("budi"), event)).toThrow(
      'An attempt of learner "ana" was given the state of learner "budi".',
    );
    expect(() => model.initialState("")).toThrow('A learner id must be a non-empty string, found "".');
  });

  it("rejects malformed events and attempts", () => {
    const model = elo();
    const state = model.initialState("ana");
    expect(() => model.update(state, { ...event, outcome: 2 })).toThrow(
      "The event needs an outcome from 0 to 1 or null, found 2.",
    );
    expect(() => model.predict(state, { ...event, skills: ["x", "x"] })).toThrow(
      'The attempt lists the skill "x" twice.',
    );
  });
});

describe("elo options from JavaScript", () => {
  it("rejects options that are not an object", () => {
    expect(() => elo(JSON.parse("null"))).toThrow("The options must be an object, found null.");
  });

  it("rejects parameters of null, which do not stand for the default", () => {
    expect(() => elo(JSON.parse('{"parameters": null}'))).toThrow("Elo parameters must be an object, found null.");
  });
});

describe("elo with events that have no outcome", () => {
  const lesson = { learner: "ana", item: "video", skills: ["fractions"], time: 9, outcome: null };

  it.each(cases)("is unaffected by them: $name", (reference) => {
    expectUnaffectedByLessons(elo(reference.options), buildEventLog(reference.events));
  });

  it("changes no rating: the state stays, and the item's rating is its own, or 0 for an item it does not know", () => {
    const model = elo({ skills: true, parameters: { items: { q1: { difficulty: 0.4, count: 3 } } } });
    const state = model.update(model.initialState("ana"), { ...lesson, item: "q1", outcome: 1 });
    expect(model.update(state, lesson)).toEqual(state);
    expect(model.update(state, lesson)).not.toBe(state);
    expect(model.updateItem(state, { ...lesson, item: "q1" })).toEqual({ difficulty: 0.4, count: 3 });
    expect(model.updateItem(state, lesson)).toEqual({ difficulty: 0, count: 0 });
  });

  it("writes no rating for them in a fit, and gives them no prediction", () => {
    const log = buildEventLog([
      { learner: "ana", item: "video", time: 1, outcome: null },
      { learner: "ana", item: "q1", time: 2, outcome: 1 },
    ]);
    const fit = elo().fit(log);
    expect(Object.keys(fit.model.parameters.items)).toEqual(["q1"]);
    expect(fit.predictions).toEqual([null, 0.5]);
    expect(defined(fit.states.ana).count).toBe(1);
  });

  it("refuses to fit a log without an event with an outcome", () => {
    expect(() => elo().fit(buildEventLog([{ learner: "ana", item: "video", time: 1, outcome: null }]))).toThrow(
      DataError,
    );
  });
});

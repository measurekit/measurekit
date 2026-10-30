import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  createRandom,
  DataError,
  type Event,
  type EventInput,
  type EventLog,
  eventLog,
  type Fold,
  MeasurekitError,
  type Random,
  splitByItem,
  splitByLearner,
  splitByTime,
} from "../src/index.js";
import holdout from "./fixtures/elo-holdout.json" with { type: "json" };
import { defined } from "./helpers.js";

const events: EventInput[] = [
  { learner: "ana", item: "q2", time: 3, outcome: 1 },
  { learner: "budi", item: "q1", skills: ["decimals"], time: 1, outcome: 0 },
  { learner: "ana", item: "q1", time: 1, outcome: 0.5 },
];
const qMatrix = { q1: ["fractions"], q2: ["fractions", "decimals"] };

/** An arbitrary log: up to 6 learners and 5 items, times with ties, in time order. */
const arbitraryLog: fc.Arbitrary<EventLog> = fc
  .array(
    fc.record({
      learner: fc.constantFrom("a", "b", "c", "d", "e", "f"),
      item: fc.constantFrom("q1", "q2", "q3", "q4", "q5"),
      time: fc.integer({ min: 0, max: 20 }),
      outcome: fc.constantFrom(0, 1),
    }),
    { minLength: 1, maxLength: 60 },
  )
  .map((records) => eventLog(records));

describe("eventLog", () => {
  it("orders events by time, keeping the given order of equal times, and takes skills from the Q-matrix", () => {
    expect(eventLog(events, { qMatrix })).toEqual([
      { learner: "budi", item: "q1", skills: ["decimals"], time: 1, outcome: 0 },
      { learner: "ana", item: "q1", skills: ["fractions"], time: 1, outcome: 0.5 },
      { learner: "ana", item: "q2", skills: ["fractions", "decimals"], time: 3, outcome: 1 },
    ]);
  });

  it("gives events without skills and without a Q-matrix no skills, and copies the events", () => {
    const skills = ["decimals"];
    const input = [{ learner: "ana", item: "q1", skills, time: 0, outcome: 1, hint: true }];
    const log = eventLog(input);
    skills.push("fractions");
    expect(log).toEqual([{ learner: "ana", item: "q1", skills: ["decimals"], time: 0, outcome: 1 }]);
    expect(eventLog([{ learner: "ana", item: "q1", time: 0, outcome: 1 }])[0]?.skills).toEqual([]);
  });

  it("returns a permutation of the events in time order", () => {
    fc.assert(
      fc.property(arbitraryLog, (log) => {
        const shuffled = [...log].reverse();
        const rebuilt = eventLog(shuffled);
        expect(rebuilt).toHaveLength(log.length);
        for (let i = 1; i < rebuilt.length; i++) {
          expect(defined(rebuilt[i]).time).toBeGreaterThanOrEqual(defined(rebuilt[i - 1]).time);
        }
        expect(new Set(rebuilt.map((event) => JSON.stringify(event)))).toEqual(
          new Set(log.map((event) => JSON.stringify(event))),
        );
      }),
    );
  });

  it.each([
    ["no events", [], "A log needs at least one event."],
    ["an event that is not an object", [null], "Event 1 must be an object, found null."],
    [
      "a missing learner",
      [{ item: "q1", time: 0, outcome: 1 }],
      "Event 1 needs a learner id, a non-empty string, found undefined.",
    ],
    [
      "an empty item id",
      [{ learner: "a", item: "", time: 0, outcome: 1 }],
      'Event 1 needs an item id, a non-empty string, found "".',
    ],
    [
      "skills that are not an array",
      [{ learner: "a", item: "q", skills: "x", time: 0, outcome: 1 }],
      'Event 1 needs skills, an array of skill ids, found "x".',
    ],
    [
      "skills of null, which do not stand for the Q-matrix's",
      [{ learner: "a", item: "q1", skills: null, time: 0, outcome: 1 }],
      "Event 1 needs skills, an array of skill ids, found null.",
    ],
    [
      "an empty skill id",
      [{ learner: "a", item: "q", skills: [""], time: 0, outcome: 1 }],
      'Event 1 needs skill ids, a non-empty string, found "".',
    ],
    [
      "a skill listed twice",
      [{ learner: "a", item: "q", skills: ["x", "x"], time: 0, outcome: 1 }],
      'Event 1 lists the skill "x" twice.',
    ],
    [
      "a time that is not finite",
      [{ learner: "a", item: "q", time: Number.NaN, outcome: 1 }],
      "Event 1 needs a time, a finite number, found NaN.",
    ],
    [
      "an outcome above 1",
      [{ learner: "a", item: "q", time: 0, outcome: 1.5 }],
      "Event 1 needs an outcome from 0 to 1, found 1.5.",
    ],
    [
      "an outcome that is a string",
      [{ learner: "a", item: "q", time: 0, outcome: "1" }],
      'Event 1 needs an outcome from 0 to 1, found "1".',
    ],
  ] satisfies [string, unknown[], string][])("rejects %s", (_, input, message) => {
    // Malformed events enter as untyped data, as from JSON.
    const build = (): EventLog => eventLog(input as EventInput[]);
    expect(build).toThrow(DataError);
    expect(build).toThrow(message);
  });

  it("rejects an event whose item the Q-matrix lacks, the unknown skills of an item", () => {
    expect(() => eventLog([{ learner: "a", item: "q9", time: 0, outcome: 1 }], { qMatrix })).toThrow(
      'Event 1 has no skills, and the Q-matrix lacks its item "q9".',
    );
  });

  it("rejects a malformed Q-matrix", () => {
    expect(() => eventLog(events, { qMatrix: { q1: ["x", "x"] } })).toThrow(
      'The Q-matrix row of item "q1" lists the skill "x" twice.',
    );
    expect(() => eventLog(events, { qMatrix: JSON.parse('[["s"]]') })).toThrow(
      "A Q-matrix must be an object of skill lists by item id, found an array.",
    );
  });
});

describe("splits", () => {
  const log = eventLog(holdout.events);

  /** Expects the folds to put every event in exactly one test log, and each fold's training log to be the rest. */
  function expectPartition(folds: readonly Fold[]): void {
    const tested = folds.flatMap((fold) => fold.test);
    expect(tested).toHaveLength(log.length);
    expect(new Set(tested)).toEqual(new Set(log));
    for (const fold of folds) {
      expect(fold.train).toEqual(log.filter((event) => !fold.test.includes(event)));
    }
  }

  it("splits by learner without a learner in both logs of a fold, in folds of equal numbers of learners", () => {
    const folds = splitByLearner(log, 4, createRandom(7));
    expectPartition(folds);
    const counts = folds.map((fold) => new Set(fold.test.map((event) => event.learner)).size);
    expect(counts.toSorted()).toEqual([1, 1, 2, 2]);
    for (const fold of folds) {
      const testLearners = new Set(fold.test.map((event) => event.learner));
      expect(fold.train.some((event) => testLearners.has(event.learner))).toBe(false);
    }
  });

  it("splits by item without an item in both logs of a fold", () => {
    const folds = splitByItem(log, 2, createRandom(7));
    expectPartition(folds);
    for (const fold of folds) {
      const testItems = new Set(fold.test.map((event) => event.item));
      expect(fold.train.some((event) => testItems.has(event.item))).toBe(false);
    }
  });

  it("partitions any log by learner and by item, and reproduces a split from the same seed", () => {
    fc.assert(
      fc.property(arbitraryLog, fc.integer(), (arbitrary, seed) => {
        for (const [split, key] of [
          [splitByLearner, (event: Event) => event.learner],
          [splitByItem, (event: Event) => event.item],
        ] as const) {
          fc.pre(new Set(arbitrary.map(key)).size >= 2);
          const folds = split(arbitrary, 2, createRandom(seed));
          expect(split(arbitrary, 2, createRandom(seed))).toEqual(folds);
          const tested = folds.flatMap((fold) => fold.test);
          expect(tested).toHaveLength(arbitrary.length);
          expect(new Set(tested)).toEqual(new Set(arbitrary));
          for (const fold of folds) {
            const testKeys = new Set(fold.test.map(key));
            expect(fold.train.some((event) => testKeys.has(key(event)))).toBe(false);
            expect(fold.train.length + fold.test.length).toBe(arbitrary.length);
          }
        }
      }),
    );
  });

  it("splits by time into the blocks of R's reference, moving boundaries past equal times", () => {
    const folds = splitByTime(log, 2);
    const [start, first, second, end] = holdout.timeBounds;
    expect(folds).toEqual([
      { train: log.slice(start, first), test: log.slice(first, second) },
      { train: log.slice(start, second), test: log.slice(second, end) },
    ]);
  });

  it("never trains on events later than those it tests", () => {
    fc.assert(
      fc.property(arbitraryLog, fc.integer({ min: 1, max: 3 }), (arbitrary, count) => {
        let folds: Fold[];
        try {
          folds = splitByTime(arbitrary, count);
        } catch (error) {
          expect(error).toBeInstanceOf(DataError);
          return;
        }
        for (const fold of folds) {
          const lastTrained = Math.max(...fold.train.map((event) => event.time));
          expect(Math.min(...fold.test.map((event) => event.time))).toBeGreaterThan(lastTrained);
        }
      }),
    );
  });

  it("rejects too many folds, a number of folds that is not an integer, and an unordered log", () => {
    const one = eventLog([{ learner: "a", item: "q", time: 0, outcome: 1 }]);
    expect(() => splitByLearner(one, 2, createRandom(1))).toThrow(
      "Splitting by learner into 2 folds needs at least 2 learners, found 1.",
    );
    expect(() => splitByItem(log, 5, createRandom(1))).toThrow("needs at least 5 items, found 4.");
    expect(() => splitByLearner(log, 1, createRandom(1))).toThrow(
      "The number of folds must be an integer of at least 2, found 1.",
    );
    expect(() => splitByTime(log, 1.5)).toThrow("The number of folds must be an integer of at least 1, found 1.5.");
    expect(() => splitByTime(log, JSON.parse('"2"'))).toThrow(
      'The number of folds must be an integer of at least 1, found "2".',
    );
    expect(() => splitByTime(one, 1)).toThrow("The log's times do not divide into 2 blocks for 1 folds");
    expect(() => splitByTime([...log].reverse(), 1)).toThrow(
      "Event 5, at time 15, is earlier than event 4, at time 16; a log is in time order.",
    );
    expect(() => splitByTime([], 1)).toThrow("A log needs at least one event.");
  });
});

describe("data input from JavaScript", () => {
  it("throws DataError, a MeasurekitError, named for its class", () => {
    const error = (() => {
      try {
        eventLog([]);
      } catch (thrown) {
        return thrown;
      }
    })();
    expect(error).toBeInstanceOf(DataError);
    expect(error).toBeInstanceOf(MeasurekitError);
    expect(error).toHaveProperty("name", "DataError");
  });

  it("rejects options and random number generators that are not objects", () => {
    const events = [{ learner: "ana", item: "q", skills: [], time: 0, outcome: 1 }];
    expect(() => eventLog(events, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => splitByLearner(eventLog(events), 2, JSON.parse("null"))).toThrow(
      "A random number generator must be one from createRandom, found null.",
    );
    // A generator from JavaScript, which TypeScript would reject: it has no normal method.
    const uniformOnly = { uniform: () => 0.5 } as unknown as Random;
    expect(() => splitByLearner(eventLog(events), 2, uniformOnly)).toThrow(
      "A random number generator must be one from createRandom, found an object.",
    );
  });

  it("rejects logs and fold counts of other types", () => {
    const missing = JSON.parse("null");
    for (const split of [splitByLearner, splitByItem]) {
      expect(() => split(missing, 2, createRandom(1))).toThrow("A log must be an array of events, found null.");
    }
    const log = eventLog([{ learner: "ana", item: "q", skills: [], time: 0, outcome: 1 }]);
    expect(() => splitByItem(log, JSON.parse('"2"'), createRandom(1))).toThrow(
      'The number of folds must be an integer of at least 2, found "2".',
    );
    expect(() => splitByTime(log, missing)).toThrow(
      "The number of folds must be an integer of at least 1, found null.",
    );
  });
});

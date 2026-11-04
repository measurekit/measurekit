import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  buildEventLog,
  createRandom,
  DataError,
  type Event,
  type EventInput,
  type EventLog,
  type FirstAttemptsOptions,
  type Fold,
  MeasurekitError,
  type Random,
  selectFirstAttempts,
  splitByItem,
  splitByLearner,
  splitByTime,
} from "../src/index.js";
import { defined } from "./assertions.js";
import { type EloHoldoutFixture, readFixture } from "./fixtures.js";

interface DataFirstAttemptsFixture {
  numbered: {
    learner: string;
    item: string;
    skills: never[];
    time: number;
    outcome: number | null;
    hints: number;
    attemptNumber: number;
  }[];
  unnumbered: { learner: string; item: string; skills: never[]; time: number; outcome: number | null; hints: number }[];
  expected: {
    numbered: {
      incorrect: { positions: number[]; outcomes: (number | null)[] };
      null: { positions: number[]; outcomes: (number | null)[] };
      keep: { positions: number[]; outcomes: (number | null)[] };
    };
    unnumbered: {
      incorrect: { positions: number[]; outcomes: (number | null)[] };
      null: { positions: number[]; outcomes: (number | null)[] };
      keep: { positions: number[]; outcomes: (number | null)[] };
    };
  };
}
const firstAttemptFixture = readFixture<DataFirstAttemptsFixture>("data-first-attempts.json");

// biome-ignore-start lint/style/useNamingConvention: the keys are those of the fixture file.
interface DataRowsFixture {
  assistments: {
    rows: { order_id: number; user_id: number; problem_id: number; skill_name: string; correct: number }[];
    sequences: {
      Addition: Record<string, number[]>;
      "Place value": Record<string, number[]>;
      Rounding: Record<string, number[]>;
      Subtraction: Record<string, number[]>;
    };
  };
  dataShop: {
    rows: {
      Row: number;
      "Anon Student Id": string;
      "Problem Name": string;
      "Step Name": string;
      "KC (Default)": string;
      "First Attempt": string;
    }[];
    sequences: {
      Addition: Record<string, number[]>;
      "Place value": Record<string, number[]>;
      Rounding: Record<string, number[]>;
      Subtraction: Record<string, number[]>;
    };
  };
}
// biome-ignore-end lint/style/useNamingConvention: the keys are those of the fixture file.
const rowFixture = readFixture<DataRowsFixture>("data-rows.json");

const holdout = readFixture<EloHoldoutFixture>("elo-holdout.json");

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
  .map((records) => buildEventLog(records));

describe("buildEventLog", () => {
  it("orders events by time, keeping the given order of equal times, and takes skills from the Q-matrix", () => {
    expect(buildEventLog(events, { qMatrix })).toEqual([
      { learner: "budi", item: "q1", skills: ["decimals"], time: 1, outcome: 0 },
      { learner: "ana", item: "q1", skills: ["fractions"], time: 1, outcome: 0.5 },
      { learner: "ana", item: "q2", skills: ["fractions", "decimals"], time: 3, outcome: 1 },
    ]);
  });

  it("gives events without skills and without a Q-matrix no skills, and copies the events", () => {
    const skills = ["decimals"];
    const input = [{ learner: "ana", item: "q1", skills, time: 0, outcome: 1, hint: true }];
    const log = buildEventLog(input);
    skills.push("fractions");
    expect(log).toEqual([{ learner: "ana", item: "q1", skills: ["decimals"], time: 0, outcome: 1 }]);
    expect(buildEventLog([{ learner: "ana", item: "q1", time: 0, outcome: 1 }])[0]?.skills).toEqual([]);
  });

  it("keeps an event without an outcome, as null, with its resource, covariates and attempt metadata", () => {
    const covariates = { load: 2 };
    const input = [
      {
        learner: "ana",
        item: "video",
        time: 0,
        outcome: null,
        resource: "lesson",
        covariates,
        hints: 1,
        duration: 30,
        attemptNumber: 1,
      },
      { learner: "ana", item: "q1", time: 1, outcome: 1, extra: true },
    ];
    const log = buildEventLog(input);
    covariates.load = 9;
    expect(log).toEqual([
      {
        learner: "ana",
        item: "video",
        skills: [],
        time: 0,
        outcome: null,
        resource: "lesson",
        covariates: { load: 2 },
        hints: 1,
        duration: 30,
        attemptNumber: 1,
      },
      { learner: "ana", item: "q1", skills: [], time: 1, outcome: 1 },
    ]);
    expect(Object.keys(log[1] ?? {})).toEqual(["learner", "item", "skills", "time", "outcome"]);
  });

  it("returns a permutation of the events in time order", () => {
    fc.assert(
      fc.property(arbitraryLog, (log) => {
        const shuffled = [...log].reverse();
        const rebuilt = buildEventLog(shuffled);
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
      "Event 1 needs an outcome from 0 to 1 or null, found 1.5.",
    ],
    [
      "an outcome that is a string",
      [{ learner: "a", item: "q", time: 0, outcome: "1" }],
      'Event 1 needs an outcome from 0 to 1 or null, found "1".',
    ],
    [
      "a missing outcome",
      [{ learner: "a", item: "q", time: 0 }],
      "Event 1 needs an outcome from 0 to 1 or null, found undefined.",
    ],
    [
      "fractional hints",
      [{ learner: "a", item: "q", time: 0, outcome: 1, hints: 1.5 }],
      "The hints of Event 1 must be an integer from 0, found 1.5.",
    ],
    [
      "negative hints",
      [{ learner: "a", item: "q", time: 0, outcome: 1, hints: -1 }],
      "The hints of Event 1 must be an integer from 0, found -1.",
    ],
    [
      "a duration of 0",
      [{ learner: "a", item: "q", time: 0, outcome: 1, duration: 0 }],
      "The duration of Event 1 must be a positive number, found 0.",
    ],
    [
      "attempt number 0",
      [{ learner: "a", item: "q", time: 0, outcome: 1, attemptNumber: 0 }],
      "The attempt number of Event 1 must be an integer from 1, found 0.",
    ],
    [
      "an empty resource",
      [{ learner: "a", item: "q", time: 0, outcome: 1, resource: "" }],
      'Event 1 needs a resource id, a non-empty string, found "".',
    ],
    [
      "covariates that are an array",
      [{ learner: "a", item: "q", time: 0, outcome: 1, covariates: [1] }],
      "Event 1 needs covariates, an object of numbers by name, found an array.",
    ],
    [
      "a covariate that is not finite",
      [{ learner: "a", item: "q", time: 0, outcome: 1, covariates: { load: Number.NaN } }],
      'The covariate "load" of Event 1 must be a finite number, found NaN.',
    ],
  ] satisfies [string, unknown[], string][])("rejects %s", (_, input, message) => {
    // Malformed events enter as untyped data, as from JSON.
    const build = (): EventLog => buildEventLog(input as EventInput[]);
    expect(build).toThrow(DataError);
    expect(build).toThrow(message);
  });

  it("rejects an event whose item the Q-matrix lacks, the unknown skills of an item", () => {
    expect(() => buildEventLog([{ learner: "a", item: "q9", time: 0, outcome: 1 }], { qMatrix })).toThrow(
      'Event 1 has no skills, and the Q-matrix lacks its item "q9".',
    );
  });

  it("rejects a malformed Q-matrix", () => {
    expect(() => buildEventLog(events, { qMatrix: { q1: ["x", "x"] } })).toThrow(
      'The Q-matrix row of item "q1" lists the skill "x" twice.',
    );
    expect(() => buildEventLog(events, { qMatrix: JSON.parse('[["s"]]') })).toThrow(
      "A Q-matrix must be an object of skill lists by item id, found an array.",
    );
  });
});

describe("selectFirstAttempts", () => {
  const numbered = buildEventLog(firstAttemptFixture.numbered);
  const unnumbered = buildEventLog(firstAttemptFixture.unnumbered);
  // buildEventLog orders by time, and the fixture's events are in time order.
  const reference = (
    log: EventLog,
    numberedLog: boolean,
    hinted: NonNullable<FirstAttemptsOptions["hinted"]>,
  ): EventLog => {
    const expected = firstAttemptFixture.expected[numberedLog ? "numbered" : "unnumbered"][hinted];
    return expected.positions.map((position, index) => ({
      ...defined(log[position]),
      outcome: expected.outcomes[index] ?? null,
    }));
  };

  it.each(["incorrect", "null", "keep"] as const)(
    "keeps the first attempts as base R does, with hinted %s",
    (hinted) => {
      expect(selectFirstAttempts(numbered, { hinted })).toEqual(reference(numbered, true, hinted));
      expect(selectFirstAttempts(unnumbered, { hinted })).toEqual(reference(unnumbered, false, hinted));
    },
  );

  it("counts a first attempt with hints as incorrect, as DataShop does, unless told otherwise", () => {
    const log = buildEventLog([
      { learner: "a", item: "q", time: 0, outcome: 1, hints: 2 },
      { learner: "a", item: "q", time: 1, outcome: 1 },
    ]);
    expect(selectFirstAttempts(log).map((event) => event.outcome)).toEqual([0]);
    expect(selectFirstAttempts(log, { hinted: "null" }).map((event) => event.outcome)).toEqual([null]);
    expect(selectFirstAttempts(log, { hinted: "keep" }).map((event) => event.outcome)).toEqual([1]);
  });

  it("keeps events without an outcome, and takes the first observed event when attempts are not numbered", () => {
    const log = buildEventLog([
      { learner: "a", item: "q", time: 0, outcome: null },
      { learner: "a", item: "q", time: 1, outcome: 0 },
      { learner: "a", item: "q", time: 2, outcome: 1 },
    ]);
    expect(selectFirstAttempts(log).map((event) => event.time)).toEqual([0, 1]);
  });

  it("tells learners and items apart whatever characters their ids hold", () => {
    const log = buildEventLog([
      { learner: "a\u0000b", item: "c", time: 0, outcome: 1 },
      { learner: "a", item: "b\u0000c", time: 1, outcome: 0 },
    ]);
    expect(selectFirstAttempts(log)).toEqual(log);
  });

  it("reads the attempt number the host records, and leaves the log it was given alone", () => {
    const log = buildEventLog([
      { learner: "a", item: "q", time: 0, outcome: 0, attemptNumber: 2 },
      { learner: "a", item: "q", time: 1, outcome: 1, attemptNumber: 1, hints: 1 },
    ]);
    const frozen = JSON.stringify(log);
    expect(selectFirstAttempts(log).map((event) => [event.time, event.outcome])).toEqual([[1, 0]]);
    expect(JSON.stringify(log)).toBe(frozen);
  });

  it("rejects an option that is not a choice and a malformed log", () => {
    const log = buildEventLog([{ learner: "a", item: "q", time: 0, outcome: 1 }]);
    expect(() => selectFirstAttempts(log, { hinted: "ignore" as "keep" })).toThrow(
      'The option hinted must be "incorrect", "null" or "keep", found "ignore".',
    );
    expect(() => selectFirstAttempts([])).toThrow("A log needs at least one event.");
    expect(() => selectFirstAttempts(log, JSON.parse("null"))).toThrow("The options must be an object");
  });
});

describe("splits", () => {
  const log = buildEventLog(holdout.events);

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

  describe("with events that have no outcome", () => {
    const withLessons = buildEventLog([
      ...["a", "b", "c", "d"].flatMap((learner, index) => [
        { learner, item: "video", time: index * 10, outcome: null },
        { learner, item: "q1", time: index * 10 + 1, outcome: 1 },
        { learner, item: "q2", time: index * 10 + 2, outcome: 0 },
        { learner, item: "q3", time: index * 10 + 3, outcome: 1 },
      ]),
      { learner: "e", item: "reading", time: 100, outcome: null },
    ]);
    const observed = (events: EventLog): number => events.filter((event) => event.outcome !== null).length;

    it("deals out the items that have an outcome, and puts every event without one in every training log", () => {
      const folds = splitByItem(withLessons, 3, createRandom(5));
      expect(folds.flatMap((fold) => fold.test)).toHaveLength(observed(withLessons));
      for (const fold of folds) {
        expect(fold.test.every((event) => event.outcome !== null)).toBe(true);
        expect(fold.train.filter((event) => event.outcome === null)).toHaveLength(
          withLessons.length - observed(withLessons),
        );
        const tested = new Set(fold.test.map((event) => event.item));
        expect(fold.train.filter((event) => event.outcome !== null && tested.has(event.item))).toEqual([]);
      }
    });

    it("counts the items with an outcome against the folds: a lesson is no item", () => {
      expect(() => splitByItem(withLessons, 4, createRandom(5))).toThrow(
        "Splitting by item into 4 folds needs at least 4 items, found 3.",
      );
    });

    it("keeps the events without an outcome of a learner with that learner", () => {
      const folds = splitByLearner(withLessons, 2, createRandom(5));
      expect(folds.flatMap((fold) => fold.test)).toHaveLength(withLessons.length);
      for (const fold of folds) {
        const testLearners = new Set(fold.test.map((event) => event.learner));
        expect(fold.train.some((event) => testLearners.has(event.learner))).toBe(false);
      }
    });

    it("divides the events with an outcome into blocks, and puts a lesson into the block that holds its time", () => {
      const log = buildEventLog([
        { learner: "a", item: "q", time: 1, outcome: 1 },
        { learner: "a", item: "q", time: 2, outcome: 0 },
        { learner: "a", item: "video", time: 3, outcome: null },
        { learner: "a", item: "q", time: 4, outcome: 1 },
        { learner: "b", item: "video", time: 4, outcome: null },
        { learner: "b", item: "q", time: 4, outcome: 0 },
        { learner: "b", item: "q", time: 5, outcome: 1 },
        { learner: "b", item: "video", time: 6, outcome: null },
      ]);
      const folds = splitByTime(log, 2);
      expect(folds.map((fold) => fold.test.map((event) => event.time))).toEqual([
        [2, 3, 4, 4, 4],
        [5, 6],
      ]);
      expect(folds.map((fold) => observed(fold.test))).toEqual([3, 1]);
      expect(folds.map((fold) => fold.train.length)).toEqual([1, 6]);
    });

    it("rejects times of events with an outcome that do not divide into blocks, however many events there are", () => {
      const log = buildEventLog([
        { learner: "a", item: "video", time: 1, outcome: null },
        { learner: "a", item: "video", time: 2, outcome: null },
        { learner: "a", item: "q", time: 3, outcome: 1 },
      ]);
      expect(() => splitByTime(log, 1)).toThrow("The log's times do not divide into 2 blocks for 1 folds");
    });
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
    const one = buildEventLog([{ learner: "a", item: "q", time: 0, outcome: 1 }]);
    expect(() => splitByLearner(one, 2, createRandom(1))).toThrow(
      "Splitting by learner into 2 folds needs at least 2 learners, found 1.",
    );
    expect(() => splitByItem(log, 5, createRandom(1))).toThrow("needs at least 5 items, found 4.");
    expect(() => splitByLearner(log, 1, createRandom(1))).toThrow(
      "The number of folds must be an integer from 2, found 1.",
    );
    expect(() => splitByTime(log, 1.5)).toThrow("The number of folds must be an integer from 1, found 1.5.");
    expect(() => splitByTime(log, JSON.parse('"2"'))).toThrow(
      'The number of folds must be an integer from 1, found "2".',
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
        buildEventLog([]);
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
    expect(() => buildEventLog(events, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => splitByLearner(buildEventLog(events), 2, JSON.parse("null"))).toThrow(
      "A random number generator must be one from createRandom, found null.",
    );
    // A generator from JavaScript, which TypeScript would reject: it has no normal method.
    const uniformOnly = { uniform: () => 0.5 } as unknown as Random;
    expect(() => splitByLearner(buildEventLog(events), 2, uniformOnly)).toThrow(
      "A random number generator must be one from createRandom, found an object.",
    );
  });

  it("rejects logs and fold counts of other types", () => {
    const missing = JSON.parse("null");
    for (const split of [splitByLearner, splitByItem]) {
      expect(() => split(missing, 2, createRandom(1))).toThrow("A log must be an array of events, found null.");
    }
    const log = buildEventLog([{ learner: "ana", item: "q", skills: [], time: 0, outcome: 1 }]);
    expect(() => splitByItem(log, JSON.parse('"2"'), createRandom(1))).toThrow(
      'The number of folds must be an integer from 2, found "2".',
    );
    expect(() => splitByTime(log, missing)).toThrow("The number of folds must be an integer from 1, found null.");
  });
});

describe("the recipe for the rows of an export", () => {
  /** An ASSISTments row: one for each skill of an attempt, with the attempt's `order_id`. */
  type AssistmentsRow = (typeof rowFixture.assistments.rows)[number];
  /** A DataShop row. */
  type DataShopRow = (typeof rowFixture.dataShop.rows)[number];

  const assistmentsLog = (rows: readonly AssistmentsRow[]): EventLog => {
    const attempts = new Map<
      number,
      { learner: string; item: string; skills: string[]; time: number; outcome: number }
    >();
    for (const row of rows) {
      const attempt = attempts.get(row.order_id) ?? {
        learner: String(row.user_id),
        item: String(row.problem_id),
        skills: [],
        time: row.order_id,
        outcome: row.correct,
      };
      attempt.skills.push(row.skill_name);
      attempts.set(row.order_id, attempt);
    }
    return buildEventLog([...attempts.values()]);
  };

  const dataShopLog = (rows: readonly DataShopRow[]): EventLog =>
    buildEventLog(
      rows.map((row) => ({
        learner: row["Anon Student Id"],
        item: `${row["Problem Name"]} ${row["Step Name"]}`,
        skills: row["KC (Default)"] === "" ? [] : row["KC (Default)"].split("~~"),
        time: row.Row,
        outcome: row["First Attempt"] === "correct" ? 1 : 0,
      })),
    );

  /** The outcomes of each learner for each skill, in the order of the log. */
  const sequencesBySkill = (log: EventLog): Record<string, Record<string, number[]>> => {
    const sequences: Record<string, Record<string, number[]>> = {};
    for (const event of log) {
      for (const skill of event.skills) {
        const learners = sequences[skill] ?? {};
        const outcomes = learners[event.learner] ?? [];
        outcomes.push(event.outcome ?? 0);
        learners[event.learner] = outcomes;
        sequences[skill] = learners;
      }
    }
    return sequences;
  };

  it("makes the sequences of outcomes that pyBKT's convert_data makes of ASSISTments rows", () => {
    expect(sequencesBySkill(assistmentsLog(rowFixture.assistments.rows))).toEqual(rowFixture.assistments.sequences);
  });

  it("makes the sequences that convert_data makes of DataShop rows with the skills of a step split", () => {
    const log = dataShopLog(rowFixture.dataShop.rows);
    expect(sequencesBySkill(log)).toEqual(rowFixture.dataShop.sequences);
    // The steps that no skill covers are attempts at no skill, and a hint or an error counts as incorrect.
    expect(log.filter((event) => event.skills.length === 0)).toHaveLength(3);
    expect(log.filter((event) => event.skills.length === 2).length).toBeGreaterThan(0);
  });
});

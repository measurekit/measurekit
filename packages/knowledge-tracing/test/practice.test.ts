import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  type Attempt,
  bkt,
  DataError,
  type Event,
  elo,
  eventLog,
  type LearnerModel,
  type MasteryTable,
  movingAverageMastery,
  opportunitiesToStreak,
  pfa,
  predictedMastery,
  recommend,
  streakMastery,
  thresholdMastery,
  windowMastery,
} from "../src/index.js";
import { readFixture } from "./fixtures.js";

interface PracticeMasteryFixture {
  events: { learner: string; item: string; skills: string[]; time: number; outcome: number | null }[];
  pfa: { intercepts: { s1: number; s4: number }; successes: number; failures: number };
  tables: {
    criterion: string;
    options: { length?: number; decay?: number; threshold?: number; correct?: number; window?: number };
    table: Record<string, { s1: number | null; s4: number | null }>;
  }[];
}
const fixture = readFixture<PracticeMasteryFixture>("practice-mastery.json");

/** A model written outside the package: a fixed prediction per item. */
function fixed(predictions: Record<string, number>): LearnerModel<string, null> {
  return {
    parameters: null,
    fit: () => {
      throw new Error("Not fitted.");
    },
    initialState: (learner) => learner,
    update: (state) => state,
    predict: (_, attempt) => predictions[attempt.item] ?? 0,
  };
}

describe("recommend", () => {
  const model = fixed({ a: 0.95, b: 0.7, c: 0.5, d: 0.75, e: 0.3, f: 0.8, g: 0.6 });
  const attempts: Attempt[] = ["a", "b", "c", "d", "e", "f", "g"].map((item) => ({
    learner: "ana",
    item,
    skills: [],
    time: 0,
  }));

  it("ranks the attempts inside the band by their distance to its middle, then the others by theirs to the band", () => {
    const ranked = recommend(model, "ana", attempts, { band: [0.6, 0.9] });
    expect(ranked.map((row) => row.attempt.item)).toEqual(["d", "b", "f", "g", "a", "c", "e"]);
    expect(ranked[0]?.prediction).toBe(0.75);
  });

  it("aims at 0.65 to 0.85 by default, and returns nothing for no candidates", () => {
    expect(recommend(model, "ana", attempts)[0]?.attempt.item).toBe("d");
    expect(recommend(model, "ana", [])).toEqual([]);
  });

  it("rejects a band outside 0 to 1, upside down or not of numbers", () => {
    expect(() => recommend(model, "ana", attempts, { band: [0.9, 0.6] })).toThrow(
      "The band must be [lower, upper] with 0 ≤ lower ≤ upper ≤ 1, found [0.9,0.6].",
    );
    expect(() => recommend(model, "ana", attempts, { band: [-0.1, 0.5] })).toThrow(DataError);
    expect(() => recommend(model, "ana", attempts, JSON.parse('{"band": ["0.6", "0.9"]}'))).toThrow(
      'The band must be [lower, upper] with 0 ≤ lower ≤ upper ≤ 1, found ["0.6","0.9"].',
    );
  });

  it("rejects candidates that are not attempts and predictions that are not probabilities", () => {
    expect(() => recommend(model, "ana", JSON.parse('"a"'))).toThrow(
      'The candidates must be an array of attempts, found "a".',
    );
    expect(() => recommend(fixed({ a: Number.NaN }), "ana", attempts)).toThrow(
      "The model's prediction of candidate 1 must be a number from 0 to 1, found NaN.",
    );
  });
});

describe("mastery criteria", () => {
  const outcomes = (learner: string, skills: string[], values: number[], start = 0) =>
    values.map((outcome, i) => ({ learner, item: "q", skills, time: start + i, outcome }));

  it("finds the first run of correct answers, counting each skill of an event and only outcomes of 1", () => {
    const log = eventLog([
      ...outcomes("ana", ["s"], [1, 1, 0, 1, 1, 1]),
      ...outcomes("ana", ["t", "s"], [1, 1, 1], 10),
      ...outcomes("budi", ["s"], [1, 0.9, 1, 1]),
    ]);
    expect(streakMastery(log)).toEqual({ ana: { s: 6, t: 3 }, budi: { s: null } });
    expect(streakMastery(log, { length: 2 })).toEqual({ ana: { s: 2, t: 2 }, budi: { s: 4 } });
  });

  it("finds the first attempt after which the model's mastery reaches the threshold", () => {
    // P(L) after a correct answer: 0.5 → 0.45 / 0.55 = 0.8182, then learning: 0.8545; after another: 0.9636,
    // then 0.9708.
    const model = bkt({
      parameters: { skills: { s: { prior: 0.5, learn: 0.2, forget: 0, guess: 0.2, slip: 0.1 } } },
    });
    const log = eventLog(outcomes("ana", ["s"], [1, 1]));
    expect(thresholdMastery(model, log)).toEqual({ ana: { s: 2 } });
    expect(thresholdMastery(model, log, { threshold: 0.85 })).toEqual({ ana: { s: 1 } });
    expect(thresholdMastery(model, log, { threshold: 0.98 })).toEqual({ ana: { s: null } });
    expect(() => thresholdMastery(model, log, { threshold: 1.5 })).toThrow(
      "The threshold must be a number from 0 to 1, found 1.5.",
    );
  });

  it("expects 14 attempts for three correct in a row at even odds, and satisfies the recurrence of the run", () => {
    expect(opportunitiesToStreak(0.5)).toBe(14);
    expect(opportunitiesToStreak(0.5, { run: 2 })).toBe(8);
    expect(opportunitiesToStreak(1, { length: 5, run: 1 })).toBe(4);
    expect(opportunitiesToStreak(0)).toBe(Number.POSITIVE_INFINITY);
    expect(opportunitiesToStreak(0, { run: 3 })).toBe(0);
    // 2²⁰⁰⁰ overflows: the expected attempts are Infinity, not Infinity - Infinity.
    expect(opportunitiesToStreak(0.5, { length: 2000, run: 1999 })).toBe(Number.POSITIVE_INFINITY);
    expect(opportunitiesToStreak(0.5, { length: 2000, run: 2000 })).toBe(0);
    fc.assert(
      fc.property(
        fc.double({ min: 0.05, max: 0.99, noNaN: true }),
        fc.integer({ min: 1, max: 8 }),
        fc.nat(),
        (p, length, r) => {
          const run = r % length;
          // E_r = 1 + p E_(r+1) + (1 - p) E_0.
          const expected = (k: number) => opportunitiesToStreak(p, { length, run: k });
          const recurrence = 1 + p * expected(run + 1) + (1 - p) * expected(0);
          expect(Math.abs(expected(run) - recurrence)).toBeLessThanOrEqual(1e-9 * recurrence);
        },
      ),
    );
  });

  it("rejects an empty log, invalid lengths, runs and probabilities", () => {
    expect(() => streakMastery([])).toThrow("A log needs at least one event.");
    expect(() => streakMastery(eventLog(outcomes("ana", ["s"], [1])), { length: 0 })).toThrow(
      "The length must be a positive integer, found 0.",
    );
    expect(() => opportunitiesToStreak(0.5, { run: 4 })).toThrow("The run must be an integer from 0 to the length 3");
    expect(() => opportunitiesToStreak(Number.NaN)).toThrow("The probability must be a number from 0 to 1, found NaN.");
    // A model from JavaScript, which TypeScript would reject: it defines no mastery.
    const withoutMastery = elo() as unknown as Parameters<typeof thresholdMastery>[0];
    for (const model of [withoutMastery, JSON.parse("null")]) {
      expect(() => thresholdMastery(model, eventLog(outcomes("ana", ["s"], [1])))).toThrow(
        "thresholdMastery needs a model that defines mastery, as BKT and KST do, found",
      );
    }
  });
});

describe("practice options from JavaScript", () => {
  it("rejects options that are not an object", () => {
    const missing = JSON.parse("null");
    const model = bkt({ parameters: { skills: { s: { prior: 0.3, learn: 0.2, forget: 0, guess: 0.2, slip: 0.1 } } } });
    const log = eventLog([{ learner: "ana", item: "q", skills: ["s"], time: 0, outcome: 1 }]);
    for (const use of [
      () => recommend(model, model.initialState("ana"), [], missing),
      () => streakMastery(log, missing),
      () => thresholdMastery(model, log, missing),
      () => opportunitiesToStreak(0.5, missing),
    ]) {
      expect(use).toThrow("The options must be an object, found null.");
    }
  });

  it("rejects a log of null", () => {
    expect(() => streakMastery(JSON.parse("null"))).toThrow("A log must be an array of events, found null.");
  });

  it("rejects a model that is not a learner model", () => {
    expect(() => recommend(JSON.parse('{"predict": 1}'), null, [])).toThrow(
      "A recommendation needs a learner model, such as bkt(), found an object.",
    );
  });

  it("rejects a band of null, which does not stand for the default", () => {
    const model = elo();
    expect(() => recommend(model, model.initialState("ana"), [], JSON.parse('{"band": null}'))).toThrow(
      "The band must be [lower, upper] with 0 ≤ lower ≤ upper ≤ 1, found null.",
    );
  });
});

describe("mastery criteria with events that have no outcome", () => {
  const answer = (learner: string, time: number, outcome: number | null, skills = ["s"]) => ({
    learner,
    item: outcome === null ? "video" : "q",
    skills,
    time,
    outcome,
  });

  it("leaves a streak as it is for an event without an outcome, and records no mastery from lessons alone", () => {
    const log = eventLog([
      answer("ana", 1, 1),
      answer("ana", 2, null),
      answer("ana", 3, 1),
      answer("ana", 4, 1),
      answer("budi", 1, null),
      answer("budi", 2, null, ["t"]),
    ]);
    expect(streakMastery(log)).toEqual({ ana: { s: 3 } });
  });

  it("updates the model with lessons, and tests the threshold only after events with an outcome", () => {
    const model = bkt({ parameters: { skills: { s: { prior: 0.1, learn: 0.6, forget: 0, guess: 0.2, slip: 0.1 } } } });
    const log = eventLog([
      answer("ana", 1, null),
      answer("ana", 2, null),
      answer("ana", 3, null),
      answer("ana", 4, 1),
      answer("budi", 1, 0),
      answer("budi", 2, 0),
    ]);
    // Three lessons take P(L) from 0.1 past 0.95 at ana's first attempt: opportunity 1, as the lessons are no opportunity.
    expect(thresholdMastery(model, log)).toEqual({ ana: { s: 1 }, budi: { s: null } });
    expect(thresholdMastery(model, eventLog([answer("ana", 1, null)]))).toEqual({});
  });
});

/** A criterion of `validation/knowledge-tracing/practice.R`: its options and its table, replayed in base R. */
interface MasteryReference {
  readonly criterion: "streak" | "movingAverage" | "window" | "predicted";
  readonly options: Record<string, number>;
  readonly table: MasteryTable;
}
const references: readonly MasteryReference[] = JSON.parse(JSON.stringify(fixture.tables));
const masteryLog = eventLog(JSON.parse(JSON.stringify(fixture.events)));

/** The performance factors analysis of the fixture, with the parameters written out in the script. */
const performance = pfa({
  parameters: {
    skills: Object.fromEntries(
      Object.entries(fixture.pfa.intercepts).map(([skill, intercept]) => [
        skill,
        { intercept, coefficients: { successes: fixture.pfa.successes, failures: fixture.pfa.failures } },
      ]),
    ),
  },
});

/** The table that a criterion gives for the log of the fixture. */
function tableOf({ criterion, options }: MasteryReference): MasteryTable {
  const at = (name: string): number => options[name] ?? Number.NaN;
  switch (criterion) {
    case "streak":
      return streakMastery(masteryLog, { length: at("length") });
    case "movingAverage":
      return movingAverageMastery(masteryLog, { decay: at("decay"), threshold: at("threshold") });
    case "window":
      return windowMastery(masteryLog, { correct: at("correct"), window: at("window") });
    default:
      return predictedMastery(performance, masteryLog, { threshold: at("threshold") });
  }
}

describe("mastery criteria against base R", () => {
  it.each(references)("replays the $criterion criterion of a seeded log as base R does: $options", (reference) => {
    expect(tableOf(reference)).toEqual(reference.table);
  });

  it("meets each criterion for some learners and skills and not for others, so that the comparison tests both", () => {
    for (const reference of references.filter(({ criterion }) => criterion !== "streak")) {
      const cells = Object.values(reference.table).flatMap((skills) => Object.values(skills));
      const met = cells.filter((opportunity) => opportunity !== null).length;
      expect(met, JSON.stringify(reference.options)).toBeGreaterThan(0);
      if (reference.options.window !== 1 && reference.options.threshold !== 0.8) expect(met).toBeLessThan(cells.length);
    }
  });
});

describe("mastery criteria of the moving average, the window and the prediction", () => {
  const answers = (values: (number | null)[], skills = ["s"], learner = "ana"): Event[] =>
    values.map((outcome, i) => ({ learner, item: outcome === null ? "video" : "q", skills, time: i, outcome }));

  it("moves the average of a skill to d m + (1 - d) y from 0, and counts partial credit as it is", () => {
    // 0.5, 0.75, 0.875; and for outcomes of 0.5, 0.25, 0.375, 0.4375.
    const log = eventLog(answers([1, 1, 1]));
    expect(movingAverageMastery(log, { decay: 0.5, threshold: 0.5 })).toEqual({ ana: { s: 1 } });
    expect(movingAverageMastery(log, { decay: 0.5, threshold: 0.75 })).toEqual({ ana: { s: 2 } });
    expect(movingAverageMastery(log, { decay: 0.5, threshold: 0.9 })).toEqual({ ana: { s: null } });
    expect(movingAverageMastery(eventLog(answers([0.5, 0.5, 0.5])), { decay: 0.5, threshold: 0.4 })).toEqual({
      ana: { s: 3 },
    });
  });

  it("takes the last outcome at a decay of 0, and never moves at a decay of 1 unless the threshold is 0", () => {
    const log = eventLog(answers([0, 0.6, 0]));
    expect(movingAverageMastery(log, { decay: 0, threshold: 0.6 })).toEqual({ ana: { s: 2 } });
    expect(movingAverageMastery(log, { decay: 1, threshold: 0.01 })).toEqual({ ana: { s: null } });
    expect(movingAverageMastery(log, { decay: 1, threshold: 0 })).toEqual({ ana: { s: 1 } });
  });

  it("counts the outcomes of 1 among the last attempts, the attempts so far where there are fewer than the window", () => {
    const log = eventLog(answers([1, 0, 1, 1, 0, 0]));
    expect(windowMastery(log, { correct: 2, window: 3 })).toEqual({ ana: { s: 3 } });
    expect(windowMastery(log, { correct: 3, window: 4 })).toEqual({ ana: { s: 4 } });
    expect(windowMastery(log, { correct: 3, window: 3 })).toEqual({ ana: { s: null } });
    expect(windowMastery(eventLog(answers([1, 0.9, 1])), { correct: 2, window: 2 })).toEqual({ ana: { s: null } });
  });

  it("leaves the average and the window as they are for an event without an outcome", () => {
    const log = eventLog(answers([1, null, null, 1, null]));
    expect(movingAverageMastery(log, { decay: 0.5, threshold: 0.75 })).toEqual({ ana: { s: 2 } });
    expect(windowMastery(log, { correct: 2, window: 2 })).toEqual({ ana: { s: 2 } });
    expect(windowMastery(eventLog(answers([null, null])), { correct: 1, window: 1 })).toEqual({});
  });

  it("credits an event to each of its skills, and each learner has a table of their own", () => {
    const log = eventLog([...answers([1, 1], ["s", "t"]), ...answers([0, 1], ["s"], "budi")]);
    expect(windowMastery(log, { correct: 2, window: 2 })).toEqual({ ana: { s: 2, t: 2 }, budi: { s: null } });
    expect(movingAverageMastery(log, { decay: 0, threshold: 1 })).toEqual({ ana: { s: 1, t: 1 }, budi: { s: 2 } });
  });

  it("tells learners and skills apart whatever characters their ids hold", () => {
    const log = eventLog([
      { learner: "a\u0000b", item: "q", skills: ["c"], time: 0, outcome: 1 },
      { learner: "a", item: "q", skills: ["b\u0000c"], time: 1, outcome: 1 },
    ]);
    const apart = { "a\u0000b": { c: null }, a: { "b\u0000c": null } };
    expect(streakMastery(log, { length: 2 })).toEqual(apart);
    expect(windowMastery(log, { correct: 2, window: 2 })).toEqual(apart);
    expect(movingAverageMastery(log, { decay: 0.5, threshold: 0.7 })).toEqual(apart);
  });

  it("predicts the attempt after its own outcome, for each of its skills, and never for an event without one", () => {
    const predictions: string[] = [];
    const model: LearnerModel<number> = {
      parameters: null,
      fit: () => {
        throw new Error("Not fitted.");
      },
      initialState: () => 0,
      update: (state, event) => state + (event.outcome ?? 0),
      predict: (state, attempt) => {
        predictions.push(`${attempt.time}:${state}`);
        return state / 4;
      },
    };
    const log = eventLog(answers([1, null, 1, 1], ["s", "t"]));
    // The state is the outcomes so far, and the prediction a quarter of it: 0.25 after the first, 0.5 after the second.
    expect(predictedMastery(model, log, { threshold: 0.5 })).toEqual({ ana: { s: 2, t: 2 } });
    expect(predictions).toEqual(["0:1", "2:2", "3:3"]);
    expect(predictedMastery(model, eventLog(answers([null])), { threshold: 0.5 })).toEqual({});
  });

  it("meets a stricter criterion no earlier, for any log", () => {
    const outcome = fc.constantFrom(0, 0.5, 1, null);
    fc.assert(
      fc.property(fc.array(outcome, { minLength: 1, maxLength: 30 }), (values) => {
        const log = eventLog(answers(values));
        const later = (strict: MasteryTable, loose: MasteryTable) => {
          const a = strict.ana?.s;
          const b = loose.ana?.s;
          return b === undefined || b === null ? true : a === null || (a ?? 0) >= b;
        };
        expect(
          later(windowMastery(log, { correct: 3, window: 3 }), windowMastery(log, { correct: 2, window: 3 })),
        ).toBe(true);
        expect(
          later(
            movingAverageMastery(log, { decay: 0.6, threshold: 0.9 }),
            movingAverageMastery(log, { decay: 0.6, threshold: 0.7 }),
          ),
        ).toBe(true);
        expect(
          later(
            predictedMastery(performance, withPfa(log), { threshold: 0.8 }),
            predictedMastery(performance, withPfa(log), { threshold: 0.6 }),
          ),
        ).toBe(true);
      }),
    );
  });

  it("gives the streak as a window that is full of correct answers", () => {
    const outcome = fc.constantFrom(0, 0.9, 1, null);
    fc.assert(
      fc.property(
        fc.array(outcome, { minLength: 1, maxLength: 40 }),
        fc.integer({ min: 1, max: 5 }),
        (values, length) => {
          const log = eventLog(answers(values, ["s", "t"]));
          expect(windowMastery(log, { correct: length, window: length })).toEqual(streakMastery(log, { length }));
        },
      ),
    );
  });
});

/** The events of a log with the skills of the fixture's model, which has parameters for s1 to s4 only. */
function withPfa(log: readonly Event[]) {
  return eventLog(log.map((event) => ({ ...event, skills: event.skills.map(() => "s1") })));
}

describe("mastery criteria, errors", () => {
  const log = eventLog([{ learner: "ana", item: "q", skills: ["s"], time: 0, outcome: 1 }]);

  it("rejects options of other types and values from JavaScript", () => {
    const missing = JSON.parse("null");
    expect(() => movingAverageMastery(log, missing)).toThrow("The options must be an object, found null.");
    expect(() => windowMastery(log, missing)).toThrow("The options must be an object, found null.");
    expect(() => predictedMastery(performance, log, missing)).toThrow("The options must be an object, found null.");
    expect(() => movingAverageMastery(log, JSON.parse('{"decay": 0.5}'))).toThrow(
      "The option threshold must be a number from 0 to 1, found undefined.",
    );
    expect(() => movingAverageMastery(log, { decay: 1.5, threshold: 0.5 })).toThrow(
      "The option decay must be a number from 0 to 1, found 1.5.",
    );
    expect(() => movingAverageMastery(log, JSON.parse('{"decay": "0.5", "threshold": 0.5}'))).toThrow(
      'The option decay must be a number from 0 to 1, found "0.5".',
    );
    expect(() => predictedMastery(performance, log, { threshold: -0.1 })).toThrow(
      "The option threshold must be a number from 0 to 1, found -0.1.",
    );
    expect(() => predictedMastery(performance, log, JSON.parse("{}"))).toThrow("The option threshold must be a number");
  });

  it("rejects a window and a number correct that do not fit", () => {
    expect(() => windowMastery(log, { correct: 2, window: 1 })).toThrow(
      "The option correct must be at most the window 1, found 2.",
    );
    expect(() => windowMastery(log, { correct: 0, window: 3 })).toThrow("The option correct must be an integer from 1");
    expect(() => windowMastery(log, { correct: 1, window: 2.5 })).toThrow(
      "The option window must be a positive integer",
    );
    expect(() => windowMastery(log, JSON.parse('{"correct": 1}'))).toThrow(
      "The option window must be a positive integer",
    );
  });

  it("rejects a log that is not one, a model that is not one, and an event with an outcome and no skills", () => {
    const missing = JSON.parse("null");
    expect(() => movingAverageMastery(missing, { decay: 0.5, threshold: 0.5 })).toThrow(
      "A log must be an array of events, found null.",
    );
    expect(() => windowMastery([], { correct: 1, window: 1 })).toThrow("A log needs at least one event.");
    expect(() => predictedMastery(missing, log, { threshold: 0.5 })).toThrow(
      "predictedMastery needs a learner model, found null.",
    );
    const unskilled = eventLog([{ learner: "ana", item: "q", skills: [], time: 0, outcome: 1 }]);
    expect(() => predictedMastery(performance, unskilled, { threshold: 0.5 })).toThrow(
      "Event 1 has no skills; a mastery table needs at least one.",
    );
    expect(
      predictedMastery(performance, eventLog([{ ...unskilled[0], outcome: null } as Event]), { threshold: 0.5 }),
    ).toEqual({});
  });
});

describe("mastery criteria of the guide", () => {
  it("meets the criteria of the guide's log, as its sample shows", () => {
    const attempt = (learner: string, item: string, skill: string, time: number, outcome: number) => ({
      learner,
      item,
      skills: [skill],
      time,
      outcome,
    });
    const log = eventLog([
      attempt("ana", "f1", "fractions", 1, 0),
      attempt("ana", "f2", "fractions", 2, 1),
      attempt("ana", "f3", "fractions", 3, 1),
      attempt("ana", "d1", "decimals", 4, 0),
      attempt("ana", "f1", "fractions", 5, 1),
      attempt("budi", "d1", "decimals", 6, 1),
    ]);
    const rates = { successes: 0.5, failures: 0.1 };
    const model = pfa({
      parameters: {
        skills: {
          fractions: { intercept: -0.5, coefficients: rates },
          decimals: { intercept: 0, coefficients: rates },
        },
      },
    });
    const at = (fractions: number | null) => ({ ana: { fractions, decimals: null }, budi: { decimals: null } });
    expect(movingAverageMastery(log, { decay: 0.5, threshold: 0.75 })).toEqual(at(3));
    expect(windowMastery(log, { correct: 2, window: 3 })).toEqual(at(3));
    expect(predictedMastery(model, log, { threshold: 0.7 })).toEqual(at(4));
  });
});

import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  type Attempt,
  bkt,
  DataError,
  elo,
  eventLog,
  type LearnerModel,
  opportunitiesToStreak,
  recommend,
  streakMastery,
  thresholdMastery,
} from "../src/index.js";

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
    // P(L) after a correct answer: 0.5 → 0.45 / 0.55 = 0.8182, then learning: 0.8545; after another: 0.9636, then 0.9708.
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

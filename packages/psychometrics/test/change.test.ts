import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  DataError,
  differenceReliability,
  normalizedGain,
  reliableChange,
  reliableChangeFromTheta,
  residualGain,
  simpleGain,
} from "../src/index.js";
import { defined, expectClose } from "./assertions.js";
import gains from "./fixtures/change-gain.json" with { type: "json" };
import reliable from "./fixtures/change-reliable.json" with { type: "json" };

// Tolerances against R, explained in docs/validation.md.
/** Gains, residuals of lm and Hake's gains, in base R. */
const GAIN_TOLERANCE = 1e-12;
/** Lord's reliability of a difference, written out in base R. */
const DIFFERENCE_RELIABILITY_TOLERANCE = 1e-12;
/** The reliable change index of clinicalsignificance's `cs_distribution` with `rci_method = "JT"`. */
const REPORTED_INDEX_TOLERANCE = 1e-10;
/** The reliable change index from ability estimates, written out in base R. */
const THETA_INDEX_TOLERANCE = 1e-12;

const TEST = { sd: 9, reliability: 0.88 };

/** Expects each value within `tolerance` of the one at its position in `expected`. */
function expectAllClose(
  actual: readonly (number | null)[],
  expected: readonly (number | null)[],
  tolerance: number,
): void {
  expect(actual).toHaveLength(expected.length);
  for (const [position, value] of actual.entries()) expectClose(value, expected[position] ?? null, tolerance);
}

describe("simpleGain", () => {
  it("matches the differences of base R", () => {
    for (const { pre, post, simple } of gains.cases) {
      const gain = simpleGain(pre, post);
      expect(gain).toHaveLength(pre.length);
      expectAllClose(gain, simple, GAIN_TOLERANCE);
    }
  });

  it("is positive for an increase and negative for a decline", () => {
    expect(simpleGain([12, 20, 31], [18, 19, 40])).toEqual([6, -1, 9]);
  });

  it("leaves the scores unchanged and returns a new array", () => {
    const pre = [1, 2];
    const post = [3, 5];
    const gain = simpleGain(pre, post);
    expect(pre).toEqual([1, 2]);
    expect(post).toEqual([3, 5]);
    expect(gain).not.toBe(post);
  });

  it("reports the gain of a single person", () => {
    expect(simpleGain([4], [4])).toEqual([0]);
  });

  it("rejects scores that are not paired finite numbers", () => {
    expect(() => simpleGain(JSON.parse("null"), [1])).toThrow(
      new DataError("The pre scores must be an array of scores, found null."),
    );
    expect(() => simpleGain([1], JSON.parse('{"a":1}'))).toThrow(
      new DataError("The post scores must be an array of scores, found an object."),
    );
    expect(() => simpleGain([1, 2], [1])).toThrow(
      new DataError("The pre and post scores must come in pairs, found 2 pre scores and 1 post scores."),
    );
    expect(() => simpleGain([], [])).toThrow(new DataError("The scores need at least 1 person, found 0."));
    expect(() => simpleGain([1, Number.NaN], [1, 2])).toThrow(
      new DataError("The pre score of person 2 must be a finite number, found NaN."),
    );
    expect(() => simpleGain([1, 2], JSON.parse('[1, "2"]'))).toThrow(
      new DataError('The post score of person 2 must be a finite number, found "2".'),
    );
    expect(() => simpleGain([1, 2], JSON.parse("[1, null]"))).toThrow(
      new DataError("The post score of person 2 must be a finite number, found null."),
    );
  });
});

describe("residualGain", () => {
  it("matches the residuals of lm in base R", () => {
    for (const { pre, post, residual } of gains.cases) {
      const gain = residualGain(pre, post);
      expect(gain).toHaveLength(pre.length);
      expectAllClose(gain, residual, GAIN_TOLERANCE);
    }
  });

  it("is 0 for scores on a line, and for two persons", () => {
    expectAllClose(residualGain([1, 2, 3, 4], [3, 5, 7, 9]), [0, 0, 0, 0], 1e-12);
    expectAllClose(residualGain([1, 2], [5, 9]), [0, 0], 1e-12);
  });

  it("is positive for a person above the regression line", () => {
    const gain = residualGain([1, 2, 3, 4], [2, 2, 5, 5]);
    expectClose(defined(gain[0]), 0.3, 1e-12);
    expectClose(defined(gain[1]), -0.9, 1e-12);
    expectClose(defined(gain[2]), 0.9, 1e-12);
    expectClose(defined(gain[3]), -0.3, 1e-12);
  });

  it("keeps its precision for scores far from 0", () => {
    const shift = 1e6;
    const gain = residualGain(
      [1, 2, 3, 4].map((s) => s + shift),
      [2, 2, 5, 5].map((s) => s + shift),
    );
    expectClose(defined(gain[1]), -0.9, 1e-9);
  });

  it("rejects pre scores that are all equal, and fewer than two persons", () => {
    expect(() => residualGain([3, 3, 3], [1, 2, 3])).toThrow(
      new DataError("The pre scores are all 3, so the post scores cannot be regressed on them."),
    );
    expect(() => residualGain([3], [1])).toThrow(new DataError("The scores need at least 2 persons, found 1."));
    expect(() => residualGain(JSON.parse('"abc"'), [1, 2])).toThrow(
      new DataError('The pre scores must be an array of scores, found "abc".'),
    );
    expect(() => residualGain([1, 2], [1, Number.POSITIVE_INFINITY])).toThrow(
      new DataError("The post score of person 2 must be a finite number, found Infinity."),
    );
  });
});

describe("normalizedGain", () => {
  it("matches Hake's gains in base R, with null for a person at the maximum", () => {
    for (const { maximum, pre, post, normalized } of gains.cases) {
      const gain = normalizedGain(pre, post, { maximum });
      expectClose(gain.group, normalized.group, GAIN_TOLERANCE);
      expect(gain.persons).toHaveLength(pre.length);
      expectAllClose(gain.persons, normalized.persons, GAIN_TOLERANCE);
    }
    const raw = defined(gains.cases[0]);
    expect(
      normalizedGain(raw.pre, raw.post, { maximum: raw.maximum }).persons.filter((value) => value === null),
    ).toHaveLength(raw.pre.filter((score) => score === raw.maximum).length);
  });

  it("takes the group's gain from the means, not from the persons' gains", () => {
    expect(normalizedGain([10, 30, 50], [30, 40, 50], { maximum: 50 })).toEqual({
      group: 0.5,
      persons: [0.5, 0.5, null],
    });
    const spread = normalizedGain([10, 40], [20, 50], { maximum: 50 });
    expect(spread.persons).toEqual([0.25, 1]);
    expectClose(spread.group, 0.4, 1e-15);
  });

  it("is 1 for a person who reaches the maximum and negative for a decline", () => {
    expect(normalizedGain([20, 20], [50, 10], { maximum: 50 }).persons).toEqual([1, -1 / 3]);
  });

  it("has no gain for a group that starts at the maximum", () => {
    expect(normalizedGain([50, 50], [50, 50], { maximum: 50 })).toEqual({ group: null, persons: [null, null] });
    expect(normalizedGain([0.1, 0.1, 0.1], [0.1, 0.1, 0.1], { maximum: 0.1 }).group).toBeNull();
  });

  it("rejects a score above the maximum, and an invalid maximum or options", () => {
    expect(() => normalizedGain([10, 55], [12, 20], { maximum: 50 })).toThrow(
      new DataError("The pre score of person 2 is 55, above the maximum 50."),
    );
    expect(() => normalizedGain([10, 20], [12, 51], { maximum: 50 })).toThrow(
      new DataError("The post score of person 2 is 51, above the maximum 50."),
    );
    expect(() => normalizedGain([10], [12], JSON.parse("null"))).toThrow(
      new DataError("The options must be an object, found null."),
    );
    expect(() => normalizedGain([10], [12], JSON.parse("{}"))).toThrow(
      new DataError("The option maximum must be a finite number, found undefined."),
    );
    expect(() => normalizedGain([10], [12], JSON.parse('{"maximum":"50"}'))).toThrow(
      new DataError('The option maximum must be a finite number, found "50".'),
    );
    expect(() => normalizedGain([], [], { maximum: 50 })).toThrow(
      new DataError("The scores need at least 1 person, found 0."),
    );
    expect(() => normalizedGain(JSON.parse("null"), [1], { maximum: 50 })).toThrow(
      new DataError("The pre scores must be an array of scores, found null."),
    );
  });
});

describe("differenceReliability", () => {
  it("matches Lord's formula in base R, a negative reliability included", () => {
    expect(gains.differenceReliability.some((entry) => entry.reliability < 0)).toBe(true);
    for (const { pre, post, correlation, reliability } of gains.differenceReliability) {
      expectClose(differenceReliability({ pre, post, correlation }), reliability, DIFFERENCE_RELIABILITY_TOLERANCE);
    }
  });

  it("is (r - rho) / (1 - rho) for equal deviations and reliabilities", () => {
    const score = { sd: 10, reliability: 0.9 };
    expectClose(differenceReliability({ pre: score, post: score, correlation: 0.5 }), 0.8, 1e-15);
    expectClose(differenceReliability({ pre: score, post: score, correlation: 0.9 }), 0, 1e-15);
    expectClose(differenceReliability({ pre: score, post: score, correlation: 0 }), 0.9, 1e-15);
  });

  it("is null when the difference has no variance, however small the deviations", () => {
    for (const sd of [10, 0.1, 1e-8, 3.3]) {
      const score = { sd, reliability: 0.9 };
      expect(differenceReliability({ pre: score, post: score, correlation: 1 })).toBeNull();
    }
    const first = { sd: 10, reliability: 0.9 };
    expect(differenceReliability({ pre: first, post: { sd: 12, reliability: 0.9 }, correlation: 1 })).not.toBeNull();
  });

  it("accepts the limits of correlation and reliability", () => {
    const perfect = { sd: 10, reliability: 1 };
    const none = { sd: 10, reliability: 0 };
    expectClose(differenceReliability({ pre: perfect, post: perfect, correlation: -1 }), 1, 1e-15);
    expectClose(differenceReliability({ pre: none, post: none, correlation: 0 }), 0, 1e-15);
  });

  it("rejects invalid scores and correlations", () => {
    const post = { sd: 12, reliability: 0.85 };
    expect(() => differenceReliability(JSON.parse("null"))).toThrow(
      new DataError("The input must be an object { pre, post, correlation }, found null."),
    );
    expect(() => differenceReliability({ pre: JSON.parse("null"), post, correlation: 0.5 })).toThrow(
      new DataError("The pre scores must be an object { sd, reliability }, found null."),
    );
    expect(() => differenceReliability({ pre: { sd: 0, reliability: 0.9 }, post, correlation: 0.5 })).toThrow(
      new DataError("The sd of the pre scores must be a positive number, found 0."),
    );
    expect(() =>
      differenceReliability({ pre: JSON.parse('{"sd":"10","reliability":0.9}'), post, correlation: 0.5 }),
    ).toThrow(new DataError('The sd of the pre scores must be a positive number, found "10".'));
    expect(() =>
      differenceReliability({
        pre: { sd: 10, reliability: 0.9 },
        post: { sd: 12, reliability: 1.2 },
        correlation: 0.5,
      }),
    ).toThrow(new DataError("The reliability of the post scores must be a number from 0 to 1, found 1.2."));
    expect(() =>
      differenceReliability({ pre: { sd: 10, reliability: 0.9 }, post: JSON.parse('{"sd":12}'), correlation: 0.5 }),
    ).toThrow(new DataError("The reliability of the post scores must be a number from 0 to 1, found undefined."));
    expect(() => differenceReliability({ pre: { sd: 10, reliability: 0.9 }, post, correlation: 1.2 })).toThrow(
      new DataError("The correlation must be a number from -1 to 1, found 1.2."),
    );
    expect(() =>
      differenceReliability({ pre: { sd: 10, reliability: 0.9 }, post, correlation: JSON.parse('"0.5"') }),
    ).toThrow(new DataError('The correlation must be a finite number, found "0.5".'));
  });
});

describe("reliableChange", () => {
  it("matches the index and the classification of clinicalsignificance at every level", () => {
    for (const { pre, post, sd, reliability, index, levels } of reliable.scores) {
      expect(levels.flatMap(({ classification }) => classification)).toContain("improved");
      pre.forEach((score, person) => {
        const input = { sd, reliability };
        expectClose(
          reliableChange(score, defined(post[person]), input).index,
          defined(index[person]),
          REPORTED_INDEX_TOLERANCE,
        );
        for (const { level, classification } of levels) {
          const result = reliableChange(score, defined(post[person]), input, { level });
          expectClose(result.index, defined(index[person]), REPORTED_INDEX_TOLERANCE);
          expect(result.classification, `person ${person + 1} at level ${level}`).toBe(classification[person]);
        }
      });
    }
    expect(reliable.scores.flatMap(({ levels }) => levels.flatMap(({ classification }) => classification))).toContain(
      "declined",
    );
  });

  it("divides the change by the standard error of the difference, sqrt(2 SEM²)", () => {
    const result = reliableChange(12, 21, TEST);
    expectClose(result.index, 9 / Math.sqrt(2 * (9 * Math.sqrt(0.12)) ** 2), 1e-14);
    expect(result).toEqual({ index: 2.041241452319315, classification: "improved" });
    expect(reliableChange(21, 12, TEST)).toEqual({ index: -2.041241452319315, classification: "declined" });
    expect(reliableChange(12, 12, TEST)).toEqual({ index: 0, classification: "unchanged" });
  });

  it("calls a change reliable when the index is beyond the critical value of the level", () => {
    const change = (level: number) => reliableChange(12, 21, TEST, { level }).classification;
    expect(change(0.95)).toBe("improved");
    expect(change(0.99)).toBe("unchanged");
    expect(change(0.9)).toBe("improved");
    expect(reliableChange(12, 21, TEST, {})).toEqual(reliableChange(12, 21, TEST, { level: 0.95 }));
    expect(reliableChange(12, 21, TEST)).toEqual(reliableChange(12, 21, TEST, { level: 0.95 }));
  });

  it("uses the whole standard deviation when the test is not reliable at all", () => {
    expectClose(reliableChange(0, 5, { sd: 2, reliability: 0 }).index, 5 / Math.sqrt(2 * 4), 1e-15);
  });

  it("rejects invalid scores, tests and options", () => {
    expect(() => reliableChange(JSON.parse('"12"'), 21, TEST)).toThrow(
      new DataError('The pre score must be a finite number, found "12".'),
    );
    expect(() => reliableChange(12, Number.NaN, TEST)).toThrow(
      new DataError("The post score must be a finite number, found NaN."),
    );
    expect(() => reliableChange(12, 21, JSON.parse("null"))).toThrow(
      new DataError("The test must be an object { sd, reliability }, found null."),
    );
    expect(() => reliableChange(12, 21, { sd: -1, reliability: 0.9 })).toThrow(
      new DataError("The sd of the test must be a positive number, found -1."),
    );
    expect(() => reliableChange(12, 21, { sd: 9, reliability: -0.1 })).toThrow(
      new DataError("The reliability of the test must be a number from 0 to 1, found -0.1."),
    );
    expect(() => reliableChange(12, 21, { sd: 9, reliability: 1 })).toThrow(
      new DataError("The reliability of the test must be below 1, as a test without error has no standard error."),
    );
    expect(() => reliableChange(12, 21, TEST, JSON.parse("null"))).toThrow(
      new DataError("The options must be an object, found null."),
    );
    for (const level of ["0", "1", "-0.2", "1.5", '"0.95"', "null"]) {
      expect(() => reliableChange(12, 21, TEST, JSON.parse(`{"level":${level}}`))).toThrow(
        /The option level must be a number between 0 and 1, exclusive, found/,
      );
    }
  });
});

describe("reliableChangeFromTheta", () => {
  it("matches the index and the classification written out in base R at every level", () => {
    const { before, after, index, levels } = reliable.theta;
    before.theta.forEach((theta, person) => {
      const first = { theta, standardError: defined(before.standardError[person]) };
      const second = { theta: defined(after.theta[person]), standardError: defined(after.standardError[person]) };
      expectClose(reliableChangeFromTheta(first, second).index, defined(index[person]), THETA_INDEX_TOLERANCE);
      for (const { level, classification } of levels) {
        expect(
          reliableChangeFromTheta(first, second, { level }).classification,
          `person ${person + 1} at ${level}`,
        ).toBe(classification[person]);
      }
    });
  });

  it("divides the change by sqrt(SE1² + SE2²)", () => {
    const result = reliableChangeFromTheta({ theta: -0.4, standardError: 0.3 }, { theta: 0.5, standardError: 0.2 });
    expectClose(result.index, 0.9 / Math.sqrt(0.13), 1e-14);
    expect(result.classification).toBe("improved");
    expect(reliableChangeFromTheta({ theta: 0.5, standardError: 0.2 }, { theta: -0.4, standardError: 0.3 })).toEqual({
      index: -result.index,
      classification: "declined",
    });
  });

  it("reads only the theta and the standard error of an estimate", () => {
    const estimate = { theta: 0.2, standardError: 0.3, extra: "ignored" };
    expect(reliableChangeFromTheta(estimate, { theta: 1.4, standardError: 0.3 })).toEqual(
      reliableChangeFromTheta({ theta: 0.2, standardError: 0.3 }, { theta: 1.4, standardError: 0.3 }),
    );
  });

  it("rejects invalid estimates and options", () => {
    const estimate = { theta: 0, standardError: 0.3 };
    expect(() => reliableChangeFromTheta(JSON.parse("null"), estimate)).toThrow(
      new DataError("The estimate before must be an object { theta, standardError }, found null."),
    );
    expect(() => reliableChangeFromTheta(estimate, JSON.parse('{"theta":"1","standardError":0.3}'))).toThrow(
      new DataError('The theta after must be a finite number, found "1".'),
    );
    expect(() => reliableChangeFromTheta({ theta: 0, standardError: 0 }, estimate)).toThrow(
      new DataError("The standard error before must be a positive number, found 0."),
    );
    expect(() => reliableChangeFromTheta(estimate, { theta: 0, standardError: Number.POSITIVE_INFINITY })).toThrow(
      new DataError("The standard error after must be a positive number, found Infinity."),
    );
    expect(() => reliableChangeFromTheta(estimate, estimate, JSON.parse("null"))).toThrow(
      new DataError("The options must be an object, found null."),
    );
    expect(() => reliableChangeFromTheta(estimate, estimate, { level: 1 })).toThrow(
      new DataError("The option level must be a number between 0 and 1, exclusive, found 1."),
    );
  });
});

describe("properties", () => {
  const scoreList = fc.array(fc.integer({ min: 0, max: 50 }), { minLength: 2, maxLength: 30 });
  const pairs = scoreList.chain((pre) =>
    fc
      .array(fc.integer({ min: 0, max: 50 }), { minLength: pre.length, maxLength: pre.length })
      .map((post) => ({ pre, post })),
  );
  const spread = pairs.filter(({ pre }) => new Set(pre).size > 1);

  it("the gain of reversed occasions is the negative gain", () => {
    fc.assert(
      fc.property(pairs, ({ pre, post }) => {
        const forward = simpleGain(pre, post);
        expectAllClose(
          simpleGain(post, pre),
          forward.map((gain) => -gain),
          0,
        );
      }),
    );
  });

  it("residual gains sum to 0 and are uncorrelated with the pre scores", () => {
    fc.assert(
      fc.property(spread, ({ pre, post }) => {
        const gain = residualGain(pre, post);
        const center = pre.reduce((sum, score) => sum + score, 0) / pre.length;
        expectClose(
          gain.reduce((sum, value) => sum + value, 0),
          0,
          1e-9,
        );
        expectClose(
          gain.reduce((sum, value, person) => sum + value * (defined(pre[person]) - center), 0),
          0,
          1e-7,
        );
      }),
    );
  });

  it("residual gains are the same in any order of the persons", () => {
    fc.assert(
      fc.property(spread, ({ pre, post }) => {
        const reversed = residualGain([...pre].reverse(), [...post].reverse()).toReversed();
        expectAllClose(residualGain(pre, post), reversed, 1e-9);
      }),
    );
  });

  it("a normalized gain is at most 1, for a person and for the group", () => {
    fc.assert(
      fc.property(pairs, ({ pre, post }) => {
        const { group, persons } = normalizedGain(pre, post, { maximum: 50 });
        for (const gain of persons) expect(gain === null || gain <= 1).toBe(true);
        expect(group === null || group <= 1 + 1e-12).toBe(true);
      }),
    );
  });

  it("the reliability of a difference is at most 1 and falls as the correlation rises", () => {
    const scores = fc.record({
      sd: fc.double({ min: 0.5, max: 50, noNaN: true }),
      reliability: fc.double({ min: 0, max: 1, noNaN: true }),
    });
    fc.assert(
      fc.property(
        scores,
        scores,
        fc.double({ min: -0.95, max: 0.9, noNaN: true }),
        fc.double({ min: 0.001, max: 0.05, noNaN: true }),
        (pre, post, correlation, step) => {
          const low = defined(differenceReliability({ pre, post, correlation }));
          const high = defined(differenceReliability({ pre, post, correlation: correlation + step }));
          expect(low).toBeLessThanOrEqual(1 + 1e-12);
          expect(high).toBeLessThanOrEqual(low + 1e-9);
        },
      ),
    );
  });

  it("swapping the occasions negates the index and swaps improved and declined", () => {
    const swapped = { improved: "declined", unchanged: "unchanged", declined: "improved" } as const;
    fc.assert(
      fc.property(
        fc.double({ min: -100, max: 100, noNaN: true }),
        fc.double({ min: -100, max: 100, noNaN: true }),
        fc.double({ min: 0.5, max: 30, noNaN: true }),
        fc.double({ min: 0, max: 0.99, noNaN: true }),
        fc.double({ min: 0.05, max: 0.99, noNaN: true }),
        (pre, post, sd, reliability, level) => {
          const forward = reliableChange(pre, post, { sd, reliability }, { level });
          const backward = reliableChange(post, pre, { sd, reliability }, { level });
          expect(backward.index + forward.index).toBe(0);
          expect(backward.classification).toBe(swapped[forward.classification]);
        },
      ),
    );
  });

  it("a higher level calls fewer changes reliable", () => {
    fc.assert(
      fc.property(
        fc.double({ min: -4, max: 4, noNaN: true }),
        fc.double({ min: 0.1, max: 0.9, noNaN: true }),
        fc.double({ min: 0, max: 0.09, noNaN: true }),
        (change, level, extra) => {
          const earlier = reliableChange(0, change, { sd: 1, reliability: 0.5 }, { level });
          const later = reliableChange(0, change, { sd: 1, reliability: 0.5 }, { level: level + extra });
          if (later.classification !== "unchanged") expect(earlier.classification).toBe(later.classification);
        },
      ),
    );
  });

  it("swapping the estimates negates the index from ability estimates", () => {
    const estimates = fc.record({
      theta: fc.double({ min: -4, max: 4, noNaN: true }),
      standardError: fc.double({ min: 0.05, max: 1, noNaN: true }),
    });
    fc.assert(
      fc.property(estimates, estimates, (before, after) => {
        expect(reliableChangeFromTheta(after, before).index + reliableChangeFromTheta(before, after).index).toBe(0);
      }),
    );
  });
});

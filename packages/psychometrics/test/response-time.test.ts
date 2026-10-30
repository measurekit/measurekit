import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  calibrateResponseTimes,
  DataError,
  effortModeratedEstimates,
  normativeThresholds,
  responseMatrix,
  responseRecords,
  responseTimeEffort,
  responseTimeFit,
  speedEstimates,
  twoParameterLogistic,
} from "../src/index.js";
import fixture from "./fixtures/response-time.json" with { type: "json" };
import { defined, expectClose } from "./helpers.js";

// Tolerances against R, explained in docs/validation.md.
/** The lognormal calibration against lme's ML fit: EM runs to a change of 1e-13. */
const CALIBRATION_TOLERANCE = 1e-8;
/** Speeds, residuals and statistics at the reference's parameters, closed forms of the same sums. */
const CLOSED_FORM_TOLERANCE = 1e-10;
/** Effort-moderated EAP estimates against mirt's fscores with 61 nodes. */
const EAP_TOLERANCE = 1e-10;

/** A response matrix of the fixture's times, each with a response of 0. */
function timed(times: readonly (readonly (number | null)[])[], itemIds: readonly string[]) {
  return responseMatrix(
    times.map((row) => row.map((time) => (time === null ? null : 0))),
    { itemIds, times },
  );
}

describe("calibrateResponseTimes", () => {
  for (const reference of fixture.cases) {
    it(`matches nlme's lme on ${reference.name}`, () => {
      const result = calibrateResponseTimes(timed(reference.times, reference.itemIds), {
        tolerance: 1e-13,
        maxIterations: 100_000,
      });
      expect(result.converged).toBe(true);
      result.items.forEach((item, i) => {
        expectClose(item.intensity, defined(reference.intensity[i]), CALIBRATION_TOLERANCE, `${item.id} intensity`);
        expectClose(item.discrimination, defined(reference.discrimination[i]), CALIBRATION_TOLERANCE, `${item.id} α`);
        expectClose(item.standardError, defined(reference.intensityErrors[i]), CALIBRATION_TOLERANCE, `${item.id} SE`);
      });
      expectClose(result.speedSd, reference.speedSd, CALIBRATION_TOLERANCE, "speed SD");
      expectClose(result.logLikelihood, reference.logLikelihood, CALIBRATION_TOLERANCE, "log-likelihood");
      const parameters = 2 * reference.itemIds.length + 1;
      expectClose(result.aic, -2 * result.logLikelihood + 2 * parameters, 1e-9, "AIC");
    });
  }

  it("calibrates response records as the matrix with the same times", () => {
    const reference = defined(fixture.cases[0]);
    const matrix = timed(reference.times, reference.itemIds);
    const records = responseRecords(
      matrix.responses.flatMap((row, p) =>
        row.flatMap((response, i) => {
          const time = matrix.times?.[p]?.[i] ?? null;
          return response === null || time === null
            ? []
            : [{ person: defined(matrix.personIds[p]), item: defined(reference.itemIds[i]), response, time }];
        }),
      ),
      { personIds: matrix.personIds, itemIds: reference.itemIds },
    );
    expect(calibrateResponseTimes(records)).toEqual(calibrateResponseTimes(matrix));
  });

  it("rejects data without times or with an item of fewer than two times", () => {
    expect(() => calibrateResponseTimes(responseMatrix([[1, 0]]))).toThrow(
      "Response time calibration needs response times.",
    );
    expect(() => calibrateResponseTimes(responseMatrix([[1, 0]]), JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    const matrix = responseMatrix(
      [
        [1, 0],
        [0, null],
      ],
      {
        times: [
          [3, 4],
          [5, null],
        ],
      },
    );
    expect(() => calibrateResponseTimes(matrix)).toThrow('Item "item2" needs at least two response times.');
    // Equal times leave the item's variance at 0, where the likelihood has no maximum; the mean of 7 equal log times
    // is off by rounding, so the check compares the times.
    const equal = responseMatrix(
      Array.from({ length: 7 }, (_, person) => [person % 2, 1]),
      { times: Array.from({ length: 7 }, (_, person) => [5, 4 + person]) },
    );
    expect(() => calibrateResponseTimes(equal)).toThrow('Item "item1" needs response times that differ.');
    expect(() => responseMatrix([[1, null]], { times: [[2, 3]] })).toThrow(
      'The time of person "person1" on item "item2" needs a response, found 3.',
    );
    expect(() => responseRecords([{ person: "p", item: "q", response: 1, time: 0 }])).toThrow(
      'The time of person "p" on item "q" must be a positive number, found 0.',
    );
  });
});

describe("rapid guessing", () => {
  const { rapidGuessing } = fixture;
  const matrix = responseMatrix(rapidGuessing.responses, {
    itemIds: rapidGuessing.itemIds,
    times: rapidGuessing.times,
  });

  it("matches base R's normative thresholds", () => {
    for (const { percentage, cap, values } of rapidGuessing.thresholds) {
      const thresholds = normativeThresholds(matrix, { percentage, cap });
      rapidGuessing.itemIds.forEach((id, i) => {
        expectClose(defined(thresholds[id]), defined(values[i]), CLOSED_FORM_TOLERANCE, id);
      });
    }
  });

  it("matches base R's response time effort and mirt's effort-moderated EAP estimates", () => {
    const thresholds = normativeThresholds(matrix);
    responseTimeEffort(matrix, thresholds).forEach((person, p) => {
      expectClose(person.effort, rapidGuessing.effort[p] ?? null, CLOSED_FORM_TOLERANCE, person.id);
    });
    const items = rapidGuessing.items.map((parameters, i) => ({
      id: defined(rapidGuessing.itemIds[i]),
      model: twoParameterLogistic,
      parameters,
    }));
    effortModeratedEstimates(matrix, items, thresholds).forEach((estimate, p) => {
      const expected = defined(rapidGuessing.moderated[p]);
      expectClose(estimate.theta, expected.theta, EAP_TOLERANCE, `${estimate.id} theta`);
      expectClose(estimate.standardError, expected.standardError, EAP_TOLERANCE, `${estimate.id} SE`);
    });
  });

  it("counts a response at its threshold as effort, and rejects invalid thresholds and options", () => {
    const data = responseMatrix(
      [
        [1, 0],
        [1, 1],
      ],
      {
        times: [
          [2, 5],
          [8, 1],
        ],
      },
    );
    expect(responseTimeEffort(data, { item1: 2, item2: 2 })).toEqual([
      { id: "person1", effort: 1, rapidGuesses: 0 },
      { id: "person2", effort: 0.5, rapidGuesses: 1 },
    ]);
    expect(normativeThresholds(data, { percentage: 50, cap: 2 })).toEqual({ item1: 2, item2: 1.5 });
    expect(() => normativeThresholds(data, { cap: 0 })).toThrow("The option cap must be a positive number, found 0.");
    expect(() => responseTimeEffort(data, { item1: -1 })).toThrow(
      'The threshold of item "item1" must be a positive number, found -1.',
    );
    expect(() => responseTimeEffort(data, JSON.parse("null"))).toThrow(
      "The thresholds must be an object of seconds by item id, found null.",
    );
    expect(() => normativeThresholds(data, JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => effortModeratedEstimates(data, JSON.parse("null"), { item1: 2 })).toThrow(
      "The items must be an array, found null.",
    );
  });
});

describe("speed and time fit", () => {
  for (const reference of fixture.cases) {
    const items = reference.itemIds.map((id, i) => ({
      id,
      intensity: defined(reference.intensity[i]),
      discrimination: defined(reference.discrimination[i]),
    }));
    const matrix = timed(reference.times, reference.itemIds);

    it(`matches lme's posterior speeds and LNIRT's person statistic on ${reference.name}`, () => {
      const options = { sd: reference.speedSd };
      speedEstimates(matrix, items, options).forEach((person, p) => {
        expectClose(person.speed, defined(reference.speed[p]), CLOSED_FORM_TOLERANCE, `${person.id} speed`);
        expectClose(person.standardError, defined(reference.speedErrors[p]), CLOSED_FORM_TOLERANCE, `${person.id} SE`);
      });
      responseTimeFit(matrix, items, options).forEach((person, p) => {
        person.residuals.forEach((residual, i) => {
          expectClose(residual, reference.residuals[p]?.[i] ?? null, CLOSED_FORM_TOLERANCE, `${person.id} z${i + 1}`);
        });
        expectClose(person.statistic, defined(reference.statistic[p]), CLOSED_FORM_TOLERANCE, `${person.id} statistic`);
        expectClose(person.significance, defined(reference.significance[p]), CLOSED_FORM_TOLERANCE, `${person.id} p`);
      });
    });
  }

  it("estimates speed by maximum likelihood without a prior, and gives nothing to a person without times", () => {
    const items = [
      { id: "item1", intensity: 4, discrimination: 2 },
      { id: "item2", intensity: 3, discrimination: 1 },
    ];
    const data = responseMatrix(
      [
        [1, 0],
        [null, null],
      ],
      {
        times: [
          [Math.exp(3.5), Math.exp(3)],
          [null, null],
        ],
      },
    );
    // τ̂ = Σ α² (β - y) / Σ α² = (4 × 0.5 + 1 × 0) / 5 = 0.4, SE = 1 / √5.
    const [first, second] = speedEstimates(data, items);
    expectClose(defined(first).speed, 0.4, 1e-12);
    expectClose(defined(first).standardError, 1 / Math.sqrt(5), 1e-12);
    expect(second).toEqual({ id: "person2", speed: null, standardError: null });
    expect(() => speedEstimates(data, items.slice(1))).toThrow(
      'The data have item "item1", which is not among the time items.',
    );
    expect(() => speedEstimates(data, [...items, defined(items[0])])).toThrow(
      'The time item id "item1" appears more than once.',
    );
    expect(() => speedEstimates(data, items, { sd: 0 })).toThrow(DataError);
    expect(() => speedEstimates(data, JSON.parse("[null]"))).toThrow(
      "Time item 1 must be an object { id, intensity, discrimination }, found null.",
    );
    expect(() => responseTimeFit(data, items, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
  });
});

describe("properties", () => {
  it("speeds shrink toward 0 under a prior, and a slower person has a lower speed", () => {
    const items = [
      { id: "item1", intensity: 4, discrimination: 1.5 },
      { id: "item2", intensity: 3.5, discrimination: 2 },
    ];
    fc.assert(
      fc.property(
        fc.double({ min: 1, max: 300, noNaN: true }),
        fc.double({ min: 1, max: 300, noNaN: true }),
        fc.double({ min: 1.01, max: 3, noNaN: true }),
        (first, second, factor) => {
          const data = responseMatrix(
            [
              [1, 1],
              [1, 1],
            ],
            {
              times: [
                [first, second],
                [first * factor, second * factor],
              ],
            },
          );
          const [fast, slow] = speedEstimates(data, items);
          const [prior] = speedEstimates(data, items, { sd: 0.3 });
          expect(defined(slow?.speed)).toBeLessThan(defined(fast?.speed));
          expect(Math.abs(defined(prior?.speed))).toBeLessThanOrEqual(Math.abs(defined(fast?.speed)) + 1e-12);
        },
      ),
    );
  });
});

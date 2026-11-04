import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  buildResponseMatrix,
  buildResponseRecords,
  calibrate,
  calibrateResponseTimes,
  calibrateSpeedAccuracy,
  createRandom,
  DataError,
  effortModeratedEstimates,
  gradedResponse,
  type IrtModel,
  type LogisticItem,
  logNormalPrior,
  normativeThresholds,
  posteriorMeanEstimate,
  type ResponseMatrix,
  responseTimeEffort,
  responseTimeFit,
  type SpeedAccuracyCalibration,
  simulateItems,
  simulateResponses,
  simulateThetas,
  speedAccuracyEstimates,
  speedEstimates,
  twoParameterLogistic,
} from "../src/index.js";
import { defined, expectClose } from "./assertions.js";
import { readFixture } from "./fixtures.js";

interface ResponseTimeFixture {
  cases: {
    name: string;
    itemIds: string[];
    times: (number | null)[][];
    residuals: (number | null)[][];
    statistic: number[];
    significance: number[];
    intensity: number[];
    discrimination: number[];
    intensityErrors: number[];
    speedSd: number;
    logLikelihood: number;
    speed: number[];
    speedErrors: number[];
  }[];
  rapidGuessing: {
    itemIds: string[];
    times: (number | null)[][];
    responses: number[][];
    thresholds: { percentage: number; cap: number; values: number[] }[];
    effort: number[];
    items: { a: number; b: number; c: number; d: number }[];
    moderated: { theta: number; standardError: number }[];
  };
}
const fixture = readFixture<ResponseTimeFixture>("response-time.json");

interface ResponseTimeSpeedAccuracyFixture {
  cases: {
    name: string;
    itemIds: string[];
    responses: (number | null)[][];
    times: (number | null)[][];
    a: number[];
    b: number[];
    intensity: number[];
    discrimination: number[];
    speedSd: number;
    correlation: number;
    logLikelihood: number;
    standardErrors: {
      a: number[];
      b: number[];
      intensity: number[];
      discrimination: number[];
      speedSd: number;
      correlation: number;
    };
    estimates: { theta: number[]; thetaSd: number[]; speed: number[]; speedSd: number[] };
    lnirt: {
      persons: number;
      correlation: number;
      speedSd: number;
      intensity: number[];
      discrimination: number[];
      ogive: { correlation: number; logLikelihood: number; standardError: number };
    };
  }[];
}
const jointFixture = readFixture<ResponseTimeSpeedAccuracyFixture>("response-time-speed-accuracy.json");

// Tolerances against R, explained in docs/validation.md.
/** The lognormal calibration against lme's ML fit: EM runs to a change of 1e-13. */
const CALIBRATION_TOLERANCE = 1e-8;
/** Speeds, residuals and statistics at the reference's parameters, closed forms of the same sums. */
const CLOSED_FORM_TOLERANCE = 1e-10;
/** Effort-moderated EAP estimates against mirt's fscores with 61 nodes. */
const EAP_TOLERANCE = 1e-10;
/** The joint model's parameters against the base R maximum of the written-out likelihood, as EM stops at 1e-11. */
const JOINT_TOLERANCE = 1e-8;
/** The joint model's log-likelihood at the estimates. */
const JOINT_LOG_LIKELIHOOD_TOLERANCE = 1e-8;
/** The joint model's standard errors, relative, against the Richardson Hessian of the exact gradient. */
const JOINT_ERROR_TOLERANCE = 1e-6;
/** The correlation and the time parameters against LNIRT's Gibbs sampler, within its Monte Carlo error. */
const LNIRT_TOLERANCE = 0.1;

/** A response matrix of the fixture's times, each with a response of 0. */
function timed(times: readonly (readonly (number | null)[])[], itemIds: readonly string[]) {
  return buildResponseMatrix(
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
    const records = buildResponseRecords(
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

  it("leaves out the standard errors of more than 2,000 items, whose information holds every pair", () => {
    // 2,001 items, each answered by four of a ring of persons in times that differ.
    const records = Array.from({ length: 2001 }, (_, item) =>
      [0, 1, 2, 3].map((k) => ({
        person: `p${(item + k) % 50}`,
        item: `i${item}`,
        response: 1,
        time: 5 + k + (item % 3),
      })),
    ).flat();
    const calibration = calibrateResponseTimes(buildResponseRecords(records));
    expect(calibration.items).toHaveLength(2001);
    for (const item of calibration.items) expect(item.standardError).toBeNull();
  });

  it("rejects data without times or with an item of fewer than two times", () => {
    expect(() => calibrateResponseTimes(buildResponseMatrix([[1, 0]]))).toThrow(
      "Response time calibration needs response times.",
    );
    expect(() => calibrateResponseTimes(buildResponseMatrix([[1, 0]]), JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    const matrix = buildResponseMatrix(
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
    const equal = buildResponseMatrix(
      Array.from({ length: 7 }, (_, person) => [person % 2, 1]),
      { times: Array.from({ length: 7 }, (_, person) => [5, 4 + person]) },
    );
    expect(() => calibrateResponseTimes(equal)).toThrow('Item "item1" needs response times that differ.');
    expect(() => buildResponseMatrix([[1, null]], { times: [[2, 3]] })).toThrow(
      'The time of person "person1" on item "item2" needs a response, found 3.',
    );
    expect(() => buildResponseRecords([{ person: "p", item: "q", response: 1, time: 0 }])).toThrow(
      'The time of person "p" on item "q" must be a positive number, found 0.',
    );
  });
});

describe("rapid guessing", () => {
  const { rapidGuessing } = fixture;
  const matrix = buildResponseMatrix(rapidGuessing.responses, {
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
    const data = buildResponseMatrix(
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
    // An item named after a property of every object, without a threshold, has none, so its responses do not count.
    const named = {
      ...data,
      itemIds: ["item1", "constructor"],
      times: [
        [2, 5],
        [1, 8],
      ],
    };
    expect(responseTimeEffort(named, { item1: 2 })).toEqual([
      { id: "person1", effort: 1, rapidGuesses: 0 },
      { id: "person2", effort: 0, rapidGuesses: 1 },
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

  it("estimates the speeds of a sparse log in time that grows with its times, as responseTimeFit's speeds", () => {
    // 40,000 persons who each answered an item of their own: a table of residuals would hold 1.6 billion values.
    const count = 40_000;
    const records = buildResponseRecords(
      Array.from({ length: count }, (_, k) => ({
        person: `p${k}`,
        item: `i${k}`,
        response: k % 2,
        time: 10 + (k % 7),
      })),
    );
    const items = records.itemIds.map((id, k) => ({ id, intensity: 2 + (k % 3) / 2, discrimination: 1.5 }));
    const estimates = speedEstimates(records, items, { sd: 0.5 });
    expect(estimates).toHaveLength(count);
    const small = buildResponseRecords(
      Array.from({ length: 30 }, (_, k) => ({
        person: `p${k % 6}`,
        item: `i${k % 5}`,
        response: 1,
        time: 3 + (k % 4),
      })),
    );
    const smallItems = small.itemIds.map((id, k) => ({ id, intensity: 1 + k / 4, discrimination: 1 + k / 10 }));
    const fit = responseTimeFit(small, smallItems, { sd: 0.5 });
    speedEstimates(small, smallItems, { sd: 0.5 }).forEach((estimate, p) => {
      expect(estimate).toEqual({ id: fit[p]?.id, speed: fit[p]?.speed, standardError: fit[p]?.standardError });
    });
  });

  it("estimates speed by maximum likelihood without a prior, and gives nothing to a person without times", () => {
    const items = [
      { id: "item1", intensity: 4, discrimination: 2 },
      { id: "item2", intensity: 3, discrimination: 1 },
    ];
    const data = buildResponseMatrix(
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

type JointCase = (typeof jointFixture.cases)[number];

/** The data of a joint fixture case: the responses and times, with their ids. */
function jointData(reference: JointCase): ResponseMatrix {
  return buildResponseMatrix(reference.responses, { itemIds: reference.itemIds, times: reference.times });
}

/** The fixture's maximum as a calibration, for the estimates and for checks that need known parameters. */
function jointCalibration(
  reference: JointCase,
  correlation = reference.correlation,
): SpeedAccuracyCalibration<LogisticItem> {
  return {
    items: reference.itemIds.map((id, i) => ({
      id,
      model: twoParameterLogistic,
      parameters: { a: defined(reference.a[i]), b: defined(reference.b[i]), c: 0, d: 1 },
      standardErrors: null,
      intensity: defined(reference.intensity[i]),
      discrimination: defined(reference.discrimination[i]),
      timeStandardErrors: null,
    })),
    speedSd: reference.speedSd,
    correlation,
    correlationStandardError: null,
    logLikelihood: reference.logLikelihood,
    aic: 0,
    bic: 0,
    converged: true,
    iterations: 0,
  };
}

/** Options that run EM to the precision of the base R maximum. */
const PRECISE = { tolerance: 1e-11, maxIterations: 50_000 } as const;

/**
 * Simulates `persons` persons answering `items` items of a model with times: theta ~ N(0, 1), speed with the
 * correlation `rho` to it and SD `sigma`, ln t = β - τ + ε / α.
 */
function simulateJoint<TParameters>(
  model: IrtModel<TParameters>,
  seed: number,
  persons: number,
  items: number,
  categoryCount: number,
  rho: number,
) {
  const random = createRandom(seed);
  const pool = simulateItems(model, items, random, { categoryCount });
  const thetas = simulateThetas(persons, random);
  const sigma = 0.4;
  const intensity = Array.from({ length: items }, () => 3 + random.uniform());
  const discrimination = Array.from({ length: items }, () => 1.2 + random.uniform());
  const speeds = thetas.map((theta) => sigma * (rho * theta + Math.sqrt(1 - rho * rho) * random.normal()));
  const times = speeds.map((speed) =>
    intensity.map((beta, i) => Math.exp(beta - speed + random.normal() / (discrimination[i] ?? 1))),
  );
  const { responses } = simulateResponses(pool, thetas, random);
  return { matrix: buildResponseMatrix(responses, { times }), sigma, intensity, discrimination, pool };
}

/** Expects the estimates of simulated data to be within four standard errors of the simulated values. */
function expectRecovery<TParameters>(model: IrtModel<TParameters>, categoryCount: number, seed: number): void {
  const { matrix, sigma, intensity, discrimination, pool } = simulateJoint(model, seed, 1500, 6, categoryCount, 0.5);
  const result = calibrateSpeedAccuracy(matrix, { model, standardErrors: "hessian" });
  expect(result.converged).toBe(true);
  expect(Math.abs(result.correlation - 0.5)).toBeLessThan(4 * defined(result.correlationStandardError));
  expect(Math.abs(result.speedSd - sigma)).toBeLessThan(0.03);
  result.items.forEach((item, i) => {
    const errors = defined(item.timeStandardErrors);
    expect(Math.abs(item.intensity - defined(intensity[i]))).toBeLessThan(4 * errors.intensity);
    expect(Math.abs(item.discrimination - defined(discrimination[i]))).toBeLessThan(4 * errors.discrimination);
    const estimated = model.toVector(item.parameters);
    const simulated = model.toVector(defined(pool[i]).parameters);
    Object.values(defined(item.standardErrors)).forEach((error, j) => {
      expect(Math.abs(defined(estimated[j]) - defined(simulated[j])), `${item.id} parameter ${j + 1}`).toBeLessThan(
        4 * error,
      );
    });
  });
}

describe("calibrateSpeedAccuracy", () => {
  for (const reference of jointFixture.cases) {
    it(`matches the base R maximum of the marginal likelihood on ${reference.name}`, () => {
      const result = calibrateSpeedAccuracy(jointData(reference), {
        model: twoParameterLogistic,
        standardErrors: "hessian",
        ...PRECISE,
      });
      expect(result.converged).toBe(true);
      expectClose(result.logLikelihood, reference.logLikelihood, JOINT_LOG_LIKELIHOOD_TOLERANCE, "log-likelihood");
      expectClose(result.speedSd, reference.speedSd, JOINT_TOLERANCE, "speed SD");
      expectClose(result.correlation, reference.correlation, JOINT_TOLERANCE, "correlation");
      const errors = reference.standardErrors;
      expectClose(
        result.correlationStandardError,
        errors.correlation,
        JOINT_ERROR_TOLERANCE * errors.correlation,
        "correlation SE",
      );
      result.items.forEach((item, i) => {
        const at = `${item.id}`;
        expectClose(item.parameters.a, defined(reference.a[i]), JOINT_TOLERANCE, `${at} a`);
        expectClose(item.parameters.b, defined(reference.b[i]), JOINT_TOLERANCE, `${at} b`);
        expectClose(item.intensity, defined(reference.intensity[i]), JOINT_TOLERANCE, `${at} intensity`);
        expectClose(item.discrimination, defined(reference.discrimination[i]), JOINT_TOLERANCE, `${at} α`);
        const checks: [number | undefined, number | undefined, string][] = [
          [item.standardErrors?.a, errors.a[i], "a"],
          [item.standardErrors?.b, errors.b[i], "b"],
          [item.timeStandardErrors?.intensity, errors.intensity[i], "intensity"],
          [item.timeStandardErrors?.discrimination, errors.discrimination[i], "α"],
        ];
        for (const [actual, expected, name] of checks) {
          const error = defined(expected);
          expectClose(defined(actual), error, JOINT_ERROR_TOLERANCE * error, `${at} ${name} SE`);
        }
      });
      const parameters = 4 * reference.itemIds.length + 2;
      expectClose(result.aic, -2 * result.logLikelihood + 2 * parameters, 1e-9, "AIC");
    });

    it(`gives the correlation and the time parameters of LNIRT's Gibbs sampler within its Monte Carlo error on ${reference.name}`, () => {
      const result = calibrateSpeedAccuracy(jointData(reference), { model: twoParameterLogistic });
      const { lnirt } = reference;
      expectClose(result.speedSd, lnirt.speedSd, LNIRT_TOLERANCE, "speed SD");
      result.items.forEach((item, i) => {
        expectClose(item.intensity, defined(lnirt.intensity[i]), LNIRT_TOLERANCE, `${item.id} intensity`);
        const discrimination = defined(lnirt.discrimination[i]);
        expectClose(item.discrimination, discrimination, LNIRT_TOLERANCE * discrimination, `${item.id} α`);
      });
    });
  }

  it("gives LNIRT's correlation where the normal ogive and the logistic model agree", () => {
    const reference = defined(jointFixture.cases.find(({ name }) => name === "simLNIRT"));
    const result = calibrateSpeedAccuracy(jointData(reference), { model: twoParameterLogistic });
    expectClose(result.correlation, reference.lnirt.correlation, LNIRT_TOLERANCE, "correlation");
  });

  it("calibrates response records as the matrix with the same responses and times", () => {
    const reference = defined(jointFixture.cases[1]);
    const matrix = buildResponseMatrix(reference.responses.slice(0, 120), {
      itemIds: reference.itemIds,
      times: reference.times.slice(0, 120),
    });
    const records = buildResponseRecords(
      matrix.responses.flatMap((row, p) =>
        row.flatMap((response, i) => {
          const time = matrix.times?.[p]?.[i] ?? null;
          if (response === null) return [];
          return [
            {
              person: defined(matrix.personIds[p]),
              item: defined(reference.itemIds[i]),
              response,
              ...(time === null ? {} : { time }),
            },
          ];
        }),
      ),
      { personIds: matrix.personIds, itemIds: reference.itemIds },
    );
    expect(calibrateSpeedAccuracy(records, { model: twoParameterLogistic })).toEqual(
      calibrateSpeedAccuracy(matrix, { model: twoParameterLogistic }),
    );
  });

  it("recovers the correlation, the speed SD and the parameters of simulated 2PL items", () => {
    expectRecovery(twoParameterLogistic, 2, 13);
  });

  it("recovers the correlation, the speed SD and the parameters of simulated graded items", () => {
    expectRecovery(gradedResponse, 3, 14);
  });

  it("estimates the items as calibrate does when ability and speed are uncorrelated, and gives the same estimates in a model list", () => {
    const { matrix } = simulateJoint(twoParameterLogistic, 31, 1200, 5, 2, 0);
    const joint = calibrateSpeedAccuracy(matrix, { model: twoParameterLogistic, ...PRECISE });
    const listed = calibrateSpeedAccuracy(matrix, {
      model: Array.from({ length: 5 }, () => twoParameterLogistic),
      ...PRECISE,
    });
    expect(listed).toEqual(joint);
    const separate = calibrate(matrix, { model: twoParameterLogistic, tolerance: 1e-10 });
    expect(Math.abs(joint.correlation)).toBeLessThan(0.15);
    joint.items.forEach((item, i) => {
      const alone = defined(separate.items[i]);
      expect(Math.abs(item.parameters.a - alone.parameters.a)).toBeLessThan(0.1);
      expect(Math.abs(item.parameters.b - alone.parameters.b)).toBeLessThan(0.1);
    });
    const times = calibrateResponseTimes(matrix, { tolerance: 1e-10 });
    expect(Math.abs(joint.speedSd - times.speedSd)).toBeLessThan(0.01);
  });

  it("refuses standard errors of more than 2,000 estimated parameters, before it calibrates", () => {
    // 600 2PL items of four persons: 1,200 item parameters, 1,200 time parameters, σ_τ and ρ.
    const records = Array.from({ length: 600 }, (_, item) =>
      [0, 1, 2, 3].map((k) => ({
        person: `p${k}`,
        item: `i${item}`,
        response: (item + k) % 2,
        time: 5 + k + (item % 3),
      })),
    ).flat();
    expect(() =>
      calibrateSpeedAccuracy(buildResponseRecords(records), { model: twoParameterLogistic, standardErrors: "hessian" }),
    ).toThrow(
      'Standard errors take at most 2000 estimated parameters, as their information holds every pair of them; found 2402. Leave standardErrors "none", or calibrate fewer items at a time.',
    );
  });

  it("shrinks the estimates with a prior, and leaves the standard errors out unless asked", () => {
    const { matrix } = simulateJoint(twoParameterLogistic, 41, 300, 5, 2, 0.4);
    const plain = calibrateSpeedAccuracy(matrix, { model: twoParameterLogistic });
    const priors = { a: logNormalPrior(0, 0.05) };
    const shrunk = calibrateSpeedAccuracy(matrix, { model: twoParameterLogistic, priors });
    const distance = (calibration: SpeedAccuracyCalibration<LogisticItem>): number =>
      Math.max(...calibration.items.map((item) => Math.abs(item.parameters.a - 1)));
    expect(distance(shrunk)).toBeLessThan(distance(plain));
    expect(plain.correlationStandardError).toBeNull();
    expect(plain.items.every((item) => item.standardErrors === null && item.timeStandardErrors === null)).toBe(true);
  });

  it("uses a response without a time for the responses only, and a person without times for theta only", () => {
    const { matrix } = simulateJoint(twoParameterLogistic, 51, 400, 5, 2, 0.5);
    const times = matrix.times?.map((row, p) =>
      row.map((time, i) => (p % 7 === 0 || (p + i) % 11 === 0 ? null : time)),
    );
    const sparse = buildResponseMatrix(matrix.responses, times === undefined ? {} : { times });
    const fit = calibrateSpeedAccuracy(sparse, { model: twoParameterLogistic });
    expect(fit.converged).toBe(true);
    const estimates = speedAccuracyEstimates(sparse, fit);
    expect(estimates[0]).toMatchObject({ speed: null, speedStandardError: null });
    expect(defined(estimates[0]?.theta)).not.toBeNull();
    expect(defined(estimates[1]?.speed)).not.toBeNull();
  });

  it("rejects data without times, an item with fewer than two times, and an item every person answered correctly", () => {
    const model = twoParameterLogistic;
    expect(() => calibrateSpeedAccuracy(buildResponseMatrix([[1, 0]]), { model })).toThrow(
      "Speed-accuracy calibration needs response times.",
    );
    const rows = [
      [1, 0],
      [0, 1],
      [1, 1],
    ];
    const single = buildResponseMatrix(rows, {
      times: [
        [10, 20],
        [12, null],
        [14, null],
      ].map((row) => row.map((time) => time)),
    });
    expect(() => calibrateSpeedAccuracy(single, { model })).toThrow('Item "item2" needs at least two response times.');
    const constant = buildResponseMatrix(
      [
        [1, 1],
        [1, 0],
        [1, 1],
      ],
      {
        times: [
          [10, 20],
          [12, 25],
          [14, 31],
        ],
      },
    );
    expect(() => calibrateSpeedAccuracy(constant, { model })).toThrow(
      'Item "item1" has no response in category 0; calibration needs responses in every category up to the highest.',
    );
  });

  it("rejects options of the wrong type or value", () => {
    const { matrix } = simulateJoint(twoParameterLogistic, 61, 40, 3, 2, 0.3);
    const model = twoParameterLogistic;
    expect(() => calibrateSpeedAccuracy(matrix, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => calibrateSpeedAccuracy(matrix, JSON.parse("{}"))).toThrow(
      "The option model must be an IRT model such as twoParameterLogistic, found undefined.",
    );
    expect(() => calibrateSpeedAccuracy(matrix, { model, standardErrors: JSON.parse('"exact"') })).toThrow(
      'The option standardErrors must be "none" or "hessian", found "exact".',
    );
    expect(() => calibrateSpeedAccuracy(matrix, { model, tolerance: -1 })).toThrow(DataError);
    expect(() => calibrateSpeedAccuracy(matrix, { model, maxIterations: JSON.parse('"many"') })).toThrow(DataError);
    expect(() => calibrateSpeedAccuracy(matrix, { model, quadrature: { points: 1 } })).toThrow(
      "The quadrature points must be an integer from 2, found 1.",
    );
    expect(() => calibrateSpeedAccuracy(matrix, { model, priors: JSON.parse('{"a": 3}') })).toThrow(DataError);
    expect(() => calibrateSpeedAccuracy(matrix, { model: [model] })).toThrow("Found 1 models for 3 items.");
    expect(() => calibrateSpeedAccuracy(JSON.parse("null"), { model })).toThrow(DataError);
  });

  it("reports that EM has not converged when it runs out of iterations", () => {
    const { matrix } = simulateJoint(twoParameterLogistic, 71, 200, 4, 2, 0.5);
    const result = calibrateSpeedAccuracy(matrix, { model: twoParameterLogistic, tolerance: 1e-14, maxIterations: 3 });
    expect(result.converged).toBe(false);
    expect(result.iterations).toBe(3);
  });
});

describe("speedAccuracyEstimates", () => {
  for (const reference of jointFixture.cases) {
    it(`matches base R's joint EAP of theta and speed at the maximum on ${reference.name}`, () => {
      const estimates = speedAccuracyEstimates(jointData(reference), jointCalibration(reference));
      expect(estimates).toHaveLength(reference.responses.length);
      estimates.forEach((person, p) => {
        const { estimates: expected } = reference;
        expectClose(person.theta, expected.theta[p] ?? null, EAP_TOLERANCE, `${person.id} theta`);
        expectClose(person.thetaStandardError, expected.thetaSd[p] ?? null, EAP_TOLERANCE, `${person.id} theta SE`);
        expectClose(person.speed, expected.speed[p] ?? null, EAP_TOLERANCE, `${person.id} speed`);
        expectClose(person.speedStandardError, expected.speedSd[p] ?? null, EAP_TOLERANCE, `${person.id} speed SE`);
      });
    });
  }

  it("gives the responses-only EAP of theta and the lognormal speed when the correlation is 0", () => {
    const reference = defined(jointFixture.cases[1]);
    const data = jointData(reference);
    const calibration = jointCalibration(reference, 0);
    const timeItems = calibration.items.map(({ id, intensity, discrimination }) => ({ id, intensity, discrimination }));
    const speeds = speedEstimates(data, timeItems, { sd: reference.speedSd });
    speedAccuracyEstimates(data, calibration).forEach((person, p) => {
      const row = defined(data.responses[p]);
      const alone = posteriorMeanEstimate(calibration.items, row);
      expectClose(person.theta, alone.theta, EAP_TOLERANCE, `${person.id} theta`);
      expectClose(person.thetaStandardError, alone.standardError, EAP_TOLERANCE, `${person.id} theta SE`);
      expectClose(person.speed, defined(speeds[p]).speed, EAP_TOLERANCE, `${person.id} speed`);
      expectClose(person.speedStandardError, defined(speeds[p]).standardError, EAP_TOLERANCE, `${person.id} speed SE`);
    });
  });

  it("sharpens theta with the times when ability and speed are correlated", () => {
    const reference = defined(jointFixture.cases[1]);
    const data = jointData(reference);
    const mean = (values: readonly (number | null)[]): number =>
      values.reduce<number>((sum, value) => sum + defined(value), 0) / values.length;
    const joint = speedAccuracyEstimates(data, jointCalibration(reference));
    const alone = speedAccuracyEstimates(data, jointCalibration(reference, 0));
    expect(mean(joint.map(({ thetaStandardError }) => thetaStandardError))).toBeLessThan(
      mean(alone.map(({ thetaStandardError }) => thetaStandardError)),
    );
  });

  it("estimates theta alone for a person without times, and gives nothing to a person without responses", () => {
    const reference = defined(jointFixture.cases[1]);
    const rows = reference.responses.slice(0, 3).map((row) => [...row]);
    const times = reference.times.slice(0, 3).map((row) => [...row]);
    times[0] = times[0]?.map(() => null) ?? [];
    rows[2] = rows[2]?.map(() => null) ?? [];
    times[2] = times[2]?.map(() => null) ?? [];
    const data = buildResponseMatrix(rows, { itemIds: reference.itemIds, times });
    const calibration = jointCalibration(reference);
    const [noTimes, full, nothing] = speedAccuracyEstimates(data, calibration);
    const alone = speedAccuracyEstimates(data, jointCalibration(reference, 0))[0];
    expect(noTimes).toMatchObject({ speed: null, speedStandardError: null });
    expectClose(defined(noTimes).theta, defined(alone).theta, EAP_TOLERANCE, "theta of the person without times");
    expect(defined(full).speed).not.toBeNull();
    expect(nothing).toEqual({
      id: "person3",
      theta: null,
      thetaStandardError: null,
      speed: null,
      speedStandardError: null,
    });
  });

  it("rejects data without times, a calibration that is not one, an unknown item and a response beyond the categories", () => {
    const reference = defined(jointFixture.cases[1]);
    const data = jointData(reference);
    const calibration = jointCalibration(reference);
    expect(() => speedAccuracyEstimates(buildResponseMatrix([[1, 0]]), calibration)).toThrow(
      "Speed-accuracy estimation needs response times.",
    );
    expect(() => speedAccuracyEstimates(data, JSON.parse("null"))).toThrow(DataError);
    expect(() => speedAccuracyEstimates(data, { ...calibration, items: calibration.items.slice(1) })).toThrow(
      'The data have item "s01", which is not among the calibration\'s items.',
    );
    expect(() => speedAccuracyEstimates(data, { ...calibration, items: JSON.parse('"items"') })).toThrow(
      'The calibration needs items, an array, found "items".',
    );
    expect(() => speedAccuracyEstimates(data, { ...calibration, items: JSON.parse("[null]") })).toThrow(
      "Item 1 of the calibration must be an object { id, model, parameters, intensity, discrimination }, found null.",
    );
    expect(() => speedAccuracyEstimates(data, { ...calibration, correlation: 1 })).toThrow(
      "The calibration's correlation must be between -1 and 1, found 1.",
    );
    expect(() => speedAccuracyEstimates(data, { ...calibration, speedSd: 0 })).toThrow(DataError);
    const beyond = buildResponseMatrix([[3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], {
      itemIds: reference.itemIds,
      times: [Array.from({ length: 12 }, () => 30)],
    });
    expect(() => speedAccuracyEstimates(beyond, calibration)).toThrow(
      'The response of person "person1" to item "s01" is 3, but the item has categories 0 to 1.',
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
          const data = buildResponseMatrix(
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

  it("joint estimates do not depend on the order of persons", () => {
    const reference = defined(jointFixture.cases[1]);
    const calibration = jointCalibration(reference);
    fc.assert(
      fc.property(fc.integer({ min: 0, max: 10_000 }), (shift) => {
        const count = 30;
        const order = Array.from({ length: count }, (_, i) => (i * 7 + shift) % count);
        const pick = <T>(rows: readonly T[]): T[] => order.map((p) => defined(rows[p]));
        const options = { itemIds: reference.itemIds };
        const forward = speedAccuracyEstimates(
          buildResponseMatrix(reference.responses.slice(0, count), {
            ...options,
            times: reference.times.slice(0, count),
          }),
          calibration,
        );
        const shuffled = speedAccuracyEstimates(
          buildResponseMatrix(pick(reference.responses.slice(0, count)), {
            ...options,
            times: pick(reference.times.slice(0, count)),
          }),
          calibration,
        );
        order.forEach((p, i) => {
          expectClose(defined(shuffled[i]).theta, defined(forward[p]).theta, 1e-12, "theta");
          expectClose(defined(shuffled[i]).speed, defined(forward[p]).speed, 1e-12, "speed");
        });
      }),
    );
  });

  it("joint estimates are finite with positive standard errors for any responses and times", () => {
    const reference = defined(jointFixture.cases[0]);
    const items = reference.itemIds.slice(0, 3);
    const calibration = { ...jointCalibration(reference), items: jointCalibration(reference).items.slice(0, 3) };
    fc.assert(
      fc.property(
        fc.array(
          fc.tuple(
            fc.array(fc.integer({ min: 0, max: 1 }), { minLength: 3, maxLength: 3 }),
            fc.array(fc.double({ min: 0.5, max: 600, noNaN: true }), { minLength: 3, maxLength: 3 }),
          ),
          { minLength: 1, maxLength: 12 },
        ),
        (persons) => {
          const data = buildResponseMatrix(
            persons.map(([row]) => row),
            { itemIds: items, times: persons.map(([, row]) => row) },
          );
          for (const estimate of speedAccuracyEstimates(data, calibration)) {
            for (const value of [estimate.theta, estimate.speed]) expect(Number.isFinite(value)).toBe(true);
            for (const error of [estimate.thetaStandardError, estimate.speedStandardError]) {
              expect(defined(error)).toBeGreaterThan(0);
            }
          }
        },
      ),
    );
  });

  it("the joint log-likelihood does not depend on the order of persons", () => {
    // 200 persons and 6 items converge within milliseconds for every seed; 60 and 4 leave some seeds short of the
    // tolerance after 50,000 iterations, which take half a minute.
    fc.assert(
      fc.property(fc.integer({ min: 1, max: 1000 }), (seed) => {
        const { matrix } = simulateJoint(twoParameterLogistic, seed, 200, 6, 2, 0.4);
        const order = Array.from({ length: 200 }, (_, i) => (i * 13 + seed) % 200);
        const pick = <T>(rows: readonly T[]): T[] => order.map((p) => defined(rows[p]));
        const shuffled = buildResponseMatrix(pick(matrix.responses), { times: pick(matrix.times ?? []) });
        const options = { model: twoParameterLogistic, ...PRECISE };
        expectClose(
          calibrateSpeedAccuracy(shuffled, options).logLikelihood,
          calibrateSpeedAccuracy(matrix, options).logLikelihood,
          1e-7,
        );
      }),
      { numRuns: 4 },
    );
  });
});

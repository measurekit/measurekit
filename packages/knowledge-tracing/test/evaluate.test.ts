import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  accuracy,
  afm,
  alignPredictions,
  auc,
  bkt,
  buildEventLog,
  calibrationCurve,
  categorizeCurve,
  compareByLearner,
  compareModels,
  createRandom,
  crossValidate,
  DataError,
  type EventLog,
  elo,
  type Fold,
  fitPowerLaw,
  kappa,
  type LearnerModel,
  type LearningCurvePoint,
  learningCurve,
  logLoss,
  type Metrics,
  metrics,
  metricsByOpportunity,
  rmse,
  rSquared,
  splitByItem,
  splitByLearner,
  splitByTime,
} from "../src/index.js";
import { allPredicted, defined, expectAllClose, expectClose } from "./assertions.js";
import { type EloHoldoutFixture, readFixture } from "./fixtures.js";

const holdout = readFixture<EloHoldoutFixture>("elo-holdout.json");

interface EvaluateAgreementFixture {
  cases: {
    name: string;
    predictions: number[];
    outcomes: number[];
    count: number;
    auc: number | null;
    rmse: number;
    logLoss: number;
    accuracy: number | null;
    rSquared: number | null;
    kappa: number | null;
  }[];
}
const agreement = readFixture<EvaluateAgreementFixture>("evaluate-agreement.json");

interface EvaluateComparisonFixture {
  events: { learner: string; item: string; skills: never[]; time: number; outcome: number | null }[];
  predictions: { first: (number | null)[]; second: (number | null)[] };
  cases: {
    name: string;
    folds: { train: number[]; test: number[] }[];
    tests: {
      loss: string;
      learners: number;
      meanDifference: number;
      standardError: number;
      statistic: number;
      degreesOfFreedom: number;
      pValue: number;
    }[];
  }[];
}
const comparison = readFixture<EvaluateComparisonFixture>("evaluate-comparison.json");

interface EvaluateCurvesFixture {
  cases: {
    name: string;
    points: { opportunity: number; count: number; errorRate: number }[];
    powerLaw: { scale: number; exponent: number; rSquared: number | null } | null;
    categories: {
      options: {
        minLearners: number;
        minPoints: number;
        slope: number;
        minSlope: number;
        lowError?: number;
        highError?: number;
      };
      category: string;
    }[];
  }[];
}
const powerLaws = readFixture<EvaluateCurvesFixture>("evaluate-curves.json");

interface EvaluateLearningCurveFixture {
  events: { learner: string; item: string; skills: string[]; time: number; outcome: number }[];
  predictions: number[];
  overall: { opportunity: number; count: number; errorRate: number; predictedErrorRate: number }[];
  skills: {
    a: { opportunity: number; count: number; errorRate: number; predictedErrorRate: number }[];
    b: { opportunity: number; count: number; errorRate: number; predictedErrorRate: number }[];
    c: { opportunity: number; count: number; errorRate: number; predictedErrorRate: number }[];
  };
}
const curves = readFixture<EvaluateLearningCurveFixture>("evaluate-learning-curve.json");

interface EvaluateMetricsFixture {
  cases: {
    name: string;
    predictions: number[];
    outcomes: number[];
    auc: number | null;
    rmse: number;
    logLoss: number;
    accuracy: number | null;
    calibration: {
      lower: number;
      upper: number;
      count: number;
      meanPrediction: number | null;
      meanOutcome: number | null;
    }[];
    calibration4: {
      lower: number;
      upper: number;
      count: number;
      meanPrediction: number | null;
      meanOutcome: number | null;
    }[];
  }[];
}
const fixture = readFixture<EvaluateMetricsFixture>("evaluate-metrics.json");

interface EvaluateOpportunityFixture {
  events: { learner: string; item: string; skills: string[]; time: number; outcome: number | null }[];
  predictions: (number | null)[];
  curves: {
    minOpportunities: number;
    curve: {
      overall: {
        opportunity: number;
        count: number;
        errorRate: number;
        predictedErrorRate: number;
        metrics: {
          count: number;
          auc: number | null;
          rmse: number;
          logLoss: number;
          accuracy: number;
          rSquared: number | null;
          kappa: number;
        };
      }[];
      skills: {
        a: {
          opportunity: number;
          count: number;
          errorRate: number;
          predictedErrorRate: number;
          metrics: {
            count: number;
            auc: number | null;
            rmse: number;
            logLoss: number;
            accuracy: number;
            rSquared: number | null;
            kappa: number | null;
          };
        }[];
        b: {
          opportunity: number;
          count: number;
          errorRate: number;
          predictedErrorRate: number;
          metrics: {
            count: number;
            auc: number | null;
            rmse: number;
            logLoss: number;
            accuracy: number;
            rSquared: number | null;
            kappa: number;
          };
        }[];
        c: {
          opportunity: number;
          count: number;
          errorRate: number;
          predictedErrorRate: number;
          metrics: {
            count: number;
            auc: number | null;
            rmse: number;
            logLoss: number;
            accuracy: number;
            rSquared: number | null;
            kappa: number;
          };
        }[];
      };
    };
  }[];
}
const opportunities = readFixture<EvaluateOpportunityFixture>("evaluate-opportunity.json");

interface EvaluateUnseenFixture {
  events: { learner: string; item: string; skills: string[]; time: number; outcome: number | null }[];
  folds: { train: number[]; test: number[] }[];
  throw: {
    folds: {
      predictions: (number | null)[];
      skipped: number;
      metrics: {
        count: number;
        auc: number;
        rmse: number;
        logLoss: number;
        accuracy: number;
        rSquared: number;
        kappa: number;
      };
    }[];
    metrics: {
      count: number;
      auc: number;
      rmse: number;
      logLoss: number;
      accuracy: number;
      rSquared: number;
      kappa: number;
    };
  };
  skip: {
    folds: {
      predictions: (number | null)[];
      skipped: number;
      metrics: {
        count: number;
        auc: number;
        rmse: number;
        logLoss: number;
        accuracy: number;
        rSquared: number;
        kappa: number;
      };
    }[];
    metrics: {
      count: number;
      auc: number;
      rmse: number;
      logLoss: number;
      accuracy: number;
      rSquared: number;
      kappa: number;
    };
  };
}
const unseen = readFixture<EvaluateUnseenFixture>("evaluate-unseen.json");

/** Absolute tolerance of metrics, calibration bins and learning curves against R. */
const METRIC_TOLERANCE = 1e-12;
/** Absolute tolerance of held-out Elo predictions against R. */
const PREDICTION_TOLERANCE = 1e-12;
/** Absolute tolerance of the test between models against `t.test`. */
const TEST_TOLERANCE = 1e-9;
/** Absolute tolerance of power laws against `lm`. */
const CURVE_TOLERANCE = 1e-10;

/** A model written outside the package from its steps; there is nothing for its fit to estimate. */
function fromSteps<TState>(
  steps: Pick<LearnerModel<TState>, "initialState" | "update" | "predict">,
): LearnerModel<TState> {
  const model: LearnerModel<TState> = {
    ...steps,
    parameters: null,
    fit(log) {
      const states = new Map<string, TState>();
      const predictions = log.map((event) => {
        const state = states.get(event.learner) ?? steps.initialState(event.learner);
        states.set(event.learner, steps.update(state, event));
        return event.outcome === null ? null : steps.predict(state, event);
      });
      return {
        model,
        states: Object.fromEntries(states),
        predictions,
        logLikelihood: 0,
        aic: null,
        bic: null,
        converged: true,
        iterations: 1,
      };
    },
  };
  return model;
}

/** The answers and correct answers of a learner so far. */
interface Tally {
  readonly answered: number;
  readonly correct: number;
}

/** P(correct) = (1 + correct) / (2 + answered) over the learner's own events with an outcome, whatever the skill. */
function laplace(): LearnerModel<Tally> {
  return fromSteps<Tally>({
    initialState: () => ({ answered: 0, correct: 0 }),
    update: (state, event) =>
      event.outcome === null ? state : { answered: state.answered + 1, correct: state.correct + event.outcome },
    predict: (state) => (1 + state.correct) / (2 + state.answered),
  });
}

/** A model that predicts what a table gives for the time of the attempt, whoever the learner is. */
function scripted(table: readonly (number | null)[]): LearnerModel<null> {
  return fromSteps<null>({
    initialState: () => null,
    update: () => null,
    predict: (_, attempt) => defined(table[attempt.time - 1]),
  });
}

/** Returns the folds a fixture lists, as indices into its log. */
function foldsOf(
  log: EventLog,
  indices: readonly { readonly train: readonly number[]; readonly test: readonly number[] }[],
): Fold[] {
  return indices.map((fold) => ({
    train: fold.train.map((index) => defined(log[index])),
    test: fold.test.map((index) => defined(log[index])),
  }));
}

/** A model written outside the package: every learner answers correctly at the rate of the training log. */
function baseRate(rate = 0.5): LearnerModel<string, { rate: number }> {
  return {
    parameters: { rate },
    fit(log) {
      const model = baseRate(log.reduce((sum, event) => sum + defined(event.outcome), 0) / log.length);
      const predictions = log.map(() => model.parameters.rate);
      const logLikelihood = log.reduce(
        (sum, event) => sum + Math.log(event.outcome === 1 ? model.parameters.rate : 1 - model.parameters.rate),
        0,
      );
      return {
        model,
        states: Object.fromEntries(log.map((event) => [event.learner, event.learner])),
        predictions,
        logLikelihood,
        aic: 2 - 2 * logLikelihood,
        bic: Math.log(log.length) - 2 * logLikelihood,
        converged: true,
        iterations: 1,
      };
    },
    initialState: (learner) => learner,
    update: (state) => state,
    predict: () => rate,
  };
}

/** The metrics of a reference. */
interface ReferenceMetrics {
  readonly count: number;
  readonly auc: number | null;
  readonly rmse: number;
  readonly logLoss: number;
  readonly accuracy: number | null;
  readonly rSquared: number | null;
  readonly kappa: number | null;
}

/** A point of a reference learning curve. */
interface ReferencePoint {
  readonly opportunity: number;
  readonly count: number;
  readonly errorRate: number;
  readonly predictedErrorRate: number | null;
}

/** Expects metrics to be the reference's, or both to be `null`. */
function expectMetrics(actual: Metrics | null, expected: ReferenceMetrics | null, label: string): void {
  if (expected === null) {
    expect(actual, label).toBeNull();
    return;
  }
  const table = defined(actual);
  expect(table.count, label).toBe(expected.count);
  for (const key of ["auc", "rmse", "logLoss", "accuracy", "rSquared", "kappa"] as const) {
    expectClose(table[key], expected[key], METRIC_TOLERANCE, `${label} ${key}`);
  }
}

describe("metrics against R", () => {
  it.each(fixture.cases)("match pROC and base R: $name", (reference) => {
    const { predictions, outcomes } = reference;
    const table = metrics(predictions, outcomes);
    expect(table.count).toBe(predictions.length);
    expectClose(table.auc, reference.auc, METRIC_TOLERANCE, "AUC");
    expectClose(table.rmse, reference.rmse, METRIC_TOLERANCE, "RMSE");
    expectClose(table.logLoss, reference.logLoss, METRIC_TOLERANCE, "log-loss");
    expectClose(table.accuracy, reference.accuracy, METRIC_TOLERANCE, "accuracy");
    expectClose(rmse(predictions, outcomes), reference.rmse, METRIC_TOLERANCE, "rmse()");
    expectClose(logLoss(predictions, outcomes), reference.logLoss, METRIC_TOLERANCE, "logLoss()");
    if (reference.auc !== null) expectClose(auc(predictions, outcomes), reference.auc, METRIC_TOLERANCE, "auc()");
    if (reference.accuracy !== null) {
      expectClose(accuracy(predictions, outcomes), reference.accuracy, METRIC_TOLERANCE, "accuracy()");
    }
    for (const [bins, expected] of [
      [10, reference.calibration],
      [4, reference.calibration4],
    ] as const) {
      const table = calibrationCurve(predictions, outcomes, { binCount: bins });
      expect(table).toHaveLength(bins);
      table.forEach((bin, index) => {
        const target = defined(expected[index]);
        expect([bin.lower, bin.upper, bin.count]).toEqual([target.lower, target.upper, target.count]);
        expectClose(bin.meanPrediction, target.meanPrediction, METRIC_TOLERANCE, `bin ${index}`);
        expectClose(bin.meanOutcome, target.meanOutcome, METRIC_TOLERANCE, `bin ${index}`);
      });
    }
  });
});

describe("R² and kappa against R", () => {
  it.each(agreement.cases)("match base R and irr: $name", (reference) => {
    const { predictions, outcomes } = reference;
    expectMetrics(metrics(predictions, outcomes), reference, "metrics");
    if (reference.rSquared === null) expect(() => rSquared(predictions, outcomes)).toThrow(DataError);
    else expectClose(rSquared(predictions, outcomes), reference.rSquared, METRIC_TOLERANCE, "rSquared()");
    if (reference.kappa === null) expect(() => kappa(predictions, outcomes)).toThrow(DataError);
    else expectClose(kappa(predictions, outcomes), reference.kappa, METRIC_TOLERANCE, "kappa()");
  });

  it("explain nothing with the mean outcome, and less than nothing with worse predictions", () => {
    expect(rSquared([0.5, 0.5, 0.5, 0.5], [1, 0, 0.5, 0.5])).toBe(0);
    expect(rSquared([0.5, 0.5], [0, 1])).toBe(0);
    expect(rSquared([1, 0], [1, 0])).toBe(1);
    expect(rSquared([0, 1], [1, 0])).toBe(-3);
  });

  it("give a kappa of 1 for predictions that agree with the outcomes, and -1 for the opposite", () => {
    expect(kappa([0.9, 0.1, 0.8, 0.2], [1, 0, 1, 0])).toBe(1);
    expect(kappa([0.1, 0.9, 0.2, 0.8], [1, 0, 1, 0])).toBe(-1);
  });

  it("leave R² and kappa undefined where they have no meaning: a DataError, or null in the table", () => {
    expect(() => rSquared([0.2, 0.9], [1, 1])).toThrow("R² needs outcomes that are not all equal; all 2 are 1.");
    expect(() => kappa([0.5], [0.5])).toThrow("Kappa needs outcomes of 0 or 1, not partial credit.");
    expect(() => kappa([0.9, 0.8], [1, 1])).toThrow(
      "Kappa is undefined when every prediction and every outcome is correct.",
    );
    expect(() => kappa([0.1, 0.2], [0, 0])).toThrow(
      "Kappa is undefined when every prediction and every outcome is incorrect.",
    );
    expect(metrics([0.5, 0.2], [0.5, 0.5])).toMatchObject({ rSquared: null, kappa: null });
  });

  it("keep kappa within -1 and 1, and R² at most 1", () => {
    const rows = fc.array(fc.record({ p: fc.double({ min: 0, max: 1, noNaN: true }), y: fc.constantFrom(0, 1) }), {
      minLength: 1,
      maxLength: 60,
    });
    fc.assert(
      fc.property(rows, (data) => {
        const table = metrics(
          data.map((row) => row.p),
          data.map((row) => row.y),
        );
        if (table.kappa !== null) expect(Math.abs(table.kappa)).toBeLessThanOrEqual(1 + 1e-12);
        if (table.rSquared !== null) expect(table.rSquared).toBeLessThanOrEqual(1 + 1e-12);
      }),
    );
  });
});

describe("metrics", () => {
  it("leave AUC unchanged under increasing transformations of the predictions", () => {
    const cases = fc
      .array(fc.record({ step: fc.integer({ min: 1, max: 999 }), outcome: fc.constantFrom(0, 1) }), {
        minLength: 2,
        maxLength: 80,
      })
      .filter((rows) => new Set(rows.map((row) => row.outcome)).size === 2);
    const transforms = [(p: number) => p ** 3, (p: number) => Math.sqrt(p), (p: number) => 0.2 + 0.5 * p];
    fc.assert(
      fc.property(cases, fc.constantFrom(...transforms), (rows, transform) => {
        const predictions = rows.map((row) => row.step / 1000);
        const outcomes = rows.map((row) => row.outcome);
        expect(auc(predictions.map(transform), outcomes)).toBe(auc(predictions, outcomes));
      }),
    );
  });

  it("give finite results from 0 to 1 for AUC and accuracy, never NaN", () => {
    const rows = fc.array(
      fc.record({ p: fc.double({ min: 0.001, max: 0.999, noNaN: true }), y: fc.constantFrom(0, 1, 0.5) }),
      { minLength: 1, maxLength: 50 },
    );
    fc.assert(
      fc.property(rows, (data) => {
        const table = metrics(
          data.map((row) => row.p),
          data.map((row) => row.y),
        );
        for (const value of [table.rmse, table.logLoss]) expect(Number.isFinite(value)).toBe(true);
        for (const value of [table.auc, table.accuracy]) {
          if (value !== null) expect(value >= 0 && value <= 1).toBe(true);
        }
      }),
    );
  });

  it("leave AUC undefined when the outcomes are all correct or all wrong: a DataError, or null in the table", () => {
    expect(() => auc([0.2, 0.9], [1, 1])).toThrow("AUC needs both correct and incorrect outcomes; all 2 are 1.");
    expect(() => auc([0.2, 0.9], [0, 0])).toThrow("AUC needs both correct and incorrect outcomes; all 2 are 0.");
    expect(metrics([0.2, 0.9], [1, 1])).toEqual({
      count: 2,
      auc: null,
      rmse: rmse([0.2, 0.9], [1, 1]),
      logLoss: logLoss([0.2, 0.9], [1, 1]),
      accuracy: 0.5,
      rSquared: null,
      kappa: 0,
    });
  });

  it("clip a certain prediction to the machine epsilon in log-loss, as scikit-learn's log_loss does", () => {
    // A wrong one costs -ln 2^-52 = 52 ln 2, a right one -ln(1 - 2^-52), which is 2^-52 to double precision.
    expectClose(logLoss([1, 0.5], [0, 1]), (52 * Math.LN2 + Math.LN2) / 2, 1e-14);
    expectClose(logLoss([0, 0.5], [1, 0]), (52 * Math.LN2 + Math.LN2) / 2, 1e-14);
    expectClose(logLoss([1, 0], [1, 0]), Number.EPSILON, 1e-30);
    expectClose(logLoss([1e-300], [0.5]), (52 * Math.LN2) / 2, 1e-14);
  });

  it("keep the log-loss of a logistic model finite where its prediction rounds to 1", () => {
    // AFM with a slope of 1 a practice predicts logistic(40) after 40 attempts, which is 1 in double precision.
    const events = Array.from({ length: 41 }, (_, time) => ({
      learner: "ana",
      item: "q",
      skills: ["fractions"],
      time,
      outcome: time < 40 ? 1 : 0,
    }));
    const model = afm({
      proficiency: "none",
      parameters: { skills: { fractions: { intercept: 0, coefficients: { opportunities: 1 } } } },
    });
    let state = model.initialState("ana");
    for (const event of events.slice(0, 40)) state = model.update(state, event);
    const prediction = model.predict(state, defined(events[40]));
    expect(prediction).toBe(1);
    expectClose(logLoss([prediction], [0]), 52 * Math.LN2, 1e-14);
  });

  it("predict a correct outcome above 0.5 only", () => {
    expect(accuracy([0.5, 0.51, 0.49], [0, 1, 0])).toBe(1);
  });

  it.each([
    [() => auc([0.5], [0.5]), "AUC needs outcomes of 0 or 1, not partial credit."],
    [() => accuracy([0.5], [0.5]), "Accuracy needs outcomes of 0 or 1, not partial credit."],
    [() => rmse([], []), "Metrics need at least one prediction."],
    [() => rmse([0.5], [1, 0]), "Found 1 predictions and 2 outcomes."],
    [() => logLoss([1.2], [1]), "Prediction 1 must be a probability from 0 to 1, found 1.2."],
    [() => metrics([0.5, Number.NaN], [1, 1]), "Prediction 2 must be a probability from 0 to 1, found NaN."],
    [() => metrics([0.5], [-1]), "Outcome 1 must be from 0 to 1, found -1."],
    [() => calibrationCurve([0.5], [1], { binCount: 0 }), "The number of bins must be a positive integer, found 0."],
  ])("reject malformed input %#", (call, message) => {
    expect(call).toThrow(DataError);
    expect(call).toThrow(message);
  });
});

describe("calibrationCurve", () => {
  it("puts a prediction on a bin bound into the upper bin, and 1 into the last bin", () => {
    const table = calibrationCurve([0, 0.3, 0.7, 1], [0, 1, 1, 1]);
    expect(table.map((bin) => bin.count)).toEqual([1, 0, 0, 1, 0, 0, 0, 1, 0, 1]);
    expect(table[1]).toEqual({ lower: 0.1, upper: 0.2, count: 0, meanPrediction: null, meanOutcome: null });
  });
});

describe("crossValidate", () => {
  const log = buildEventLog(holdout.events);
  const at = (indices: readonly number[]): EventLog => indices.map((index) => defined(log[index]));

  it.each(holdout.folds)("predicts the held-out events as R does: $name", (reference) => {
    const folds: Fold[] = [{ train: at(reference.train), test: at(reference.test) }];
    for (const [name, model] of [
      ["basic", elo()],
      ["skills", elo({ skills: true })],
    ] as const) {
      const evaluation = crossValidate(model, folds);
      expectAllClose(
        allPredicted(defined(evaluation.folds[0]).predictions),
        reference.predictions[name],
        PREDICTION_TOLERANCE,
        name,
      );
    }
  });

  it("pools the predictions of every fold into the overall metrics", () => {
    const folds = splitByLearner(log, 3, createRandom(1));
    const evaluation = crossValidate(elo(), folds);
    expect(evaluation.folds).toHaveLength(3);
    expect(evaluation.metrics).toEqual(
      metrics(
        evaluation.folds.flatMap((fold) => allPredicted(fold.predictions)),
        folds.flatMap((fold) => fold.test.map((event) => defined(event.outcome))),
      ),
    );
    evaluation.folds.forEach((fold, index) => {
      expect(fold.predictions).toHaveLength(defined(folds[index]).test.length);
    });
  });

  it("evaluates a model written outside the package", () => {
    const evaluation = crossValidate(baseRate(0.5), [{ train: at([0, 1, 2, 3]), test: at([4, 5]) }]);
    const rate = at([0, 1, 2, 3]).reduce((sum, event) => sum + defined(event.outcome), 0) / 4;
    expect(defined(evaluation.folds[0]).predictions).toEqual([rate, rate]);
  });

  it("rejects no folds, a fold that is not an object and a malformed test log", () => {
    expect(() => crossValidate(elo(), [])).toThrow("Cross-validation needs at least one fold.");
    expect(() => crossValidate(elo(), [{ train: log, test: [] }])).toThrow("A log needs at least one event.");
    expect(() => crossValidate(elo(), JSON.parse("[null]"))).toThrow(
      "Fold 1 must be an object { train, test } of two logs, found null.",
    );
  });
});

describe("crossValidate with skills the training log has no outcome of", () => {
  const log = buildEventLog(unseen.events);
  const folds = foldsOf(log, unseen.folds);

  it("splits the curriculum by time at the blocks that R forms", () => {
    expect(splitByTime(log, 4)).toEqual(folds);
  });

  it.each(["throw", "skip"] as const)("predicts as R does with unseenSkills %s", (mode) => {
    const reference = unseen[mode];
    const evaluation = crossValidate(laplace(), folds, { unseenSkills: mode });
    expect(evaluation.folds).toHaveLength(reference.folds.length);
    evaluation.folds.forEach((fold, index) => {
      const expected = defined(reference.folds[index]);
      expect(fold.skipped, `skipped ${index}`).toBe(expected.skipped);
      expect(fold.predictions.map((p) => p === null)).toEqual(expected.predictions.map((p) => p === null));
      expectAllClose(
        fold.predictions.map((p) => p ?? 0),
        expected.predictions.map((p) => p ?? 0),
        PREDICTION_TOLERANCE,
        `fold ${index}`,
      );
      expectMetrics(fold.metrics, expected.metrics, `fold ${index}`);
    });
    expectMetrics(evaluation.metrics, reference.metrics, "pooled");
  });

  it("skips a different share of the test events from the folds, only under skip", () => {
    const skipped = (mode: "throw" | "skip") =>
      crossValidate(laplace(), folds, { unseenSkills: mode }).folds.map((fold) => fold.skipped);
    expect(skipped("throw")).toEqual([0, 0, 0, 0]);
    expect(skipped("skip").every((count, index) => count === defined(unseen.skip.folds[index]).skipped)).toBe(true);
    expect(skipped("skip").reduce((sum, count) => sum + count, 0)).toBeGreaterThan(0);
    expect(crossValidate(laplace(), folds)).toEqual(crossValidate(laplace(), folds, { unseenSkills: "throw" }));
  });

  it("lets BKT and AFM, which have parameters for the seen skills only, be evaluated by skipping the others", () => {
    const evaluateBySkipping = <TState>(model: LearnerModel<TState>): void => {
      expect(() => crossValidate(model, folds)).toThrow(DataError);
      const evaluation = crossValidate(model, folds, { unseenSkills: "skip" });
      expect(evaluation.folds.map((fold) => fold.skipped)).toEqual(unseen.skip.folds.map((fold) => fold.skipped));
      expect(evaluation.metrics.count).toBe(unseen.skip.metrics.count);
    };
    evaluateBySkipping(bkt());
    evaluateBySkipping(afm());
  });

  it("predicts a skill new to a test block with Elo, which starts it from its initial ability", () => {
    const evaluation = crossValidate(elo({ skills: true }), folds);
    expect(evaluation.metrics.count).toBe(unseen.throw.metrics.count);
  });

  it("counts a skill met only through a lesson as unseen, and an event without skills as seen", () => {
    const small = buildEventLog([
      { learner: "a", item: "video", skills: ["s"], time: 1, outcome: null },
      { learner: "a", item: "q1", skills: ["t"], time: 2, outcome: 1 },
      { learner: "a", item: "q2", skills: ["s"], time: 3, outcome: 1 },
      { learner: "a", item: "q3", skills: [], time: 4, outcome: 0 },
    ]);
    const fold = { train: small.slice(0, 2), test: small.slice(2) };
    const evaluation = crossValidate(laplace(), [fold], { unseenSkills: "skip" });
    // The skipped event leaves the learner's state: 1 correct of 1 answered, not 2 of 2.
    expect(defined(evaluation.folds[0])).toEqual({
      predictions: [null, 2 / 3],
      metrics: metrics([2 / 3], [0]),
      skipped: 1,
    });
  });

  it("refuses a run whose test events with an outcome are all skipped", () => {
    const small = buildEventLog([
      { learner: "a", item: "q1", skills: ["t"], time: 1, outcome: 1 },
      { learner: "a", item: "q2", skills: ["s"], time: 2, outcome: 1 },
      { learner: "a", item: "q3", skills: ["s"], time: 3, outcome: null },
    ]);
    expect(() =>
      crossValidate(laplace(), [{ train: small.slice(0, 1), test: small.slice(1) }], { unseenSkills: "skip" }),
    ).toThrow(
      "Cross-validation needs a test event with a prediction; unseenSkills skipped all 1 test events with an outcome.",
    );
  });

  it("gives every model of a comparison the same skipped events", () => {
    const rows = compareModels({ laplace: laplace(), elo: elo({ skills: true }) }, folds, { unseenSkills: "skip" });
    for (const row of rows)
      expect(row.folds.map((fold) => fold.skipped)).toEqual(unseen.skip.folds.map((f) => f.skipped));
    const [first, second] = rows;
    first?.folds.forEach((fold, index) => {
      expect(fold.predictions.map((p) => p === null)).toEqual(
        defined(second?.folds[index]).predictions.map((p) => p === null),
      );
    });
  });

  it("rejects an option that is not one of the choices", () => {
    expect(() => crossValidate(laplace(), folds, { unseenSkills: JSON.parse('"drop"') })).toThrow(
      'The option unseenSkills must be "throw" or "skip", found "drop".',
    );
    expect(() => crossValidate(laplace(), folds, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => crossValidate(laplace(), folds, { unseenSkills: JSON.parse("null") })).toThrow(DataError);
  });
});

describe("compareModels", () => {
  it("evaluates every model on the same folds, in the order given", () => {
    const folds = splitByLearner(buildEventLog(holdout.events), 2, createRandom(3));
    const comparison = compareModels(
      { elo: elo(), "concept elo": elo({ skills: true }), "base rate": baseRate() },
      folds,
    );
    expect(comparison.map((row) => row.name)).toEqual(["elo", "concept elo", "base rate"]);
    expect(comparison[0]).toEqual({ name: "elo", ...crossValidate(elo(), folds) });
    expect(comparison[1]).toEqual({ name: "concept elo", ...crossValidate(elo({ skills: true }), folds) });
  });

  it("rejects an empty comparison", () => {
    expect(() => compareModels({}, [])).toThrow(
      "A comparison needs at least one model, as an object of models by name.",
    );
  });
});

describe("compareByLearner", () => {
  const log = buildEventLog(comparison.events);
  const first = scripted(comparison.predictions.first);
  const second = scripted(comparison.predictions.second);

  it.each(comparison.cases)("matches t.test over the learners' mean losses: $name", (reference) => {
    const folds = foldsOf(log, reference.folds);
    const one = crossValidate(first, folds);
    const two = crossValidate(second, folds);
    for (const expected of reference.tests) {
      const result = compareByLearner(one, two, folds, { loss: expected.loss === "squared" ? "squared" : "log" });
      const label = expected.loss;
      expect(result.learners, label).toBe(expected.learners);
      expect(result.degreesOfFreedom, label).toBe(expected.degreesOfFreedom);
      expectClose(result.meanDifference, expected.meanDifference, TEST_TOLERANCE, `${label} difference`);
      expectClose(result.standardError, expected.standardError, TEST_TOLERANCE, `${label} standard error`);
      expectClose(result.statistic, expected.statistic, TEST_TOLERANCE, `${label} statistic`);
      expectClose(result.pValue, expected.pValue, TEST_TOLERANCE, `${label} p-value`);
    }
  });

  it("compares by log loss unless asked for the squared one", () => {
    const folds = foldsOf(log, defined(comparison.cases[0]).folds);
    const one = crossValidate(first, folds);
    const two = crossValidate(second, folds);
    expect(compareByLearner(one, two, folds)).toEqual(compareByLearner(one, two, folds, { loss: "log" }));
    expect(compareByLearner(one, two, folds)).not.toEqual(compareByLearner(one, two, folds, { loss: "squared" }));
  });

  it("negates the difference and the statistic, and keeps the p-value, when the models swap", () => {
    const folds = foldsOf(log, defined(comparison.cases[1]).folds);
    const one = crossValidate(first, folds);
    const two = crossValidate(second, folds);
    const forward = compareByLearner(one, two, folds);
    const backward = compareByLearner(two, one, folds);
    expectClose(backward.meanDifference, -defined(forward.meanDifference), 1e-14);
    expectClose(backward.statistic, -defined(forward.statistic), 1e-12);
    expectClose(backward.standardError, forward.standardError, 1e-14);
    expectClose(backward.pValue, forward.pValue, 1e-14);
  });

  it("leaves a model's test against itself without a statistic, with a difference and error of 0", () => {
    const folds = foldsOf(log, defined(comparison.cases[0]).folds);
    const one = crossValidate(first, folds);
    expect(compareByLearner(one, one, folds)).toEqual({
      learners: 40,
      meanDifference: 0,
      standardError: 0,
      statistic: null,
      degreesOfFreedom: 39,
      pValue: null,
    });
  });

  describe("on three events", () => {
    const small = buildEventLog([
      { learner: "a", item: "q", skills: [], time: 1, outcome: 1 },
      { learner: "b", item: "q", skills: [], time: 2, outcome: 1 },
      { learner: "c", item: "q", skills: [], time: 3, outcome: 0 },
    ]);
    const fold = { train: small, test: small };
    const evaluate = (table: readonly number[]) => crossValidate(scripted(table), [fold]);

    it("tests squared losses by hand: differences of 0.75, 0 and 0 have a mean of 0.25 and an error of 0.25", () => {
      const result = compareByLearner(evaluate([0, 0.5, 0.5]), evaluate([0.5, 0.5, 0.5]), [fold], { loss: "squared" });
      expect(result.learners).toBe(3);
      expectClose(result.meanDifference, 0.25, 1e-15);
      expectClose(result.standardError, 0.25, 1e-15);
      expectClose(result.statistic, 1, 1e-14);
      // P(|T| > 1) with 2 degrees of freedom is 1 - 1 / √3.
      expectClose(result.pValue, 1 - 1 / Math.sqrt(3), 1e-12);
    });

    it("tests a certain prediction that is wrong at the log loss logLoss gives it, 52 ln 2", () => {
      // Differences of 51 ln 2, 0 and 0 have a mean of 17 ln 2 and an error of 17 ln 2, so t = 1.
      const result = compareByLearner(evaluate([0, 0.5, 0.5]), evaluate([0.5, 0.5, 0.5]), [fold]);
      expect(result.learners).toBe(3);
      expectClose(result.meanDifference, 17 * Math.LN2, 1e-14);
      expectClose(result.standardError, 17 * Math.LN2, 1e-14);
      expectClose(result.statistic, 1, 1e-14);
      expectClose(result.pValue, 1 - 1 / Math.sqrt(3), 1e-12);
    });

    it("has a difference but no variance for one learner", () => {
      const alone = { train: small.slice(0, 1), test: small.slice(0, 1) };
      const result = compareByLearner(
        crossValidate(scripted([0.9]), [alone]),
        crossValidate(scripted([0.6]), [alone]),
        [alone],
        { loss: "squared" },
      );
      expect(result).toEqual({
        learners: 1,
        meanDifference: expect.closeTo(0.01 - 0.16, 14),
        standardError: null,
        statistic: null,
        degreesOfFreedom: 0,
        pValue: null,
      });
    });

    it("uses the events that both models predicted, and refuses when there are none", () => {
      const evaluation = (predictions: readonly (number | null)[]) => ({
        folds: [{ predictions, metrics: null, skipped: predictions.filter((p) => p === null).length }],
        metrics: metrics([0.5], [1]),
      });
      const result = compareByLearner(evaluation([0.9, null, 0.2]), evaluation([0.5, 0.5, null]), [fold], {
        loss: "squared",
      });
      expect(result.learners).toBe(1);
      expectClose(result.meanDifference, 0.01 - 0.25, 1e-14);
      expect(() => compareByLearner(evaluation([null, null, null]), evaluation([0.5, 0.5, 0.5]), [fold])).toThrow(
        "A comparison needs a test event with an outcome that both models predicted; found none.",
      );
    });
  });

  it("rejects evaluations that do not belong to the folds, and options that are not valid", () => {
    const folds = foldsOf(log, defined(comparison.cases[1]).folds);
    const one = crossValidate(first, folds);
    const two = crossValidate(second, folds);
    expect(() => compareByLearner(one, two, folds.slice(0, 1))).toThrow(
      "The first evaluation has 2 folds, but 1 were given.",
    );
    expect(() =>
      compareByLearner(
        one,
        two,
        folds.map((fold) => ({ ...fold, test: fold.test.slice(1) })),
      ),
    ).toThrow(
      `Found ${defined(one.folds[0]).predictions.length} predictions for ${defined(folds[0]).test.length - 1} events.`,
    );
    expect(() => compareByLearner(one, two, folds, { loss: JSON.parse('"hinge"') })).toThrow(
      'The option loss must be "log" or "squared", found "hinge".',
    );
    expect(() => compareByLearner(one, two, folds, JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
    expect(() => compareByLearner(JSON.parse("null"), two, folds)).toThrow(
      "The first evaluation must be an object returned by crossValidate, found null.",
    );
    expect(() => compareByLearner(one, JSON.parse('{"folds": 1}'), folds)).toThrow(
      "The second evaluation must have an array of folds, found 1.",
    );
    expect(() => compareByLearner(one, two, JSON.parse('"folds"'))).toThrow(
      "The folds must be an array of folds, found",
    );
  });
});

describe("comparing Q-matrices", () => {
  const items = ["q1", "q2", "q3", "q4"];
  const events = ["ana", "budi", "citra", "dewi", "eka", "fiona"].flatMap((learner, i) =>
    items.map((item, j) => ({ learner, item, time: 10 * i + j, outcome: (i * 3 + j * 5) % 7 < 4 ? 1 : 0 })),
  );
  // The same items with two readings of their skills.
  const coarse = { q1: ["math"], q2: ["math"], q3: ["math"], q4: ["math"] };
  const fine = { q1: ["add"], q2: ["add"], q3: ["multiply"], q4: ["multiply"] };

  it("compares two logs of the same events by the learners, with one seed for the folds", () => {
    const [coarseLog, fineLog] = [coarse, fine].map((qMatrix) => buildEventLog(events, { qMatrix }));
    const coarseFolds = splitByLearner(defined(coarseLog), 3, createRandom(7));
    const fineFolds = splitByLearner(defined(fineLog), 3, createRandom(7));
    expect(coarseFolds.map((fold) => fold.test.map((event) => event.learner))).toEqual(
      fineFolds.map((fold) => fold.test.map((event) => event.learner)),
    );
    const model = { elo: elo({ skills: true }) };
    const [coarseRow] = compareModels(model, coarseFolds);
    const [fineRow] = compareModels(model, fineFolds);
    const result = compareByLearner(defined(coarseRow), defined(fineRow), coarseFolds);
    expect(result.learners).toBe(6);
    expect(result.degreesOfFreedom).toBe(5);
  });
});

describe("learningCurve", () => {
  const log = buildEventLog(curves.events);

  it("matches base R's error rates by opportunity, over all skills and per skill", () => {
    const curve = learningCurve(log, { predictions: curves.predictions });
    const expectPoints = (actual: typeof curve.overall, expected: typeof curves.overall, label: string): void => {
      expect(
        actual.map((point) => [point.opportunity, point.count]),
        label,
      ).toEqual(expected.map((point) => [point.opportunity, point.count]));
      actual.forEach((point, index) => {
        const target = defined(expected[index]);
        expectClose(point.errorRate, target.errorRate, METRIC_TOLERANCE, `${label} ${index}`);
        expectClose(point.predictedErrorRate, target.predictedErrorRate, METRIC_TOLERANCE, `${label} ${index}`);
      });
    };
    expectPoints(curve.overall, curves.overall, "overall");
    expect(Object.keys(curve.skills).toSorted()).toEqual(Object.keys(curves.skills));
    for (const [skill, points] of Object.entries(curves.skills)) {
      expectPoints(defined(curve.skills[skill]), points, skill);
    }
  });

  it("has no predicted error rates without predictions", () => {
    const curve = learningCurve(log);
    expect(curve.overall.every((point) => point.predictedErrorRate === null)).toBe(true);
  });

  it.each(opportunities.curves)(
    "matches base R's curves of learner and skill pairs with at least $minOpportunities opportunities",
    ({ minOpportunities, curve: reference }) => {
      const events = buildEventLog(opportunities.events);
      const curve = learningCurve(events, { predictions: opportunities.predictions, minOpportunities });
      const expectPoints = (actual: typeof curve.overall, expected: readonly ReferencePoint[], label: string): void => {
        expect(
          actual.map((point) => [point.opportunity, point.count]),
          label,
        ).toEqual(expected.map((point) => [point.opportunity, point.count]));
        actual.forEach((point, index) => {
          const target = defined(expected[index]);
          expectClose(point.errorRate, target.errorRate, METRIC_TOLERANCE, `${label} ${index}`);
          expectClose(point.predictedErrorRate, target.predictedErrorRate, METRIC_TOLERANCE, `${label} ${index}`);
        });
      };
      expectPoints(curve.overall, reference.overall, "overall");
      expect(Object.keys(curve.skills).toSorted()).toEqual(Object.keys(reference.skills).toSorted());
      for (const [skill, points] of Object.entries(reference.skills)) {
        expectPoints(defined(curve.skills[skill]), points, skill);
      }
    },
  );

  it("keeps only learner and skill pairs with enough practice, at every opportunity", () => {
    const small = buildEventLog([
      { learner: "a", item: "q", skills: ["s"], time: 1, outcome: 0 },
      { learner: "a", item: "q", skills: ["s"], time: 2, outcome: 1 },
      { learner: "b", item: "q", skills: ["s"], time: 3, outcome: 1 },
    ]);
    expect(learningCurve(small).overall).toEqual([
      { opportunity: 1, count: 2, errorRate: 0.5, predictedErrorRate: null },
      { opportunity: 2, count: 1, errorRate: 0, predictedErrorRate: null },
    ]);
    expect(learningCurve(small, { minOpportunities: 2 }).overall).toEqual([
      { opportunity: 1, count: 1, errorRate: 1, predictedErrorRate: null },
      { opportunity: 2, count: 1, errorRate: 0, predictedErrorRate: null },
    ]);
  });

  it("rejects a minimum that is not a positive integer, or that no learner and skill reaches", () => {
    expect(() => learningCurve(log, { minOpportunities: 0 })).toThrow(
      "The option minOpportunities must be a positive integer, found 0.",
    );
    expect(() => learningCurve(log, { minOpportunities: 1000 })).toThrow(
      "A learning curve needs a learner and skill with at least 1000 opportunities; none has.",
    );
    expect(() => learningCurve(log, JSON.parse("null"))).toThrow("The options must be an object, found null.");
  });

  it("rejects a log without skills and predictions that are not one per event", () => {
    const plain = buildEventLog([{ learner: "ana", item: "q1", time: 0, outcome: 1 }]);
    expect(() => learningCurve(plain)).toThrow(
      "A learning curve needs events with skills; no event of the log has any.",
    );
    expect(() => learningCurve(log, { predictions: [0.5] })).toThrow(`Found 1 predictions for ${log.length} events.`);
    expect(() => learningCurve([])).toThrow("A log needs at least one event.");
  });
});

describe("metricsByOpportunity", () => {
  const log = buildEventLog(opportunities.events);
  const reference = defined(opportunities.curves[0]).curve;

  it("matches pROC and base R by opportunity, over all skills and for each skill", () => {
    const table = metricsByOpportunity(log, opportunities.predictions);
    const expectPoints = (
      actual: typeof table.overall,
      expected: readonly { readonly opportunity: number; readonly metrics: ReferenceMetrics | null }[],
      label: string,
    ): void => {
      expect(actual.map((point) => point.opportunity)).toEqual(expected.map((point) => point.opportunity));
      actual.forEach((point, index) => {
        expectMetrics(point.metrics, defined(expected[index]).metrics, `${label} ${index}`);
      });
    };
    expectPoints(table.overall, reference.overall, "overall");
    expect(Object.keys(table.skills).toSorted()).toEqual(Object.keys(reference.skills).toSorted());
    for (const [skill, points] of Object.entries(reference.skills)) {
      expectPoints(defined(table.skills[skill]), points, skill);
    }
  });

  it("has the opportunities of the learning curve, and metrics null where no event has a prediction", () => {
    const table = metricsByOpportunity(log, opportunities.predictions);
    const curve = learningCurve(log, { predictions: opportunities.predictions });
    expect(table.overall.map((point) => point.opportunity)).toEqual(curve.overall.map((point) => point.opportunity));
    expect(Object.keys(table.skills)).toEqual(Object.keys(curve.skills));
    expect(table.overall.some((point) => point.metrics === null)).toBe(false);
    const none = metricsByOpportunity(
      log,
      opportunities.predictions.map(() => null),
    );
    expect(none.overall.every((point) => point.metrics === null)).toBe(true);
    expect(none.overall).toHaveLength(curve.overall.length);
  });

  it("counts an event of two skills at an opportunity of each, and leaves out lessons and events without skills", () => {
    const small = buildEventLog([
      { learner: "a", item: "q1", skills: ["s", "t"], time: 1, outcome: 1 },
      { learner: "a", item: "video", skills: ["s"], time: 2, outcome: null },
      { learner: "a", item: "q2", skills: [], time: 3, outcome: 0 },
      { learner: "a", item: "q3", skills: ["s"], time: 4, outcome: 0 },
    ]);
    const table = metricsByOpportunity(small, [0.8, null, 0.5, 0.3]);
    expect(table.overall.map((point) => point.metrics?.count)).toEqual([2, 1]);
    expect(table.skills.s?.map((point) => point.metrics?.count)).toEqual([1, 1]);
    expect(table.skills.t?.map((point) => point.metrics?.count)).toEqual([1]);
    expect(table.overall[0]?.metrics).toEqual(metrics([0.8, 0.8], [1, 1]));
  });

  it("rejects a log without skills and predictions that are not one per event", () => {
    const plain = buildEventLog([{ learner: "ana", item: "q1", time: 0, outcome: 1 }]);
    expect(() => metricsByOpportunity(plain, [0.5])).toThrow(
      "Metrics by opportunity need events with skills; no event of the log has any.",
    );
    expect(() => metricsByOpportunity(log, [0.5])).toThrow(`Found 1 predictions for ${log.length} events.`);
    expect(() => metricsByOpportunity(log, JSON.parse("null"))).toThrow(
      "The predictions must be an array with a prediction or null for each event, found null.",
    );
    expect(() => metricsByOpportunity(JSON.parse("null"), [])).toThrow("A log must be an array of events, found null.");
  });
});

describe("the analysis of learning curves", () => {
  it.each(powerLaws.cases)("fits the power law as lm does: $name", (reference) => {
    const fit = fitPowerLaw(reference.points);
    if (reference.powerLaw === null) {
      expect(fit).toBeNull();
      return;
    }
    expectClose(defined(fit).scale, reference.powerLaw.scale, CURVE_TOLERANCE, "scale");
    expectClose(defined(fit).exponent, reference.powerLaw.exponent, CURVE_TOLERANCE, "exponent");
    expectClose(defined(fit).rSquared, reference.powerLaw.rSquared, CURVE_TOLERANCE, "R²");
  });

  it.each(powerLaws.cases)("categorizes as DataShop's rules do: $name", (reference) => {
    reference.categories.forEach(({ options, category }, index) => {
      expect(categorizeCurve(reference.points, options), `options ${index}`).toBe(category);
    });
  });

  it("recovers the scale and exponent of an exact power law, whatever the counts", () => {
    const laws = fc.record({
      scale: fc.double({ min: 0.2, max: 0.95, noNaN: true }),
      exponent: fc.double({ min: -0.2, max: 1.5, noNaN: true }),
      counts: fc.array(fc.integer({ min: 1, max: 500 }), { minLength: 2, maxLength: 12 }),
    });
    fc.assert(
      fc.property(laws, ({ scale, exponent, counts }) => {
        const points = counts.map((count, index) => ({
          opportunity: index + 1,
          count,
          errorRate: Math.min(scale * (index + 1) ** -exponent, 1),
          predictedErrorRate: null,
        }));
        fc.pre(points.every((point) => point.errorRate < 1));
        const fit = defined(fitPowerLaw(points));
        expect(Math.abs(fit.exponent - exponent)).toBeLessThan(1e-9);
        expect(Math.abs(fit.scale - scale)).toBeLessThan(1e-9);
      }),
    );
  });

  describe("categorizeCurve at its thresholds", () => {
    const curve = (counts: readonly number[], errors: readonly number[]): LearningCurvePoint[] =>
      counts.map((count, index) => ({
        opportunity: index + 1,
        count,
        errorRate: defined(errors[index]),
        predictedErrorRate: null,
      }));
    const options = { minLearners: 10, minPoints: 3, slope: 0.1, minSlope: 0.1 };

    it("calls a curve good at exactly the thresholds: only a value beyond one is bad", () => {
      expect(categorizeCurve(curve([10, 10, 10], [0.9, 0.6, 0.4]), options)).toBe("good");
      expect(categorizeCurve(curve([10, 10, 10], [0.9, 0.5, 0.2]), options)).toBe("good");
      expect(categorizeCurve(curve([10, 10, 10], [0.9, 0.6, 0.41]), options)).toBe("still high");
      expect(categorizeCurve(curve([10, 10, 10], [0.9, 0.6, 0.19]), options)).toBe("low and flat");
      expect(categorizeCurve(curve([10, 10, 10], [0.9, 0.6, 0.4]), { ...options, slope: 0.09 })).toBe("no learning");
    });

    it("discards points of fewer learners before it counts the points and reads the last", () => {
      expect(categorizeCurve(curve([9, 10, 10], [0.9, 0.6, 0.4]), options)).toBe("too little data");
      expect(categorizeCurve(curve([9, 10, 10], [0.9, 0.6, 0.4]), { ...options, minPoints: 2 })).toBe("good");
      expect(categorizeCurve(curve([10, 2, 10], [0.9, 0.05, 0.3]), { ...options, minLearners: 5, minPoints: 2 })).toBe(
        "good",
      );
      expect(categorizeCurve(curve([10, 10, 2], [0.9, 0.35, 0.95]), { ...options, minLearners: 5, minPoints: 2 })).toBe(
        "good",
      );
      expect(categorizeCurve(curve([2, 2], [0.9, 0.5]), options)).toBe("too little data");
    });

    it("tests the categories in order: too little data, low and flat, still high, no learning", () => {
      expect(categorizeCurve(curve([10, 10], [0.1, 0.9]), { ...options, minPoints: 3, slope: 0 })).toBe(
        "too little data",
      );
      expect(categorizeCurve(curve([10, 10, 10], [0.1, 0.5, 0.9]), { ...options, slope: 0 })).toBe("low and flat");
      expect(categorizeCurve(curve([10, 10, 10], [0.9, 0.5, 0.9]), { ...options, slope: 0 })).toBe("still high");
    });

    it("takes the thresholds of DataShop unless they are given", () => {
      const points = curve([10, 10, 10], [0.9, 0.5, 0.3]);
      expect(categorizeCurve(points, options)).toBe("good");
      expect(categorizeCurve(points, { ...options, lowError: 0.35 })).toBe("low and flat");
      expect(categorizeCurve(points, { ...options, highError: 0.25 })).toBe("still high");
    });
  });

  it("rejects points and options that are not valid", () => {
    const point = { opportunity: 1, count: 5, errorRate: 0.5, predictedErrorRate: null };
    const options = { minLearners: 1, minPoints: 1, slope: 0, minSlope: 0 };
    expect(() => fitPowerLaw([{ ...point, opportunity: 0 }])).toThrow(
      "Point 1 needs an opportunity that is an integer from 1, found 0.",
    );
    expect(() => fitPowerLaw([point, point])).toThrow("Point 2 repeats opportunity 1.");
    expect(() => fitPowerLaw([{ ...point, count: 0 }])).toThrow(
      "Point 1 needs a count that is a safe integer from 1, found 0.",
    );
    // Counts this large would make the sums of the fit overflow, and its results NaN.
    expect(() => fitPowerLaw([point, { ...point, opportunity: 2, count: 1e308 }])).toThrow(
      "Point 2 needs a count that is a safe integer from 1, found 1e+308.",
    );
    expect(() => fitPowerLaw([{ ...point, errorRate: 1.5 }])).toThrow(
      "Point 1 needs an error rate from 0 to 1, found 1.5.",
    );
    expect(() => fitPowerLaw(JSON.parse("null"))).toThrow(
      "The points must be an array of learning curve points, found null.",
    );
    expect(() => fitPowerLaw(JSON.parse("[null]"))).toThrow(
      "Point 1 must be an object { opportunity, count, errorRate }, found null.",
    );
    expect(() => categorizeCurve([point], { ...options, minLearners: 0 })).toThrow(
      "The option minLearners must be a positive integer, found 0.",
    );
    expect(() => categorizeCurve([point], { ...options, slope: JSON.parse('"1"') })).toThrow(
      'The option slope must be a finite number, found "1".',
    );
    expect(() => categorizeCurve([point], { ...options, highError: 2 })).toThrow(DataError);
    expect(() => categorizeCurve([point], JSON.parse("null"))).toThrow("The options must be an object, found null.");
    expect(() => categorizeCurve([point], JSON.parse("{}"))).toThrow(DataError);
    expect(fitPowerLaw([])).toBeNull();
  });
});

describe("evaluate options from JavaScript", () => {
  it("rejects options that are not an object", () => {
    expect(() => calibrationCurve([0.5], [1], JSON.parse("null"))).toThrow(
      "The options must be an object, found null.",
    );
  });

  it("rejects predictions, outcomes and logs of other types", () => {
    for (const use of [
      () => auc(JSON.parse("null"), [1]),
      () => rmse([0.5], JSON.parse('"1"')),
      () => logLoss(JSON.parse("{}"), [1]),
      () => accuracy(JSON.parse("null"), JSON.parse("null")),
      () => metrics([0.5], JSON.parse("null")),
    ]) {
      expect(use).toThrow("Predictions and outcomes must be arrays of numbers.");
    }
    expect(() => learningCurve(JSON.parse("null"))).toThrow("A log must be an array of events, found null.");
  });

  it("rejects a model that is not a learner model", () => {
    expect(() => crossValidate(JSON.parse("null"), [])).toThrow(
      "Cross-validation needs a learner model, such as bkt(), found null.",
    );
    expect(() => compareModels({ elo: JSON.parse('{"fit": 1}') }, [])).toThrow(
      "Cross-validation needs a learner model, such as bkt(), found an object.",
    );
  });
});

describe("evaluation with events that have no outcome", () => {
  const base = buildEventLog(holdout.events);
  // A lesson after every third event, at the learner and time of the event, which Elo ignores.
  const lessons = base.flatMap(
    (event, index): EventLog =>
      index % 3 === 0
        ? [
            event,
            { learner: event.learner, item: `lesson${index}`, skills: event.skills, time: event.time, outcome: null },
          ]
        : [event],
  );

  it("aligns the events with an outcome and a prediction, and leaves the others to the metrics", () => {
    const log = buildEventLog([
      { learner: "a", item: "q", time: 1, outcome: 1 },
      { learner: "a", item: "video", time: 2, outcome: null },
      { learner: "a", item: "q", time: 3, outcome: 0 },
      { learner: "a", item: "q", time: 4, outcome: 1 },
    ]);
    expect(alignPredictions(log, [0.9, null, 0.2, null])).toEqual({ predictions: [0.9, 0.2], outcomes: [1, 0] });
    expect(alignPredictions(log, [0.9, 0.5, 0.2, 0.7])).toEqual({ predictions: [0.9, 0.2, 0.7], outcomes: [1, 0, 1] });
    const fit = elo().fit(log);
    const pairs = alignPredictions(log, fit.predictions);
    expect(pairs.outcomes).toEqual([1, 0, 1]);
    expect(metrics(pairs.predictions, pairs.outcomes).count).toBe(3);
  });

  it("rejects predictions that are not one per event, or are neither probabilities nor null", () => {
    const log = buildEventLog([{ learner: "a", item: "q", time: 1, outcome: 1 }]);
    expect(() => alignPredictions(log, [])).toThrow("Found 0 predictions for 1 events.");
    expect(() => alignPredictions(log, [1.5])).toThrow(
      "Prediction 1 must be a probability from 0 to 1 or null, found 1.5.",
    );
    expect(() => alignPredictions(log, JSON.parse('"0.5"'))).toThrow(
      "The predictions must be an array with a prediction or null for each event, found",
    );
    expect(() => alignPredictions([], [])).toThrow("A log needs at least one event.");
  });

  it.each([
    ["learner", (log: EventLog) => splitByLearner(log, 3, createRandom(3))],
    ["item", (log: EventLog) => splitByItem(log, 3, createRandom(3))],
    ["time", (log: EventLog) => splitByTime(log, 3)],
  ])("evaluates on folds by %s as it does without the events, which have null predictions", (_, split) => {
    const folds = split(lessons);
    const withNulls = crossValidate(elo(), folds);
    const without = crossValidate(elo(), split(base));
    expect(withNulls.metrics).toEqual(without.metrics);
    expect(withNulls.folds.map((fold) => fold.metrics)).toEqual(without.folds.map((fold) => fold.metrics));
    folds.forEach((fold, index) => {
      const nulls = defined(withNulls.folds[index]).predictions.filter((prediction) => prediction === null);
      expect(nulls).toHaveLength(fold.test.filter((event) => event.outcome === null).length);
    });
  });

  it("keeps the predictions of a test log aligned with it, a null for each event without an outcome", () => {
    const folds = splitByLearner(lessons, 3, createRandom(3));
    const evaluation = crossValidate(elo(), folds);
    folds.forEach((fold, index) => {
      expect(defined(evaluation.folds[index]).predictions.map((p) => p === null)).toEqual(
        fold.test.map((event) => event.outcome === null),
      );
    });
    expect(folds.flatMap((fold) => fold.test).some((event) => event.outcome === null)).toBe(true);
  });

  it("gives a fold with no event with an outcome no metrics, and refuses when no fold has one", () => {
    const lesson = (learner: string, time: number) => ({ learner, item: "video", skills: [], time, outcome: null });
    const answer = (learner: string, time: number) => ({ learner, item: "q", skills: [], time, outcome: 1 });
    const train = buildEventLog([answer("a", 1), answer("b", 2)]);
    const folds: Fold[] = [
      { train, test: buildEventLog([lesson("c", 3)]) },
      { train, test: buildEventLog([answer("c", 4)]) },
    ];
    const evaluation = crossValidate(elo(), folds);
    expect(evaluation.folds.map((fold) => fold.metrics?.count ?? null)).toEqual([null, 1]);
    expect(evaluation.metrics.count).toBe(1);
    expect(() => crossValidate(elo(), [{ train, test: buildEventLog([lesson("c", 3)]) }])).toThrow(
      "Cross-validation needs an event with an outcome in a test log; no test log has one.",
    );
  });

  it("counts opportunities and errors over the events with an outcome in learning curves", () => {
    const fit = elo({ skills: true }).fit(lessons);
    const curve = learningCurve(lessons, { predictions: fit.predictions });
    const plain = learningCurve(base, { predictions: elo({ skills: true }).fit(base).predictions });
    expect(curve).toEqual(plain);
    expect(learningCurve(lessons).overall.map((point) => point.count)).toEqual(
      learningCurve(base).overall.map((point) => point.count),
    );
  });

  it("counts an event with an outcome and no prediction in the error rate, and leaves the predicted one", () => {
    const log = buildEventLog([
      { learner: "a", item: "q", skills: ["s"], time: 1, outcome: 1 },
      { learner: "a", item: "video", skills: ["s"], time: 2, outcome: null },
      { learner: "a", item: "q", skills: ["s"], time: 3, outcome: 0 },
      { learner: "b", item: "q", skills: ["s"], time: 4, outcome: 1 },
    ]);
    expect(learningCurve(log, { predictions: [0.5, null, null, 0.75] }).overall).toEqual([
      { opportunity: 1, count: 2, errorRate: 0, predictedErrorRate: 0.375 },
      { opportunity: 2, count: 1, errorRate: 1, predictedErrorRate: null },
    ]);
    expect(learningCurve(log, { predictions: [null, null, null, null] }).overall[0]).toEqual({
      opportunity: 1,
      count: 2,
      errorRate: 0,
      predictedErrorRate: null,
    });
    expect(() => learningCurve(buildEventLog([{ ...defined(log[1]) }]))).toThrow(
      "A learning curve needs events with skills; no event of the log has any.",
    );
  });
});

import fc from "fast-check";
import { describe, expect, it } from "vitest";
import {
  accuracy,
  auc,
  calibration,
  compareModels,
  createRandom,
  crossValidate,
  DataError,
  type EventLog,
  elo,
  eventLog,
  type Fold,
  type LearnerModel,
  learningCurve,
  logLoss,
  metrics,
  rmse,
  splitByLearner,
} from "../src/index.js";
import holdout from "./fixtures/elo-holdout.json" with { type: "json" };
import curves from "./fixtures/evaluate-learning-curve.json" with { type: "json" };
import fixture from "./fixtures/evaluate-metrics.json" with { type: "json" };
import { defined, expectAllClose, expectClose } from "./helpers.js";

/** Absolute tolerance of metrics, calibration bins and learning curves against R. */
const METRIC_TOLERANCE = 1e-12;
/** Absolute tolerance of held-out Elo predictions against R. */
const PREDICTION_TOLERANCE = 1e-12;

/** A model written outside the package: every learner answers correctly at the rate of the training log. */
function baseRate(rate = 0.5): LearnerModel<string, { rate: number }> {
  return {
    parameters: { rate },
    fit(log) {
      const model = baseRate(log.reduce((sum, event) => sum + event.outcome, 0) / log.length);
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
      const table = calibration(predictions, outcomes, { binCount: bins });
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
    });
  });

  it("give a log-loss of Infinity for a certain prediction that is wrong, and 0 for one that is right", () => {
    expect(logLoss([1, 0], [1, 0])).toBe(0);
    expect(logLoss([1, 0.5], [0, 1])).toBe(Number.POSITIVE_INFINITY);
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
    [() => calibration([0.5], [1], { binCount: 0 }), "The number of bins must be a positive integer, found 0."],
  ])("reject malformed input %#", (call, message) => {
    expect(call).toThrow(DataError);
    expect(call).toThrow(message);
  });
});

describe("calibration", () => {
  it("puts a prediction on a bin bound into the upper bin, and 1 into the last bin", () => {
    const table = calibration([0, 0.3, 0.7, 1], [0, 1, 1, 1]);
    expect(table.map((bin) => bin.count)).toEqual([1, 0, 0, 1, 0, 0, 0, 1, 0, 1]);
    expect(table[1]).toEqual({ lower: 0.1, upper: 0.2, count: 0, meanPrediction: null, meanOutcome: null });
  });
});

describe("crossValidate", () => {
  const log = eventLog(holdout.events);
  const at = (indices: readonly number[]): EventLog => indices.map((index) => defined(log[index]));

  it.each(holdout.folds)("predicts the held-out events as R does: $name", (reference) => {
    const folds: Fold[] = [{ train: at(reference.train), test: at(reference.test) }];
    for (const [name, model] of [
      ["basic", elo()],
      ["skills", elo({ skills: true })],
    ] as const) {
      const evaluation = crossValidate(model, folds);
      expectAllClose(defined(evaluation.folds[0]).predictions, reference.predictions[name], PREDICTION_TOLERANCE, name);
    }
  });

  it("pools the predictions of every fold into the overall metrics", () => {
    const folds = splitByLearner(log, 3, createRandom(1));
    const evaluation = crossValidate(elo(), folds);
    expect(evaluation.folds).toHaveLength(3);
    expect(evaluation.metrics).toEqual(
      metrics(
        evaluation.folds.flatMap((fold) => fold.predictions),
        folds.flatMap((fold) => fold.test.map((event) => event.outcome)),
      ),
    );
    evaluation.folds.forEach((fold, index) => {
      expect(fold.predictions).toHaveLength(defined(folds[index]).test.length);
    });
  });

  it("evaluates a model written outside the package", () => {
    const evaluation = crossValidate(baseRate(0.5), [{ train: at([0, 1, 2, 3]), test: at([4, 5]) }]);
    const rate = at([0, 1, 2, 3]).reduce((sum, event) => sum + event.outcome, 0) / 4;
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

describe("compareModels", () => {
  it("evaluates every model on the same folds, in the order given", () => {
    const folds = splitByLearner(eventLog(holdout.events), 2, createRandom(3));
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

describe("learningCurve", () => {
  const log = eventLog(curves.events);

  it("matches base R's error rates by opportunity, over all skills and per skill", () => {
    const curve = learningCurve(log, curves.predictions);
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

  it("rejects a log without skills and predictions that are not one per event", () => {
    const plain = eventLog([{ learner: "ana", item: "q1", time: 0, outcome: 1 }]);
    expect(() => learningCurve(plain)).toThrow(
      "A learning curve needs events with skills; no event of the log has any.",
    );
    expect(() => learningCurve(log, [0.5])).toThrow(`Found 1 predictions and ${log.length} outcomes.`);
    expect(() => learningCurve([])).toThrow("A log needs at least one event.");
  });
});

describe("evaluate options from JavaScript", () => {
  it("rejects options that are not an object", () => {
    expect(() => calibration([0.5], [1], JSON.parse("null"))).toThrow("The options must be an object, found null.");
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

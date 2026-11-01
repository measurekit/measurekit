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
  observedPairs,
  rmse,
  splitByItem,
  splitByLearner,
  splitByTime,
} from "../src/index.js";
import holdout from "./fixtures/elo-holdout.json" with { type: "json" };
import curves from "./fixtures/evaluate-learning-curve.json" with { type: "json" };
import fixture from "./fixtures/evaluate-metrics.json" with { type: "json" };
import { allPredicted, defined, expectAllClose, expectClose } from "./helpers.js";

/** Absolute tolerance of metrics, calibration bins and learning curves against R. */
const METRIC_TOLERANCE = 1e-12;
/** Absolute tolerance of held-out Elo predictions against R. */
const PREDICTION_TOLERANCE = 1e-12;

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
    expect(() => learningCurve(log, [0.5])).toThrow(`Found 1 predictions for ${log.length} events.`);
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

describe("evaluation with events that have no outcome", () => {
  const base = eventLog(holdout.events);
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
    const log = eventLog([
      { learner: "a", item: "q", time: 1, outcome: 1 },
      { learner: "a", item: "video", time: 2, outcome: null },
      { learner: "a", item: "q", time: 3, outcome: 0 },
      { learner: "a", item: "q", time: 4, outcome: 1 },
    ]);
    expect(observedPairs(log, [0.9, null, 0.2, null])).toEqual({ predictions: [0.9, 0.2], outcomes: [1, 0] });
    expect(observedPairs(log, [0.9, 0.5, 0.2, 0.7])).toEqual({ predictions: [0.9, 0.2, 0.7], outcomes: [1, 0, 1] });
    const fit = elo().fit(log);
    const pairs = observedPairs(log, fit.predictions);
    expect(pairs.outcomes).toEqual([1, 0, 1]);
    expect(metrics(pairs.predictions, pairs.outcomes).count).toBe(3);
  });

  it("rejects predictions that are not one per event, or are neither probabilities nor null", () => {
    const log = eventLog([{ learner: "a", item: "q", time: 1, outcome: 1 }]);
    expect(() => observedPairs(log, [])).toThrow("Found 0 predictions for 1 events.");
    expect(() => observedPairs(log, [1.5])).toThrow(
      "Prediction 1 must be a probability from 0 to 1 or null, found 1.5.",
    );
    expect(() => observedPairs(log, JSON.parse('"0.5"'))).toThrow(
      "The predictions must be an array with a prediction or null for each event, found",
    );
    expect(() => observedPairs([], [])).toThrow("A log needs at least one event.");
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
    const train = eventLog([answer("a", 1), answer("b", 2)]);
    const folds: Fold[] = [
      { train, test: eventLog([lesson("c", 3)]) },
      { train, test: eventLog([answer("c", 4)]) },
    ];
    const evaluation = crossValidate(elo(), folds);
    expect(evaluation.folds.map((fold) => fold.metrics?.count ?? null)).toEqual([null, 1]);
    expect(evaluation.metrics.count).toBe(1);
    expect(() => crossValidate(elo(), [{ train, test: eventLog([lesson("c", 3)]) }])).toThrow(
      "Cross-validation needs an event with an outcome in a test log; no test log has one.",
    );
  });

  it("counts opportunities and errors over the events with an outcome in learning curves", () => {
    const fit = elo({ skills: true }).fit(lessons);
    const curve = learningCurve(lessons, fit.predictions);
    const plain = learningCurve(base, elo({ skills: true }).fit(base).predictions);
    expect(curve).toEqual(plain);
    expect(learningCurve(lessons).overall.map((point) => point.count)).toEqual(
      learningCurve(base).overall.map((point) => point.count),
    );
  });

  it("refuses a learning curve whose predictions lack an event with an outcome", () => {
    const log = eventLog([
      { learner: "a", item: "q", skills: ["s"], time: 1, outcome: 1 },
      { learner: "a", item: "video", skills: ["s"], time: 2, outcome: null },
    ]);
    expect(learningCurve(log, [0.5, null]).overall[0]?.count).toBe(1);
    expect(() => learningCurve(log, [null, null])).toThrow(
      "A learning curve needs a prediction for every event with an outcome.",
    );
    expect(() => learningCurve(eventLog([{ ...defined(log[1]) }]))).toThrow(
      "A learning curve needs events with skills; no event of the log has any.",
    );
  });
});

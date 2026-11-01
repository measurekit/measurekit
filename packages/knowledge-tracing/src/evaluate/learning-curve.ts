import { DataError } from "@measurekit/core";
import { isObserved } from "../data/event.js";
import { checkLog, type EventLog } from "../data/event-log.js";
import { observedPairs } from "./observed-pairs.js";

/** The error rate at one opportunity of a learning curve. */
export interface LearningCurvePoint {
  /** The opportunity: 1 for a learner's first event with the skill, 2 for the second, and so on. */
  readonly opportunity: number;
  /** The number of events at this opportunity: of learner and skill pairs, so an event with two skills counts twice. */
  readonly count: number;
  /** The observed error rate, 1 - mean outcome, at this opportunity. */
  readonly errorRate: number;
  /** The predicted error rate, 1 - mean prediction, or `null` when no predictions were given. */
  readonly predictedErrorRate: number | null;
}

/** Learning curves: error rate by opportunity, over all skills and for each skill. */
export interface LearningCurve {
  /** The curve over all skills: each event counts once for each of its skills. */
  readonly overall: readonly LearningCurvePoint[];
  /** The curve of each skill, by skill id, in the order the skills first appear. */
  readonly skills: Readonly<Record<string, readonly LearningCurvePoint[]>>;
}

/**
 * Returns learning curves: the error rate of events by opportunity, the number of times the learner has practiced the
 * skill, counting the event itself.
 *
 * @remarks
 * Error rates that fall with practice indicate that the skills model what learners learn. With predictions, such as a
 * fit's, each point also has the predicted error rate. Opportunities count the events of the log given, so pass whole
 * learner histories: a full log, or the test log of a split by learner. Events without skills are left out, and so are
 * events without an outcome: they are no opportunity and have no error rate.
 *
 * @param predictions - The prediction of each event, in log order, such as a fit's: a probability for each event with
 * an outcome, and `null` for each event without one.
 * @throws `DataError` when the log is malformed, no event with an outcome has skills, or the predictions are malformed,
 * not one per event, or lacking for an event with an outcome.
 */
export function learningCurve(log: EventLog, predictions?: readonly (number | null)[]): LearningCurve {
  checkLog(log);
  const observed = log.filter(isObserved);
  if (predictions !== undefined && observedPairs(log, predictions).outcomes.length < observed.length) {
    throw new DataError("A learning curve needs a prediction for every event with an outcome.");
  }
  if (observed.every((event) => event.skills.length === 0)) {
    throw new DataError("A learning curve needs events with skills; no event of the log has any.");
  }
  const overall = emptyTotals();
  const bySkill = new Map<string, Totals>();
  const practiced = new Map<string, Map<string, number>>();
  log.forEach((event, index) => {
    if (!isObserved(event)) return;
    const counts = practiced.get(event.learner) ?? new Map<string, number>();
    practiced.set(event.learner, counts);
    for (const skill of event.skills) {
      const opportunity = (counts.get(skill) ?? 0) + 1;
      counts.set(skill, opportunity);
      const totals = bySkill.get(skill) ?? emptyTotals();
      bySkill.set(skill, totals);
      const prediction = predictions?.[index] ?? 0;
      add(overall, opportunity, event.outcome, prediction);
      add(totals, opportunity, event.outcome, prediction);
    }
  });
  const hasPredictions = predictions !== undefined;
  return {
    overall: points(overall, hasPredictions),
    skills: Object.fromEntries([...bySkill].map(([skill, totals]) => [skill, points(totals, hasPredictions)])),
  };
}

/** Sums of outcomes and predictions by opportunity, from opportunity 1 at index 0. */
interface Totals {
  readonly counts: number[];
  readonly outcomes: number[];
  readonly predictions: number[];
}

function emptyTotals(): Totals {
  return { counts: [], outcomes: [], predictions: [] };
}

function add(totals: Totals, opportunity: number, outcome: number, prediction: number): void {
  const index = opportunity - 1;
  totals.counts[index] = (totals.counts[index] ?? 0) + 1;
  totals.outcomes[index] = (totals.outcomes[index] ?? 0) + outcome;
  totals.predictions[index] = (totals.predictions[index] ?? 0) + prediction;
}

function points(totals: Totals, hasPredictions: boolean): LearningCurvePoint[] {
  // A learner reaches opportunity k of a skill only after k - 1, so the counts have no gaps.
  return totals.counts.map((count, index) => ({
    opportunity: index + 1,
    count,
    errorRate: 1 - (totals.outcomes[index] ?? 0) / count,
    predictedErrorRate: hasPredictions ? 1 - (totals.predictions[index] ?? 0) / count : null,
  }));
}

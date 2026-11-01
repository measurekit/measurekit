import { checkLog, type EventLog } from "../data/event-log.js";
import { type Metrics, metrics } from "./metrics.js";
import { checkPredictionList } from "./observed-pairs.js";
import { checkHasSkills, collectByOpportunity } from "./opportunities.js";

/** The metrics of the events at one opportunity. */
export interface OpportunityPoint {
  /** The opportunity: 1 for a learner's first event with the skill, 2 for the second, and so on. */
  readonly opportunity: number;
  /** The metrics of the events at this opportunity with a prediction: pairs of learner and skill, as in a learning curve. */
  readonly metrics: Metrics | null;
}

/** Metrics by opportunity, over all skills and for each skill, in the shape of {@link LearningCurve}. */
export interface OpportunityMetrics {
  /** Over all skills: each event counts once for each of its skills. */
  readonly overall: readonly OpportunityPoint[];
  /** For each skill, by skill id, in the order the skills first appear. */
  readonly skills: Readonly<Record<string, readonly OpportunityPoint[]>>;
}

/** The predictions and outcomes at one opportunity. */
interface Cell {
  readonly predictions: number[];
  readonly outcomes: number[];
}

/**
 * Returns the metrics of predictions by opportunity, the number of times the learner has practiced the skill, counting
 * the event itself: AUC, RMSE and the other metrics of the events at each opportunity, over all skills and for each.
 *
 * @remarks
 * It shows where a model errs: a model that ignores learning predicts the first opportunities best, and one that
 * overrates it, the late ones. Opportunities are those of a learning curve, counted over the events with an outcome
 * of the log given, so pass whole learner histories. An event with two skills is at an opportunity of each, so the
 * overall metrics count it twice. A point has `metrics: null` when none of its events has a prediction, and its AUC is
 * `null` for outcomes all equal, as in `metrics`.
 *
 * @param predictions - The prediction of each event, in log order, such as a fit's: a probability, or `null` for an
 * event without one.
 * @throws `DataError` when the log is malformed, no event with an outcome has skills, or the predictions are malformed
 * or not one per event.
 */
export function metricsByOpportunity(log: EventLog, predictions: readonly (number | null)[]): OpportunityMetrics {
  checkLog(log);
  checkPredictionList(log, predictions);
  checkHasSkills(log, "Metrics by opportunity need");
  const { overall, skills } = collectByOpportunity(
    log,
    1,
    (): Cell => ({ predictions: [], outcomes: [] }),
    (cell, index) => {
      const prediction = predictions[index];
      const outcome = log[index]?.outcome;
      if (prediction === null || prediction === undefined || outcome === null || outcome === undefined) return;
      cell.predictions.push(prediction);
      cell.outcomes.push(outcome);
    },
  );
  return {
    overall: pointsOf(overall),
    skills: Object.fromEntries([...skills].map(([skill, cells]) => [skill, pointsOf(cells)])),
  };
}

function pointsOf(cells: readonly Cell[]): OpportunityPoint[] {
  return cells.map(({ predictions, outcomes }, index) => ({
    opportunity: index + 1,
    metrics: outcomes.length === 0 ? null : metrics(predictions, outcomes),
  }));
}

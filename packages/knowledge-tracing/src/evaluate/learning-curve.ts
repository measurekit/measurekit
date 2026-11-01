import { checkOptions, checkPositiveInteger, DataError } from "@measurekit/core";
import { checkLog, type EventLog } from "../data/event-log.js";
import { checkPredictionList } from "./observed-pairs.js";
import { checkHasSkills, collectByOpportunity } from "./opportunities.js";

/** The error rate at one opportunity of a learning curve. */
export interface LearningCurvePoint {
  /** The opportunity: 1 for a learner's first event with the skill, 2 for the second, and so on. */
  readonly opportunity: number;
  /** The number of events at this opportunity: of learner and skill pairs, so an event with two skills counts twice. */
  readonly count: number;
  /** The observed error rate, 1 - mean outcome, at this opportunity. */
  readonly errorRate: number;
  /**
   * The predicted error rate, 1 - mean prediction over the events with a prediction, or `null` when no predictions
   * were given or none is at this opportunity.
   */
  readonly predictedErrorRate: number | null;
}

/** Learning curves: error rate by opportunity, over all skills and for each skill. */
export interface LearningCurve {
  /** The curve over all skills: each event counts once for each of its skills. */
  readonly overall: readonly LearningCurvePoint[];
  /** The curve of each skill, by skill id, in the order the skills first appear. */
  readonly skills: Readonly<Record<string, readonly LearningCurvePoint[]>>;
}

const DEFAULT_MIN_OPPORTUNITIES = 1;

/** Options of {@link learningCurve}. */
export interface LearningCurveOptions {
  /**
   * The prediction of each event, in log order, such as a fit's: a probability, or `null` for an event without one.
   *
   * @defaultValue none
   */
  readonly predictions?: readonly (number | null)[];
  /**
   * The fewest opportunities that a learner and skill pair has in the log to take part, so that a curve is not shaped
   * by who left: the pairs of learners who stopped early are kept out of every opportunity, not only the late ones.
   *
   * @defaultValue `1`
   */
  readonly minOpportunities?: number;
}

/**
 * Returns learning curves: the error rate of events by opportunity, the number of times the learner has practiced the
 * skill, counting the event itself.
 *
 * @remarks
 * Error rates that fall with practice indicate that the skills model what learners learn. With predictions, such as a
 * fit's, each point also has the predicted error rate. Opportunities count the events of the log given, so pass whole
 * learner histories: a full log, or the test log of a split by learner. Events without skills are left out, and so are
 * events without an outcome: they are no opportunity and have no error rate. An event with an outcome and a `null`
 * prediction, such as one that `unseenSkills` of `crossValidate` skipped, counts in the observed error rate and leaves
 * the predicted one.
 *
 * @throws `DataError` when the log is malformed, no event with an outcome has skills, an option is invalid, the
 * predictions are malformed or not one per event, or no learner and skill pair has `minOpportunities` opportunities.
 */
export function learningCurve(log: EventLog, options: LearningCurveOptions = {}): LearningCurve {
  checkOptions(options);
  const { predictions, minOpportunities = DEFAULT_MIN_OPPORTUNITIES } = options;
  checkPositiveInteger(minOpportunities, "The option minOpportunities");
  checkLog(log);
  if (predictions !== undefined) checkPredictionList(log, predictions);
  checkHasSkills(log, "A learning curve needs");
  const { overall, skills } = collectByOpportunity(log, minOpportunities, emptyTotals, (totals, index) => {
    const prediction = predictions?.[index];
    totals.count++;
    totals.outcomes += log[index]?.outcome ?? 0;
    if (prediction !== null && prediction !== undefined) {
      totals.predictions += prediction;
      totals.predicted++;
    }
  });
  if (overall.length === 0) {
    throw new DataError(
      `A learning curve needs a learner and skill with at least ${minOpportunities} opportunities; none has.`,
    );
  }
  return {
    overall: points(overall),
    skills: Object.fromEntries([...skills].map(([skill, totals]) => [skill, points(totals)])),
  };
}

/** Sums of outcomes and predictions at one opportunity. */
interface Totals {
  count: number;
  outcomes: number;
  predictions: number;
  /** The events with a prediction. */
  predicted: number;
}

function emptyTotals(): Totals {
  return { count: 0, outcomes: 0, predictions: 0, predicted: 0 };
}

function points(totals: readonly Totals[]): LearningCurvePoint[] {
  return totals.map(({ count, outcomes, predictions, predicted }, index) => ({
    opportunity: index + 1,
    count,
    errorRate: 1 - outcomes / count,
    predictedErrorRate: predicted === 0 ? null : 1 - predictions / predicted,
  }));
}

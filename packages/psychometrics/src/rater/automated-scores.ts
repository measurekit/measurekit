import {
  checkArray,
  checkFields,
  checkRecord,
  DataError,
  describe,
  type FieldNames,
  isRecord,
  kappaOfPairs,
  mean,
} from "@measurekit/core";
import { correlation, standardDeviation } from "../math/internal.js";

/** One response scored by an automated scorer and by a person, a subject of the evaluation of automated scores. */
export interface AutomatedScore {
  /** The automated score: a finite number, which the option `scale` rounds and trims to the scale's points. */
  readonly machine: number;
  /** A person's score of the same response: a category, an integer from 0, for the agreement statistics. */
  readonly human: number;
  /** A second person's score, for the agreement of two people; `null` or absent without one. */
  readonly second?: number | null;
  /** The subgroup of the response's writer, for the subgroup statistics; `null` or absent without one. */
  readonly group?: string | null;
  /** The scorer's confidence in its score, any finite number where higher is surer, for the routing curve. */
  readonly confidence?: number | null;
  /** When the scorer scored it, any finite number such as epoch milliseconds, for the windows of drift. */
  readonly time?: number | null;
}

/** The points of a score scale, from the lowest to the highest integer score. */
export interface ScoreRange {
  /** The lowest score, an integer from 0. */
  readonly lowest: number;
  /** The highest score, an integer above `lowest`. */
  readonly highest: number;
}

const SCORE_RANGE_FIELDS: FieldNames<ScoreRange> = { lowest: true, highest: true };

/**
 * The agreement of one scoring with another on the same responses, the first against the second, as RSMTool's
 * `metrics_helper` gives it: the machine against a person, or a second person against the first.
 */
export interface ScoreAgreement {
  /** The responses with both scores. */
  readonly subjects: number;
  /** The mean of the first scores. */
  readonly firstMean: number;
  /** The standard deviation of the first scores, on n - 1; `null` for one subject. */
  readonly firstSd: number | null;
  /** The mean of the second scores. */
  readonly secondMean: number;
  /** The standard deviation of the second scores, on n - 1; `null` for one subject. */
  readonly secondSd: number | null;
  /** The share of the responses with the same score in both. */
  readonly exactAgreement: number;
  /** The share of the responses whose scores differ by at most 1. */
  readonly adjacentAgreement: number;
  /** Cohen's kappa; `null` when the chance agreement is 1. */
  readonly kappa: number | null;
  /** Cohen's kappa with quadratic weights, QWK; `null` when the chance agreement is 1. */
  readonly quadraticKappa: number | null;
  /** The Pearson correlation; `null` for fewer than two subjects or a constant scoring. */
  readonly correlation: number | null;
  /**
   * The standardized mean difference, the first mean less the second over the pooled SD √((s₁² + s₂²) / 2), or, for a
   * subgroup, over the population's; `null` where that SD is 0 or undefined.
   */
  readonly smd: number | null;
  /** The root mean squared difference of the scores. */
  readonly rmse: number;
  /** 1 - Σ (second - first)² / Σ (second - mean of second)², as scikit-learn's `r2_score`; `null` where the second scores are constant. */
  readonly rSquared: number | null;
}

/** The scores of the subjects, checked, as arrays: `NaN` where a subject has no second score, confidence or time. */
export interface CheckedScores {
  /** The machine scores, rounded and trimmed when a scale is given. */
  readonly machine: Float64Array;
  readonly human: Float64Array;
  readonly second: Float64Array;
  readonly groups: readonly (string | null)[];
  readonly confidence: Float64Array;
  readonly time: Float64Array;
}

/** Returns whether a value is a category: an integer from 0. */
function isCategory(value: unknown): boolean {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0;
}

/**
 * Returns a field of a record that may be absent: `NaN` for `null` or absent, after checking it is a finite number, or a
 * category with `category`, otherwise.
 *
 * @param label - Names the field in the message, such as `a second score`.
 */
function optionalNumber(value: unknown, label: string, subject: number, category: boolean): number {
  if (value === undefined || value === null) return Number.NaN;
  if (category ? !isCategory(value) : !Number.isFinite(value)) {
    const kind = category ? "a category, an integer from 0," : "a finite number";
    throw new DataError(`Subject ${subject + 1} needs ${label} that is ${kind} or null, found ${describe(value)}.`);
  }
  return Number(value);
}

/**
 * Checks a score scale.
 *
 * @throws `DataError` when it is not `{ lowest, highest }` of integers from 0, the lowest below the highest.
 */
function checkScoreRange(scale: ScoreRange): void {
  checkRecord(scale, "The option scale must be an object { lowest, highest }");
  checkFields(scale, SCORE_RANGE_FIELDS, "The option scale");
  if (!(isCategory(scale.lowest) && isCategory(scale.highest) && scale.lowest < scale.highest)) {
    throw new DataError(
      `The option scale needs integers from 0 with lowest below highest, found ${describe(scale.lowest)} and ${describe(scale.highest)}.`,
    );
  }
}

/**
 * Checks the scores of an evaluation and returns them as arrays. With `scale`, machine scores are trimmed to it and
 * rounded to the nearest integer, a half up.
 *
 * @param categories - Whether the scores enter agreement statistics, which need categories: the human and second scores,
 *   and the machine scores once `scale` has rounded them.
 * @throws `DataError` when the scores are not an array of at least `minSubjects` records, a score is not a finite number,
 * or not a category where one is needed, a group is not a string or `null`, or the scale is invalid.
 */
export function checkAutomatedScores(
  scores: readonly AutomatedScore[],
  { categories, scale, minSubjects }: { categories: boolean; scale?: ScoreRange; minSubjects: number },
): CheckedScores {
  checkArray(scores, "The scores must be an array of { machine, human } records");
  if (scale !== undefined) checkScoreRange(scale);
  if (scores.length < minSubjects) {
    throw new DataError(`The evaluation needs at least ${minSubjects} scored subjects, found ${scores.length}.`);
  }
  const n = scores.length;
  const checked = {
    machine: new Float64Array(n),
    human: new Float64Array(n),
    second: new Float64Array(n),
    groups: new Array<string | null>(n),
    confidence: new Float64Array(n),
    time: new Float64Array(n),
  };
  for (let subject = 0; subject < n; subject++) {
    const record: unknown = scores[subject];
    if (!isRecord(record)) {
      throw new DataError(`Subject ${subject + 1} must be a record { machine, human }, found ${describe(record)}.`);
    }
    checked.machine[subject] = machineScore(record.machine, subject, categories, scale);
    if (record.human === null || record.human === undefined) {
      throw new DataError(`Subject ${subject + 1} needs a human score, found ${describe(record.human)}.`);
    }
    checked.human[subject] = optionalNumber(record.human, "a human score", subject, categories);
    checked.second[subject] = optionalNumber(record.second, "a second score", subject, categories);
    checked.confidence[subject] = optionalNumber(record.confidence, "a confidence", subject, false);
    checked.time[subject] = optionalNumber(record.time, "a time", subject, false);
    const group = record.group ?? null;
    if (!(group === null || typeof group === "string")) {
      throw new DataError(`Subject ${subject + 1} needs a group that is a string or null, found ${describe(group)}.`);
    }
    checked.groups[subject] = group;
  }
  return checked;
}

/** Returns a subject's machine score, rounded and trimmed to the scale when one is given. */
function machineScore(value: unknown, subject: number, categories: boolean, scale: ScoreRange | undefined): number {
  if (!(typeof value === "number" && Number.isFinite(value))) {
    throw new DataError(
      `Subject ${subject + 1} needs a machine score that is a finite number, found ${describe(value)}.`,
    );
  }
  if (scale !== undefined) return Math.min(scale.highest, Math.max(scale.lowest, Math.round(value)));
  if (categories && !isCategory(value)) {
    throw new DataError(
      `Subject ${subject + 1} has the machine score ${value}, which is not a category, an integer from 0: pass the option scale to round and trim the machine scores.`,
    );
  }
  return value;
}

/**
 * Returns the agreement of the first scores with the second, at least one pair.
 *
 * @param population - The standard deviations of the first and second scores over the population, for a subgroup's SMD.
 */
export function scoreAgreement(
  first: Float64Array,
  second: Float64Array,
  population?: { readonly first: number; readonly second: number },
): ScoreAgreement {
  const n = first.length;
  let exact = 0;
  let adjacent = 0;
  let squares = 0;
  for (let k = 0; k < n; k++) {
    const difference = Math.abs((first[k] ?? 0) - (second[k] ?? 0));
    if (difference === 0) exact++;
    if (difference <= 1) adjacent++;
    squares += difference * difference;
  }
  const firstMean = mean(first);
  const secondMean = mean(second);
  const firstSd = n > 1 ? standardDeviation(first) : null;
  const secondSd = n > 1 ? standardDeviation(second) : null;
  let spread = 0;
  for (const score of second) spread += (score - secondMean) ** 2;
  return {
    subjects: n,
    firstMean,
    firstSd,
    secondMean,
    secondSd,
    exactAgreement: exact / n,
    adjacentAgreement: adjacent / n,
    kappa: kappaOfPairs(first, second),
    quadraticKappa: kappaOfPairs(first, second, { weights: "quadratic" }),
    correlation: clamped(correlation(first, second)),
    smd: meanDifference(
      firstMean - secondMean,
      population ?? (firstSd === null ? null : { first: firstSd, second: secondSd ?? 0 }),
    ),
    rmse: Math.sqrt(squares / n),
    rSquared: spread > 0 ? 1 - squares / spread : null,
  };
}

/** Returns a difference of means over √((first² + second²) / 2), or `null` where that is 0 or the SDs are unknown. */
function meanDifference(difference: number, deviations: { readonly first: number; readonly second: number } | null) {
  if (deviations === null) return null;
  const pooled = Math.sqrt((deviations.first ** 2 + deviations.second ** 2) / 2);
  return pooled > 0 ? difference / pooled : null;
}

/** Returns a correlation held within -1 and 1, which rounding of perfectly correlated scores could pass, as R's `cor` does. */
function clamped(value: number | null): number | null {
  return value === null ? null : Math.min(1, Math.max(-1, value));
}

import { checkOptions, checkPositive, DataError, type FieldNames } from "@measurekit/core";
import {
  type AutomatedScore,
  checkAutomatedScores,
  type ScoreAgreement,
  type ScoreRange,
  scoreAgreement,
} from "./automated-scores.js";

/** Options of {@link automatedScoreDrift}. */
export interface AutomatedScoreDriftOptions {
  /**
   * The score scale, as `automatedScoreAgreement` takes it.
   *
   * @defaultValue none: machine scores are taken as given, and must be categories
   */
  readonly scale?: ScoreRange;
}

const AUTOMATED_SCORE_DRIFT_OPTIONS: FieldNames<AutomatedScoreDriftOptions> = { scale: true };

/** The agreement of the machine with people within one window of time. */
export interface DriftWindow {
  /** The window's start, included. */
  readonly start: number;
  /** Its end, excluded: the start plus the width. */
  readonly end: number;
  /** The machine-human agreement of the subjects scored in the window. */
  readonly agreement: ScoreAgreement;
}

/**
 * Returns the agreement of automated scores with human scores in each window of time, from which a drift of the scorer
 * or of the responses shows.
 *
 * @remarks
 * Windows of the same width follow each other from the earliest `time`; a subject belongs to the window that holds its `time`, and
 * a window without a subject is left out. Each window holds the machine-human statistics of `automatedScoreAgreement`,
 * of its subjects alone: compare them with those of the acceptance, or judge each with `validateAutomatedScores`. A
 * window of few subjects gives wide statistics; the guide gives their standard errors.
 *
 * @param scores - The subjects, each with its `time`.
 * @param width - The width of a window, in the unit of the subjects' `time`, such as 604,800,000 for a week of
 * milliseconds: above 0.
 * @param options - The score scale.
 * @returns The windows with a subject, in order of time.
 * @throws `DataError` when there is no subject, a subject has no finite `time`, a score is invalid as
 * `automatedScoreAgreement` says, the width is not a positive number, or the times span more widths than a number holds.
 */
export function automatedScoreDrift(
  scores: readonly AutomatedScore[],
  width: number,
  options: AutomatedScoreDriftOptions = {},
): readonly DriftWindow[] {
  checkOptions(options, AUTOMATED_SCORE_DRIFT_OPTIONS);
  const { scale } = options;
  checkPositive(width, "The window width");
  const { machine, human, time } = checkAutomatedScores(scores, {
    categories: true,
    minSubjects: 1,
    ...(scale ? { scale } : {}),
  });
  let origin = Number.POSITIVE_INFINITY;
  let latest = Number.NEGATIVE_INFINITY;
  time.forEach((at, subject) => {
    if (Number.isNaN(at)) throw new DataError(`Subject ${subject + 1} needs a time for the windows of drift.`);
    origin = Math.min(origin, at);
    latest = Math.max(latest, at);
  });
  // The end of the last window is the largest number of the windows.
  if (!Number.isFinite(origin + (Math.floor((latest - origin) / width) + 1) * width)) {
    throw new DataError("The windows of drift are not finite: the times span too many widths; rescale them.");
  }
  const windows = new Map<number, number[]>();
  time.forEach((at, subject) => {
    const index = Math.floor((at - origin) / width);
    const members = windows.get(index) ?? [];
    members.push(subject);
    windows.set(index, members);
  });
  return [...windows.entries()]
    .sort(([x], [y]) => x - y)
    .map(([index, members]) => ({
      start: origin + index * width,
      end: origin + (index + 1) * width,
      agreement: scoreAgreement(
        Float64Array.from(members, (subject) => machine[subject] ?? 0),
        Float64Array.from(members, (subject) => human[subject] ?? 0),
      ),
    }));
}

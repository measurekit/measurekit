import { checkFinite, checkOptions, checkPositive, DataError, type FieldNames } from "@measurekit/core";
import {
  type AutomatedScore,
  checkAutomatedScores,
  type ScoreAgreement,
  type ScoreRange,
  scoreAgreement,
} from "./automated-scores.js";

/** Options of {@link automatedScoreDrift}. */
export interface AutomatedScoreDriftOptions {
  /** The width of a window, in the unit of the subjects' `time`, such as 604,800,000 for a week of milliseconds: above 0. */
  readonly width: number;
  /**
   * The start of the first window.
   *
   * @defaultValue the earliest `time`
   */
  readonly start?: number;
  /**
   * The score scale, as `automatedScoreAgreement` takes it.
   *
   * @defaultValue none: machine scores are taken as given, and must be categories
   */
  readonly scale?: ScoreRange;
}

const AUTOMATED_SCORE_DRIFT_OPTIONS: FieldNames<AutomatedScoreDriftOptions> = { width: true, start: true, scale: true };

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
 * Windows of the same width follow each other from `start`; a subject belongs to the window that holds its `time`, and
 * a window without a subject is left out. Each window holds the machine-human statistics of `automatedScoreAgreement`,
 * of its subjects alone: compare them with those of the acceptance, or judge each with `validateAutomatedScores`. A
 * window of few subjects gives wide statistics; the guide gives their standard errors.
 *
 * @param scores - The subjects, each with its `time`.
 * @param options - The width of a window, the start and the score scale.
 * @returns The windows with a subject, in order of time.
 * @throws `DataError` when there is no subject, a subject has no finite `time` or one before `start`, a score is
 * invalid as `automatedScoreAgreement` says, or the width is not a positive number.
 */
export function automatedScoreDrift(
  scores: readonly AutomatedScore[],
  options: AutomatedScoreDriftOptions,
): readonly DriftWindow[] {
  checkOptions(options, AUTOMATED_SCORE_DRIFT_OPTIONS);
  const { width, start, scale } = options;
  checkPositive(width, "The option width");
  if (start !== undefined) checkFinite(start, "The option start");
  const { machine, human, time } = checkAutomatedScores(scores, {
    categories: true,
    minSubjects: 1,
    ...(scale ? { scale } : {}),
  });
  let earliest = Number.POSITIVE_INFINITY;
  time.forEach((at, subject) => {
    if (Number.isNaN(at)) throw new DataError(`Subject ${subject + 1} needs a time for the windows of drift.`);
    earliest = Math.min(earliest, at);
  });
  const origin = start ?? earliest;
  if (earliest < origin) throw new DataError(`A subject's time, ${earliest}, is before the option start, ${origin}.`);
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

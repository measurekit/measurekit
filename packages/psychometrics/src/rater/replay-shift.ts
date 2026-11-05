import { checkNonNegative, mean } from "@measurekit/core";
import { correlation, standardDeviation } from "../math/internal.js";
import { pairOfRaters } from "./ratings.js";

/** The shift of a scorer's replayed scores from the locked scores of the same responses. */
export interface ReplayShift {
  /** The responses with both scores. */
  readonly subjects: number;
  /** The mean of the replayed scores less the locked ones. */
  readonly meanShift: number;
  /** The standard deviation of the locked scores, on n - 1. */
  readonly lockedSd: number;
  /** The mean shift over `lockedSd`; `null` when the locked scores are constant. */
  readonly standardizedShift: number | null;
  /** The standard error of `standardizedShift`: the SD of the differences over √n, over `lockedSd`; `null` with it. */
  readonly standardError: number | null;
  /** The share of the responses whose replayed score equals the locked one. */
  readonly exactAgreement: number;
  /** The correlation of the replayed with the locked scores; `null` when either is constant. */
  readonly correlation: number | null;
  /** Whether the absolute standardized shift is at most `maxShiftSd`; `false` where it is `null`. */
  readonly passed: boolean;
}

/**
 * Returns the shift of an automated scorer's scores on responses rated again, an anchor replay: the mean change from the
 * locked scores in their standard deviations, and whether it stays within `maxShiftSd`.
 *
 * @remarks
 * Responses whose scores are locked, those of the machine at its acceptance or those people gave, are scored again by
 * the machine: a changed model, prompt or pipeline shows as a shift of the mean that people's scores of new responses
 * cannot separate from a change of the responses. The shift is mean(replayed - locked) over the SD of the locked scores;
 * its standard error is that of a paired mean, the SD of the differences over √n, in the same unit, which for scores of
 * equal SD that correlate at r is √(2 (1 - r) / n). Scores are finite numbers, not only categories. It uses the subjects
 * with both scores.
 *
 * @param ratings - One row for each response: the locked score, then the replayed score; `null` when missing.
 * @param maxShiftSd - The largest mean shift, in standard deviations of the locked scores, either way, at which the
 * replay passes: a program's policy, required as no source sets it. A third of the 0.15 that acceptance allows a
 * machine's SMD is 0.05.
 * @returns The shift, its standard error, the agreement of the replayed with the locked scores and whether the replay passes.
 * @throws `DataError` when the rows do not hold two scores, a score is not a finite number or `null`, fewer than two
 * responses have both, or `maxShiftSd` is not a non-negative number.
 * @see {@link https://doi.org/10.1111/j.1745-3992.2011.00223.x | Williamson, Xi and Breyer (2012)}
 *
 * @example
 * ```ts
 * import { replayShift } from "@measurekit/psychometrics/rater";
 *
 * // Eight responses scored at acceptance, then again after a new model.
 * const locked = [2, 3, 3, 4, 1, 2, 4, 3];
 * const replayed = [2, 3, 4, 4, 1, 3, 4, 3];
 * const replay = replayShift(
 *   locked.map((score, i) => [score, replayed[i] ?? null]),
 *   0.05,
 * );
 * replay.standardizedShift; // => 0.24, a quarter of an SD higher
 * replay.passed; // => false
 * ```
 */
export function replayShift(ratings: readonly (readonly (number | null)[])[], maxShiftSd: number): ReplayShift {
  checkNonNegative(maxShiftSd, "The maxShiftSd");
  const { first: locked, second: replayed } = pairOfRaters(ratings, "The replay shift");
  const n = locked.length;
  const differences = replayed.map((score, k) => score - (locked[k] ?? 0));
  let exact = 0;
  for (const difference of differences) if (difference === 0) exact++;
  const meanShift = mean(differences);
  const lockedSd = standardDeviation(locked);
  const standardizedShift = lockedSd > 0 ? meanShift / lockedSd : null;
  const standardError = lockedSd > 0 ? standardDeviation(differences) / Math.sqrt(n) / lockedSd : null;
  return {
    subjects: n,
    meanShift,
    lockedSd,
    standardizedShift,
    standardError,
    exactAgreement: exact / n,
    correlation: correlation(locked, replayed),
    passed: standardizedShift !== null && Math.abs(standardizedShift) <= maxShiftSd,
  };
}

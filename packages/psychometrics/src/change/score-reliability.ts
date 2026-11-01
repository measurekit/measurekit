import { checkPositive, checkRecord, DataError, describe, isUnitInterval } from "@measurekit/core";

/** The standard deviation of a test's scores and their reliability. */
export interface ScoreReliability {
  /** The standard deviation of the scores. */
  readonly sd: number;
  /** The reliability of the scores, from 0 to 1, such as Cronbach's alpha. */
  readonly reliability: number;
}

/**
 * Checks the standard deviation and reliability of the scores of one occasion.
 *
 * @param scores - Names the scores in the error message, such as `pre scores`.
 * @throws `DataError` when the record is not an object `{ sd, reliability }` with a positive `sd` and a `reliability`
 * from 0 to 1.
 */
export function checkScoreReliability(value: ScoreReliability, scores: string): void {
  checkRecord(value, `The ${scores} must be an object { sd, reliability }`);
  checkPositive(value.sd, `The sd of the ${scores}`);
  if (!isUnitInterval(value.reliability)) {
    throw new DataError(
      `The reliability of the ${scores} must be a number from 0 to 1, found ${describe(value.reliability)}.`,
    );
  }
}

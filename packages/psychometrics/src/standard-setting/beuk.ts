import { checkFinite, checkPositive, checkRecord, checkUnitInterval, DataError } from "@measurekit/core";
import { checkDistribution } from "../equating/internal.js";
import { crossFailureRate } from "./hofstee.js";

/** A panel's judgments in the Beuk method: the means and standard deviations of its answers. */
export interface BeukJudgments {
  /** The mean of the panelists' minimum passing scores, on the total score scale. */
  readonly cutScore: number;
  /** The mean of the panelists' expected pass rates: a share of persons, from 0 to 1. */
  readonly passRate: number;
  /** The standard deviation of the panelists' minimum passing scores. */
  readonly cutScoreSd: number;
  /** The standard deviation of the panelists' expected pass rates. */
  readonly passRateSd: number;
}

/** A Beuk cut score with its failure rate. */
export interface BeukCutScore {
  /** The cut score on the total score scale. */
  readonly cutScore: number;
  /** The failure rate at the cut score: the share of persons below it. */
  readonly failureRate: number;
}

/**
 * Returns the Beuk compromise cut score: where the line through the panel's mean cut score and pass rate, of slope
 * passRateSd / cutScoreSd in pass rate, crosses the observed pass rate.
 *
 * @remarks
 * In failure rates, the line falls from (k̄, 1 - p̄) by passRateSd / cutScoreSd per score, so the panel's surer answer
 * moves less, and the observed failure rate F(c), the share of persons with a total score below c, rises: they cross
 * once. F is taken at each score from 0 to the highest and is linear between them, as in {@link hofsteeCutScore}.
 *
 * @param distribution - The number of persons at each total score, from 0 to the highest.
 * @param judgments - The means and standard deviations of the panelists' cut scores and pass rates.
 * @returns The cut score and the failure rate there.
 * @throws `DataError` when the distribution is invalid, the mean cut score is outside the scores, the pass rate outside
 * [0, 1], a standard deviation not positive, or the line does not cross the failure rates within the scores.
 * @see {@link https://doi.org/10.1111/j.1745-3984.1984.tb00226.x | Beuk (1984)}
 */
export function beukCutScore(distribution: readonly number[], judgments: BeukJudgments): BeukCutScore {
  const total = checkDistribution(distribution, "observed");
  const highest = distribution.length - 1;
  checkRecord(judgments, "The Beuk judgments must be an object { cutScore, passRate, cutScoreSd, passRateSd }");
  const { cutScore, passRate, cutScoreSd, passRateSd } = judgments;
  checkFinite(cutScore, "The judgment cutScore");
  checkUnitInterval(passRate, "The judgment passRate");
  checkPositive(cutScoreSd, "The judgment cutScoreSd");
  checkPositive(passRateSd, "The judgment passRateSd");
  if (!(cutScore >= 0 && cutScore <= highest)) {
    throw new DataError(`The judgment cutScore must be a score from 0 to ${highest}, found ${cutScore}.`);
  }
  const slope = passRateSd / cutScoreSd;
  const failureRate = 1 - passRate;
  return crossFailureRate(
    distribution,
    total,
    {
      lowestCut: 0,
      highestCut: highest,
      highestRate: failureRate + slope * cutScore,
      lowestRate: failureRate - slope * (highest - cutScore),
    },
    `The Beuk line does not cross the observed failure rates between the scores 0 and ${highest}.`,
  );
}

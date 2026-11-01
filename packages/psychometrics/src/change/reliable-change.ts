import { checkFinite, checkPositive, checkRecord, DataError } from "@measurekit/core";
import { criticalValue, type ReliableChangeOptions } from "./critical-value.js";
import { checkScoreReliability, type ScoreReliability } from "./score-reliability.js";

/** The change of one person between two occasions, and whether it is more than the error of measurement explains. */
export interface ReliableChange {
  /** The reliable change index: the change in units of its standard error, positive for an increase. */
  readonly index: number;
  /** `"improved"` or `"declined"` when the index lies beyond the critical value of the level, else `"unchanged"`. */
  readonly classification: "improved" | "unchanged" | "declined";
}

/** The fields of an ability estimate that reliable change reads. */
export interface ThetaEstimate {
  /** The estimate of theta. */
  readonly theta: number;
  /** Its standard error. */
  readonly standardError: number;
}

/** Returns the index and the classification of a change, from the standard error of the change. */
function classify(change: number, standardError: number, critical: number): ReliableChange {
  const index = change / standardError;
  return { index, classification: index > critical ? "improved" : index < -critical ? "declined" : "unchanged" };
}

/**
 * Returns whether a person's change between two test scores is reliable: more than the error of measurement of the two
 * scores explains.
 *
 * @remarks
 * The reliable change index of Jacobson and Truax is (post - pre) / sqrt(2 SEM²), with SEM = sd sqrt(1 - reliability)
 * the standard error of measurement of the test, the same on both occasions. The change is reliable when the index is
 * beyond the critical value, the quantile of the standard normal distribution at (1 + level) / 2, 1.96 at the default
 * level. It assumes that higher scores are better: for a scale where lower is better, the change is a decline when
 * `classification` says improved. The standard deviation and reliability are those of the test, such as the standard
 * deviation of a norm group's scores and the reliability in it, not of the persons being assessed.
 *
 * Valid input: finite scores, a positive standard deviation, a reliability from 0 to below 1 and a level between 0 and
 * 1, exclusive.
 *
 * @example
 * ```ts
 * import { reliableChange } from "@measurekit/psychometrics/change";
 *
 * reliableChange(12, 21, { sd: 9, reliability: 0.88 }); // { index: 2.041241452319315, classification: "improved" }
 * ```
 *
 * @throws `DataError` when a score is not a finite number, the test is not an object `{ sd, reliability }` of valid
 * values, the reliability is 1, the options are not an object or the level is not between 0 and 1, exclusive.
 * @see {@link https://doi.org/10.1037/0022-006X.59.1.12 | Jacobson and Truax (1991)}
 */
export function reliableChange(
  pre: number,
  post: number,
  input: ScoreReliability,
  options: ReliableChangeOptions = {},
): ReliableChange {
  checkFinite(pre, "The pre score");
  checkFinite(post, "The post score");
  checkScoreReliability(input, "test");
  if (input.reliability === 1) {
    throw new DataError("The reliability of the test must be below 1, as a test without error has no standard error.");
  }
  const critical = criticalValue(options);
  const standardErrorOfMeasurement = input.sd * Math.sqrt(1 - input.reliability);
  return classify(post - pre, Math.sqrt(2 * standardErrorOfMeasurement ** 2), critical);
}

/**
 * Returns whether a person's change between two ability estimates is reliable: more than their standard errors
 * explain.
 *
 * @remarks
 * The index is (theta2 - theta1) / sqrt(SE1² + SE2²), and the change is reliable when it is beyond the critical value of
 * the level, as for {@link reliableChange}. The standard errors are the person's own, so the estimates may come from
 * different items, such as two adaptive tests. An `AbilityEstimate` of the `irt` module fits.
 *
 * Valid input: finite estimates with positive, finite standard errors, and a level between 0 and 1, exclusive.
 *
 * @throws `DataError` when an estimate is not an object `{ theta, standardError }` with a finite theta and a positive,
 * finite standard error, the options are not an object or the level is not between 0 and 1, exclusive.
 * @see {@link https://doi.org/10.1177/0146621616664046 | Jabrayilov, Emons and Sijtsma (2016)}
 */
export function reliableChangeFromTheta(
  before: ThetaEstimate,
  after: ThetaEstimate,
  options: ReliableChangeOptions = {},
): ReliableChange {
  checkEstimate(before, "before");
  checkEstimate(after, "after");
  const critical = criticalValue(options);
  return classify(
    after.theta - before.theta,
    Math.sqrt(before.standardError ** 2 + after.standardError ** 2),
    critical,
  );
}

/** Checks an ability estimate. */
function checkEstimate(value: ThetaEstimate, occasion: string): void {
  checkRecord(value, `The estimate ${occasion} must be an object { theta, standardError }`);
  checkFinite(value.theta, `The theta ${occasion}`);
  checkPositive(value.standardError, `The standard error ${occasion}`);
}

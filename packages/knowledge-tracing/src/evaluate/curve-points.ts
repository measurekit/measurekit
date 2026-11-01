import { checkArray, checkRecord, DataError, describe, isUnitInterval } from "@measurekit/core";
import type { LearningCurvePoint } from "./learning-curve.js";

/** The fields of a learning curve point that {@link powerLawFit} and {@link categorizeCurve} read. */
export type ErrorRatePoint = Pick<LearningCurvePoint, "opportunity" | "count" | "errorRate">;

/**
 * Checks the points of a learning curve: integer opportunities from 1, each once, counts above 0 and error rates from
 * 0 to 1.
 */
export function checkCurvePoints(points: readonly ErrorRatePoint[]): void {
  checkArray(points, "The points must be an array of learning curve points");
  const seen = new Set<number>();
  points.forEach((point, index) => {
    const label = `Point ${index + 1}`;
    checkRecord(point, `${label} must be an object { opportunity, count, errorRate }`);
    const { opportunity, count, errorRate } = point;
    if (!(Number.isInteger(opportunity) && opportunity >= 1)) {
      throw new DataError(`${label} needs an opportunity that is an integer from 1, found ${describe(opportunity)}.`);
    }
    if (seen.has(opportunity)) throw new DataError(`${label} repeats opportunity ${opportunity}.`);
    seen.add(opportunity);
    if (!(Number.isInteger(count) && count >= 1)) {
      throw new DataError(`${label} needs a count that is an integer from 1, found ${describe(count)}.`);
    }
    if (!isUnitInterval(errorRate)) {
      throw new DataError(`${label} needs an error rate from 0 to 1, found ${describe(errorRate)}.`);
    }
  });
}

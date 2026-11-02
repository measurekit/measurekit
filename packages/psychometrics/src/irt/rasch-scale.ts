import type { LogisticItem, ThresholdItem } from "@measurekit/core";
import type { CalibratedItem } from "./calibration.js";
import { oneParameterLogistic } from "./logistic.js";
import { partialCredit } from "./partial-credit.js";

/**
 * Returns an item on the Rasch scale, a = 1, from its thresholds and their standard errors: a 1PL item for one
 * threshold, a partial credit item for more.
 */
export function raschScaleItem(
  id: string,
  thresholds: readonly number[],
  errors: readonly number[] | null,
): CalibratedItem<LogisticItem | ThresholdItem> {
  if (thresholds.length === 1) {
    return {
      id,
      model: oneParameterLogistic,
      parameters: { a: 1, b: thresholds[0] ?? 0, c: 0, d: 1 },
      standardErrors: errors === null ? null : { b: errors[0] ?? 0 },
    };
  }
  return {
    id,
    model: partialCredit,
    parameters: { a: 1, b: [...thresholds] },
    standardErrors: errors === null ? null : Object.fromEntries(errors.map((error, k) => [`b${k + 1}`, error])),
  };
}

import { checkOptions, type FieldNames, normalQuantile } from "@measurekit/core";
import { checkOpenUnitInterval } from "../checks.js";

/** Options of {@link reliableChange} and {@link reliableChangeFromTheta}. */
export interface ReliableChangeOptions {
  /**
   * The probability of the central interval of no reliable change, between 0 and 1, exclusive.
   *
   * @defaultValue `0.95`
   */
  readonly level?: number;
}

const RELIABLE_CHANGE_OPTIONS: FieldNames<ReliableChangeOptions> = { level: true };

const DEFAULT_LEVEL = 0.95;

/**
 * Checks the options of a reliable change and returns the critical value of its index: the quantile of the standard
 * normal distribution at (1 + level) / 2.
 *
 * @throws `DataError` when the options are not an object or the level is not a number between 0 and 1, exclusive.
 */
export function criticalValue(options: ReliableChangeOptions): number {
  checkOptions(options, RELIABLE_CHANGE_OPTIONS);
  const { level = DEFAULT_LEVEL } = options;
  checkOpenUnitInterval(level, "The option level");
  return normalQuantile((1 + level) / 2);
}

import { DataError, describe } from "@measurekit/core";

/**
 * Checks that a value is a function, such as an estimator or a distribution.
 *
 * @param message - Says what the value should be, such
 * as `The option estimator must be a function of items and responses`.
 * @throws `DataError` when it is not.
 */
export function checkFunction(value: unknown, message: string): void {
  if (typeof value !== "function") throw new DataError(`${message}, found ${describe(value)}.`);
}

/**
 * Checks that a value is a number above 0 and below 1, such as the level of an interval.
 *
 * @param label - Names the value in the error message, such as `The option level`.
 * @throws `DataError` when it is not.
 */
export function checkOpenUnitInterval(value: unknown, label: string): asserts value is number {
  if (!(typeof value === "number" && value > 0 && value < 1)) {
    throw new DataError(`${label} must be a number between 0 and 1, exclusive, found ${describe(value)}.`);
  }
}

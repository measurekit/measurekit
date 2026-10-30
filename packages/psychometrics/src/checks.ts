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

import { DataError } from "./errors.js";

/**
 * Describes a value in a message: strings quoted; arrays, functions and other objects as "an array", "a function" and
 * "an object"; other values as `String` prints them.
 */
export function describe(value: unknown): string {
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return "an array";
  if (typeof value === "function") return "a function";
  if (typeof value === "object" && value !== null) return "an object";
  return String(value);
}

/** Returns whether a value is a string of at least one character, as every id is. */
export function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

/** Returns whether a value is a finite number above 0. */
export function isPositive(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

/** Returns whether a value is an object that is neither `null` nor an array. */
export function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/** Returns whether a value is a number from 0 to 1, such as a probability. */
export function isUnitInterval(value: unknown): value is number {
  return typeof value === "number" && value >= 0 && value <= 1;
}

/**
 * Checks that a value is a record, such as an options object or a lookup table.
 *
 * @param message - Says what the value should be, such as `The states must be an object of states by learner`.
 * @throws `DataError` when it is not.
 */
export function checkRecord(value: unknown, message: string): void {
  if (!isRecord(value)) throw new DataError(`${message}, found ${describe(value)}.`);
}

/**
 * Checks that a value is an array, such as a list of items or events.
 *
 * @param message - Says what the value should be, such as `The items must be an array`.
 * @throws `DataError` when it is not.
 */
export function checkArray(value: unknown, message: string): void {
  if (!Array.isArray(value)) throw new DataError(`${message}, found ${describe(value)}.`);
}

/**
 * Checks a function's options: an object, as a caller from JavaScript may pass `null` or another value.
 *
 * @throws `DataError` when they are not.
 */
export function checkOptions(options: unknown): void {
  checkRecord(options, "The options must be an object");
}

/**
 * Checks that a value is an object with a method, such as a stopping rule.
 *
 * @param message - Says what the value should be, such as `The option stopping must be an object with a stops method`.
 * @throws `DataError` when it is not.
 */
export function checkMethod(value: unknown, method: string, message: string): void {
  if (!(isRecord(value) && typeof value[method] === "function")) {
    throw new DataError(`${message}, found ${describe(value)}.`);
  }
}

/**
 * Checks a random number generator, as `createRandom` returns it.
 *
 * @throws `DataError` when it is not an object with the methods `uniform` and `normal`.
 */
export function checkRandom(random: unknown): void {
  if (!(isRecord(random) && typeof random.uniform === "function" && typeof random.normal === "function")) {
    throw new DataError(`A random number generator must be one from createRandom, found ${describe(random)}.`);
  }
}

/**
 * Checks that a value is a boolean.
 *
 * @param label - Names the value in the error message, such as `The option purify`.
 * @throws `DataError` when it is not.
 */
export function checkBoolean(value: unknown, label: string): asserts value is boolean {
  if (typeof value !== "boolean") throw new DataError(`${label} must be a boolean, found ${describe(value)}.`);
}

/**
 * Checks that a value is one of the given strings.
 *
 * @param label - Names the value in the error message, such as `The option routing`.
 * @throws `DataError` when it is not.
 */
export function checkChoice(value: unknown, label: string, choices: readonly string[]): void {
  if (!choices.some((choice) => choice === value)) {
    const quoted = choices.map((choice) => JSON.stringify(choice));
    const list = `${quoted.slice(0, -1).join(", ")} or ${quoted.at(-1)}`;
    throw new DataError(`${label} must be ${list}, found ${describe(value)}.`);
  }
}

/**
 * Checks that a value is a finite number.
 *
 * @param label - Names the value in the error message, such as `Theta`.
 * @throws `DataError` when it is not.
 */
export function checkFinite(value: unknown, label: string): asserts value is number {
  if (!(typeof value === "number" && Number.isFinite(value))) {
    throw new DataError(`${label} must be a finite number, found ${describe(value)}.`);
  }
}

/**
 * Checks that a value is a finite number above 0.
 *
 * @param label - Names the value in the error message, such as `The option sd`.
 * @throws `DataError` when it is not.
 */
export function checkPositive(value: unknown, label: string): asserts value is number {
  if (!isPositive(value)) throw new DataError(`${label} must be a positive number, found ${describe(value)}.`);
}

/**
 * Checks that a value is a finite number of at least 0.
 *
 * @param label - Names the value in the error message, such as `The rating of item 2 by panelist 1`.
 * @throws `DataError` when it is not.
 */
export function checkNonNegative(value: unknown, label: string): asserts value is number {
  if (!(typeof value === "number" && Number.isFinite(value) && value >= 0)) {
    throw new DataError(`${label} must be a non-negative number, found ${describe(value)}.`);
  }
}

/**
 * Checks that a value is a safe integer of at least 1.
 *
 * @param label - Names the value in the error message, such as `The option maxIterations`.
 * @throws `DataError` when it is not.
 */
export function checkPositiveInteger(value: unknown, label: string): asserts value is number {
  if (!(typeof value === "number" && Number.isSafeInteger(value) && value >= 1)) {
    throw new DataError(`${label} must be a positive integer, found ${describe(value)}.`);
  }
}

/**
 * Checks that a value is a number from 0 to 1.
 *
 * @param label - Names the value in the error message, such as `The option missingProbability`.
 * @throws `DataError` when it is not.
 */
export function checkUnitInterval(value: unknown, label: string): asserts value is number {
  if (!isUnitInterval(value)) throw new DataError(`${label} must be a number from 0 to 1, found ${describe(value)}.`);
}

/**
 * Checks a seed.
 *
 * @throws `DataError` when it is not a safe integer.
 */
export function checkSeed(seed: unknown): asserts seed is number {
  if (!(typeof seed === "number" && Number.isSafeInteger(seed))) {
    throw new DataError(`A seed must be a safe integer, found ${describe(seed)}.`);
  }
}

/**
 * Checks the options that stop an iterative method.
 *
 * @throws `DataError` when `tolerance` is not a non-negative number or `maxIterations` not a positive integer.
 */
export function checkStopping(tolerance: unknown, maxIterations: unknown): void {
  checkNonNegative(tolerance, "The option tolerance");
  checkPositiveInteger(maxIterations, "The option maxIterations");
}

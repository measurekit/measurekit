import { DataError } from "./errors.js";

/**
 * Describes a value in a message: strings quoted; arrays, functions and other objects as "an array", "a function" and
 * "an object"; other values as `String` prints them.
 *
 * @param value - The value to describe.
 * @returns The description: a quoted string, a number as written, or `an array`, `a function`, `an object`, `null` or
 *   `undefined`.
 */
export function describe(value: unknown): string {
  if (typeof value === "string") return JSON.stringify(value);
  if (Array.isArray(value)) return "an array";
  if (typeof value === "function") return "a function";
  if (typeof value === "object" && value !== null) return "an object";
  return String(value);
}

/**
 * Returns whether a value is a string of at least one character, as every id is.
 *
 * @param value - The value to test.
 * @returns Whether the value is a string of at least one character; `true` narrows its type.
 */
export function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

/**
 * Returns whether a value is a finite number above 0.
 *
 * @param value - The value to test.
 * @returns Whether the value is a finite number above 0; `true` narrows its type.
 */
export function isPositive(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

/**
 * Returns whether a value is an object that is neither `null` nor an array.
 *
 * @param value - The value to test.
 * @returns Whether the value is an object other than `null` and an array; `true` narrows its type.
 */
export function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Returns whether a value is a number from 0 to 1, such as a probability.
 *
 * @param value - The value to test.
 * @returns Whether the value is a number from 0 to 1; `true` narrows its type.
 */
export function isUnitInterval(value: unknown): value is number {
  return typeof value === "number" && value >= 0 && value <= 1;
}

/**
 * Checks that a value is a record, such as an options object or a lookup table.
 *
 * @param value - The value to check.
 * @param message - Says what the value should be, such as `The states must be an object of states by learner`.
 * @throws `DataError` when it is not.
 */
export function checkRecord(value: unknown, message: string): void {
  if (!isRecord(value)) throw new DataError(`${message}, found ${describe(value)}.`);
}

/**
 * Checks that a value is an array without holes, such as a list of items or events.
 *
 * @remarks
 * A hole, an index below the length that holds no element, as `new Array(3)` and `[1, , 2]` have, is no value of the
 * array that JSON gives, and `forEach`, `map` and `every` skip it unseen: it is refused.
 *
 * @param value - The value to check.
 * @param message - Says what the value should be, such as `The items must be an array`.
 * @throws `DataError` when it is not an array, or has a hole.
 */
export function checkArray(value: unknown, message: string): void {
  if (!Array.isArray(value)) throw new DataError(`${message}, found ${describe(value)}.`);
  const hole = firstHole(value);
  if (hole >= 0) throw new DataError(`${message}, found an array with a hole at index ${hole}.`);
}

/** Returns the first index below an array's length that holds no element, or -1 when every index holds one. */
function firstHole(values: readonly unknown[]): number {
  for (let index = 0; index < values.length; index++) if (!(index in values)) return index;
  return -1;
}

/**
 * The names of every field of a record type, each mapped to `true`: the options a function takes, or the fields of a
 * record of settings inside its options.
 *
 * @remarks
 * A constant of this type, declared against the options type, is a list that the type checker holds complete: it
 * refuses a name missing from it and a name the type does not have.
 */
export type FieldNames<TRecord> = { readonly [TName in keyof TRecord]-?: true };

/**
 * Checks a function's options: an object, as a caller from JavaScript may pass `null` or another value, that holds no
 * name the function does not take, as a misspelled option would otherwise leave its default in place unseen.
 *
 * @param options - The options a caller passed.
 * @param names - The names of the options the function takes.
 * @throws `DataError` when the options are not an object, or hold a name that is not one of `names`.
 */
export function checkOptions<TOptions extends object>(options: TOptions, names: FieldNames<NoInfer<TOptions>>): void {
  checkRecord(options, "The options must be an object");
  const unknown = unknownName(options, names);
  if (unknown !== undefined) {
    throw new DataError(`There is no option ${JSON.stringify(unknown)}: the options are ${listOf(names)}.`);
  }
}

/**
 * Checks that a record of settings inside a function's options, such as its quadrature or a prior, holds no field but
 * the given ones, as a misspelled field would otherwise leave its default in place unseen.
 *
 * @remarks
 * Records of data, such as items, events, states and parameters, are not checked this way: they may carry the fields
 * a host adds.
 *
 * @param record - The record, already checked to be an object.
 * @param names - The names of its fields.
 * @param label - Names the record in the error message, such as `The option quadrature`.
 * @throws `DataError` when the record holds a name that is not one of `names`.
 */
export function checkFields<TRecord extends object>(
  record: TRecord,
  names: FieldNames<NoInfer<TRecord>>,
  label: string,
): void {
  const unknown = unknownName(record, names);
  if (unknown !== undefined) {
    throw new DataError(`${label} may hold only ${listOf(names)}, found ${JSON.stringify(unknown)}.`);
  }
}

/**
 * Returns the first own enumerable name of a record that `names` lacks, or `undefined` when it has every one. It
 * allocates nothing, as options are checked on hot paths such as a line search.
 */
function unknownName(record: object, names: object): string | undefined {
  for (const name in record) if (Object.hasOwn(record, name) && !Object.hasOwn(names, name)) return name;
  return undefined;
}

/** Lists the names of a record of names in a sentence: `a, b and c`. */
function listOf(names: object): string {
  const all = Object.keys(names);
  return all.length < 2 ? all.join("") : `${all.slice(0, -1).join(", ")} and ${all.at(-1)}`;
}

/**
 * Checks that a value is an object with a method, such as a stopping rule.
 *
 * @param value - The value to check.
 * @param method - The name of the method the value must have.
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
 * @param random - The generator a caller passed.
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
 * @param value - The value to check.
 * @param label - Names the value in the error message, such as `The option purify`.
 * @throws `DataError` when it is not.
 */
export function checkBoolean(value: unknown, label: string): asserts value is boolean {
  if (typeof value !== "boolean") throw new DataError(`${label} must be a boolean, found ${describe(value)}.`);
}

/**
 * Checks that a value is one of the given strings.
 *
 * @param value - The value to check.
 * @param label - Names the value in the error message, such as `The option routing`.
 * @param choices - The strings the value may be.
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
 * @param value - The value to check.
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
 * @param value - The value to check.
 * @param label - Names the value in the error message, such as `The option sd`.
 * @throws `DataError` when it is not.
 */
export function checkPositive(value: unknown, label: string): asserts value is number {
  if (!isPositive(value)) throw new DataError(`${label} must be a positive number, found ${describe(value)}.`);
}

/**
 * Checks that a value is a finite number of at least 0.
 *
 * @param value - The value to check.
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
 * @param value - The value to check.
 * @param label - Names the value in the error message, such as `The option maxIterations`.
 * @throws `DataError` when it is not.
 */
export function checkPositiveInteger(value: unknown, label: string): asserts value is number {
  if (!(typeof value === "number" && Number.isSafeInteger(value) && value >= 1)) {
    throw new DataError(`${label} must be a positive integer, found ${describe(value)}.`);
  }
}

/**
 * Checks that a value is a safe integer of at least `minimum`.
 *
 * @param value - The value to check.
 * @param label - Names the value in the error message, such as `The option points`.
 * @param minimum - The least value allowed.
 * @throws `DataError` when it is not.
 */
export function checkIntegerFrom(value: unknown, label: string, minimum: number): asserts value is number {
  if (!(typeof value === "number" && Number.isSafeInteger(value) && value >= minimum)) {
    throw new DataError(`${label} must be an integer from ${minimum}, found ${describe(value)}.`);
  }
}

/**
 * Checks that a value is a number from 0 to 1.
 *
 * @param value - The value to check.
 * @param label - Names the value in the error message, such as `The option missingProbability`.
 * @throws `DataError` when it is not.
 */
export function checkUnitInterval(value: unknown, label: string): asserts value is number {
  if (!isUnitInterval(value)) throw new DataError(`${label} must be a number from 0 to 1, found ${describe(value)}.`);
}

/**
 * Checks a seed.
 *
 * @param seed - The seed a caller passed.
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
 * @param tolerance - The tolerance a caller passed.
 * @param maxIterations - The most iterations a caller passed.
 * @throws `DataError` when `tolerance` is not a non-negative number or `maxIterations` not a positive integer.
 */
export function checkStopping(tolerance: unknown, maxIterations: unknown): void {
  checkNonNegative(tolerance, "The option tolerance");
  checkPositiveInteger(maxIterations, "The option maxIterations");
}

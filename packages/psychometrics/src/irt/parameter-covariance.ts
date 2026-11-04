import { checkArray, checkFinite, checkRecord, DataError, describe } from "@measurekit/core";
import type { TestItem } from "./model.js";

/**
 * The covariance matrix of the estimated parameters of a calibration's items, labeled by item id and parameter name.
 * Plain data, which JSON serializes.
 */
export interface ParameterCovariance {
  /**
   * The item and the parameter name of each row and column: the estimated items in item order, each with the names of
   * its `standardErrors`. Items whose model shares a parameter, as the rating scale model shares its steps, repeat the
   * shared parameter, with the same variance and a correlation of 1.
   */
  readonly parameters: readonly {
    /** The item id. */
    readonly item: string;
    /** The parameter's name in the item's parameter vector. */
    readonly name: string;
  }[];
  /**
   * The covariance matrix by rows: `matrix[i][j]` is the covariance of `parameters[i]` and `parameters[j]`, on the
   * parameters' own scale.
   */
  readonly matrix: readonly (readonly number[])[];
}

/**
 * The step of the central differences, relative to a parameter of at least 1 in size: the cube root of the machine
 * epsilon, which balances truncation and rounding error.
 */
const DIFFERENCE_STEP = 6e-6;
/** The share of the room to a parameter's bound that a difference may use. */
const BOUND_ROOM = 0.5;
/** The negative variance, relative to the size of its terms, that rounding of a positive semidefinite matrix can give. */
export const VARIANCE_ROUNDING = 1e-9;

/**
 * Checks a covariance and returns its labels.
 *
 * @param noun - Names the covariance in error messages, such as `the covariance`.
 * @throws `DataError` when it is not an object with a label of two strings for each row and a square matrix of finite
 * numbers of that size.
 */
export function checkCovariance(
  covariance: ParameterCovariance,
  noun: string,
): readonly { readonly item: string; readonly name: string }[] {
  const subject = `${noun.charAt(0).toUpperCase()}${noun.slice(1)}`;
  checkRecord(covariance, `${subject} must be an object { parameters, matrix }`);
  const { parameters, matrix } = covariance;
  checkArray(parameters, `The parameters of ${noun} must be an array of labels { item, name }`);
  checkArray(matrix, `The matrix of ${noun} must be an array of rows`);
  if (matrix.length !== parameters.length) {
    throw new DataError(
      `${subject} needs a row for each of its ${parameters.length} parameters, found ${matrix.length}.`,
    );
  }
  parameters.forEach((label, k) => {
    checkRecord(label, `Parameter ${k + 1} of ${noun} must be an object { item, name }`);
    if (!(typeof label.item === "string" && typeof label.name === "string")) {
      throw new DataError(
        `Parameter ${k + 1} of ${noun} needs a string item and name, found ${describe(label.item)} and ${describe(label.name)}.`,
      );
    }
  });
  matrix.forEach((row, i) => {
    checkArray(row, `Row ${i + 1} of the matrix of ${noun} must be an array of numbers`);
    if (row.length !== parameters.length) {
      throw new DataError(
        `The matrix of ${noun} must be square: row ${i + 1} has ${row.length} entries for ${parameters.length} parameters.`,
      );
    }
    row.forEach((value, j) => {
      checkFinite(value, `Entry (${i + 1}, ${j + 1}) of the matrix of ${noun}`);
    });
  });
  return parameters;
}

/**
 * Returns the index of a named parameter in an item's parameter vector.
 *
 * @param noun - Names the covariance that labels the parameter in the error message, such as `the covariance`.
 * @throws `DataError` when the item's model does not estimate a parameter of that name.
 */
export function parameterEntry(item: TestItem<unknown>, name: string, noun: string): number {
  const { model, parameters, id } = item;
  const names = model.parameters(model.categoryCount(parameters)).map((parameter) => parameter.name);
  const entry = names.indexOf(name);
  if (entry < 0) {
    const subject = `${noun.charAt(0).toUpperCase()}${noun.slice(1)}`;
    throw new DataError(
      `${subject} has the parameter "${name}" of item "${id}", which its ${model.name} model does not estimate; it estimates ${names.join(", ")}.`,
    );
  }
  return entry;
}

/**
 * Returns the step of the central differences in one parameter of an item, shrunk to stay inside the bounds of its
 * range.
 *
 * @throws `DataError` when the parameter lies on a bound of its range, where a function of it has no derivative.
 */
export function differenceStep(item: TestItem<unknown>, entry: number): number {
  const value = item.model.toVector(item.parameters)[entry] ?? 0;
  const parameter = item.model.parameters(item.model.categoryCount(item.parameters))[entry];
  const { lower = Number.NEGATIVE_INFINITY, upper = Number.POSITIVE_INFINITY } = parameter ?? {};
  const room = BOUND_ROOM * Math.min(value - lower, upper - value);
  if (!(room > 0)) {
    throw new DataError(
      `The parameter "${parameter?.name}" of item "${item.id}" is ${value}, on a bound of its range; the standard error needs it inside the range.`,
    );
  }
  return Math.min(DIFFERENCE_STEP * Math.max(1, Math.abs(value)), room);
}

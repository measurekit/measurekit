import { checkArray, checkOptions, DataError, describe, isPositive } from "@measurekit/core";
import { indexIds } from "./ids.js";
import { isResponse, MAX_RESPONSE } from "./response.js";

/** Scored responses of persons to items, with their ids. */
export interface ResponseMatrix {
  /** Person ids, one per row of `responses`. */
  readonly personIds: readonly string[];
  /** Item ids, one per column of `responses`. */
  readonly itemIds: readonly string[];
  /** Responses by person, then item: a score or category from 0, or `null` when missing. */
  readonly responses: readonly (readonly (number | null)[])[];
  /**
   * Response times in seconds by person, then item, aligned with `responses`: positive, `null` when missing. A time
   * needs a response.
   */
  readonly times?: readonly (readonly (number | null)[])[];
}

/** Ids of the persons and items of a response matrix. */
export interface ResponseMatrixOptions {
  /**
   * Person ids.
   *
   * @defaultValue `person1`, `person2`, and so on
   */
  readonly personIds?: readonly string[];
  /**
   * Item ids.
   *
   * @defaultValue `item1`, `item2`, and so on
   */
  readonly itemIds?: readonly string[];
  /**
   * Response times in seconds by person, then item, `null` when missing.
   *
   * @defaultValue none: the matrix has no times
   */
  readonly times?: readonly (readonly (number | null)[])[];
}

/**
 * Builds a response matrix from one row of responses per person, with response times if given. The rows are copied.
 *
 * @param rows - The responses of each person, one entry for each item: an integer from 0, or `null` when missing.
 * @param options - The ids of the persons and items, and the response times.
 * @returns The matrix, whose ids are `person1`, `item1` and so on without the options.
 * @throws `DataError` when the rows are not an array, there are no persons or no items, the rows differ in length, a
 * response is not an integer from 0 to 2³¹ - 1 or `null`, the ids do not match the rows and columns or repeat, or the
 * times do not match the responses or are not positive finite numbers or `null`.
 */
export function buildResponseMatrix(
  rows: readonly (readonly (number | null)[])[],
  options: ResponseMatrixOptions = {},
): ResponseMatrix {
  return buildResponseMatrixWith(rows, options, copyRows);
}

/**
 * Builds a response matrix that holds the rows given, not copies of them: for a function that made the rows and knows
 * that every response in them is an integer from 0 to 2³¹ - 1 or `null`. Checks the rest of what {@link
 * buildResponseMatrix} checks.
 */
export function buildResponseMatrixOfRows(
  rows: readonly (readonly (number | null)[])[],
  options: ResponseMatrixOptions = {},
): ResponseMatrix {
  return buildResponseMatrixWith(rows, options, (owned) => owned, { responses: false });
}

function buildResponseMatrixWith(
  rows: readonly (readonly (number | null)[])[],
  options: ResponseMatrixOptions,
  own: (rows: readonly (readonly (number | null)[])[]) => readonly (readonly (number | null)[])[],
  checks: Checks = {},
): ResponseMatrix {
  checkArray(rows, "The rows of a response matrix must be an array");
  checkOptions(options);
  // A first row that is not an array has no length to count items by: the check of the rows names it.
  const first = rows[0];
  const {
    personIds = defaultIds("person", rows.length),
    itemIds = defaultIds("item", Array.isArray(first) ? first.length : 0),
  } = options;
  const matrix: ResponseMatrix = {
    personIds,
    itemIds,
    responses: rows,
    ...(options.times !== undefined && { times: options.times }),
  };
  // Ids that are made here are unique strings, so they need no check.
  checkMatrix(matrix, {
    personIds: options.personIds !== undefined,
    itemIds: options.itemIds !== undefined,
    ...checks,
  });
  return {
    ...matrix,
    responses: own(rows),
    ...(matrix.times !== undefined && { times: copyRows(matrix.times) }),
  };
}

/**
 * Builds a response matrix from one column of responses per item.
 *
 * @param columns - The responses to each item, one entry for each person: an integer from 0, or `null` when missing.
 * @param options - The ids of the persons and items, and the response times.
 * @returns The matrix, whose ids are `person1`, `item1` and so on without the options.
 * @throws `DataError` when the columns are not arrays of equal length, or in the cases listed for {@link
 * buildResponseMatrix}.
 */
export function buildResponseMatrixFromColumns(
  columns: readonly (readonly (number | null)[])[],
  options: ResponseMatrixOptions = {},
): ResponseMatrix {
  checkArray(columns, "The columns of a response matrix must be an array");
  // Every column is checked before the length of the first sizes the rows.
  columns.forEach((column, item) => {
    checkArray(column, `Column ${item + 1} must be an array of responses`);
  });
  const personCount = columns[0]?.length ?? 0;
  const rows: (number | null)[][] = Array.from({ length: personCount }, () => []);
  columns.forEach((column, item) => {
    if (column.length !== personCount) {
      throw new DataError(`Column ${item + 1} has ${column.length} responses, column 1 has ${personCount}.`);
    }
    column.forEach((response, person) => {
      rows[person]?.push(response);
    });
  });
  return buildResponseMatrix(rows, options);
}

/** Checks that a response matrix is well formed, throwing a `DataError` that names the first problem found. */
export function checkResponseMatrix(matrix: ResponseMatrix): void {
  checkMatrix(matrix);
}

/** The parts of a matrix that take a pass over its ids or its cells: each is checked unless it is `false`. */
interface Checks {
  readonly personIds?: boolean;
  readonly itemIds?: boolean;
  readonly responses?: boolean;
}

function checkMatrix(
  matrix: ResponseMatrix,
  { personIds: checkPersonIds = true, itemIds: checkItemIds = true, responses: checkCells = true }: Checks = {},
): void {
  // Matrices built by hand or loaded from JSON may be another value, lack a field or hold another type.
  const { personIds, itemIds, responses } = matrix ?? {};
  if (!(Array.isArray(personIds) && Array.isArray(itemIds) && Array.isArray(responses))) {
    throw new DataError("A response matrix needs the arrays personIds, itemIds and responses.");
  }
  if (personIds.length !== responses.length) {
    throw new DataError(`Found ${responses.length} rows of responses and ${personIds.length} person ids.`);
  }
  // By index, so that a hole in the rows is a row that is not an array, which forEach would skip.
  for (let person = 0; person < responses.length; person++) {
    const row: unknown = responses[person];
    if (!Array.isArray(row)) throw new DataError(`The responses of person "${personIds[person]}" must be an array.`);
    if (row.length !== itemIds.length) {
      throw new DataError(
        `Person "${personIds[person]}" has ${row.length} responses, but there are ${itemIds.length} item ids.`,
      );
    }
  }
  if (responses.length === 0 || itemIds.length === 0) {
    throw new DataError("A response matrix needs at least one person and one item.");
  }
  if (checkPersonIds) indexIds("person", personIds);
  if (checkItemIds) indexIds("item", itemIds);
  if (checkCells) {
    responses.forEach((row, person) => {
      checkRow(row, personIds[person], itemIds);
    });
  }
  if (matrix.times !== undefined) checkTimes(matrix);
}

/** Checks that each response of a person is an integer from 0 to {@link MAX_RESPONSE} or `null`. */
function checkRow(row: readonly unknown[], personId: string | undefined, itemIds: readonly string[]): void {
  for (let item = 0; item < row.length; item++) {
    const response = row[item];
    if (response !== null && !isResponse(response)) {
      throw new DataError(
        `The response of person "${personId}" to item "${itemIds[item]}" must be an integer from 0 to ${MAX_RESPONSE} or null, found ${describe(response)}.`,
      );
    }
  }
}

/**
 * Checks that a matrix's times have one row per person and one time per item, each a positive finite number where
 * there is a response, or `null`.
 */
function checkTimes(matrix: ResponseMatrix): void {
  const { itemIds, responses, times } = matrix;
  if (!(Array.isArray(times) && times.length === responses.length)) {
    throw new DataError(`A response matrix needs one row of times per person, ${responses.length} rows.`);
  }
  // By index, so that a hole in the rows of times is a row that is not an array, which forEach would skip.
  for (let person = 0; person < times.length; person++) {
    const row: unknown = times[person];
    if (!(Array.isArray(row) && row.length === itemIds.length)) {
      throw new DataError(`Person "${matrix.personIds[person]}" needs one time per item, ${itemIds.length} times.`);
    }
    checkTimeRow(matrix, person, row);
  }
}

/** Checks that each time of a person is a positive finite number where there is a response, or `null`. */
function checkTimeRow(
  { personIds, itemIds, responses }: ResponseMatrix,
  person: number,
  row: readonly unknown[],
): void {
  for (const [item, time] of row.entries()) {
    if (time === null) continue;
    const place = `The time of person "${personIds[person]}" on item "${itemIds[item]}"`;
    if (!isPositive(time)) {
      throw new DataError(`${place} must be a positive number or null, found ${describe(time)}.`);
    }
    if (responses[person]?.[item] === null) throw new DataError(`${place} needs a response, found ${time}.`);
  }
}

function copyRows<T>(rows: readonly (readonly T[])[]): T[][] {
  return rows.map((row) => row.slice());
}

function defaultIds(prefix: string, count: number): string[] {
  return Array.from({ length: count }, (_, index) => `${prefix}${index + 1}`);
}

import { checkArray, checkOptions, DataError, describe, isPositive } from "@measurekit/core";
import { indexIds } from "./ids.js";

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
 * @throws `DataError` when the rows are not an array, there are no persons or no items, the rows differ in length, a
 * response is not a non-negative integer or `null`, the ids do not match the rows and columns or repeat, or the times
 * do not match the responses or are not positive finite numbers or `null`.
 */
export function responseMatrix(
  rows: readonly (readonly (number | null)[])[],
  options: ResponseMatrixOptions = {},
): ResponseMatrix {
  return buildResponseMatrix(rows, options, copyRows);
}

/**
 * Builds a response matrix that holds the rows given, not copies of them: for a function that made the rows and gives
 * them up. Checks what {@link responseMatrix} checks.
 */
export function responseMatrixOfRows(
  rows: readonly (readonly (number | null)[])[],
  options: ResponseMatrixOptions = {},
): ResponseMatrix {
  return buildResponseMatrix(rows, options, (owned) => owned);
}

function buildResponseMatrix(
  rows: readonly (readonly (number | null)[])[],
  options: ResponseMatrixOptions,
  own: (rows: readonly (readonly (number | null)[])[]) => readonly (readonly (number | null)[])[],
): ResponseMatrix {
  checkArray(rows, "The rows of a response matrix must be an array");
  checkOptions(options);
  const { personIds = defaultIds("person", rows.length), itemIds = defaultIds("item", rows[0]?.length ?? 0) } = options;
  const matrix: ResponseMatrix = {
    personIds,
    itemIds,
    responses: rows,
    ...(options.times !== undefined && { times: options.times }),
  };
  checkResponseMatrix(matrix);
  return {
    ...matrix,
    responses: own(rows),
    ...(matrix.times !== undefined && { times: copyRows(matrix.times) }),
  };
}

/**
 * Builds a response matrix from one column of responses per item.
 *
 * @throws `DataError` when the columns are not arrays of equal length, or in the cases listed for {@link
 * responseMatrix}.
 */
export function responseMatrixFromColumns(
  columns: readonly (readonly (number | null)[])[],
  options: ResponseMatrixOptions = {},
): ResponseMatrix {
  checkArray(columns, "The columns of a response matrix must be an array");
  const personCount = columns[0]?.length ?? 0;
  const rows: (number | null)[][] = Array.from({ length: personCount }, () => []);
  columns.forEach((column, item) => {
    checkArray(column, `Column ${item + 1} must be an array of responses`);
    if (column.length !== personCount) {
      throw new DataError(`Column ${item + 1} has ${column.length} responses, column 1 has ${personCount}.`);
    }
    column.forEach((response, person) => {
      rows[person]?.push(response);
    });
  });
  return responseMatrix(rows, options);
}

/** Checks that a response matrix is well formed, throwing a `DataError` that names the first problem found. */
export function checkResponseMatrix(matrix: ResponseMatrix): void {
  // Matrices built by hand or loaded from JSON may be another value, lack a field or hold another type.
  const { personIds, itemIds, responses } = matrix ?? {};
  if (!(Array.isArray(personIds) && Array.isArray(itemIds) && Array.isArray(responses))) {
    throw new DataError("A response matrix needs the arrays personIds, itemIds and responses.");
  }
  if (responses.length === 0 || itemIds.length === 0) {
    throw new DataError("A response matrix needs at least one person and one item.");
  }
  if (personIds.length !== responses.length) {
    throw new DataError(`Found ${responses.length} rows of responses and ${personIds.length} person ids.`);
  }
  indexIds("person", personIds);
  indexIds("item", itemIds);
  responses.forEach((row, person) => {
    if (!Array.isArray(row)) throw new DataError(`The responses of person "${personIds[person]}" must be an array.`);
    if (row.length !== itemIds.length) {
      throw new DataError(
        `Person "${personIds[person]}" has ${row.length} responses, but there are ${itemIds.length} item ids.`,
      );
    }
    for (let item = 0; item < row.length; item++) {
      const response = row[item];
      if (response !== null && !(typeof response === "number" && Number.isSafeInteger(response) && response >= 0)) {
        throw new DataError(
          `The response of person "${personIds[person]}" to item "${itemIds[item]}" must be a non-negative integer or null, found ${describe(response)}.`,
        );
      }
    }
  });
  if (matrix.times !== undefined) checkTimes(matrix);
}

/**
 * Checks that a matrix's times have one row per person and one time per item, each a positive finite number where
 * there is a response, or `null`.
 */
function checkTimes({ personIds, itemIds, responses, times }: ResponseMatrix): void {
  if (!(Array.isArray(times) && times.length === responses.length)) {
    throw new DataError(`A response matrix needs one row of times per person, ${responses.length} rows.`);
  }
  times.forEach((row, person) => {
    if (!(Array.isArray(row) && row.length === itemIds.length)) {
      throw new DataError(`Person "${personIds[person]}" needs one time per item, ${itemIds.length} times.`);
    }
    for (const [item, time] of row.entries()) {
      if (time === null) continue;
      const place = `The time of person "${personIds[person]}" on item "${itemIds[item]}"`;
      if (!isPositive(time)) {
        throw new DataError(`${place} must be a positive number or null, found ${describe(time)}.`);
      }
      if (responses[person]?.[item] === null) throw new DataError(`${place} needs a response, found ${time}.`);
    }
  });
}

function copyRows<T>(rows: readonly (readonly T[])[]): T[][] {
  return rows.map((row) => row.slice());
}

function defaultIds(prefix: string, count: number): string[] {
  return Array.from({ length: count }, (_, index) => `${prefix}${index + 1}`);
}

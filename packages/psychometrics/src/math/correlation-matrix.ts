import { checkArray, DataError, describe } from "@measurekit/core";

/** How far from 1 the diagonal of a correlation matrix, and from symmetric, may be, and how far past 1 an entry. */
const MATRIX_TOLERANCE = 1e-8;

/**
 * Checks that a value is a matrix of correlations, as rows: a square array of numbers from -1 to 1, symmetric, with 1
 * on the diagonal, each within 1e-8.
 *
 * @param label - Names the matrix in the errors, such as "the correlation".
 * @param size - The number of rows it must have; by default, any number from 1.
 * @throws `DataError` naming the first row or entry that is not.
 */
export function checkCorrelationMatrix(matrix: unknown, label: string, size?: number): void {
  const name = `${label.charAt(0).toUpperCase()}${label.slice(1)}`;
  checkArray(matrix, `${name} must be an array of rows`);
  const rows = matrix as readonly unknown[];
  if (rows.length === 0 || (size !== undefined && rows.length !== size)) {
    throw new DataError(
      `${name} needs ${size === undefined ? "at least one row" : `${size} rows`}, found ${rows.length}.`,
    );
  }
  const square: (readonly unknown[])[] = [];
  rows.forEach((row, i) => {
    checkArray(row, `Row ${i + 1} of ${label} must be an array`);
    const entries = row as readonly unknown[];
    if (entries.length !== rows.length) {
      throw new DataError(`Row ${i + 1} of ${label} needs ${rows.length} entries, found ${entries.length}.`);
    }
    square.push(entries);
  });
  for (let i = 0; i < square.length; i++) {
    for (let j = i; j < square.length; j++) checkEntry(square, i, j, label, name);
  }
}

/** Checks the entry (`i`, `j`) of a square matrix, `j` from `i`, against its mirror. */
function checkEntry(matrix: readonly (readonly unknown[])[], i: number, j: number, label: string, name: string): void {
  const value = matrix[i]?.[j];
  const mirror = matrix[j]?.[i];
  if (!(typeof value === "number" && Math.abs(value) <= 1 + MATRIX_TOLERANCE)) {
    throw new DataError(
      `Entry (${i + 1}, ${j + 1}) of ${label} must be a number from -1 to 1, found ${describe(value)}.`,
    );
  }
  if (i === j && !(Math.abs(value - 1) <= MATRIX_TOLERANCE)) {
    throw new DataError(`Entry (${i + 1}, ${i + 1}) of ${label} must be 1, found ${value}.`);
  }
  if (!(typeof mirror === "number" && Math.abs(value - mirror) <= MATRIX_TOLERANCE)) {
    throw new DataError(
      `${name} must be symmetric, but entry (${i + 1}, ${j + 1}) is ${value} and entry (${j + 1}, ${i + 1}) is ${describe(mirror)}.`,
    );
  }
}

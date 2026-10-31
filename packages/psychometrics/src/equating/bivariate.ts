import { checkArray, DataError, describe } from "@measurekit/core";

/**
 * The number of persons at each total score and anchor score: a row for each total score from 0, holding a count for
 * each anchor score from 0. Counts may be fractional, as after smoothing.
 */
export type BivariateDistribution = readonly (readonly number[])[];

/** A bivariate distribution checked and flattened: counts by row, then column. */
export interface BivariateTable {
  /** The number of rows, the total scores. */
  readonly rows: number;
  /** The number of columns, the anchor scores. */
  readonly columns: number;
  /** The counts, row by row. */
  readonly counts: Float64Array;
  /** The number of persons. */
  readonly persons: number;
}

/** The words that name the two scores of a table in messages. */
export interface ScoreNames {
  /** What a row is, such as `total score`. */
  readonly row: string;
  /** What a column is, such as `anchor score`. */
  readonly column: string;
}

/** The scores of a NEAT design. */
const NEAT_SCORES: ScoreNames = { row: "total score", column: "anchor score" };

/**
 * Checks a bivariate distribution and flattens it.
 *
 * @param label - Names the distribution in error messages, such as `from`.
 * @throws `DataError` when it is not an array of equally long arrays of counts, has fewer than two rows or columns, a
 * count that is negative or not finite, or no persons.
 */
export function bivariateTable(
  distribution: BivariateDistribution,
  label: string,
  names: ScoreNames = NEAT_SCORES,
): BivariateTable {
  checkArray(distribution, `The ${label} distribution must be an array with a row of counts for each ${names.row}`);
  const rows = distribution.length;
  distribution.forEach((row, x) => {
    checkArray(row, `Row ${x + 1} of the ${label} distribution must be an array of counts for each ${names.column}`);
  });
  const columns = distribution[0]?.length ?? 0;
  if (rows < 2 || columns < 2) {
    throw new DataError(
      `The ${label} distribution needs at least the ${names.row}s 0 and 1 and the ${names.column}s 0 and 1, found ${rows} and ${columns} scores.`,
    );
  }
  const counts = new Float64Array(rows * columns);
  let persons = 0;
  distribution.forEach((row, x) => {
    if (row.length !== columns) {
      throw new DataError(
        `The ${label} distribution needs the same number of ${names.column}s in every row, found ${columns} in row 1 and ${row.length} in row ${x + 1}.`,
      );
    }
    row.forEach((count, v) => {
      if (!(Number.isFinite(count) && count >= 0)) {
        throw new DataError(
          `The ${label} distribution needs a non-negative count at each pair of scores, found ${describe(count)} at the ${names.row} ${x} and the ${names.column} ${v}.`,
        );
      }
      counts[x * columns + v] = count;
      persons += count;
    });
  });
  if (!(persons > 0)) throw new DataError(`The ${label} distribution needs at least one person.`);
  return { rows, columns, counts, persons };
}

/**
 * Checks the distributions of two groups that took forms of one test and a common anchor, and flattens them.
 *
 * @throws `DataError` when either is invalid, or they have different numbers of anchor scores.
 */
export function neatTables(from: BivariateDistribution, to: BivariateDistribution): [BivariateTable, BivariateTable] {
  const source = bivariateTable(from, "from");
  const target = bivariateTable(to, "to");
  if (source.columns !== target.columns) {
    throw new DataError(
      `The from and to distributions need the same anchor scores, found ${source.columns} and ${target.columns}.`,
    );
  }
  return [source, target];
}

/** The persons at each row score and at each column score of a table. */
export interface Margins {
  /** The persons at each row score. */
  readonly rows: Float64Array;
  /** The persons at each column score. */
  readonly columns: Float64Array;
}

/** Returns the numbers of persons at each score of a table, one score at a time. */
export function margins(table: BivariateTable): Margins {
  const rows = new Float64Array(table.rows);
  const columns = new Float64Array(table.columns);
  for (let x = 0; x < table.rows; x++) {
    for (let v = 0; v < table.columns; v++) {
      const count = table.counts[x * table.columns + v] ?? 0;
      rows[x] = (rows[x] ?? 0) + count;
      columns[v] = (columns[v] ?? 0) + count;
    }
  }
  return { rows, columns };
}

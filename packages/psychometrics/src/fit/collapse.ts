// The collapsing of S-X² tables of mirt 1.48's itemfit (collapseCells, mincell = 1), step by step, so that the
// statistic and its degrees of freedom match mirt's.

/** Observed and expected counts of one item by score group (rows) and category (columns). */
export interface CellTable {
  /** Observed counts. */
  readonly observed: number[][];
  /** Expected counts; NaN for a cell left out. */
  readonly expected: number[][];
}

/** mirt's smallest expected count of a cell. */
const MIN_EXPECTED = 1;
/** mirt's share of the score groups, in which the last category's expected total must reach {@link MIN_EXPECTED}. */
const RARE_CATEGORY_SHARE = 0.1;
/** mirt's smallest share of the score groups in which a category has an expected count. */
const EMPTY_CATEGORY_SHARE = 0.05;

/**
 * Collapses a table as mirt's itemfit does before computing S-X² and returns it, with NaN for the expected counts of
 * the cells left out, or `null` when fewer than two score groups or two categories remain. Expected counts are NaN
 * for impossible cells on input.
 */
export function collapseCells(observed: readonly number[][], expected: readonly number[][]): CellTable | null {
  const table: CellTable = { observed: observed.map((row) => [...row]), expected: expected.map((row) => [...row]) };
  mergeImpossibleRows(table);
  if (table.observed.length === 0) return null;
  removeRows(
    table,
    table.observed.map((row) => sum(row) === 0),
  );
  // mirt flags the rows with a small expected count before merging single persons, and then reads the flags by the
  // positions of the merged rows.
  const flagged = table.expected.map((row) => row.some((value) => value < MIN_EXPECTED));
  mergeSinglePersons(table);
  const columns = table.expected[0]?.length ?? 0;
  if (columns > 2) {
    table.expected.forEach((_, j) => {
      if (flagged[j] === true) collapseCategories(table, j, columns);
    });
  }
  mapExpected(table, (value) => (Number.isNaN(value) ? 0 : value));
  if (columns > 2) mergeRareLastCategories(table);
  mergeEmptyCategories(table);
  mergeSmallRows(table);
  mapExpected(table, (value) => (value === 0 ? Number.NaN : value));
  const rows = table.expected.length;
  return rows >= 2 && (table.expected[0]?.length ?? 0) >= 2 ? table : null;
}

/**
 * Merges the rows with impossible cells, the first half of them into the next row and the second half into the row
 * before them, treating the impossible cells as 0.
 */
function mergeImpossibleRows(table: CellTable): void {
  const impossible = table.expected.flatMap((row, j) => (row.some(Number.isNaN) ? [j] : []));
  mapExpected(table, (value) => (Number.isNaN(value) ? 0 : value));
  if (impossible.length === 0) return;
  const half = Math.floor(impossible.length / 2);
  const first = impossible[0] ?? 0;
  const last = impossible[impossible.length - 1] ?? 0;
  const remove = table.expected.map(() => false);
  if (half > 0) {
    const upTo = impossible[half - 1] ?? 0;
    for (let j = first; j <= upTo; j++) {
      addRow(table, j, upTo + 1);
      remove[j] = true;
    }
  }
  const from = impossible[half] ?? 0;
  for (let j = from; j <= last; j++) {
    addRow(table, j, from - 1);
    remove[j] = true;
  }
  removeRows(table, remove);
}

/** Merges each row with one observed person into the next, and a last such row into the one before. */
function mergeSinglePersons(table: CellTable): void {
  const remove = table.observed.map(() => false);
  for (let j = 0; j < table.observed.length - 1; j++) {
    if (sum(table.observed[j] ?? []) === 1) {
      addRow(table, j, j + 1);
      remove[j] = true;
    }
  }
  removeRows(table, remove);
  const last = table.observed.length - 1;
  if (last >= 1 && sum(table.observed[last] ?? []) === 1) {
    addRow(table, last, last - 1);
    removeRows(
      table,
      table.observed.map((_, j) => j === last),
    );
  }
}

/**
 * Collapses the categories of row j: merges the category with the smallest expected count, the last of equal ones,
 * into its neighbor with the smaller expected count, the left one on ties, until every count is at least
 * `MIN_EXPECTED` or two remain. The row keeps `columns` entries: NaN expected and 0 observed counts fill its end.
 */
function collapseCategories(table: CellTable, j: number, columns: number): void {
  const expected = [...(table.expected[j] ?? [])];
  const observed = [...(table.observed[j] ?? [])];
  while (expected.length > 2) {
    const smallest = expected.reduce((least, value) => Math.min(least, value), Number.POSITIVE_INFINITY);
    const at = expected.lastIndexOf(smallest);
    const target = mergeTarget(expected, at);
    expected[target] = (expected[target] ?? 0) + (expected[at] ?? 0);
    observed[target] = (observed[target] ?? 0) + (observed[at] ?? 0);
    expected.splice(at, 1);
    observed.splice(at, 1);
    if (expected.every((value) => value >= MIN_EXPECTED)) break;
  }
  table.expected[j] = [...expected, ...Array.from({ length: columns - expected.length }, () => Number.NaN)];
  table.observed[j] = [...observed, ...Array.from({ length: columns - observed.length }, () => 0)];
}

/** Returns the neighbor a category merges into: the only one at an end, otherwise the smaller, the left on ties. */
function mergeTarget(expected: readonly number[], at: number): number {
  if (at === 0) return 1;
  if (at === expected.length - 1) return at - 1;
  const left = expected[at - 1] ?? 0;
  const right = expected[at + 1] ?? 0;
  return Math.min(left, right) === left ? at - 1 : at + 1;
}

/**
 * Merges the last category into the one before while its expected total is below `MIN_EXPECTED` per tenth of the
 * rows.
 */
function mergeRareLastCategories(table: CellTable): void {
  let columns = table.expected[0]?.length ?? 0;
  while (columns > 2) {
    const total = table.expected.reduce((sum, row) => sum + (row[columns - 1] ?? 0), 0);
    if (!(total < MIN_EXPECTED * Math.ceil(table.expected.length * RARE_CATEGORY_SHARE))) return;
    mergeColumn(table, columns - 1);
    columns--;
  }
}

/** Merges, from the last category down to the second, each category with an expected count in fewer than 5% of rows. */
function mergeEmptyCategories(table: CellTable): void {
  const rows = table.expected.length;
  for (let k = (table.expected[0]?.length ?? 0) - 1; k >= 1; k--) {
    const filled = table.expected.filter((row) => (row[k] ?? 0) > 0).length;
    if (filled / rows < EMPTY_CATEGORY_SHARE) mergeColumn(table, k);
  }
}

/**
 * Merges score groups while a cell's expected count is above 0 and below `MIN_EXPECTED`: the first row with one into
 * its neighbor, the only one at an end, otherwise the one with fewer observed persons, the row before on ties.
 */
function mergeSmallRows(table: CellTable): void {
  for (;;) {
    const rows = table.expected.length;
    const j = table.expected.findIndex((row) => row.some((value) => value < MIN_EXPECTED && value !== 0));
    if (j < 0 || rows < 2 || (table.expected[0]?.length ?? 0) < 2) return;
    let target = j + 1;
    if (j === rows - 1) target = j - 1;
    else if (j > 0) target = sum(table.observed[j - 1] ?? []) <= sum(table.observed[j + 1] ?? []) ? j - 1 : j + 1;
    addRow(table, j, target);
    removeRows(
      table,
      table.expected.map((_, row) => row === j),
    );
  }
}

/** Adds column k to column k - 1 and removes it. */
function mergeColumn(table: CellTable, k: number): void {
  for (const rows of [table.expected, table.observed]) {
    for (const row of rows) {
      row[k - 1] = (row[k - 1] ?? 0) + (row[k] ?? 0);
      row.splice(k, 1);
    }
  }
}

/** Adds row `from` to row `to`, in both tables. */
function addRow(table: CellTable, from: number, to: number): void {
  for (const rows of [table.expected, table.observed]) {
    const source = rows[from] ?? [];
    const target = rows[to] ?? [];
    for (let k = 0; k < source.length; k++) target[k] = (target[k] ?? 0) + (source[k] ?? 0);
  }
}

/** Replaces every expected count by `f` of it. */
function mapExpected(table: CellTable, f: (value: number) => number): void {
  for (const row of table.expected) {
    for (let k = 0; k < row.length; k++) row[k] = f(row[k] ?? 0);
  }
}

/** Removes the flagged rows from both tables. */
function removeRows(table: CellTable, remove: readonly boolean[]): void {
  for (const rows of [table.expected, table.observed]) {
    let kept = 0;
    rows.forEach((row, j) => {
      if (remove[j] !== true) rows[kept++] = row;
    });
    rows.length = kept;
  }
}

function sum(values: readonly number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

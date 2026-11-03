import { checkArray, DataError, describe } from "@measurekit/core";

/** The checklist marks and global ratings of a performance test, by station, then person. */
export interface Stations {
  /** Each station's marks, one per person. */
  readonly marks: readonly Float64Array[];
  /** Each station's global ratings, one per person. */
  readonly ratings: readonly Float64Array[];
  /** The number of persons. */
  readonly persons: number;
}

/**
 * Checks the checklist marks and global ratings of the persons at the stations, by person, then station, and returns
 * them by station.
 *
 * @throws `DataError` when they are not arrays of rows of finite numbers, one row per person and one entry per station,
 * the same for both, or there is no person or station.
 */
export function readStations(marks: readonly (readonly number[])[], ratings: readonly (readonly number[])[]): Stations {
  checkArray(marks, "The marks must be an array of rows, one per person");
  checkArray(ratings, "The ratings must be an array of rows, one per person");
  if (marks.length !== ratings.length) {
    throw new DataError(
      `The marks and the ratings need a row for each person, found ${marks.length} and ${ratings.length}.`,
    );
  }
  const first = marks[0];
  if (first === undefined) throw new DataError("A borderline method needs the marks of at least one person.");
  checkArray(first, "The marks of person 1 must be an array, one per station");
  const stations = first.length;
  if (stations === 0) throw new DataError("A borderline method needs at least one station.");
  const persons = marks.length;
  const columns = (rows: readonly (readonly number[])[], name: string): Float64Array[] => {
    const out = Array.from({ length: stations }, () => new Float64Array(persons));
    // checkArray refused holes in the rows, so forEach visits every person.
    rows.forEach((row, person) => {
      checkArray(row, `The ${name} of person ${person + 1} must be an array, one per station`);
      if (row.length !== stations) {
        throw new DataError(`Person ${person + 1} needs ${stations} ${name}, one per station, found ${row.length}.`);
      }
      for (let station = 0; station < stations; station++) {
        const value = row[station];
        if (!Number.isFinite(value)) {
          throw new DataError(
            `Person ${person + 1} needs a finite ${name.slice(0, -1)} at station ${station + 1}, found ${describe(value)}.`,
          );
        }
        (out[station] ?? new Float64Array(0))[person] = value ?? 0;
      }
    });
    return out;
  };
  return { marks: columns(marks, "marks"), ratings: columns(ratings, "ratings"), persons };
}

import { checkArray, checkRecord, DataError, describe } from "@measurekit/core";
import { checkResponseMatrix, indexIds, type ResponseMatrix } from "../data/internal.js";
import { variance } from "../math/internal.js";
import { type CompleteCases, completeCases, itemScores } from "./complete-cases.js";
import { alphaFromVariances } from "./reliability.js";

/** The columns of the matrix that make up each section, by name. */
interface SectionItems {
  readonly section: string;
  readonly columns: readonly number[];
}

/** The persons who answered every item of the sections, with the columns of each section among the cases' items. */
export interface SectionCases {
  /** The complete cases, whose items are those of the sections in the order given. */
  readonly cases: CompleteCases;
  /** The sections' names, in the order given. */
  readonly names: readonly string[];
  /** The columns of the cases that hold each section's items. */
  readonly indices: readonly (readonly number[])[];
}

/** A section's subscores, their variance and the coefficient alpha of its items. */
export interface SectionStatistics {
  /** The sum of the section's items for each complete case. */
  readonly scores: Float64Array;
  /** The variance of the subscores. */
  readonly variance: number;
  /** Coefficient alpha of the section's items; 0 where the subscores are constant. */
  readonly alpha: number;
}

/**
 * Checks a matrix and its sections and returns the persons who answered every item of the sections.
 *
 * @param subject - Names the analysis in error messages, such as `Subscore value`.
 * @throws `DataError` when the matrix is invalid, the sections are not an object of arrays of item ids, there are fewer
 * than two sections, a section has fewer than two items, an id is not an item of the matrix or is in two places, or
 * fewer than two persons answered every item of the sections.
 */
export function sectionCases(
  matrix: ResponseMatrix,
  sections: Readonly<Record<string, readonly string[]>>,
  subject: string,
): SectionCases {
  checkResponseMatrix(matrix);
  const items = sectionItems(matrix, sections, subject);
  const used = items.flatMap(({ columns }) => columns);
  const cases = completeCases({
    ...matrix,
    itemIds: used.map((column) => matrix.itemIds[column] ?? ""),
    responses: matrix.responses.map((row) => used.map((column) => row[column] ?? null)),
  });
  if (cases.persons.length < 2) {
    throw new DataError(`${subject} needs at least two persons who answered every item of the sections.`);
  }
  let start = 0;
  const indices = items.map(({ columns }) => {
    const result = columns.map((_, i) => start + i);
    start += columns.length;
    return result;
  });
  return { cases, names: items.map(({ section }) => section), indices };
}

/** Returns a section's subscores, their variance and the coefficient alpha of its items, which are `indices` of the cases. */
export function sectionStatistics(cases: CompleteCases, indices: readonly number[]): SectionStatistics {
  const scores = itemScores(cases, indices);
  const scoreVariance = variance(scores);
  let itemVariance = 0;
  for (const item of indices) itemVariance += variance(itemScores(cases, [item]));
  const alpha = scoreVariance === 0 ? 0 : alphaFromVariances(indices.length, itemVariance, scoreVariance);
  return { scores, variance: scoreVariance, alpha };
}

/** Checks the sections against the matrix and returns each one's item columns, in the order given. */
function sectionItems(
  matrix: ResponseMatrix,
  sections: Readonly<Record<string, readonly string[]>>,
  subject: string,
): SectionItems[] {
  checkRecord(sections, "The sections must be an object of item ids by section name");
  const columns = indexIds("item", matrix.itemIds);
  const owner = new Map<string, string>();
  const result = Object.entries(sections).map(([section, ids]) => {
    checkArray(ids, `The items of section "${section}" must be an array of item ids`);
    if (ids.length < 2) {
      throw new DataError(`Section "${section}" needs at least two items, found ${ids.length}.`);
    }
    return {
      section,
      columns: ids.map((id) => {
        const column = columns.get(id);
        if (column === undefined) {
          throw new DataError(`Section "${section}" has the item ${describe(id)}, which is not an item of the matrix.`);
        }
        const first = owner.get(id);
        if (first !== undefined) {
          throw new DataError(
            `The item "${id}" is in the sections "${first}" and "${section}"; a section's items are its own.`,
          );
        }
        owner.set(id, section);
        return column;
      }),
    };
  });
  if (result.length < 2) {
    throw new DataError(`${subject} needs at least two sections, found ${result.length}.`);
  }
  return result;
}

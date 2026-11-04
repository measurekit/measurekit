import { checkArray, checkOptions, DataError, describe, type FieldNames } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { type DifData, type DifGroup, itemCasesOf, polytomousData } from "./groups.js";
import { type DifAnalysis, type PurificationOptions, purified } from "./purification.js";

/** Options of the regressions that take a matching score of the caller's: {@link ordinalLogisticRegressionDif} and {@link logisticDiscriminantDif}. */
export interface MatchingOptions extends PurificationOptions {
  /**
   * One matching score per person, such as θ from a graded response calibration, or `null` to leave a person out. It
   * replaces the total score as the criterion and cannot be purified, so it cannot be used with `purify`.
   *
   * @defaultValue the total score over the matching items
   */
  readonly matching?: readonly (number | null)[];
}

const MATCHING_OPTIONS: FieldNames<MatchingOptions> = { matching: true, purify: true };

/** The persons of an item's regressions, with their matching scores. */
export interface MatchedCases {
  /** Each person's response to the item. */
  readonly responses: Int32Array;
  /** Each person's matching score. */
  readonly scores: Float64Array;
  /** 1 for each person of the focal group. */
  readonly focal: Uint8Array;
}

/**
 * Returns the statistics of every item on the persons {@link matchedCases} gives, purified on request.
 *
 * @param statistic - Names the statistic in error messages.
 * @param statistics - Computes one item's statistics from its persons and its id.
 * @param flagged - Whether an item's statistics flag it in purification.
 * @throws `DataError` when the matrix, the groups or the options are invalid, either group has no person with a
 * response, `matching` is not an array of finite numbers and `null` with one entry per person, or `purify` is `true`
 * with `matching`.
 */
export function matchedAnalysis<TItem>(
  matrix: ResponseMatrix,
  groups: readonly DifGroup[],
  options: MatchingOptions,
  statistic: string,
  statistics: (cases: MatchedCases, id: string) => TItem,
  flagged: (item: TItem) => boolean,
): DifAnalysis<{ readonly id: string } & TItem> {
  checkOptions(options, MATCHING_OPTIONS);
  const { purify = false, matching } = options;
  const data = polytomousData(matrix, groups, statistic);
  if (matching !== undefined) {
    checkMatching(matrix, matching);
    if (purify === true) {
      throw new DataError("The option purify purifies the total score, so it cannot be used with the option matching.");
    }
  }
  return purified(
    matrix.itemIds,
    purify,
    (anchors) => {
      const casesOf = matching === undefined ? itemCasesOf(data, anchors) : null;
      return matrix.itemIds.map((id, item) => ({
        id,
        ...statistics(casesOf === null ? givenCases(data, item, matching ?? []) : casesOf(item), id),
      }));
    },
    flagged,
  );
}

function checkMatching(matrix: ResponseMatrix, matching: readonly (number | null)[]): void {
  checkArray(matching, "The option matching must be an array");
  if (matching.length !== matrix.personIds.length) {
    throw new DataError(`Found ${matching.length} matching scores for ${matrix.personIds.length} persons.`);
  }
  matching.forEach((score, person) => {
    if (score !== null && !Number.isFinite(score)) {
      throw new DataError(
        `The matching score of person "${matrix.personIds[person]}" must be a finite number or null, found ${describe(score)}.`,
      );
    }
  });
}

/** Returns the persons who answered an item and have a matching score given. */
function givenCases(data: DifData, item: number, matching: readonly (number | null)[]): MatchedCases {
  const persons: number[] = [];
  matching.forEach((score, person) => {
    if (score !== null && (data.responses[person * data.itemCount + item] ?? -1) >= 0) persons.push(person);
  });
  return {
    responses: Int32Array.from(persons, (person) => data.responses[person * data.itemCount + item] ?? 0),
    scores: Float64Array.from(persons, (person) => matching[person] ?? 0),
    focal: Uint8Array.from(persons, (person) => data.focal[person] ?? 0),
  };
}

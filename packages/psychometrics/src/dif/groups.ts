import { checkArray, checkChoice, DataError } from "@measurekit/core";
import { checkDichotomous } from "../data/dichotomous.js";
import { checkResponseMatrix, type ResponseMatrix } from "../data/response-matrix.js";

/** A person's group in a DIF analysis: the reference group, or the focal group whose items are studied. */
export type DifGroup = "reference" | "focal";

const DIF_GROUPS: readonly DifGroup[] = ["reference", "focal"];

/**
 * Checks a response matrix and that `groups` gives each person's group.
 *
 * @throws `DataError` when the matrix is invalid, the groups are not an array, their number differs from the number of
 * persons, or a group is neither `"reference"` nor `"focal"`.
 */
export function checkGroups(matrix: ResponseMatrix, groups: readonly DifGroup[]): void {
  checkResponseMatrix(matrix);
  checkArray(groups, "The groups must be an array");
  if (groups.length !== matrix.personIds.length) {
    throw new DataError(`Found ${groups.length} groups for ${matrix.personIds.length} persons.`);
  }
  groups.forEach((group, person) => {
    checkChoice(group, `The group of person "${matrix.personIds[person]}"`, DIF_GROUPS);
  });
}

/** A dichotomous response matrix with each person's group. */
export interface DifData {
  /** The number of items. */
  readonly itemCount: number;
  /** Responses by person, then item: 0, 1, or -1 when missing. */
  readonly responses: Int8Array;
  /** 1 for each person of the focal group, 0 for the reference group. */
  readonly focal: Uint8Array;
}

/** The persons an item's statistics use, with their matching scores. */
export interface ItemCases {
  /** Each person's response to the item: 0 or 1. */
  readonly responses: Uint8Array;
  /** Each person's matching score, from 0 to `maxScore`. */
  readonly scores: Int32Array;
  /** 1 for each person of the focal group. */
  readonly focal: Uint8Array;
  /** The highest matching score: the number of matching items. */
  readonly maxScore: number;
}

/**
 * Checks a dichotomous response matrix with its groups and returns it as {@link DifData}.
 *
 * @param statistic - Names the statistic in error messages.
 * @throws `DataError` in the cases of {@link checkGroups}, when a response is neither 0 nor 1, or when either group
 * has no person with a response.
 */
export function dichotomousData(matrix: ResponseMatrix, groups: readonly DifGroup[], statistic: string): DifData {
  checkGroups(matrix, groups);
  checkDichotomous(matrix, statistic);
  const itemCount = matrix.itemIds.length;
  const responses = new Int8Array(matrix.responses.length * itemCount).fill(-1);
  const answered = [false, false];
  matrix.responses.forEach((row, person) => {
    row.forEach((response, item) => {
      if (response === null) return;
      responses[person * itemCount + item] = response;
      answered[groups[person] === "focal" ? 1 : 0] = true;
    });
  });
  if (!(answered[0] && answered[1])) throw new DataError(`${statistic} needs persons with responses in both groups.`);
  return { itemCount, responses, focal: Uint8Array.from(groups, (group) => (group === "focal" ? 1 : 0)) };
}

/** Returns a person's number of answered and correct matching items other than `item`, or `null` without any. */
function restScore(
  { itemCount, responses }: DifData,
  person: number,
  item: number,
  anchors: Uint8Array,
): { answered: number; correct: number } | null {
  let answered = 0;
  let correct = 0;
  for (let j = 0; j < itemCount; j++) {
    const other = responses[person * itemCount + j] ?? -1;
    if (j === item || anchors[j] !== 1 || other < 0) continue;
    answered++;
    correct += other;
  }
  return answered === 0 ? null : { answered, correct };
}

/**
 * Returns the persons who answered an item and at least one other matching item, with their matching scores.
 *
 * @remarks
 * The matching items are the anchors and the studied item. A person's score is the response to the studied item plus
 * the total over the other matching items the person answered, scaled to all of them and rounded to the nearest
 * integer, halves up: x + round(m s / n) for s correct of n answered of the m others. With every response present it is
 * the total score over the matching items.
 */
export function itemCases(data: DifData, item: number, anchors: Uint8Array): ItemCases {
  const { itemCount, responses, focal } = data;
  let others = 0;
  for (let j = 0; j < itemCount; j++) if (j !== item && anchors[j] === 1) others++;
  const persons: number[] = [];
  const scores: number[] = [];
  for (let person = 0; person < focal.length; person++) {
    const response = responses[person * itemCount + item] ?? -1;
    const rest = response < 0 ? null : restScore(data, person, item, anchors);
    if (rest === null) continue;
    persons.push(person);
    scores.push(response + Math.floor((others * rest.correct) / rest.answered + 0.5));
  }
  return {
    responses: Uint8Array.from(persons, (person) => responses[person * itemCount + item] ?? 0),
    scores: Int32Array.from(scores),
    focal: Uint8Array.from(persons, (person) => focal[person] ?? 0),
    maxScore: others + 1,
  };
}

import { checkArray, checkChoice, DataError } from "@measurekit/core";
import { checkDichotomous, checkResponseMatrix, type ResponseMatrix } from "../data/internal.js";

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

/** A response matrix with each person's group. */
export interface DifData {
  /** The number of items. */
  readonly itemCount: number;
  /** Responses by person, then item: a score from 0, or -1 when missing. */
  readonly responses: Int32Array;
  /** 1 for each person of the focal group, 0 for the reference group. */
  readonly focal: Uint8Array;
  /**
   * The highest possible score of each item: its highest response in the matrix, at least 1, as a matrix does not
   * declare more. It is 1 for a dichotomous item.
   */
  readonly maxScores: Int32Array;
}

/** The persons an item's statistics use, with their matching scores. */
export interface ItemCases {
  /** Each person's response to the item, from 0 to the item's highest possible score. */
  readonly responses: Int32Array;
  /** Each person's matching score, from 0 to `maxScore`: in 64-bit floats, as a sum of responses may pass 2³¹. */
  readonly scores: Float64Array;
  /** 1 for each person of the focal group. */
  readonly focal: Uint8Array;
  /** The highest matching score: the highest possible scores of the matching items, the studied item's included. */
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
  return difData(matrix, groups, statistic);
}

/**
 * Checks a response matrix of items with any number of categories with its groups and returns it as {@link DifData}.
 *
 * @param statistic - Names the statistic in error messages.
 * @throws `DataError` in the cases of {@link checkGroups}, or when either group has no person with a response.
 */
export function polytomousData(matrix: ResponseMatrix, groups: readonly DifGroup[], statistic: string): DifData {
  checkGroups(matrix, groups);
  return difData(matrix, groups, statistic);
}

/** Returns a checked matrix and its groups as {@link DifData}. */
function difData(matrix: ResponseMatrix, groups: readonly DifGroup[], statistic: string): DifData {
  const itemCount = matrix.itemIds.length;
  const responses = new Int32Array(matrix.responses.length * itemCount).fill(-1);
  const maxScores = new Int32Array(itemCount).fill(1);
  const answered = [false, false];
  matrix.responses.forEach((row, person) => {
    row.forEach((response, item) => {
      if (response === null) return;
      responses[person * itemCount + item] = response;
      maxScores[item] = Math.max(maxScores[item] ?? 1, response);
      answered[groups[person] === "focal" ? 1 : 0] = true;
    });
  });
  if (!(answered[0] && answered[1])) throw new DataError(`${statistic} needs persons with responses in both groups.`);
  return { itemCount, responses, focal: Uint8Array.from(groups, (group) => (group === "focal" ? 1 : 0)), maxScores };
}

/**
 * Returns a function that gives the persons who answered an item and at least one other matching item, with their
 * matching scores, for the anchors given.
 *
 * @remarks
 * The matching items are the anchors and the studied item. A person's score is the response to the studied item plus
 * the total over the other matching items the person answered, scaled to all of them and rounded to the nearest
 * integer, halves up: x + round(M s / m) for s the total over the answered others, m their highest possible sum and M
 * that of all the others. With every response present it is the total score over the matching items. Each person's
 * total and highest possible sum over the answered anchors are summed once, and an anchor's own share taken away for
 * its statistics: sums of integers, exact, so that every item takes time with the persons, not the persons times the
 * items.
 */
export function itemCasesOf(data: DifData, anchors: Uint8Array): (item: number) => ItemCases {
  const { itemCount, responses, focal, maxScores } = data;
  const { possible, totals, anchorPossible } = anchorSums(data, anchors);
  return (item) => {
    // 1 when the item is an anchor: the person answered it, so its share of the person's sums is its own.
    const inAnchors = anchors[item] === 1 ? 1 : 0;
    const own = maxScores[item] ?? 1;
    const others = anchorPossible - inAnchors * own;
    const persons: number[] = [];
    const scores: number[] = [];
    for (let person = 0; person < focal.length; person++) {
      const response = responses[person * itemCount + item] ?? -1;
      const restPossible = (possible[person] ?? 0) - inAnchors * own;
      if (response < 0 || restPossible === 0) continue;
      const rest = (totals[person] ?? 0) - inAnchors * response;
      persons.push(person);
      scores.push(response + Math.floor((others * rest) / restPossible + 0.5));
    }
    return {
      responses: Int32Array.from(persons, (person) => responses[person * itemCount + item] ?? 0),
      scores: Float64Array.from(scores),
      focal: Uint8Array.from(persons, (person) => focal[person] ?? 0),
      maxScore: others + own,
    };
  };
}

/** Returns each person's highest possible sum and total over the answered anchors, and the anchors' possible sum. */
function anchorSums(
  { itemCount, responses, focal, maxScores }: DifData,
  anchors: Uint8Array,
): { possible: Float64Array; totals: Float64Array; anchorPossible: number } {
  let anchorPossible = 0;
  for (let j = 0; j < itemCount; j++) if (anchors[j] === 1) anchorPossible += maxScores[j] ?? 1;
  const possible = new Float64Array(focal.length);
  const totals = new Float64Array(focal.length);
  for (let person = 0; person < focal.length; person++) {
    for (let j = 0; j < itemCount; j++) {
      const response = responses[person * itemCount + j] ?? -1;
      if (anchors[j] !== 1 || response < 0) continue;
      possible[person] = (possible[person] ?? 0) + (maxScores[j] ?? 1);
      totals[person] = (totals[person] ?? 0) + response;
    }
  }
  return { possible, totals, anchorPossible };
}

/**
 * The most categories an item's persons may use in the polytomous statistics, whose generalized Mantel test and ordinal
 * regression hold a matrix of every pair of them: FACETS's limit of a rating scale, 255.
 */
const MAX_CATEGORIES = 255;

/**
 * Checks the number of categories an item's persons used.
 *
 * @param statistic - Names the statistic in the message.
 * @throws `DataError` when there are more than 255.
 */
export function checkCategoryCount(categories: number, statistic: string, id: string): void {
  if (categories > MAX_CATEGORIES) {
    throw new DataError(
      `${statistic} takes items of at most ${MAX_CATEGORIES} categories, found ${categories} used for item "${id}".`,
    );
  }
}

/** The responses and matching scores of the persons a statistic uses, which {@link countByScore} tabulates. */
export interface ScoredCases {
  /** Each person's response or studied score: an integer from 0. */
  readonly responses: ArrayLike<number>;
  /** Each person's matching score. */
  readonly scores: ArrayLike<number>;
  /** 1 for each person of the focal group. */
  readonly focal: Uint8Array;
}

/**
 * Persons counted by matching score and group: the sums the statistics on a matching score are made of. The score groups
 * are the matching scores that occur and the categories the responses that occur, each in ascending order, and each cell
 * lists only the categories that occur in it, so that the tables grow with the persons, not with the highest score or
 * response, nor with the score groups times the categories.
 */
export interface ScoreGroups {
  /** The number of categories that occur. */
  readonly categories: number;
  /** The number of score groups. */
  readonly scoreCount: number;
  /** The matching score of each score group. */
  readonly scores: Float64Array;
  /** The persons in score group s and group g (1 for focal), at `2 s + g`. */
  readonly sizes: Float64Array;
  /** The sums of their responses, at `2 s + g`. */
  readonly totals: Float64Array;
  /** The sums of their squared responses, at `2 s + g`. */
  readonly squares: Float64Array;
  /**
   * Where the categories of score group s and group g start in `cellCategories` and `cellCounts`, at `2 s + g`, and end,
   * at `2 s + g + 1`.
   */
  readonly cellStarts: Int32Array;
  /** The categories that occur in each cell, by their rank among all, ascending within the cell. */
  readonly cellCategories: Int32Array;
  /** The persons of each cell in each of its categories. */
  readonly cellCounts: Float64Array;
}

/** Counts persons by matching score and group, with the sums of their responses and the categories they responded. */
export function countByScore({ responses, scores, focal }: ScoredCases): ScoreGroups {
  const groupOf = ranksOf(scores);
  const categoryOf = ranksOf(responses);
  const scoreCount = groupOf.size;
  const categories = categoryOf.size;
  const sizes = new Float64Array(2 * scoreCount);
  const totals = new Float64Array(2 * scoreCount);
  const squares = new Float64Array(2 * scoreCount);
  // Each person's cell and category as one key, cell × categories + category: an integer below 2 persons², exact.
  const keys = new Float64Array(scores.length);
  for (let person = 0; person < scores.length; person++) {
    const response = responses[person] ?? 0;
    const cell = 2 * (groupOf.get(scores[person] ?? 0) ?? 0) + (focal[person] ?? 0);
    sizes[cell] = (sizes[cell] ?? 0) + 1;
    totals[cell] = (totals[cell] ?? 0) + response;
    squares[cell] = (squares[cell] ?? 0) + response * response;
    keys[person] = cell * categories + (categoryOf.get(response) ?? 0);
  }
  return {
    categories,
    scoreCount,
    scores: Float64Array.from(groupOf.keys()),
    sizes,
    totals,
    squares,
    ...cellLists(keys.sort(), categories, 2 * scoreCount),
  };
}

/** Returns the categories of each cell with their persons, from the sorted keys of the persons. */
function cellLists(
  keys: Float64Array,
  categories: number,
  cells: number,
): Pick<ScoreGroups, "cellStarts" | "cellCategories" | "cellCounts"> {
  const cellStarts = new Int32Array(cells + 1);
  const cellCategories: number[] = [];
  const cellCounts: number[] = [];
  for (let at = 0; at < keys.length; ) {
    const key = keys[at] ?? 0;
    let end = at + 1;
    while (keys[end] === key) end++;
    const cell = Math.floor(key / categories);
    cellStarts[cell + 1] = (cellStarts[cell + 1] ?? 0) + 1;
    cellCategories.push(key - cell * categories);
    cellCounts.push(end - at);
    at = end;
  }
  for (let cell = 0; cell < cells; cell++) cellStarts[cell + 1] = (cellStarts[cell + 1] ?? 0) + (cellStarts[cell] ?? 0);
  return { cellStarts, cellCategories: Int32Array.from(cellCategories), cellCounts: Float64Array.from(cellCounts) };
}

/** Returns the rank of each distinct value among the distinct values in ascending order, from 0. */
function ranksOf(values: ArrayLike<number>): Map<number, number> {
  const distinct = [...new Set(Array.from(values))].sort((a, b) => a - b);
  return new Map(distinct.map((value, rank) => [value, rank]));
}

import {
  checkArray,
  checkBoolean,
  checkIntegerFrom,
  checkNonNegative,
  checkOptions,
  checkRecord,
  chiSquareUpperTail,
  DataError,
  describe,
  mean,
} from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";

import { variance } from "../math/internal.js";
import { type DifData, type DifGroup, polytomousData, ranksOf } from "./groups.js";
import { type DifAnalysis, type PurificationOptions, purified, significant } from "./purification.js";

/** Options of {@link sibtest}. */
export interface SibtestOptions extends PurificationOptions {
  /**
   * The persons of each group a score group must exceed to be used, mirt's `Jmin`.
   *
   * @defaultValue `5`
   */
  readonly minimumCell?: number;
  /**
   * Whether to apply the regression correction of the matching score. Without it, β compares the raw means.
   *
   * @defaultValue `true`
   */
  readonly correction?: boolean;
  /**
   * The guessing parameter of the correction, mirt's `guess_correction`, from 0 to below 1.
   *
   * @defaultValue `0`
   */
  readonly guessing?: number;
  /**
   * Bundles by name, each the ids of its items, tested as one set of studied items against the anchors outside it.
   *
   * @defaultValue no bundles
   * @see {@link https://doi.org/10.1111/j.1745-3984.1996.tb00502.x | Douglas, Roussos and Stout (1996)}
   */
  readonly bundles?: Readonly<Record<string, readonly string[]>>;
}

/** SIBTEST statistics of one item or bundle. */
export interface Sibtest {
  /** The item's id, or the bundle's name. */
  readonly id: string;
  /** β, the weighted difference of the corrected means: positive when the item favors the reference group. */
  readonly beta: number | null;
  /** The standard error of β. */
  readonly standardError: number | null;
  /** (β / SE)² on one degree of freedom. */
  readonly chiSquare: number | null;
  /** The probability of a chi-square at least as large without DIF. */
  readonly significance: number | null;
}

/** SIBTEST statistics of every item and bundle, with the purification of their matching items. */
export interface SibtestDif extends DifAnalysis<Sibtest> {
  /** One entry per bundle, in the order of `bundles`; empty without them. */
  readonly bundles: readonly Sibtest[];
}

/** The default of `minimumCell`: mirt's `Jmin`. */
const DEFAULT_MINIMUM_CELL = 5;
/** The least `minimumCell`: mirt stops on a `Jmin` below it. */
const LEAST_MINIMUM_CELL = 2;
/** The fewest anchors purification goes on with, as the correction's coefficient alpha needs two matching items. */
const MINIMUM_ANCHORS = 2;

/** The statistics of an item or bundle where they are undefined. */
const UNDEFINED_STATISTIC: Omit<Sibtest, "id"> = {
  beta: null,
  standardError: null,
  chiSquare: null,
  significance: null,
};

/**
 * Returns the SIBTEST statistics of every item and bundle, as mirt's `SIBTEST` and difR's `difSIBTEST(type = "udif")`,
 * for items of any number of categories, purified on request.
 *
 * @remarks
 * Persons are matched on their total over the matching items: all items or, with `purify`, the anchors, less the
 * studied item or bundle. A bundle's studied score is the sum over its items. Each analysis uses the persons who
 * answered every studied and matching item, so a person with a missing response to an anchor leaves the analysis of
 * every item.
 *
 * The score groups k are the matching scores that occur, in ascending order. With x_G the mean matching score of group
 * G, b_G the coefficient alpha of the matching items in G and m_G(k) the mean studied score of G in k, the correction
 * estimates the true score V_G(k) = x_G + b_G (k - x_G) and moves each mean to the groups' common
 * V(k) = (V_R(k) + V_F(k)) / 2, along the slope between the neighbors of k:
 * m*_G(k) = m_G(k) + M (V(k) - V_G(k)), M = (m_G(k + 1) - m_G(k - 1)) / (V_G(k + 1) - V_G(k - 1)). A neighbor without
 * persons of G has the mean 0, as in mirt.
 *
 * A score group is used when each group has more than `minimumCell` persons in it, with studied scores that vary; when
 * it is neither the lowest score nor the highest; and when k is at least `guessing` times the number of items. With
 * p(k) the share of k's persons among those of the score groups used, β = Σ p(k) (m*_R(k) - m*_F(k)) and
 * SE² = Σ p(k)² (s²_R(k) / n_R(k) + s²_F(k) / n_F(k)), s² the sample variance. In coefficient alpha, `guessing` replaces
 * the variance of an item with two different responses in a group by v (1 - v), for
 * v = max((u - guessing) / (1 - guessing), 0) and u the item's mean above its lower response.
 *
 * The statistics are `null` without a score group to use, or where the correction is undefined, as for fewer than two
 * matching items. Purification flags items whose chi-square is significant at 0.05, never bundles, and stops before
 * fewer than two anchors are left; bundles are matched on the anchors of the final round.
 *
 * @param matrix - The responses of the persons.
 * @param groups - The group of each person, in the order of the matrix.
 * @param options - The score groups used, the correction, the bundles and whether to purify the matching items.
 * @returns The statistics of every item and bundle, and the purification when it was asked for.
 * @throws `DataError` when the matrix or the groups are invalid, either group has no person with a response, `purify`
 * or `correction` is not a boolean, `minimumCell` is not an integer from 2, `guessing` is not a number from 0 to below
 * 1, `bundles` is not an object, or a bundle is not a list of at least one known item without repeats.
 * @see {@link https://doi.org/10.1007/BF02294572 | Shealy and Stout (1993)}
 * @see {@link https://doi.org/10.1111/j.1745-3984.1996.tb00502.x | Douglas, Roussos and Stout (1996)}
 * @see {@link https://doi.org/10.1111/j.1745-3984.1996.tb00496.x | Chang, Mazzeo and Roussos (1996)}
 */
export function sibtest(matrix: ResponseMatrix, groups: readonly DifGroup[], options: SibtestOptions = {}): SibtestDif {
  checkOptions(options);
  const { purify = false, minimumCell = DEFAULT_MINIMUM_CELL, correction = true, guessing = 0, bundles = {} } = options;
  checkIntegerFrom(minimumCell, "The option minimumCell", LEAST_MINIMUM_CELL);
  checkBoolean(correction, "The option correction");
  checkGuessing(guessing);
  const data = polytomousData(matrix, groups, "SIBTEST");
  const studiedBundles = bundleItems(bundles, matrix.itemIds);
  const settings: Settings = { minimumCell, correction, guessing };
  const analysis = purified(
    matrix.itemIds,
    purify,
    (anchors) => matrix.itemIds.map((id, item) => ({ id, ...statistic(data, [item], anchors, settings) })),
    (item) => significant(item.significance),
    MINIMUM_ANCHORS,
  );
  const kept = new Set(analysis.purification?.anchors ?? matrix.itemIds);
  const anchors = Uint8Array.from(matrix.itemIds, (id) => (kept.has(id) ? 1 : 0));
  return {
    ...analysis,
    bundles: studiedBundles.map(({ id, items }) => ({ id, ...statistic(data, items, anchors, settings) })),
  };
}

/** The options of one analysis. */
interface Settings {
  readonly minimumCell: number;
  readonly correction: boolean;
  readonly guessing: number;
}

function checkGuessing(guessing: number): void {
  checkNonNegative(guessing, "The option guessing");
  if (guessing >= 1) throw new DataError(`The option guessing must be below 1, found ${describe(guessing)}.`);
}

/**
 * Returns each bundle's name with its items, as positions in the matrix.
 *
 * @throws `DataError` when `bundles` is not an object, or a bundle is not a list of at least one known item without
 * repeats.
 */
function bundleItems(
  bundles: Readonly<Record<string, readonly string[]>>,
  itemIds: readonly string[],
): { id: string; items: number[] }[] {
  checkRecord(bundles, "The option bundles must be an object of item ids by bundle");
  const positions = new Map(itemIds.map((id, item) => [id, item]));
  return Object.entries(bundles).map(([id, items]) => {
    const label = `The bundle ${describe(id)}`;
    checkArray(items, `${label} must be an array of item ids`);
    if (items.length === 0) throw new DataError(`${label} needs at least one item.`);
    const seen = new Set<string>();
    for (const item of items) {
      if (!positions.has(item)) throw new DataError(`${label} names the unknown item ${describe(item)}.`);
      if (seen.has(item)) throw new DataError(`${label} names the item ${describe(item)} twice.`);
      seen.add(item);
    }
    return { id, items: items.map((item) => positions.get(item) ?? 0) };
  });
}

/** The persons of one analysis: those who answered every studied and matching item. */
interface StudiedCases {
  /** The matching items: the anchors less the studied items. */
  readonly matching: readonly number[];
  /** The persons, as positions in the matrix. */
  readonly persons: Int32Array;
  /** Each person's total over the matching items: in 64-bit floats, as a sum of responses may pass 2³¹. */
  readonly matchingScores: Float64Array;
  /** Each person's total over the studied items. */
  readonly studiedScores: Float64Array;
  /** 1 for each person of the focal group. */
  readonly focal: Uint8Array;
}

function studiedCases(
  { itemCount, responses, focal }: DifData,
  studied: readonly number[],
  anchors: Uint8Array,
): StudiedCases {
  const matching = Array.from(anchors.keys()).filter((item) => anchors[item] === 1 && !studied.includes(item));
  const persons: number[] = [];
  const matchingScores: number[] = [];
  const studiedScores: number[] = [];
  const total = (person: number, items: readonly number[]): number | null => {
    let sum = 0;
    for (const item of items) {
      const response = responses[person * itemCount + item] ?? -1;
      if (response < 0) return null;
      sum += response;
    }
    return sum;
  };
  for (let person = 0; person < focal.length; person++) {
    const matchingScore = total(person, matching);
    const studiedScore = matchingScore === null ? null : total(person, studied);
    if (matchingScore === null || studiedScore === null) continue;
    persons.push(person);
    matchingScores.push(matchingScore);
    studiedScores.push(studiedScore);
  }
  return {
    matching,
    persons: Int32Array.from(persons),
    matchingScores: Float64Array.from(matchingScores),
    studiedScores: Float64Array.from(studiedScores),
    focal: Uint8Array.from(persons, (person) => focal[person] ?? 0),
  };
}

/**
 * The studied scores by matching score and group. The score groups are the matching scores that occur, in ascending
 * order, so that the tables grow with the persons, not with the highest score.
 */
interface ScoreGroups {
  /** The matching score of each score group. */
  readonly scores: Float64Array;
  /** The persons in score group s and group g (1 for focal), at `2 s + g`. */
  readonly sizes: Float64Array;
  /** Their mean studied score, 0 without persons, at `2 s + g`. */
  readonly means: Float64Array;
  /** The sample variance of their studied scores, 0 for fewer than two persons, at `2 s + g`. */
  readonly variances: Float64Array;
}

function scoreGroups({ matchingScores, studiedScores, focal }: StudiedCases): ScoreGroups {
  const ranks = ranksOf(matchingScores);
  const cells = 2 * ranks.size;
  const sizes = new Float64Array(cells);
  const totals = new Float64Array(cells);
  const squares = new Float64Array(cells);
  for (let person = 0; person < matchingScores.length; person++) {
    const cell = 2 * (ranks.get(matchingScores[person] ?? 0) ?? 0) + (focal[person] ?? 0);
    const score = studiedScores[person] ?? 0;
    sizes[cell] = (sizes[cell] ?? 0) + 1;
    totals[cell] = (totals[cell] ?? 0) + score;
    squares[cell] = (squares[cell] ?? 0) + score * score;
  }
  const means = totals.map((total, cell) => {
    const size = sizes[cell] ?? 0;
    return size === 0 ? 0 : total / size;
  });
  const variances = totals.map((total, cell) => {
    const size = sizes[cell] ?? 0;
    return size < 2 ? 0 : (size * (squares[cell] ?? 0) - total * total) / (size * (size - 1));
  });
  return { scores: Float64Array.from(ranks.keys()), sizes, means, variances };
}

/**
 * Returns the score groups a statistic uses, by their rank. The lowest and the highest score lack a neighbor for the
 * correction, and mirt leaves them out with it or without it.
 */
function usedScoreGroups(
  { scores, sizes, variances }: ScoreGroups,
  minimumCell: number,
  lowestScore: number,
): readonly number[] {
  const used: number[] = [];
  for (let s = 1; s < scores.length - 1; s++) {
    const enough = (sizes[2 * s] ?? 0) > minimumCell && (sizes[2 * s + 1] ?? 0) > minimumCell;
    const varied = variances[2 * s] !== 0 && variances[2 * s + 1] !== 0;
    if (enough && varied && (scores[s] ?? 0) >= lowestScore) used.push(s);
  }
  return used;
}

/**
 * Returns the variance of an item's responses in coefficient alpha, as mirt's `CA`: with `guessing`, an item with two
 * different responses in the group counts as if it had no guessing.
 */
function itemVariance(responses: Float64Array, guessing: number): number {
  if (guessing === 0) return variance(responses);
  const distinct = new Set(responses);
  if (distinct.size !== 2) return variance(responses);
  const adjusted = Math.max((mean(responses) - Math.min(...distinct) - guessing) / (1 - guessing), 0);
  return adjusted * (1 - adjusted);
}

/** Returns coefficient alpha of the matching items among one group's persons, which is not finite below two items. */
function coefficientAlpha(
  { itemCount, responses }: DifData,
  matching: readonly number[],
  persons: Int32Array,
  scores: Float64Array,
  guessing: number,
): number {
  const column = new Float64Array(persons.length);
  let itemVariances = 0;
  for (const item of matching) {
    for (let i = 0; i < persons.length; i++) column[i] = responses[(persons[i] ?? 0) * itemCount + item] ?? 0;
    itemVariances += itemVariance(column, guessing);
  }
  return (matching.length / (matching.length - 1)) * (1 - itemVariances / variance(scores));
}

/**
 * Returns the true score of each score group in one group of persons (1 for focal), on the scale of the matching score:
 * mirt divides V_G by the number of matching items, which the slope M cancels.
 */
function trueScores(
  data: DifData,
  { matching, persons, matchingScores, focal }: StudiedCases,
  scores: Float64Array,
  group: number,
  guessing: number,
): Float64Array {
  const isMember = (_: number, person: number): boolean => focal[person] === group;
  const members = matchingScores.filter(isMember);
  const center = mean(members);
  const alpha = coefficientAlpha(data, matching, persons.filter(isMember), members, guessing);
  return scores.map((score) => center + alpha * (score - center));
}

/** Returns the mean of score group s and group g, moved to the true score `common` along the slope between its neighbors. */
function correctedMean(means: Float64Array, trueScore: Float64Array, s: number, g: number, common: number): number {
  const rise = (means[2 * (s + 1) + g] ?? 0) - (means[2 * (s - 1) + g] ?? 0);
  const run = (trueScore[s + 1] ?? 0) - (trueScore[s - 1] ?? 0);
  return (means[2 * s + g] ?? 0) + (rise / run) * (common - (trueScore[s] ?? 0));
}

/** Returns the mean studied scores of {@link ScoreGroups} with the regression correction, at `2 s + g`. */
function correctedMeans(
  data: DifData,
  cases: StudiedCases,
  { scores, means }: ScoreGroups,
  guessing: number,
): Float64Array {
  const reference = trueScores(data, cases, scores, 0, guessing);
  const focal = trueScores(data, cases, scores, 1, guessing);
  const corrected = means.slice();
  for (let s = 1; s < scores.length - 1; s++) {
    const common = ((reference[s] ?? 0) + (focal[s] ?? 0)) / 2;
    corrected[2 * s] = correctedMean(means, reference, s, 0, common);
    corrected[2 * s + 1] = correctedMean(means, focal, s, 1, common);
  }
  return corrected;
}

function statistic(
  data: DifData,
  studied: readonly number[],
  anchors: Uint8Array,
  { minimumCell, correction, guessing }: Settings,
): Omit<Sibtest, "id"> {
  const cases = studiedCases(data, studied, anchors);
  const groups = scoreGroups(cases);
  const used = usedScoreGroups(groups, minimumCell, guessing * data.itemCount);
  if (used.length === 0) return UNDEFINED_STATISTIC;
  const means = correction ? correctedMeans(data, cases, groups, guessing) : groups.means;
  const { sizes, variances } = groups;
  let weight = 0;
  let difference = 0;
  let sampling = 0;
  for (const s of used) {
    const reference = sizes[2 * s] ?? 0;
    const focal = sizes[2 * s + 1] ?? 0;
    const size = reference + focal;
    weight += size;
    difference += size * ((means[2 * s] ?? 0) - (means[2 * s + 1] ?? 0));
    sampling += size * size * ((variances[2 * s] ?? 0) / reference + (variances[2 * s + 1] ?? 0) / focal);
  }
  const beta = difference / weight;
  const standardError = Math.sqrt(sampling) / weight;
  const chiSquare = (beta / standardError) ** 2;
  if (!Number.isFinite(chiSquare)) return UNDEFINED_STATISTIC;
  return { beta, standardError, chiSquare, significance: chiSquareUpperTail(chiSquare, 1) };
}

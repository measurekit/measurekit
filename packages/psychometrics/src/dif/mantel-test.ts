import { checkOptions, chiSquareUpperTail } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";

import { invertPositiveDefinite, standardDeviation } from "../math/internal.js";
import {
  checkCategoryCount,
  countByScore,
  type DifGroup,
  type ItemCases,
  itemStatistics,
  polytomousData,
  type ScoreGroups,
} from "./groups.js";
import { type LiuAgresti, liuAgresti } from "./liu-agresti.js";
import {
  type DifAnalysis,
  type PurificationOptions,
  purified,
  SIGNIFICANCE_LEVEL,
  significant,
} from "./purification.js";

/** The ETS category of a polytomous item's DIF: AA negligible, BB slight to moderate, CC moderate to large. */
export type PolytomousEtsCategory = "AA" | "BB" | "CC";

/** The generalized Mantel-Haenszel test of one item: general association of group and response. */
export interface GeneralizedMantelHaenszel {
  /** The chi-square, or `null` when the categories the item's persons used leave nothing to test. */
  readonly chiSquare: number | null;
  /** The number of categories the item's persons used in score groups of more than one person, less 1. */
  readonly degreesOfFreedom: number;
  /** The probability of a chi-square at least as large without DIF. */
  readonly significance: number | null;
}

/** Mantel DIF statistics of one polytomous item. */
export interface MantelTest {
  /** The item id. */
  readonly id: string;
  /** Mantel's chi-square with one degree of freedom: the groups differ in their mean response, score group by score group. */
  readonly chiSquare: number | null;
  /** The probability of a chi-square at least as large without DIF. */
  readonly significance: number | null;
  /** The generalized Mantel-Haenszel test: the groups differ in their distribution of responses. */
  readonly generalized: GeneralizedMantelHaenszel;
  /**
   * The standardized mean difference: the focal group's mean response less the reference group's, averaged over the
   * score groups with the focal group's persons as weights. Positive when the item favors the focal group.
   */
  readonly standardizedMeanDifference: number | null;
  /** The standardized mean difference over the standard deviation of the item's responses in both groups. */
  readonly effectSize: number | null;
  /** The ETS category by `effectSize` and `significance`. */
  readonly etsCategory: PolytomousEtsCategory | null;
  /**
   * Liu and Agresti's cumulative common odds ratio of the categories, as Penfield and Algina test it; `null` where it is
   * undefined, as when no score group holds both groups.
   */
  readonly liuAgresti: LiuAgresti | null;
}

/**
 * ETS's bounds of |SMD / SD| for categories BB and CC.
 *
 * @see {@link https://doi.org/10.1002/j.2333-8504.2012.tb02290.x | Zwick (2012)}
 */
const POLYTOMOUS_ETS_THRESHOLDS = { moderate: 0.17, large: 0.25 } as const;

/**
 * Returns the Mantel DIF statistics of every polytomous item: Mantel's chi-square, the generalized Mantel-Haenszel
 * chi-square, the standardized mean difference, the ETS category and Liu and Agresti's cumulative common odds ratio.
 *
 * @remarks
 * Persons are matched on their score over the matching items, all items or, with `purify`, the anchors, and the studied
 * item; score groups of one person are left out. An item's highest possible score is its highest response in the
 * matrix, at least 1. Each item uses the persons who answered it and another matching item, their score scaled from the
 * matching items they answered (see the guide). A statistic is `null` where it is undefined, as for an item every
 * person in a score group answered alike.
 *
 * In each score group of n persons, F is the sum of the focal group's responses, n_k the number of persons who
 * responded k, and n_R and n_F the sizes of the groups: E(F) = n_F Σ k n_k / n and Var(F) = n_R n_F [n Σ k² n_k - (Σ k
 * n_k)²] / (n² (n - 1)). Mantel's chi-square is (Σ (F - E(F)))² / Σ Var(F). The generalized test is the
 * Cochran-Mantel-Haenszel statistic of general association of the 2 × K tables: Σ (a - E(a)) over the categories but
 * the last, in the quadratic form of the inverse of Σ Cov(a), with one degree of freedom less than the categories
 * used. It is `null` when that matrix is singular, as when the score groups hold disjoint sets of categories. The
 * standardized mean difference is Σ (n_Fs / n_F) (m_Fs - m_Rs) over the score groups s with persons of both groups,
 * m the mean response. The effect size divides it by the sample standard deviation of the item's responses over all
 * persons the item uses.
 *
 * ETS categories: AA when Mantel's chi-square is not significant at 0.05 or |effect size| ≤ 0.17; CC when it is
 * significant and |effect size| > 0.25; BB otherwise. Purification flags items whose Mantel chi-square is significant at
 * 0.05.
 *
 * @param matrix - The responses of the persons.
 * @param groups - The group of each person, in the order of the matrix.
 * @param options - Whether to purify the matching score.
 * @returns The statistics of every item, and the purification when it was asked for.
 * @throws `DataError` when the matrix or the groups are invalid, either group has no person with a response, `purify`
 * is not a boolean, or an item's persons use more than 255 categories.
 * @see {@link https://doi.org/10.1080/01621459.1963.10500879 | Mantel (1963)}
 * @see {@link https://doi.org/10.1111/j.1745-3984.1993.tb00425.x | Zwick, Donoghue and Grima (1993)}
 * @see {@link https://doi.org/10.1080/00031305.1986.10475371 | Somes (1986)}
 * @see {@link https://doi.org/10.3102/10769986021003187 | Zwick and Thayer (1996)}
 */
export function mantelTest(
  matrix: ResponseMatrix,
  groups: readonly DifGroup[],
  options: PurificationOptions = {},
): DifAnalysis<MantelTest> {
  checkOptions(options);
  const { purify = false } = options;
  const data = polytomousData(matrix, groups, "Mantel DIF");
  return purified(
    matrix.itemIds,
    purify,
    (anchors) => itemStatistics(matrix.itemIds, data, anchors, (cases, id) => statistics(id, cases)),
    (item) => significant(item.significance),
  );
}

function statistics(id: string, cases: ItemCases): Omit<MantelTest, "id"> {
  const groups = countByScore(cases);
  checkCategoryCount(groups.categories, "Mantel DIF", id);
  const { excess, variance } = mantelSums(groups);
  const chiSquare = variance > 0 ? (excess * excess) / variance : null;
  const significance = chiSquare === null ? null : chiSquareUpperTail(chiSquare, 1);
  const standardizedMeanDifference = meanDifference(groups);
  const spread = responseSpread(cases.responses);
  const effectSize =
    standardizedMeanDifference === null || spread === null || spread === 0 ? null : standardizedMeanDifference / spread;
  return {
    chiSquare,
    significance,
    generalized: generalizedTest(groups),
    standardizedMeanDifference,
    effectSize,
    etsCategory: etsCategory(effectSize, significance),
    liuAgresti: liuAgresti(groups),
  };
}

/** Returns Σ (F - E(F)) and Σ Var(F) over the score groups of more than one person. */
function mantelSums({ scoreCount: groupCount, sizes, totals, squares }: ScoreGroups): {
  excess: number;
  variance: number;
} {
  let excess = 0;
  let variance = 0;
  for (let s = 0; s < groupCount; s++) {
    const reference = sizes[2 * s] ?? 0;
    const focal = sizes[2 * s + 1] ?? 0;
    const size = reference + focal;
    if (size <= 1) continue;
    const total = (totals[2 * s] ?? 0) + (totals[2 * s + 1] ?? 0);
    const square = (squares[2 * s] ?? 0) + (squares[2 * s + 1] ?? 0);
    excess += (totals[2 * s + 1] ?? 0) - (focal * total) / size;
    variance += (reference * focal * (size * square - total * total)) / (size * size * (size - 1));
  }
  return { excess, variance };
}

/** Returns Σ (n_Fs / n_F) (m_Fs - m_Rs) over the score groups with persons of both groups, or `null` without any. */
function meanDifference({ scoreCount: groupCount, sizes, totals }: ScoreGroups): number | null {
  let weighted = 0;
  let focalPersons = 0;
  for (let s = 0; s < groupCount; s++) {
    const reference = sizes[2 * s] ?? 0;
    const focal = sizes[2 * s + 1] ?? 0;
    if (reference === 0 || focal === 0) continue;
    weighted += focal * ((totals[2 * s + 1] ?? 0) / focal - (totals[2 * s] ?? 0) / reference);
    focalPersons += focal;
  }
  return focalPersons === 0 ? null : weighted / focalPersons;
}

/** Returns the sample standard deviation of the responses, or `null` for fewer than two. */
function responseSpread(responses: Int32Array): number | null {
  return responses.length < 2 ? null : standardDeviation(Float64Array.from(responses));
}

/**
 * Returns the generalized Mantel-Haenszel test on the categories used in the score groups of more than one person, all
 * but the last of them.
 */
function generalizedTest(groups: ScoreGroups): GeneralizedMantelHaenszel {
  const used = categoriesUsed(groups);
  const degreesOfFreedom = Math.max(used.length - 1, 0);
  const chiSquare = degreesOfFreedom < 1 ? null : associationChiSquare(groups, used.slice(0, degreesOfFreedom));
  return {
    chiSquare,
    degreesOfFreedom,
    significance: chiSquare === null ? null : chiSquareUpperTail(chiSquare, degreesOfFreedom),
  };
}

/** Returns the categories at least one person of a score group of more than one person responded. */
function categoriesUsed({ scoreCount, sizes, cellStarts, cellCategories }: ScoreGroups): readonly number[] {
  const used = new Set<number>();
  for (let s = 0; s < scoreCount; s++) {
    if ((sizes[2 * s] ?? 0) + (sizes[2 * s + 1] ?? 0) <= 1) continue;
    for (let entry = cellStarts[2 * s] ?? 0; entry < (cellStarts[2 * s + 2] ?? 0); entry++) {
      used.add(cellCategories[entry] ?? 0);
    }
  }
  return [...used].sort((a, b) => a - b);
}

/** The sums of the generalized test over the score groups, with the columns of one score group. */
interface Association {
  /** Σ (a - E(a)), by dimension. */
  readonly excess: Float64Array;
  /** Σ Cov(a), by rows. */
  readonly covariance: Float64Array;
  /** The persons of the score group in each dimension's category, and those of the focal group. */
  readonly column: Float64Array;
  readonly focalColumn: Float64Array;
  /** The dimensions whose columns the score group fills, ascending once it is read. */
  readonly touched: number[];
  /** The dimension of each category, -1 for the last used, which the test leaves out. */
  readonly dimensionOf: Int32Array;
}

/**
 * Returns the chi-square of general association over `categories`: Σ (a - E(a)) in the quadratic form of the inverse of
 * Σ Cov(a), a the focal group's counts in the categories, or `null` for a singular covariance matrix.
 */
function associationChiSquare(groups: ScoreGroups, categories: readonly number[]): number | null {
  const dimension = categories.length;
  const sums: Association = {
    excess: new Float64Array(dimension),
    covariance: new Float64Array(dimension * dimension),
    column: new Float64Array(dimension),
    focalColumn: new Float64Array(dimension),
    touched: [],
    dimensionOf: new Int32Array(groups.categories).fill(-1),
  };
  categories.forEach((k, a) => {
    sums.dimensionOf[k] = a;
  });
  for (let s = 0; s < groups.scoreCount; s++) addScoreGroup(sums, groups, s);
  const { excess } = sums;
  const inverse = invertPositiveDefinite(sums.covariance, dimension);
  if (inverse === null) return null;
  let chiSquare = 0;
  for (let i = 0; i < dimension; i++) {
    for (let j = 0; j < dimension; j++)
      chiSquare += (excess[i] ?? 0) * (inverse[i * dimension + j] ?? 0) * (excess[j] ?? 0);
  }
  return Math.max(0, chiSquare);
}

/**
 * Adds score group s of more than one person to the sums. Its cells list only the categories that occur in them, and
 * the others add 0 to every sum.
 */
function addScoreGroup(sums: Association, groups: ScoreGroups, s: number): void {
  const { excess, column, focalColumn, touched } = sums;
  const reference = groups.sizes[2 * s] ?? 0;
  const focal = groups.sizes[2 * s + 1] ?? 0;
  if (reference + focal <= 1) return;
  fillColumns(sums, groups, s);
  for (const a of touched) {
    excess[a] = (excess[a] ?? 0) + (focalColumn[a] ?? 0) - (focal * (column[a] ?? 0)) / (reference + focal);
  }
  addCovariance(sums.covariance, column, touched, reference, focal);
  for (const a of touched) {
    column[a] = 0;
    focalColumn[a] = 0;
  }
  touched.length = 0;
}

/** Fills the columns of score group s at the dimensions its cells use, and lists them in `touched`, ascending. */
function fillColumns(
  { column, focalColumn, touched, dimensionOf }: Association,
  { cellStarts, cellCategories, cellCounts }: ScoreGroups,
  s: number,
): void {
  const focalStart = cellStarts[2 * s + 1] ?? 0;
  for (let entry = cellStarts[2 * s] ?? 0; entry < (cellStarts[2 * s + 2] ?? 0); entry++) {
    const a = dimensionOf[cellCategories[entry] ?? 0] ?? -1;
    if (a < 0) continue;
    const persons = cellCounts[entry] ?? 0;
    if (column[a] === 0) touched.push(a);
    column[a] = (column[a] ?? 0) + persons;
    if (entry >= focalStart) focalColumn[a] = (focalColumn[a] ?? 0) + persons;
  }
  touched.sort((x, y) => x - y);
}

/**
 * Adds the covariance of the focal group's counts in a score group with the given category totals, which are 0 but at
 * the dimensions `touched`.
 */
function addCovariance(
  covariance: Float64Array,
  column: Float64Array,
  touched: readonly number[],
  reference: number,
  focal: number,
): void {
  const size = reference + focal;
  const weight = (reference * focal) / (size * size * (size - 1));
  const dimension = column.length;
  for (const i of touched) {
    const a = column[i] ?? 0;
    for (const j of touched) {
      const b = column[j] ?? 0;
      covariance[i * dimension + j] =
        (covariance[i * dimension + j] ?? 0) + weight * ((i === j ? size * a : 0) - a * b);
    }
  }
}

function etsCategory(effectSize: number | null, significance: number | null): PolytomousEtsCategory | null {
  if (effectSize === null || significance === null) return null;
  const size = Math.abs(effectSize);
  if (significance >= SIGNIFICANCE_LEVEL || size <= POLYTOMOUS_ETS_THRESHOLDS.moderate) return "AA";
  return size > POLYTOMOUS_ETS_THRESHOLDS.large ? "CC" : "BB";
}

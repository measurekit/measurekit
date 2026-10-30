import { checkBoolean, checkOptions, normalQuantile } from "@measurekit/core";
import type { ResponseMatrix } from "../data/response-matrix.js";
import { chiSquareUpperTail } from "../math/chi-square.js";
import { type DifGroup, dichotomousData, type ItemCases, itemCases } from "./groups.js";
import {
  type DifAnalysis,
  type PurificationOptions,
  purified,
  SIGNIFICANCE_LEVEL,
  significant,
} from "./purification.js";

/** The ETS category of an item's DIF: A negligible, B slight to moderate, C moderate to large. */
export type EtsCategory = "A" | "B" | "C";

/** Options of {@link mantelHaenszel}. */
export interface MantelHaenszelOptions extends PurificationOptions {
  /**
   * Whether the chi-square subtracts 0.5 from |Σ (A - E(A))| when that is at least 0.5 (Yates's continuity correction).
   *
   * @defaultValue `true`
   */
  readonly continuityCorrection?: boolean;
}

/** Mantel-Haenszel DIF statistics of one item. */
export interface MantelHaenszel {
  /** The item id. */
  readonly id: string;
  /** The Mantel-Haenszel chi-square with one degree of freedom. */
  readonly chiSquare: number | null;
  /** The probability of a chi-square at least as large without DIF. */
  readonly significance: number | null;
  /**
   * The common odds ratio α of a correct response, the reference group's odds over the focal group's, across score
   * groups: above 1 when the item favors the reference group.
   */
  readonly oddsRatio: number | null;
  /** MH D-DIF, -2.35 ln α, on the ETS delta scale: negative when the item is harder for the focal group. */
  readonly delta: number | null;
  /** The standard error of `delta`, from the Robins-Breslow-Greenland variance of ln α. */
  readonly deltaStandardError: number | null;
  /** The ETS category by `delta`, its standard error and `significance`. */
  readonly etsCategory: EtsCategory | null;
}

/**
 * ETS's bounds of |MH D-DIF| for categories B and C.
 *
 * @see {@link https://doi.org/10.1002/j.2333-8504.2012.tb02290.x | Zwick (2012)}
 */
const ETS_THRESHOLDS = { moderate: 1, large: 1.5 } as const;
/** The one-sided critical value of the test that |MH D-DIF| exceeds 1. */
const ONE_SIDED_CRITICAL = normalQuantile(1 - SIGNIFICANCE_LEVEL);
/** The ETS delta scale's multiple of the log odds ratio: 4 / 1.7. */
const DELTA_SCALE = -2.35;

/**
 * Returns the Mantel-Haenszel DIF statistics of every dichotomous item, as difR's `difMH`, with the ETS category.
 *
 * @remarks
 * Persons are matched on their score over the matching items, all items or, with `purify`, the anchors, and the studied
 * item; score groups of one person are left out. Each item uses the persons who answered it and another matching item,
 * their score scaled from the matching items they answered (see the guide). A statistic is `null` where it is
 * undefined, as for an item every person in a score group answered alike.
 *
 * ETS categories: A when |MH D-DIF| < 1 or the chi-square is not significant at 0.05; C when |MH D-DIF| ≥ 1.5 and (|MH
 * D-DIF| - 1) / SE > 1.645, the one-sided test at 0.05 that |MH D-DIF| exceeds 1; B otherwise. Purification flags items
 * whose chi-square is significant at 0.05.
 *
 * @throws `DataError` when the matrix or the groups are invalid, a response is neither 0 nor 1, either group has no
 * person with a response, or `purify` or `continuityCorrection` is not a boolean.
 * @see {@link https://doi.org/10.1002/j.2333-8504.2012.tb02290.x | Zwick (2012)}
 */
export function mantelHaenszel(
  matrix: ResponseMatrix,
  groups: readonly DifGroup[],
  options: MantelHaenszelOptions = {},
): DifAnalysis<MantelHaenszel> {
  checkOptions(options);
  const { continuityCorrection = true, purify = false } = options;
  checkBoolean(continuityCorrection, "The option continuityCorrection");
  const data = dichotomousData(matrix, groups, "Mantel-Haenszel DIF");
  return purified(
    matrix.itemIds,
    purify,
    (anchors) =>
      matrix.itemIds.map((id, item) => ({
        id,
        ...statistics(scoreTables(itemCases(data, item, anchors)), continuityCorrection),
      })),
    (item) => significant(item.significance),
  );
}

/**
 * Returns, for each matching score s, the 2 × 2 table of the item at `4 s`: reference correct (A), reference incorrect
 * (B), focal correct (C) and focal incorrect (D).
 */
function scoreTables({ responses, scores, focal, maxScore }: ItemCases): Float64Array {
  const tables = new Float64Array((maxScore + 1) * 4);
  for (let person = 0; person < scores.length; person++) {
    const cell = 4 * (scores[person] ?? 0) + 2 * (focal[person] ?? 0) + 1 - (responses[person] ?? 0);
    tables[cell] = (tables[cell] ?? 0) + 1;
  }
  return tables;
}

function statistics(tables: Float64Array, continuityCorrection: boolean): Omit<MantelHaenszel, "id"> {
  const sums = stratifiedSums(tables);
  const { excess, variance, concordant, discordant } = sums;
  const correction = continuityCorrection && Math.abs(excess) >= 0.5 ? 0.5 : 0;
  const chiSquare = variance > 0 ? (Math.abs(excess) - correction) ** 2 / variance : null;
  const significance = chiSquare === null ? null : chiSquareUpperTail(chiSquare, 1);
  const oddsRatio = discordant > 0 ? concordant / discordant : null;
  const defined = concordant > 0 && discordant > 0;
  const delta = defined ? DELTA_SCALE * Math.log(concordant / discordant) : null;
  const deltaStandardError = defined ? -DELTA_SCALE * Math.sqrt(logOddsRatioVariance(sums)) : null;
  return {
    chiSquare,
    significance,
    oddsRatio,
    delta,
    deltaStandardError,
    etsCategory: etsCategory(delta, deltaStandardError, significance),
  };
}

/** Sums over the score groups of more than one person, each with its table A, B, C, D of T persons. */
interface StratifiedSums {
  /** Σ (A - E(A)), E(A) = (A + B)(A + C) / T. */
  excess: number;
  /** Σ Var(A) = Σ (A + B)(C + D)(A + C)(B + D) / (T² (T - 1)). */
  variance: number;
  /** R = Σ AD / T. */
  concordant: number;
  /** S = Σ BC / T. */
  discordant: number;
  /** Σ (A + D) AD / T². */
  concordantSquares: number;
  /** Σ ((A + D) BC + (B + C) AD) / T². */
  crossSquares: number;
  /** Σ (B + C) BC / T². */
  discordantSquares: number;
}

function stratifiedSums(tables: Float64Array): StratifiedSums {
  const sums: StratifiedSums = {
    excess: 0,
    variance: 0,
    concordant: 0,
    discordant: 0,
    concordantSquares: 0,
    crossSquares: 0,
    discordantSquares: 0,
  };
  for (let s = 0; s < tables.length; s += 4) {
    const a = tables[s] ?? 0;
    const b = tables[s + 1] ?? 0;
    const c = tables[s + 2] ?? 0;
    const d = tables[s + 3] ?? 0;
    const total = a + b + c + d;
    if (total <= 1) continue;
    const square = total * total;
    sums.excess += a - ((a + b) * (a + c)) / total;
    sums.variance += ((a + b) * (c + d) * (a + c) * (b + d)) / (square * (total - 1));
    sums.concordant += (a * d) / total;
    sums.discordant += (b * c) / total;
    sums.concordantSquares += ((a + d) * a * d) / square;
    sums.crossSquares += ((a + d) * b * c + (b + c) * a * d) / square;
    sums.discordantSquares += ((b + c) * b * c) / square;
  }
  return sums;
}

/**
 * Returns the Robins-Breslow-Greenland variance of ln α: Σ (A + D) AD / T² / (2 R²) + Σ ((A + D) BC + (B + C) AD) / T²
 * / (2 R S) + Σ (B + C) BC / T² / (2 S²).
 *
 * @see {@link https://doi.org/10.2307/2531052 | Robins, Breslow and Greenland (1986)}
 */
function logOddsRatioVariance(sums: StratifiedSums): number {
  const { concordant, discordant } = sums;
  return (
    sums.concordantSquares / (2 * concordant * concordant) +
    sums.crossSquares / (2 * concordant * discordant) +
    sums.discordantSquares / (2 * discordant * discordant)
  );
}

function etsCategory(
  delta: number | null,
  standardError: number | null,
  significance: number | null,
): EtsCategory | null {
  if (delta === null || standardError === null || significance === null) return null;
  const size = Math.abs(delta);
  if (size < ETS_THRESHOLDS.moderate || significance >= SIGNIFICANCE_LEVEL) return "A";
  return size >= ETS_THRESHOLDS.large && (size - ETS_THRESHOLDS.moderate) / standardError > ONE_SIDED_CRITICAL
    ? "C"
    : "B";
}

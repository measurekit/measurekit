import { normalCdf } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import type { CdmCalibration } from "./calibration.js";
import { checkedModel, profileMoments } from "./model-fit.js";

/** An item's share correct against the share the model expects. */
export interface ProportionFit {
  /** The share of the persons who answered correctly. */
  readonly observed: number;
  /** The share the model expects. */
  readonly expected: number;
  /** |observed - expected| / √(expected (1 - expected) / N); `null` when the expected share is 0 or 1. */
  readonly z: number | null;
}

/** The fit of one item of a cognitive diagnosis model, as GDINA's `itemfit`. */
export interface CdmItemFit {
  /** The item's id. */
  readonly id: string;
  /** Its share correct. */
  readonly proportion: ProportionFit;
  /**
   * The largest |z| of the item's pairs on the Fisher-transformed correlation, |atanh r - atanh ρ| √(N - 3); `null`
   * without a pair, with fewer than four persons, or when no correlation of its pairs is defined.
   */
  readonly correlationZ: number | null;
  /** The largest |z| of the item's pairs on the log odds ratio, over its standard error √(Σ 1 / (N π_ab)); `null` without a pair. */
  readonly logOddsZ: number | null;
  /**
   * The smaller of the Holm-adjusted p-values of the two largest |z| over the item's J - 1 pairs, each min(1, (J - 1) p);
   * `null` when both are `null`.
   */
  readonly significance: number | null;
}

/** GDINA's count of an empty cell of an observed table of two items, so that its log odds ratio is finite. */
const EMPTY_CELL = 0.5;

/**
 * Returns the item fit of a calibrated cognitive diagnosis model: each item's share correct, and the largest z of its
 * pairs on their correlation and log odds ratio, as GDINA's `itemfit`.
 *
 * @remarks
 * The expected shares are exact sums over the profiles, where GDINA simulates `N.resampling` persons. In an observed
 * table of two items an empty cell counts as 0.5, as in GDINA. Uses the persons who answered every item.
 *
 * @param matrix - The responses of the persons, dichotomous.
 * @param calibration - The calibration of the matrix's items, in its order.
 * @returns One fit for each item, in item order.
 * @throws `DataError` as {@link cdmModelFit} does.
 * @see {@link https://doi.org/10.1111/j.1745-3984.2012.00185.x | Chen, de la Torre and Zhang (2013)}
 */
export function cdmItemFit(matrix: ResponseMatrix, calibration: CdmCalibration): readonly CdmItemFit[] {
  const { design, state, models, sample } = checkedModel(matrix, calibration);
  const itemCount = models.length;
  const persons = sample.persons;
  const model = profileMoments(design, state);
  const correlation = new Float64Array(itemCount).fill(Number.NaN);
  const logOdds = new Float64Array(itemCount).fill(Number.NaN);
  const raise = (table: Float64Array, i: number, value: number): void => {
    if (Number.isFinite(value) && !(value <= (table[i] ?? Number.NaN))) table[i] = value;
  };
  for (let i = 0; i < itemCount; i++) {
    for (let j = i + 1; j < itemCount; j++) {
      const observed = pairTable(sample.means, sample.products, itemCount, i, j);
      const expected = pairTable(model.means, model.products, itemCount, i, j);
      const zr = correlationZ(observed, expected, persons);
      const zl = logOddsZ(observed, expected, persons);
      raise(correlation, i, zr);
      raise(correlation, j, zr);
      raise(logOdds, i, zl);
      raise(logOdds, j, zl);
    }
  }
  return calibration.items.map(({ id }, i) => {
    const observed = sample.means[i] ?? 0;
    const expected = model.means[i] ?? 0;
    const z = Math.abs(observed - expected) / Math.sqrt((expected * (1 - expected)) / persons);
    const largest = (table: Float64Array): number | null =>
      Number.isNaN(table[i] ?? Number.NaN) ? null : (table[i] ?? 0);
    const correlationLargest = largest(correlation);
    const logOddsLargest = largest(logOdds);
    return {
      id,
      proportion: { observed, expected, z: Number.isFinite(z) ? z : null },
      correlationZ: correlationLargest,
      logOddsZ: logOddsLargest,
      significance: adjusted(itemCount - 1, correlationLargest, logOddsLargest),
    };
  });
}

/** The shares of a table of two items: both correct, only the first, only the second, neither. */
type PairTable = readonly [both: number, first: number, second: number, neither: number];

/** Returns the table of items i < j from their shares correct and the shares of pairs correct. */
function pairTable(means: Float64Array, products: Float64Array, itemCount: number, i: number, j: number): PairTable {
  const both = products[i * itemCount + j] ?? 0;
  const first = means[i] ?? 0;
  const second = means[j] ?? 0;
  return [both, first - both, second - both, 1 - first - second + both];
}

/** Returns the correlation of a table's two items; `NaN` when either has no variance. */
function tableCorrelation([both, first, second, neither]: PairTable): number {
  const p = both + first;
  const q = both + second;
  const scale = Math.sqrt(p * (1 - p) * q * (1 - q));
  return scale > 0 ? (both * neither - first * second) / scale : Number.NaN;
}

/** Returns |atanh r - atanh ρ| √(N - 3); `NaN` with fewer than four persons or an undefined correlation. */
function correlationZ(observed: PairTable, expected: PairTable, persons: number): number {
  if (persons <= 3) return Number.NaN;
  return (
    Math.abs(Math.atanh(tableCorrelation(observed)) - Math.atanh(tableCorrelation(expected))) * Math.sqrt(persons - 3)
  );
}

/** Returns |ln OR - ln ω| / √(Σ 1 / (N π_ab)), with an empty observed cell counted as 0.5. */
function logOddsZ(observed: PairTable, expected: PairTable, persons: number): number {
  const counts = observed.map((share) => {
    const count = Math.round(share * persons);
    return count === 0 ? EMPTY_CELL : count;
  });
  const logRatio = (cells: readonly number[]): number =>
    Math.log(cells[0] ?? 0) + Math.log(cells[3] ?? 0) - Math.log(cells[1] ?? 0) - Math.log(cells[2] ?? 0);
  const variance = expected.reduce((sum, share) => sum + 1 / (persons * share), 0);
  return Math.abs(logRatio(counts) - logRatio(expected)) / Math.sqrt(variance);
}

/** Returns the smaller of the Holm-adjusted p-values of the two largest z over `pairs` pairs. */
function adjusted(pairs: number, ...largest: readonly (number | null)[]): number | null {
  const values = largest.flatMap((z) => (z === null ? [] : [Math.min(1, pairs * 2 * normalCdf(-z))]));
  return values.length === 0 ? null : values.reduce((least, value) => Math.min(least, value));
}

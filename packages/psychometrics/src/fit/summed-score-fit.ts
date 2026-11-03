import { checkOptions, chiSquareUpperTail, DataError } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import {
  type FixedParameters,
  type IrtItem,
  itemTrace,
  quadratureRule,
  summedScoreLikelihoods,
} from "../irt/internal.js";

import { collapseCells } from "./collapse.js";
import { checkItems, estimatedEntries } from "./inputs.js";

/**
 * The highest total score of the items that S-X² takes. For each item it sums the scores of the other items at each
 * node, which costs the items times the square of the total score: 300 dichotomous items take 2 to 3 s.
 */
const MAX_TOTAL_SCORE = 300;

/** Options of {@link summedScoreFit}: what the items' calibration did not estimate. */
export interface SummedScoreFitOptions {
  /**
   * The ids of the calibration's anchors, whose degrees of freedom count no parameter.
   *
   * @defaultValue none
   */
  readonly fixed?: readonly string[];
  /**
   * The parameters the calibration held at their values, as the option of `calibrate` of that name, which the degrees
   * of freedom do not count; only their names are read.
   *
   * @defaultValue none
   */
  readonly fixedParameters?: FixedParameters;
}

/** The S-X² fit of one item. */
export interface SummedScoreFit {
  /** The item id. */
  readonly id: string;
  /** S-X²: Σ (O - E)² / E over the cells of the collapsed table; `null` without degrees of freedom. */
  readonly chiSquare: number | null;
  /** The cells minus the score groups minus the item's estimated parameters, at least 0. */
  readonly degreesOfFreedom: number;
  /** The probability of a chi-square at least as large when the item fits; `null` with `chiSquare`. */
  readonly significance: number | null;
}

/**
 * Returns the S-X² item fit statistic of Orlando and Thissen, in the form of Kang and Chen for items with more than two
 * categories, for items of any model, as mirt's `itemfit(fit_stats = "S_X2")` computes it.
 *
 * @remarks
 * The persons are grouped by total score, from 1 to the highest score less 1; for each group and category of an item,
 * the observed count of persons is compared with the count the model expects, integrating over theta ~ N(0, 1) with 61
 * rectangular points and the Lord-Wingersky recursion for the total score of the other items. Cells are collapsed as
 * mirt collapses them, to expected counts of at least 1. The degrees of freedom subtract the item's estimated
 * parameters, as mirt's: every parameter of its model, none of an anchor's in `fixed`, and not those `fixedParameters`
 * holds. Uses the persons who answered every item.
 *
 * @param matrix - The responses of the persons.
 * @param items - The calibrated items, in the order of the matrix's items.
 * @param options - The anchors and the held parameters of the items' calibration.
 * @returns For each item, the S-X² statistic with its degrees of freedom and probability; `null` where the statistic is
 *   undefined.
 * @throws `DataError` when the matrix or the items are invalid, a response is not a category of its item, the items'
 * highest total score is above 300, no person answered every item, or the options are invalid for the items.
 * @see {@link https://doi.org/10.1177/01466216000241003 | Orlando and Thissen (2000)}
 * @see {@link https://doi.org/10.1111/j.1745-3984.2008.00071.x | Kang and Chen (2008)}
 */
export function summedScoreFit(
  matrix: ResponseMatrix,
  items: readonly IrtItem[],
  options: SummedScoreFitOptions = {},
): SummedScoreFit[] {
  checkOptions(options);
  const { fixed = [], fixedParameters = {} } = options;
  checkItems(matrix, items);
  const estimated = estimatedEntries(items, matrix.itemIds, fixed, fixedParameters);
  const complete = matrix.responses.filter((row) => !row.includes(null));
  if (complete.length === 0) throw new DataError("S-X² needs at least one person who answered every item.");
  const counts = items.map((item) => item.model.categoryCount(item.parameters));
  const maxScore = counts.reduce((sum, count) => sum + count - 1, 0);
  if (maxScore > MAX_TOTAL_SCORE) {
    throw new DataError(
      `S-X² sums the scores of the other items for each item, so it takes items of a highest total score of at most ${MAX_TOTAL_SCORE}, found ${maxScore}; test a part of the form.`,
    );
  }
  const totals = complete.map((row) => row.reduce<number>((sum, response) => sum + (response ?? 0), 0));
  const { nodes, weights } = quadratureRule();
  const traces = items.map((item, i) => itemTrace(item, counts[i] ?? 0, nodes));
  const all = summedScoreLikelihoods(traces, counts, -1, nodes.length);
  const marginal = Float64Array.from({ length: maxScore + 1 }, (_, s) => weighted(all, s, weights));
  const persons = new Float64Array(maxScore + 1);
  for (const total of totals) persons[total] = (persons[total] ?? 0) + 1;

  return items.map((_, i) => {
    const categories = counts[i] ?? 0;
    const rest = summedScoreLikelihoods(traces, counts, i, nodes.length);
    const trace = traces[i] ?? new Float64Array(0);
    const restMax = maxScore - categories + 1;
    const observed: number[][] = [];
    const expected: number[][] = [];
    for (let s = 1; s < maxScore; s++) {
      observed.push(new Array<number>(categories).fill(0));
      expected.push(
        Array.from({ length: categories }, (_, k) => {
          if (s - k < 0 || s - k > restMax) return Number.NaN;
          let numerator = 0;
          for (let q = 0; q < nodes.length; q++) {
            numerator += (weights[q] ?? 0) * (trace[q * categories + k] ?? 0) * (rest[(s - k) * nodes.length + q] ?? 0);
          }
          return ((persons[s] ?? 0) * numerator) / (marginal[s] ?? 0);
        }),
      );
    }
    complete.forEach((row, person) => {
      const total = totals[person] ?? 0;
      const cells = observed[total - 1];
      const response = row[i] ?? 0;
      if (cells !== undefined) cells[response] = (cells[response] ?? 0) + 1;
    });
    const parameterCount = (estimated[i] ?? []).filter(Boolean).length;
    return { id: matrix.itemIds[i] ?? "", ...statistic(observed, expected, parameterCount) };
  });
}

/** Returns S-X², its degrees of freedom and significance from an item's table before collapsing. */
function statistic(observed: number[][], expected: number[][], parameterCount: number): Omit<SummedScoreFit, "id"> {
  const table = collapseCells(observed, expected);
  if (table === null) return { chiSquare: null, degreesOfFreedom: 0, significance: null };
  let chiSquare = 0;
  let cells = 0;
  table.expected.forEach((row, j) => {
    row.forEach((value, k) => {
      if (Number.isNaN(value)) return;
      cells++;
      chiSquare += ((table.observed[j]?.[k] ?? 0) - value) ** 2 / value;
    });
  });
  const degreesOfFreedom = Math.max(0, cells - table.expected.length - parameterCount);
  if (degreesOfFreedom === 0) return { chiSquare: null, degreesOfFreedom, significance: null };
  return { chiSquare, degreesOfFreedom, significance: chiSquareUpperTail(chiSquare, degreesOfFreedom) };
}

/** Returns Σ_q w_q L(s | θ_q) of a table of score likelihoods. */
function weighted(likelihoods: Float64Array, s: number, weights: Float64Array): number {
  let sum = 0;
  for (let q = 0; q < weights.length; q++) sum += (weights[q] ?? 0) * (likelihoods[s * weights.length + q] ?? 0);
  return sum;
}

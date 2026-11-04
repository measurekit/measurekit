import type { ResponseMatrix } from "../data/internal.js";
import {
  checkItemsOfMatrix,
  checkPersonThetas,
  type IrtItem,
  largestCategoryCount,
  logProbability,
} from "../irt/internal.js";
import { isExtreme, scoreMoments } from "./inputs.js";
import { addResponse, type MeanSquareFit, meanSquares, SUM_COUNT } from "./mean-squares.js";

/** The fit of one person's responses. */
export interface PersonFit extends MeanSquareFit {
  /** The person id. */
  readonly id: string;
  /**
   * The standardized log-likelihood lz: (l0 - E(l0)) / √Var(l0), l0 the log-likelihood of the responses at theta. About
   * N(0, 1) under fit; large negative values flag misfit.
   *
   * @see {@link https://doi.org/10.1111/j.2044-8317.1985.tb00817.x | Drasgow, Levine and Williams (1985)}
   */
  readonly standardizedLogLikelihood: number | null;
}

/**
 * Returns the fit of every person's responses at their theta.
 *
 * @remarks
 * The infit and outfit mean squares with their standardized forms, as eRm's `personfit` computes them, and the lz
 * statistic, as mirt's `personfit` computes `Zh`, for items of any model. Missing responses are skipped. The mean
 * squares are `null` for a person with an extreme score, every answered response in its item's lowest category or every
 * one in its highest; every statistic is `null` for a person without responses.
 *
 * @param matrix - The responses of the persons.
 * @param items - The calibrated items, in the order of the matrix's items.
 * @param thetas - The estimate of theta of each person, in the order of the matrix.
 * @returns For each person, the infit and outfit mean squares, their standardized forms and the standardized
 *   log-likelihood lz.
 * @throws `DataError` when the matrix, the items or the thetas are invalid, or a response is not a category of its
 * item.
 */
export function personFit(
  matrix: ResponseMatrix,
  items: readonly IrtItem[],
  thetas: readonly number[],
): readonly PersonFit[] {
  checkItemsOfMatrix(matrix, items);
  checkPersonThetas(matrix, thetas);
  const sums = new Float64Array(SUM_COUNT);
  const probabilities = new Float64Array(largestCategoryCount(items));
  const moments = new Float64Array(3);
  const logLikelihood = new Float64Array(3);
  return matrix.responses.map((row, person) => {
    const theta = thetas[person] ?? 0;
    const extreme = isExtreme(items, row);
    sums.fill(0);
    logLikelihood.fill(0);
    row.forEach((response, column) => {
      const item = items[column];
      if (response === null || item === undefined) return;
      scoreMoments(item, theta, probabilities, moments);
      if (!extreme) addResponse(sums, 0, response - (moments[0] ?? 0), moments);
      addLogLikelihood(probabilities, item.model.categoryCount(item.parameters), response, logLikelihood);
    });
    const deviation = Math.sqrt(logLikelihood[2] ?? 0);
    const lz = ((logLikelihood[0] ?? 0) - (logLikelihood[1] ?? 0)) / deviation;
    return {
      id: matrix.personIds[person] ?? "",
      ...meanSquares(sums, 0),
      standardizedLogLikelihood: Number.isFinite(lz) ? lz : null,
    };
  });
}

/**
 * Adds one response's terms to the log-likelihood l0, its expectation Σ P_k ln P_k and its variance
 * Σ_(k < h) P_k P_h (ln P_k - ln P_h)², the last a sum of non-negative terms, at `out[0]` to `out[2]`.
 */
function addLogLikelihood(probabilities: Float64Array, count: number, response: number, out: Float64Array): void {
  let mean = 0;
  let variance = 0;
  for (let k = 0; k < count; k++) {
    const p = probabilities[k] ?? 0;
    const logP = logProbability(p);
    mean += p * logP;
    for (let h = k + 1; h < count; h++) {
      variance += p * (probabilities[h] ?? 0) * (logP - logProbability(probabilities[h] ?? 0)) ** 2;
    }
  }
  out[0] = (out[0] ?? 0) + logProbability(probabilities[response] ?? 0);
  out[1] = (out[1] ?? 0) + mean;
  out[2] = (out[2] ?? 0) + variance;
}

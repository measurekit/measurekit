import { normalCdf } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import type { IrtItem } from "../irt/internal.js";
import { checkModelInputs, type ModelInputs, personProbabilities, probabilitiesOf } from "./inputs.js";
import type { PersonPair } from "./pairs.js";

/** Wollack's ω of one pair. */
export interface OmegaIndex {
  /** The suspected copier and the source. */
  readonly persons: readonly [copier: string, source: string];
  /** The items both persons answered, which the index compares. */
  readonly compared: number;
  /** The compared items on which the copier gave the source's response. */
  readonly matches: number;
  /** The matches expected of the copier at the copier's theta, given the source's responses. */
  readonly expectedMatches: number;
  /** ω, the matches less the expected ones over their standard deviation; `null` when they have none. */
  readonly omega: number | null;
  /** The upper tail of the standard normal distribution at ω: small values point to copying; `null` with ω. */
  readonly significance: number | null;
}

/**
 * Returns Wollack's ω of each pair: whether the suspected copier, the first person, gave the source's responses more
 * often than the copier's ability explains.
 *
 * @remarks
 * Over the items both persons answered, the expected number of matches is the sum of the probabilities that the
 * copier gives the source's response, P_i(u_s | θ_c), at the copier's theta, and its variance the sum of P (1 - P);
 * ω is the observed matches less the expected ones, over the standard deviation, about N(0, 1) without copying. The
 * index is directional: it conditions on the source's responses and the copier's theta. With the nominal response
 * model the responses are the options chosen; with dichotomous items they are the scores, and a match is two correct
 * or two incorrect responses. Missing responses leave their item out of the pair.
 *
 * @param matrix - The responses: each person's category of each item, such as the option chosen, numbered from 0.
 * @param items - The calibrated items, in the order of the matrix's items, such as `nominalResponse` items.
 * @param thetas - Each person's theta, in the order of the matrix.
 * @param pairs - The pairs: the suspected copier first and the source second.
 * @returns For each pair, ω and its significance.
 * @throws `DataError` when the matrix, the items, the thetas or the pairs are invalid, a response is not a category of
 * its item, or a pair names a person twice or a person the matrix lacks.
 * @see {@link https://doi.org/10.1177/01466216970214002 | Wollack (1997)}
 *
 * @example
 * ```ts
 * import { buildResponseMatrix, nominalResponse } from "@measurekit/psychometrics";
 * import { omegaIndex } from "@measurekit/psychometrics/similarity";
 *
 * // Three options per item: the copier's choices repeat the source's on every item.
 * const item = { model: nominalResponse, parameters: { a: [-1, 0, 1], c: [0, 0.5, 0] } };
 * const matrix = buildResponseMatrix(
 *   [
 *     [0, 1, 0, 2, 1, 0],
 *     [0, 1, 0, 2, 1, 0],
 *   ],
 *   { personIds: ["copier", "source"] },
 * );
 * const [pair] = omegaIndex(matrix, Array(6).fill(item), [0, 0], [["copier", "source"]]);
 * pair?.matches; // => 6
 * pair?.omega; // => 3.52, three and a half standard deviations above the 2 matches expected
 * ```
 */
export function omegaIndex(
  matrix: ResponseMatrix,
  items: readonly IrtItem[],
  thetas: readonly number[],
  pairs: readonly PersonPair[],
): readonly OmegaIndex[] {
  const inputs = checkModelInputs(matrix, items, thetas, pairs);
  const copierProbabilities = personProbabilities(inputs);
  return Array.from({ length: pairs.length }, (_, k): OmegaIndex => {
    const copier = inputs.pairs[2 * k] ?? 0;
    const source = inputs.pairs[2 * k + 1] ?? 0;
    const probabilities = probabilitiesOf(inputs, copierProbabilities, copier);
    const { compared, matches, expectedMatches, variance } = matchMoments(inputs, copier, source, probabilities);
    const omega = variance > 0 ? (matches - expectedMatches) / Math.sqrt(variance) : null;
    return {
      persons: [matrix.personIds[copier] ?? "", matrix.personIds[source] ?? ""],
      compared,
      matches,
      expectedMatches,
      omega,
      significance: omega === null ? null : normalCdf(-omega),
    };
  });
}

/** The items a pair compares, its matches, and the mean and variance of the matches without copying. */
interface MatchMoments {
  readonly compared: number;
  readonly matches: number;
  readonly expectedMatches: number;
  readonly variance: number;
}

/** Returns the moments of a pair's matches at the copier's probabilities, given the source's responses. */
function matchMoments(
  { matrix, offsets }: ModelInputs,
  copier: number,
  source: number,
  probabilities: Float64Array,
): MatchMoments {
  const copied = matrix.responses[copier] ?? [];
  const given = matrix.responses[source] ?? [];
  let compared = 0;
  let matches = 0;
  let expectedMatches = 0;
  let variance = 0;
  for (let column = 0; column < given.length; column++) {
    const response = given[column] ?? null;
    const copiedResponse = copied[column] ?? null;
    if (response === null || copiedResponse === null) continue;
    const p = probabilities[(offsets[column] ?? 0) + response] ?? 0;
    compared++;
    if (copiedResponse === response) matches++;
    expectedMatches += p;
    variance += p * (1 - p);
  }
  return { compared, matches, expectedMatches, variance };
}

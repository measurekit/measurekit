import type { ResponseMatrix } from "../data/internal.js";
import type { IrtItem } from "../irt/internal.js";
import {
  checkComparedItems,
  checkModelInputs,
  type ModelInputs,
  matchProbability,
  personProbabilities,
  probabilitiesOf,
} from "./inputs.js";
import type { PersonPair } from "./pairs.js";

/** The generalized binomial test of one pair. */
export interface GeneralizedBinomialTest {
  /** The two persons. */
  readonly persons: PersonPair;
  /** The items both persons answered, which the test compares. */
  readonly compared: number;
  /** The compared items on which both gave the same response. */
  readonly matches: number;
  /** The matches expected of two persons who answer alone, at their thetas. */
  readonly expectedMatches: number;
  /** The probability of at least `matches` matches without copying: small values point to copying. */
  readonly significance: number;
}

/**
 * Returns van der Linden and Sotaridona's generalized binomial test of each pair: whether the two persons gave the same
 * responses more often than two persons who answer alone at their abilities do.
 *
 * @remarks
 * Two persons who answer alone match on item i with probability π_i = Σ_k P_ik(θ₁) P_ik(θ₂), the sum over the item's
 * categories of the product of their probabilities. Over the items both answered, the number of matches then follows
 * the generalized (compound) binomial distribution of the π_i, which the recursion of Lord and Wingersky gives exactly;
 * the significance is its upper tail at the observed matches. The test is symmetric in the persons. With the nominal
 * response model the responses are the options chosen; with dichotomous items they are the scores. Missing responses
 * leave their item out of the pair. Time grows with the square of the items a pair compares, at most 1,000.
 *
 * @param matrix - The responses: each person's category of each item, such as the option chosen, numbered from 0.
 * @param items - The calibrated items, in the order of the matrix's items, such as `nominalResponse` items.
 * @param thetas - Each person's theta, in the order of the matrix.
 * @param pairs - The pairs of persons.
 * @returns For each pair, the matches and their significance.
 * @throws `DataError` when the matrix, the items, the thetas or the pairs are invalid, a response is not a category of
 * its item, a pair names a person twice or a person the matrix lacks, or a pair compares more than 1,000 items.
 * @see {@link https://doi.org/10.3102/10769986031003283 | van der Linden and Sotaridona (2006)}
 *
 * @example
 * ```ts
 * import { buildResponseMatrix } from "@measurekit/psychometrics/data";
 * import { twoParameterLogistic } from "@measurekit/psychometrics/irt";
 * import { generalizedBinomialTest } from "@measurekit/psychometrics/similarity";
 *
 * const items = [-1, -0.5, 0, 0.5, 1, 1.5].map((b) => ({ model: twoParameterLogistic, parameters: { a: 1.2, b, c: 0, d: 1 } }));
 * const matrix = buildResponseMatrix([
 *   [1, 1, 0, 1, 0, 1],
 *   [1, 1, 0, 1, 0, 1],
 *   [0, 1, 1, 0, 1, 0],
 * ]);
 * const [same, other] = generalizedBinomialTest(matrix, items, [0, 0, 0], [
 *   ["person1", "person2"],
 *   ["person1", "person3"],
 * ]);
 * same?.significance; // => 0.0462, six matches of the 3.6 expected
 * other?.significance; // => 0.9968, one match
 * ```
 */
export function generalizedBinomialTest(
  matrix: ResponseMatrix,
  items: readonly IrtItem[],
  thetas: readonly number[],
  pairs: readonly PersonPair[],
): readonly GeneralizedBinomialTest[] {
  const inputs = checkModelInputs(matrix, items, thetas, pairs);
  const firstProbabilities = personProbabilities(inputs);
  const secondProbabilities = personProbabilities(inputs);
  const distribution = new Float64Array(items.length + 1);
  return Array.from({ length: pairs.length }, (_, k): GeneralizedBinomialTest => {
    const first = inputs.pairs[2 * k] ?? 0;
    const second = inputs.pairs[2 * k + 1] ?? 0;
    const p = probabilitiesOf(inputs, firstProbabilities, first);
    const q = probabilitiesOf(inputs, secondProbabilities, second);
    const tally = matchDistribution(inputs, first, second, p, q, distribution);
    return {
      persons: [matrix.personIds[first] ?? "", matrix.personIds[second] ?? ""],
      ...tally,
      significance: upperTail(distribution, tally.compared, tally.matches),
    };
  });
}

/** The items a pair compares, its matches and the matches expected without copying. */
interface MatchTally {
  readonly compared: number;
  readonly matches: number;
  readonly expectedMatches: number;
}

/**
 * Returns the tally of a pair and writes into `distribution` the probability of each number of matches over the items
 * it compares, without copying.
 *
 * @throws `DataError` when the pair compares more than {@link MAX_COMPARED_ITEMS} items.
 */
function matchDistribution(
  { matrix, offsets }: ModelInputs,
  first: number,
  second: number,
  p: Float64Array,
  q: Float64Array,
  distribution: Float64Array,
): MatchTally {
  const firstRow = matrix.responses[first] ?? [];
  const secondRow = matrix.responses[second] ?? [];
  let compared = 0;
  let matches = 0;
  let expectedMatches = 0;
  distribution[0] = 1;
  for (let column = 0; column < firstRow.length; column++) {
    const response = firstRow[column] ?? null;
    const other = secondRow[column] ?? null;
    if (response === null || other === null) continue;
    if (response === other) matches++;
    const match = matchProbability(p, q, offsets[column] ?? 0, offsets[column + 1] ?? 0);
    expectedMatches += match;
    compared++;
    checkComparedItems(compared, "The generalized binomial test");
    addItem(distribution, compared, match);
  }
  return { compared, matches, expectedMatches };
}

/**
 * Adds an item of match probability `match` to the distribution of the matches of `items - 1` items: Lord and
 * Wingersky's step, in place from the highest count down.
 */
function addItem(distribution: Float64Array, items: number, match: number): void {
  const miss = 1 - match;
  distribution[items] = (distribution[items - 1] ?? 0) * match;
  for (let count = items - 1; count > 0; count--) {
    distribution[count] = (distribution[count] ?? 0) * miss + (distribution[count - 1] ?? 0) * match;
  }
  distribution[0] = (distribution[0] ?? 0) * miss;
}

/** Returns P(M ≥ matches) of the distribution of `items` items, summed from the far end so that small terms add first. */
function upperTail(distribution: Float64Array, items: number, matches: number): number {
  let tail = 0;
  for (let count = items; count >= matches; count--) tail += distribution[count] ?? 0;
  return Math.min(tail, 1);
}

import { logistic } from "@measurekit/core";
import { findRoot, symmetricEigen } from "../math/internal.js";
import { type CompressedResponses, forEachRespondent } from "./responses.js";

// Roberts and Laughlin's appendix, as the GGUM package's GUM.initprm: an item's agreement is the share of its responses
// at or above its agree category. The most endorsed item sits at the origin, and τ_B is the threshold at which a
// dichotomous graded unfolding item there has that share; each other item's distance from the origin gives a person at
// the origin its share; the item's sign is that of its loading on the leading eigenvector of the item correlations.

/** The starting parameters of an unfolding item. */
interface UnfoldingStart {
  readonly a: number;
  readonly b: number;
  readonly t: readonly number[];
}

/**
 * τ_B when the GGUM package rescales the shares, so that the most endorsed item's is e² / (1 + e²): its start when the
 * most endorsed share would give τ_B below {@link LOWEST_THRESHOLD}, and the start of a single item.
 */
const RESCALED_THRESHOLD = -2;
/** The lowest τ_B the GGUM package starts from before it rescales the shares. */
const LOWEST_THRESHOLD = -3;
/** The bracket of a location's distance from the origin, the GGUM package's. */
const LOCATION_BRACKET = 6;

/**
 * Returns the lowest category of an item with `categoryCount` categories that is not a disagreement: the middle
 * category, or the first agree category when the number of categories is even.
 */
export function agreeCategory(categoryCount: number): number {
  return Math.ceil((categoryCount - 1) / 2);
}

/**
 * Returns the starting parameters of an item from its share of persons in each category alone: a = 1, and b and the
 * thresholds from its agreement with τ_B at its rescaled value. One item's shares cannot place the sign of its
 * location, so b is at least 0.
 */
export function singleItemStart(proportions: Float64Array): UnfoldingStart {
  return {
    a: 1,
    b: locationStart(agreement(proportions), RESCALED_THRESHOLD),
    t: thresholdStarts(proportions.length, RESCALED_THRESHOLD),
  };
}

/**
 * Returns the starting parameters of each item by Roberts and Laughlin's procedure, as the GGUM package: a = 1, the
 * most endorsed item at the origin, the size of each other location from the item's agreement, its sign from its
 * loading on the leading eigenvector of the pairwise item correlations, signed so that the loadings sum to at least 0,
 * and the thresholds from τ_B.
 *
 * @param responses - The responses the calibration uses.
 * @param proportions - Each item's share of responses in each category.
 */
export function unfoldingStarts(
  responses: CompressedResponses,
  proportions: readonly Float64Array[],
): UnfoldingStart[] {
  const shares = proportions.map(agreement);
  const origin = shares.reduce((best, share, i) => (share > (shares[best] ?? 0) ? i : best), 0);
  const most = shares[origin] ?? 0;
  let threshold = -Math.abs(Math.log(most / (1 - most)));
  if (threshold < LOWEST_THRESHOLD) {
    const scale = logistic(-RESCALED_THRESHOLD) / most;
    for (let i = 0; i < shares.length; i++) shares[i] = (shares[i] ?? 0) * scale;
    threshold = RESCALED_THRESHOLD;
  }
  const signs = loadingSigns(responses);
  return proportions.map((itemProportions, i) => ({
    a: 1,
    b: i === origin ? 0 : (signs[i] ?? 1) * locationStart(shares[i] ?? 0, threshold),
    t: thresholdStarts(itemProportions.length, threshold),
  }));
}

/** Returns an item's share of responses at or above its agree category. */
function agreement(proportions: Float64Array): number {
  let share = 0;
  for (let k = agreeCategory(proportions.length); k < proportions.length; k++) share += proportions[k] ?? 0;
  return share;
}

/**
 * Returns the starting thresholds of an item with `categoryCount` categories from τ_B: t_k = (C + 1 - k) τ_B / (C - h +
 * 1), with C + 1 categories and h the agree category.
 */
function thresholdStarts(categoryCount: number, threshold: number): number[] {
  const highest = categoryCount - 1;
  const span = highest - agreeCategory(categoryCount) + 1;
  return Array.from({ length: highest }, (_, k) => ((highest - k) * threshold) / span);
}

/**
 * Returns the distance of an item's location from the origin, at which a person at the origin agrees with it as often
 * as its share says: the root of (e^(-δ-τ) + e^(-2δ-τ)) / (1 + e^(-δ-τ) + e^(-2δ-τ) + e^(-3δ)) = share, the agreement
 * with a dichotomous graded unfolding item with threshold τ_B. A share at or above the agreement at the origin gives 0,
 * and one below that at the bracket's end gives the end.
 */
function locationStart(share: number, threshold: number): number {
  const excess = (delta: number): number => {
    const one = Math.exp(-delta - threshold);
    const two = Math.exp(-2 * delta - threshold);
    return (one + two) / (1 + one + two + Math.exp(-3 * delta)) - share;
  };
  const atOrigin = excess(0);
  if (atOrigin <= 0) return 0;
  const atEnd = excess(LOCATION_BRACKET);
  if (atEnd >= 0) return LOCATION_BRACKET;
  return findRoot(excess, 0, LOCATION_BRACKET, atOrigin, atEnd);
}

/**
 * Returns the sign of each item's loading on the leading eigenvector of the items' correlations over the persons who
 * answered both, as R's `cor(use = "pairwise")`, the vector signed so that its entries sum to at least 0, as psych's
 * `principal`. A correlation that is undefined, for a pair without two persons in common or without variance among
 * them, counts as 0, and a loading of 0 as positive.
 */
function loadingSigns(responses: CompressedResponses): number[] {
  const size = responses.itemIds.length;
  const leading = symmetricEigen(pairwiseCorrelations(responses), size).vectors.subarray(0, size);
  const sum = leading.reduce((total, loading) => total + loading, 0);
  return Array.from(leading, (loading) => ((sum < 0 ? -loading : loading) < 0 ? -1 : 1));
}

/**
 * Returns the correlations of the items over the persons who answered both, by rows, 0 where undefined, in space that
 * grows with the items and time that grows with the pairs of responses of each person. The sums are of integers, so
 * they are exact.
 */
function pairwiseCorrelations(responses: CompressedResponses): Float64Array {
  const size = responses.itemIds.length;
  const { counts, sums, squares, products } = pairSums(responses);
  return Float64Array.from({ length: size * size }, (_, entry) => {
    const i = Math.floor(entry / size);
    const j = entry % size;
    if (i === j) return 1;
    const mirrored = j * size + i;
    const n = counts[entry] ?? 0;
    const covariance = n * (products[entry] ?? 0) - (sums[entry] ?? 0) * (sums[mirrored] ?? 0);
    const varianceI = n * (squares[entry] ?? 0) - (sums[entry] ?? 0) ** 2;
    const varianceJ = n * (squares[mirrored] ?? 0) - (sums[mirrored] ?? 0) ** 2;
    return n >= 2 && varianceI > 0 && varianceJ > 0 ? covariance / Math.sqrt(varianceI * varianceJ) : 0;
  });
}

/**
 * Returns, at entry i * size + j, sums over the persons who answered items i and j: their number, and the sums of x_i,
 * x_i² and x_i x_j.
 */
function pairSums(responses: CompressedResponses): {
  readonly counts: Float64Array;
  readonly sums: Float64Array;
  readonly squares: Float64Array;
  readonly products: Float64Array;
} {
  const { itemIds, items, codes } = responses;
  const size = itemIds.length;
  const counts = new Float64Array(size * size);
  const sums = new Float64Array(size * size);
  const squares = new Float64Array(size * size);
  const products = new Float64Array(size * size);
  forEachRespondent(responses, undefined, (_, start, end) => {
    for (let r = start; r < end; r++) {
      const row = (items[r] ?? 0) * size;
      const x = codes[r] ?? 0;
      for (let s = start; s < end; s++) {
        const entry = row + (items[s] ?? 0);
        counts[entry] = (counts[entry] ?? 0) + 1;
        sums[entry] = (sums[entry] ?? 0) + x;
        squares[entry] = (squares[entry] ?? 0) + x * x;
        products[entry] = (products[entry] ?? 0) + x * (codes[s] ?? 0);
      }
    }
  });
  return { counts, sums, squares, products };
}

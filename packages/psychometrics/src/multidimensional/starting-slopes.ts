import type { CompressedResponses } from "../irt/internal.js";
import { symmetricEigen } from "../math/internal.js";
import type { Design } from "./design.js";
import { NORMAL_OGIVE_SCALE } from "./loadings.js";
import { type ItemResponses, responsesByItem } from "./mhrm-chain.js";

/**
 * The smallest share of an item's variance that unique variance keeps in the starting slopes, so that a starting
 * communality near 1 does not give a slope near infinity.
 */
const MIN_UNIQUENESS = 0.1;

/**
 * Returns the starting slopes of every item on every dimension.
 *
 * @remarks
 * A confirmatory design starts each loading slope at 1. An exploratory one starts from the first D eigenvectors of the
 * items' Pearson correlation matrix as loadings, converted to slopes by a = 1.702 λ / √(1 - h²); the slopes of
 * dimensions an item does not load on are 0.
 */
export function startingSlopes(design: Design, responses: CompressedResponses): Float64Array[] {
  const dimensions = design.dimensionNames.length;
  const exploratory = design.exploratory ? principalSlopes(responses, dimensions) : null;
  return design.loads.map((loads, item) =>
    Float64Array.from(loads, (loaded, d) => (loaded ? (exploratory?.[item]?.[d] ?? 1) : 0)),
  );
}

/** Returns the slopes of the first `dimensions` principal components of the items' correlations. */
function principalSlopes(responses: CompressedResponses, dimensions: number): number[][] {
  const size = responses.itemIds.length;
  const { values, vectors } = symmetricEigen(itemCorrelations(responses), size);
  return Array.from({ length: size }, (_, item) => {
    const loadings = Array.from({ length: dimensions }, (__, d) => {
      return (vectors[d * size + item] ?? 0) * Math.sqrt(Math.max(values[d] ?? 0, 0));
    });
    const communality = loadings.reduce((sum, loading) => sum + loading * loading, 0);
    const scale = NORMAL_OGIVE_SCALE / Math.sqrt(Math.max(1 - communality, MIN_UNIQUENESS));
    return loadings.map((loading) => loading * scale);
  });
}

/**
 * Returns the Pearson correlations of the items over the persons who answered both, 0 where undefined, as R's
 * `cor(use = "pairwise.complete.obs")` gives them.
 *
 * @remarks
 * Each pair merges the two items' lists of respondents, which run in the order of the persons, so that a pair costs the
 * responses to its items rather than every person: sparse data, such as adaptive tests on a large pool, answer few of
 * the pairs. The sums run over the persons in their order, in two passes, the means and then the deviations from them.
 */
export function itemCorrelations(responses: CompressedResponses): Float64Array {
  const data = responsesByItem(responses);
  const size = responses.itemIds.length;
  const result = new Float64Array(size * size);
  const common = { x: new Float64Array(data.count), y: new Float64Array(data.count) };
  for (let i = 0; i < size; i++) {
    result[i * size + i] = 1;
    for (let j = i + 1; j < size; j++) {
      const value = pairCorrelation(common.x, common.y, commonResponses(data, i, j, common.x, common.y));
      result[i * size + j] = value;
      result[j * size + i] = value;
    }
  }
  return result;
}

/**
 * Writes the responses to items i and j of each person who answered both, in the order of the persons, into `x` and
 * `y`, and returns their number.
 */
function commonResponses(
  { starts, respondents, codes }: ItemResponses,
  i: number,
  j: number,
  x: Float64Array,
  y: Float64Array,
): number {
  const endA = starts[i + 1] ?? 0;
  const endB = starts[j + 1] ?? 0;
  let count = 0;
  for (let a = starts[i] ?? 0, b = starts[j] ?? 0; a < endA && b < endB; ) {
    const personA = respondents[a] ?? 0;
    const personB = respondents[b] ?? 0;
    if (personA < personB) a++;
    else if (personB < personA) b++;
    else {
      x[count] = codes[a++] ?? 0;
      y[count++] = codes[b++] ?? 0;
    }
  }
  return count;
}

/** Returns the correlation of the first `count` pairs of `x` and `y`, by two passes, or 0 where it is undefined. */
function pairCorrelation(x: Float64Array, y: Float64Array, count: number): number {
  if (count < 2) return 0;
  let sumX = 0;
  let sumY = 0;
  for (let k = 0; k < count; k++) {
    sumX += x[k] ?? 0;
    sumY += y[k] ?? 0;
  }
  const meanX = sumX / count;
  const meanY = sumY / count;
  let squaresX = 0;
  let squaresY = 0;
  let products = 0;
  for (let k = 0; k < count; k++) {
    const a = (x[k] ?? 0) - meanX;
    const b = (y[k] ?? 0) - meanY;
    squaresX += a * a;
    squaresY += b * b;
    products += a * b;
  }
  return squaresX > 0 && squaresY > 0 ? products / Math.sqrt(squaresX * squaresY) : 0;
}

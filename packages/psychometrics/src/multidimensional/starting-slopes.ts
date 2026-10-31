import type { CompressedResponses } from "../irt/responses.js";
import { symmetricEigen } from "../math/eigen.js";
import type { Design } from "./design.js";
import { NORMAL_OGIVE_SCALE } from "./loadings.js";

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
  const { values, vectors } = symmetricEigen(correlations(responses), size);
  return Array.from({ length: size }, (_, item) => {
    const loadings = Array.from({ length: dimensions }, (__, d) => {
      return (vectors[d * size + item] ?? 0) * Math.sqrt(Math.max(values[d] ?? 0, 0));
    });
    const communality = loadings.reduce((sum, loading) => sum + loading * loading, 0);
    const scale = NORMAL_OGIVE_SCALE / Math.sqrt(Math.max(1 - communality, MIN_UNIQUENESS));
    return loadings.map((loading) => loading * scale);
  });
}

/** Returns the Pearson correlations of the items over the persons who answered both, 0 where undefined. */
function correlations({ itemIds, offsets, items, codes }: CompressedResponses): Float64Array {
  const size = itemIds.length;
  const personCount = offsets.length - 1;
  const answers = new Float64Array(personCount * size).fill(Number.NaN);
  for (let person = 0; person < personCount; person++) {
    for (let r = offsets[person] ?? 0; r < (offsets[person + 1] ?? 0); r++) {
      answers[person * size + (items[r] ?? 0)] = codes[r] ?? 0;
    }
  }
  const result = new Float64Array(size * size);
  for (let i = 0; i < size; i++) {
    for (let j = 0; j <= i; j++) {
      const value = i === j ? 1 : pairCorrelation(answers, size, personCount, i, j);
      result[i * size + j] = value;
      result[j * size + i] = value;
    }
  }
  return result;
}

function pairCorrelation(answers: Float64Array, size: number, personCount: number, i: number, j: number): number {
  let n = 0;
  let sumX = 0;
  let sumY = 0;
  let sumXx = 0;
  let sumYy = 0;
  let sumXy = 0;
  for (let person = 0; person < personCount; person++) {
    const x = answers[person * size + i] ?? Number.NaN;
    const y = answers[person * size + j] ?? Number.NaN;
    if (Number.isNaN(x) || Number.isNaN(y)) continue;
    n++;
    sumX += x;
    sumY += y;
    sumXx += x * x;
    sumYy += y * y;
    sumXy += x * y;
  }
  const varianceX = n * sumXx - sumX * sumX;
  const varianceY = n * sumYy - sumY * sumY;
  return varianceX > 0 && varianceY > 0 ? (n * sumXy - sumX * sumY) / Math.sqrt(varianceX * varianceY) : 0;
}

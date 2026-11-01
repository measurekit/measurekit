import type { CompressedResponses } from "../irt/internal.js";
import { symmetricEigen } from "../math/eigen.js";
import { pearsonMatrix } from "../math/pearson.js";
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
  const personCount = offsets.length - 1;
  const columns = itemIds.map(() => new Float64Array(personCount).fill(Number.NaN));
  for (let person = 0; person < personCount; person++) {
    for (let r = offsets[person] ?? 0; r < (offsets[person + 1] ?? 0); r++) {
      const column = columns[items[r] ?? 0];
      if (column !== undefined) column[person] = codes[r] ?? 0;
    }
  }
  return pearsonMatrix(columns).map((value) => (Number.isNaN(value) ? 0 : value));
}

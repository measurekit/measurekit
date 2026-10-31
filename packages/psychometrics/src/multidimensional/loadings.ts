/**
 * The scaling constant that makes the logistic curve close to the normal ogive, with which factor loadings and logistic
 * slopes convert: a = 1.702 λ / √(1 - h²).
 *
 * @see {@link https://doi.org/10.3102/10769986019003293 | Camilli (1994)}
 */
export const NORMAL_OGIVE_SCALE = 1.702;

/** Standardized loadings of items on the dimensions, and the share of each item's variance they explain. */
export interface StandardizedLoadings {
  /** The loadings by rows: one row per item, one entry per dimension. */
  readonly loadings: number[][];
  /** The communality of each item: λᵀ Φ λ. */
  readonly communalities: number[];
}

/**
 * Returns the standardized loadings of items with the given slopes: λ = a' / √(1 + a'ᵀ Φ a') with a' = a / 1.702, and
 * the communalities λᵀ Φ λ, for dimensions with the correlation matrix Φ.
 *
 * @param slopes - One row of slopes per item.
 * @param correlations - The correlation matrix by rows.
 */
export function standardizedLoadings(
  slopes: readonly (readonly number[])[],
  correlations: readonly (readonly number[])[],
): StandardizedLoadings {
  const loadings = slopes.map((a) => {
    const scaled = a.map((slope) => slope / NORMAL_OGIVE_SCALE);
    const norm = Math.sqrt(1 + quadraticForm(scaled, correlations));
    return scaled.map((slope) => slope / norm);
  });
  return { loadings, communalities: loadings.map((lambda) => quadraticForm(lambda, correlations)) };
}

/** Returns xᵀ M x. */
function quadraticForm(x: readonly number[], matrix: readonly (readonly number[])[]): number {
  let sum = 0;
  x.forEach((xi, i) => {
    x.forEach((xj, j) => {
      sum += xi * (matrix[i]?.[j] ?? 0) * xj;
    });
  });
  return sum;
}

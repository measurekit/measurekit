/** Rasch residual fit of an item or a person: mean squares of residuals and their standardized forms. */
export interface MeanSquareFit {
  /** Infit, the information-weighted mean square: Σ (x - E)² / Σ W. */
  readonly infitMeanSquare: number | null;
  /** Outfit, the unweighted mean square: the mean of (x - E)² / W. */
  readonly outfitMeanSquare: number | null;
  /** Infit standardized by the Wilson-Hilferty cube root transformation (ZSTD), about N(0, 1) under fit. */
  readonly infitStandardized: number | null;
  /** Outfit standardized by the Wilson-Hilferty cube root transformation (ZSTD), about N(0, 1) under fit. */
  readonly outfitStandardized: number | null;
}

/** The sums one item or person accumulates over its responses, at `SUM_COUNT` entries per item or person. */
export const SUM_COUNT = 6;

/**
 * Adds a response with residual x - E, score variance W and fourth central moment C to the sums at `offset`: the
 * count, Σ (x - E)² / W, Σ (x - E)², Σ W, Σ C / W² and Σ (C - W²).
 */
export function addResponse(sums: Float64Array, offset: number, residual: number, moments: Float64Array): void {
  const variance = moments[1] ?? 0;
  const fourth = moments[2] ?? 0;
  const square = residual * residual;
  sums[offset] = (sums[offset] ?? 0) + 1;
  sums[offset + 1] = (sums[offset + 1] ?? 0) + square / variance;
  sums[offset + 2] = (sums[offset + 2] ?? 0) + square;
  sums[offset + 3] = (sums[offset + 3] ?? 0) + variance;
  sums[offset + 4] = (sums[offset + 4] ?? 0) + fourth / (variance * variance);
  sums[offset + 5] = (sums[offset + 5] ?? 0) + fourth - variance * variance;
}

/**
 * Returns the mean squares of the sums at `offset` with their standard deviations under fit: q² = Σ (C / W²) / n² - 1 /
 * n for outfit, Σ (C - W²) / (Σ W)² for infit.
 *
 * @see {@link https://www.rasch.org/rsa.htm | Wright and Masters (1982)}
 */
export function meanSquares(sums: Float64Array, offset: number): MeanSquareFit {
  const count = sums[offset] ?? 0;
  const totalVariance = sums[offset + 3] ?? 0;
  const outfit = definedOrNull((sums[offset + 1] ?? 0) / count);
  const infit = definedOrNull((sums[offset + 2] ?? 0) / totalVariance);
  const outfitDeviation = Math.sqrt((sums[offset + 4] ?? 0) / (count * count) - 1 / count);
  const infitDeviation = Math.sqrt((sums[offset + 5] ?? 0) / (totalVariance * totalVariance));
  return {
    infitMeanSquare: infit,
    outfitMeanSquare: outfit,
    infitStandardized: standardized(infit, infitDeviation),
    outfitStandardized: standardized(outfit, outfitDeviation),
  };
}

/** Returns the Wilson-Hilferty transformation (MS^(1/3) - 1) (3 / q) + q / 3 of a mean square with deviation q. */
function standardized(meanSquare: number | null, deviation: number): number | null {
  if (meanSquare === null) return null;
  return definedOrNull((Math.cbrt(meanSquare) - 1) * (3 / deviation) + deviation / 3);
}

function definedOrNull(value: number): number | null {
  return Number.isFinite(value) ? value : null;
}

import { DataError } from "@measurekit/core";
import { findIncreasingRoot, normalCdf, normalDensity } from "../math/internal.js";

/**
 * A score distribution made continuous by a Gaussian kernel: the scores 0 to J with their probabilities r, and the
 * constants of F_h(t) = Σ_j r_j Φ(R_j(t)), R_j(t) = (t - a x_j - (1 - a) μ) / (a h), a = √(σ² / (σ² + h²)), which keeps
 * the mean μ and the variance σ² of the scores.
 */
export interface Continuized {
  readonly probabilities: Float64Array;
  readonly mean: number;
  /** σ², the variance of the scores times n / (n - 1), as kequate. */
  readonly variance: number;
  readonly bandwidth: number;
  /** a = √(σ² / (σ² + h²)). */
  readonly shrinkage: number;
}

/** The bandwidth of linear equating relative to the standard deviation of the scores, kequate's. */
export const LINEAR_BANDWIDTH = 1000;

/**
 * Returns the moments of a distribution: its mean and its variance times n / (n - 1), n its persons, as kequate.
 *
 * @throws `DataError` when the scores have no variance.
 */
export function scoreMoments(probabilities: Float64Array, persons: number, label: string): [number, number] {
  let mean = 0;
  let squares = 0;
  probabilities.forEach((p, x) => {
    mean += p * x;
    squares += p * x * x;
  });
  const variance = (persons / (persons - 1)) * (squares - mean * mean);
  if (!(variance > 0)) throw new DataError(`Kernel equating needs ${label} scores with variance.`);
  return [mean, variance];
}

/** Returns the distribution continuized with the bandwidth h. */
export function continuize(probabilities: Float64Array, moments: [number, number], bandwidth: number): Continuized {
  const [mean, variance] = moments;
  return {
    probabilities,
    mean,
    variance,
    bandwidth,
    // σ / hypot(σ, h) is √(σ² / (σ² + h²)) without the square of a large bandwidth overflowing.
    shrinkage: Math.sqrt(variance) / Math.hypot(Math.sqrt(variance), bandwidth),
  };
}

/** Returns R_j(t) of score x. */
function standardized(c: Continuized, t: number, x: number): number {
  const a = c.shrinkage;
  return (t - a * x - (1 - a) * c.mean) / (a * c.bandwidth);
}

/** Returns F_h(t). */
export function kernelCdf(c: Continuized, t: number): number {
  let sum = 0;
  c.probabilities.forEach((p, x) => {
    sum += p * normalCdf(standardized(c, t, x));
  });
  return sum;
}

/** Returns the density f_h(t) = Σ_j r_j φ(R_j(t)) / (a h). */
export function kernelDensity(c: Continuized, t: number): number {
  let sum = 0;
  c.probabilities.forEach((p, x) => {
    sum += p * normalDensity(standardized(c, t, x));
  });
  return sum / (c.shrinkage * c.bandwidth);
}

/**
 * Returns the sign of the derivative of the density at t: Σ_j -r_j φ(R_j) R_j, whose positive factor 1 / (a h)² does
 * not change it.
 */
export function densitySlopeSign(c: Continuized, t: number): number {
  let sum = 0;
  c.probabilities.forEach((p, x) => {
    const z = standardized(c, t, x);
    sum -= p * normalDensity(z) * z;
  });
  return Math.sign(sum);
}

/** Returns the t where F_h(t) = u, for u strictly between 0 and 1. */
export function kernelQuantile(c: Continuized, u: number, start: number): number {
  return findIncreasingRoot((t) => kernelCdf(c, t) - u, start);
}

/**
 * Writes ∂F_h(t)/∂r_j for every score j into `out`: Φ(R_j(t)) - M_j f_h(t), with M_j = (t - μ)(1 - a²)((x_j - μ) /
 * σ)² / 2 + (1 - a) x_j, the term that carries the change of μ and σ² with r (von Davier, Holland and Thayer, section
 * 5.2; kequate's dFdr).
 */
export function cdfGradient(c: Continuized, t: number, density: number, out: Float64Array): void {
  const a = c.shrinkage;
  const sd = Math.sqrt(c.variance);
  for (let x = 0; x < c.probabilities.length; x++) {
    const z = (x - c.mean) / sd;
    const m = 0.5 * (t - c.mean) * (1 - a * a) * z * z + (1 - a) * x;
    out[x] = normalCdf(standardized(c, t, x)) - m * density;
  }
}

/** The moments that the percent relative error compares: the first ten, as kequate's PREp. */
const PRE_MOMENTS = 10;

/**
 * Returns the percent relative error of the first ten moments of an equating: 100 (Σ_x e(x)^p r_x - Σ_y y^p s_y) /
 * Σ_y y^p s_y for p = 1 to 10.
 */
export function percentRelativeError(equated: readonly number[], from: Float64Array, to: Float64Array): number[] {
  return Array.from({ length: PRE_MOMENTS }, (_, k) => {
    const power = k + 1;
    let equatedMoment = 0;
    let target = 0;
    from.forEach((r, x) => {
      equatedMoment += (equated[x] ?? 0) ** power * r;
    });
    to.forEach((s, y) => {
      target += y ** power * s;
    });
    return (100 * (equatedMoment - target)) / target;
  });
}

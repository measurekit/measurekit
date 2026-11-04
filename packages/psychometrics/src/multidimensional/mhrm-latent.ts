import { cholesky, invertPositiveDefinite } from "@measurekit/core";
import { identity, logDeterminantOfFactor, multiply } from "../math/internal.js";
import { pairsOf } from "./correlation.js";

/** The correlation matrix of the dimensions of an MH-RM calibration, with its inverse and log determinant. */
export interface Correlations {
  readonly dimensions: number;
  /** The pairs of dimensions below the diagonal, in the order of the correlations: (1, 0), (2, 0), (2, 1), ... */
  readonly pairs: readonly (readonly [number, number])[];
  /** The correlation matrix by rows. */
  readonly matrix: Float64Array;
  /** Its inverse by rows. */
  readonly precision: Float64Array;
  /** Returns ln |Φ|. */
  logDeterminant(): number;
  /** Sets the matrix. Returns `false`, leaving the matrix unchanged, when it is not positive definite. */
  set(matrix: Float64Array): boolean;
}

/** Returns the correlations of `dimensions` dimensions, the identity at first. */
export function createCorrelations(dimensions: number): Correlations {
  const matrix = identity(dimensions);
  const precision = identity(dimensions);
  let logDeterminant = 0;
  return {
    dimensions,
    pairs: pairsOf(dimensions),
    matrix,
    precision,
    logDeterminant: () => logDeterminant,
    set(next) {
      const factor = cholesky(next, dimensions);
      const inverse = factor === null ? null : invertPositiveDefinite(next, dimensions);
      if (factor === null || inverse === null) return false;
      matrix.set(next);
      precision.set(inverse);
      logDeterminant = logDeterminantOfFactor(factor, dimensions);
      return true;
    },
  };
}

/** Writes Σ_p θ_p θ_pᵀ of the draws, `respondents` rows of `dimensions` entries, into `out`, by rows. */
export function secondMoments(theta: Float64Array, respondents: number, dimensions: number, out: Float64Array): void {
  out.fill(0, 0, dimensions * dimensions);
  for (let p = 0; p < respondents; p++) {
    for (let i = 0; i < dimensions; i++) {
      const row = theta[p * dimensions + i] ?? 0;
      for (let j = 0; j < dimensions; j++) {
        out[i * dimensions + j] = (out[i * dimensions + j] ?? 0) + row * (theta[p * dimensions + j] ?? 0);
      }
    }
  }
}

/**
 * Writes the gradient of the complete-data log-likelihood of the correlations, N(0, Φ) for each of `persons` draws with
 * second moments `second`, into `out`, in the order of the pairs: (Ω S Ω)_ij - N Ω_ij with Ω = Φ⁻¹.
 */
export function latentScore(
  correlations: Correlations,
  second: Float64Array,
  persons: number,
  out: Float64Array,
): void {
  const { dimensions, precision, pairs } = correlations;
  const middle = new Float64Array(dimensions * dimensions);
  const product = new Float64Array(dimensions * dimensions);
  multiply(precision, second, dimensions, dimensions, dimensions, product);
  multiply(product, precision, dimensions, dimensions, dimensions, middle);
  pairs.forEach(([i, j], a) => {
    out[a] = (middle[i * dimensions + j] ?? 0) - persons * (precision[i * dimensions + j] ?? 0);
  });
}

/**
 * Writes the score of each person's draw for the correlations into `out`, `pairs` entries per person, in the order of the
 * pairs: (Ω θ)_i (Ω θ)_j - Ω_ij.
 */
export function personLatentScores(
  correlations: Correlations,
  theta: Float64Array,
  respondents: number,
  out: Float64Array,
): void {
  const { dimensions, precision, pairs } = correlations;
  const product = new Float64Array(dimensions);
  for (let p = 0; p < respondents; p++) {
    for (let i = 0; i < dimensions; i++) {
      let sum = 0;
      for (let j = 0; j < dimensions; j++)
        sum += (precision[i * dimensions + j] ?? 0) * (theta[p * dimensions + j] ?? 0);
      product[i] = sum;
    }
    pairs.forEach(([i, j], a) => {
      out[p * pairs.length + a] = (product[i] ?? 0) * (product[j] ?? 0) - (precision[i * dimensions + j] ?? 0);
    });
  }
}

/**
 * Writes the observed information of the correlations in the draws with second moments `second` into `out`, by rows over
 * the pairs: -N (Ω_jk Ω_il + Ω_jl Ω_ik) plus half of tr(Ω Φ_a Ω Φ_b Ω S) and tr(Ω Φ_b Ω Φ_a Ω S), with Φ_a the matrix of
 * ones at the pair's two entries.
 */
export function latentObservedInformation(
  correlations: Correlations,
  second: Float64Array,
  persons: number,
  out: Float64Array,
): void {
  const { dimensions, precision, pairs } = correlations;
  const omega = (r: number, s: number): number => precision[r * dimensions + s] ?? 0;
  const t = new Float64Array(dimensions * dimensions);
  multiply(precision, second, dimensions, dimensions, dimensions, t);
  const products = pairs.map(([i, j]) => {
    const middle = new Float64Array(dimensions * dimensions);
    for (let r = 0; r < dimensions; r++) {
      for (let s = 0; s < dimensions; s++)
        middle[r * dimensions + s] = omega(r, i) * omega(j, s) + omega(r, j) * omega(i, s);
    }
    const product = new Float64Array(dimensions * dimensions);
    multiply(t, middle, dimensions, dimensions, dimensions, product);
    return product;
  });
  /** Returns tr(Ω Φ_a Ω Φ_b Ω S) from the product T Ω Φ_a Ω, for the pair b = (k, l). */
  const trace = (product: Float64Array | undefined, k: number, l: number): number =>
    (product?.[l * dimensions + k] ?? 0) + (product?.[k * dimensions + l] ?? 0);
  pairs.forEach(([i, j], a) => {
    pairs.forEach(([k, l], b) => {
      const expected = persons * (omega(j, k) * omega(i, l) + omega(j, l) * omega(i, k));
      out[a * pairs.length + b] = -expected + 0.5 * (trace(products[a], k, l) + trace(products[b], i, j));
    });
  });
}

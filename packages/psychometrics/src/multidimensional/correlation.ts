import { cholesky } from "@measurekit/core";
import { invertPositiveDefinite } from "../math/linear-algebra.js";
import { bfgs } from "../math/optimize.js";
import { type Grid, normalLogWeights } from "./grid.js";
import { multiply } from "./matrix.js";

/** The gradient norm, per person, at which the M-step of the correlations has converged. */
const CORRELATION_TOLERANCE = 1e-10;
const CORRELATION_MAX_ITERATIONS = 100;

/**
 * The distribution of the dimensions in a calibration: the normal distribution with mean 0 and a correlation matrix,
 * weighted on the nodes of a grid, whose correlations an M-step can estimate.
 */
export interface Latent {
  /** The correlation matrix, by rows. */
  readonly matrix: Float64Array;
  /** The log weights of the nodes, normalized so that the weights sum to 1. */
  readonly logWeights: Float64Array;
  /** The number of dimensions. */
  readonly dimensions: number;
  /** The pairs of dimensions below the diagonal, in the order of the correlations: (1, 0), (2, 0), (2, 1), ... */
  readonly pairs: readonly (readonly [number, number])[];
  /**
   * Sets the correlation matrix and the weights. Returns `false`, leaving both unchanged, when the matrix is not
   * positive definite.
   */
  set(matrix: Float64Array): boolean;
  /**
   * Writes the gradient of the expected negative complete-data log-likelihood with respect to the correlations into
   * `out`, in the order of the pairs (1, 0), (2, 0), (2, 1), ..., given the expected number of persons at each node.
   */
  gradient(nodeCounts: Float64Array, out: Float64Array): void;
  /** Sets the correlations that maximize the expected complete-data log-likelihood. */
  maximize(nodeCounts: Float64Array): void;
  /** Writes the derivative of ln w_q with respect to each correlation into `out`, by pair, then node. */
  scores(out: Float64Array): void;
}

/** Returns the pairs of dimensions below the diagonal in the order of the correlations. */
export function pairsOf(dimensions: number): [number, number][] {
  const pairs: [number, number][] = [];
  for (let i = 1; i < dimensions; i++) for (let j = 0; j < i; j++) pairs.push([i, j]);
  return pairs;
}

/** Returns the latent distribution on a grid, with the identity as its correlation matrix. */
export function latentDistribution(grid: Grid): Latent {
  const { dimensions, count } = grid;
  const size = dimensions * dimensions;
  const pairs = pairsOf(dimensions);
  const matrix = new Float64Array(size);
  const logWeights = new Float64Array(count);
  const weights = new Float64Array(count);
  const precision = new Float64Array(size);
  const zeros = new Float64Array(dimensions);
  const observed = new Float64Array(size);
  const expected = new Float64Array(size);
  const middle = new Float64Array(size);
  const full = new Float64Array(size);
  const outer = new Float64Array(size);
  const candidate = new Float64Array(size);
  const covector = new Float64Array(dimensions);
  const factor = {
    lower: new Float64Array(size),
    norms: new Float64Array(dimensions),
    gradient: new Float64Array(size),
  };
  let logNormalizer = 0;

  const set = (next: Float64Array): boolean => {
    const inverse = invertPositiveDefinite(next, dimensions);
    if (inverse === null) return false;
    precision.set(inverse);
    matrix.set(next);
    logNormalizer = normalLogWeights(grid, zeros, precision, logWeights);
    for (let q = 0; q < count; q++) weights[q] = Math.exp(logWeights[q] ?? 0);
    return true;
  };
  set(identity(dimensions));

  /**
   * Writes the gradient with respect to the correlation matrix, as a full symmetric matrix, of the negative expected
   * log-likelihood per person: -Ω (M_n - M_w) Ω / 2, with Ω the precision, M_n the observed second moments in
   * `observed` and M_w those of the weights.
   */
  const matrixGradient = (out: Float64Array): void => {
    secondMoments(expected, grid, weights, 1);
    for (let k = 0; k < size; k++) middle[k] = (observed[k] ?? 0) - (expected[k] ?? 0);
    multiply(precision, middle, dimensions, dimensions, dimensions, outer);
    multiply(outer, precision, dimensions, dimensions, dimensions, out);
    for (let k = 0; k < size; k++) out[k] = -(out[k] ?? 0) / 2;
  };

  /** Writes the second moments of the expected persons per person into `observed`; returns the number of persons. */
  const observe = (nodeCounts: Float64Array): number => {
    const persons = nodeCounts.reduce((sum, value) => sum + value, 0);
    secondMoments(observed, grid, nodeCounts, persons > 0 ? 1 / persons : 0);
    return persons;
  };

  /** Returns the negative expected log-likelihood per person at the parameters, writing its gradient. */
  const objective = (parameters: Float64Array, gradient: Float64Array): number => {
    correlationMatrix(parameters, dimensions, factor, candidate);
    // The parameters give a positive definite matrix, which can still be singular to working precision.
    if (!set(candidate)) {
      gradient.fill(0);
      return Number.POSITIVE_INFINITY;
    }
    matrixGradient(full);
    parametersGradient(full, dimensions, factor, gradient);
    return 0.5 * traceOfProduct(precision, observed, dimensions) + logNormalizer;
  };

  return {
    matrix,
    logWeights,
    dimensions,
    pairs,
    set,
    gradient(nodeCounts, out) {
      const persons = observe(nodeCounts);
      matrixGradient(full);
      // The correlation ρ_ij moves the entries (i, j) and (j, i) of the matrix.
      pairs.forEach(([i, j], p) => {
        out[p] = 2 * persons * (full[i * dimensions + j] ?? 0);
      });
    },
    maximize(nodeCounts) {
      const start = pairs.length === 0 ? null : rowParameters(matrix, dimensions);
      if (start === null || observe(nodeCounts) === 0) return;
      const { x } = bfgs(objective, start, {
        tolerance: CORRELATION_TOLERANCE,
        maxIterations: CORRELATION_MAX_ITERATIONS,
      });
      correlationMatrix(x, dimensions, factor, candidate);
      set(candidate);
    },
    scores(out) {
      secondMoments(expected, grid, weights, 1);
      multiply(precision, expected, dimensions, dimensions, dimensions, outer);
      multiply(outer, precision, dimensions, dimensions, dimensions, middle);
      for (let q = 0; q < count; q++) {
        precisionTimesNode(precision, grid, q, covector);
        pairs.forEach(([i, j], p) => {
          out[p * count + q] = (covector[i] ?? 0) * (covector[j] ?? 0) - (middle[i * dimensions + j] ?? 0);
        });
      }
    },
  };
}

/** The Cholesky-like factor of a correlation matrix with unit rows, and its scratch space. */
interface Factor {
  /** The lower triangular factor L by rows, each row of unit length. */
  readonly lower: Float64Array;
  /** The norm n_i of the unnormalized row i, 1 + the sum of the squares of its parameters. */
  readonly norms: Float64Array;
  /** The gradient with respect to the factor. */
  readonly gradient: Float64Array;
}

/** Writes Ω z_q into `out`. */
function precisionTimesNode(precision: Float64Array, { dimensions, nodes }: Grid, q: number, out: Float64Array): void {
  for (let i = 0; i < dimensions; i++) {
    let sum = 0;
    for (let j = 0; j < dimensions; j++) sum += (precision[i * dimensions + j] ?? 0) * (nodes[q * dimensions + j] ?? 0);
    out[i] = sum;
  }
}

/** Writes the second moments Σ_q scale v_q z_q z_qᵀ of the nodes under the weights `v` into `out`. */
function secondMoments(out: Float64Array, { dimensions, count, nodes }: Grid, v: Float64Array, scale: number): void {
  out.fill(0);
  for (let q = 0; q < count; q++) {
    const weight = scale * (v[q] ?? 0);
    if (weight !== 0) addOuterProduct(out, nodes, q, dimensions, weight);
  }
  for (let i = 0; i < dimensions; i++) {
    for (let j = 0; j < i; j++) out[j * dimensions + i] = out[i * dimensions + j] ?? 0;
  }
}

/** Adds `weight` times z_q z_qᵀ to the lower triangle of `out`. */
function addOuterProduct(out: Float64Array, nodes: Float64Array, q: number, dimensions: number, weight: number): void {
  for (let i = 0; i < dimensions; i++) {
    const row = weight * (nodes[q * dimensions + i] ?? 0);
    for (let j = 0; j <= i; j++) {
      out[i * dimensions + j] = (out[i * dimensions + j] ?? 0) + row * (nodes[q * dimensions + j] ?? 0);
    }
  }
}

/**
 * Writes the correlation matrix of the parameters into `out`, and the factor with it: row i of the factor is (u_i1,
 * ..., u_i,i-1, 1) scaled to unit length, so that the matrix L Lᵀ has a unit diagonal.
 */
function correlationMatrix(parameters: Float64Array, dimensions: number, factor: Factor, out: Float64Array): void {
  const { lower, norms } = factor;
  lower.fill(0);
  for (let i = 0, p = 0; i < dimensions; p += i, i++) {
    let squares = 1;
    for (let j = 0; j < i; j++) squares += (parameters[p + j] ?? 0) ** 2;
    const norm = Math.sqrt(squares);
    norms[i] = norm;
    for (let j = 0; j < i; j++) lower[i * dimensions + j] = (parameters[p + j] ?? 0) / norm;
    lower[i * dimensions + i] = 1 / norm;
  }
  for (let i = 0; i < dimensions; i++) {
    for (let k = 0; k <= i; k++) {
      const entry = rowProduct(lower, dimensions, i, k);
      out[i * dimensions + k] = entry;
      out[k * dimensions + i] = entry;
    }
  }
}

/** Returns the sum over j up to k of L[i][j] L[k][j]. */
function rowProduct(lower: Float64Array, dimensions: number, i: number, k: number): number {
  let sum = 0;
  for (let j = 0; j <= k; j++) sum += (lower[i * dimensions + j] ?? 0) * (lower[k * dimensions + j] ?? 0);
  return sum;
}

/**
 * Writes the gradient with respect to the parameters into `out`, given the gradient `g` with respect to the matrix and
 * the factor of the last {@link correlationMatrix}: G_L = 2 G L, and for the parameter u_ij of row i, (G_L[i][j] -
 * L[i][j] Σ_k G_L[i][k] L[i][k]) / n_i.
 */
function parametersGradient(g: Float64Array, dimensions: number, factor: Factor, out: Float64Array): void {
  const { lower, norms, gradient } = factor;
  factorGradient(g, dimensions, lower, gradient);
  for (let i = 1, p = 0; i < dimensions; p += i, i++) {
    const along = diagonalProduct(gradient, lower, dimensions, i);
    for (let j = 0; j < i; j++) {
      const slope = (gradient[i * dimensions + j] ?? 0) - (lower[i * dimensions + j] ?? 0) * along;
      out[p + j] = slope / (norms[i] ?? 1);
    }
  }
}

/** Writes 2 G L, the lower triangle of the gradient with respect to the factor L, into `out`. */
function factorGradient(g: Float64Array, dimensions: number, lower: Float64Array, out: Float64Array): void {
  for (let i = 0; i < dimensions; i++) {
    for (let k = 0; k <= i; k++) {
      let sum = 0;
      for (let m = k; m < dimensions; m++) sum += (g[i * dimensions + m] ?? 0) * (lower[m * dimensions + k] ?? 0);
      out[i * dimensions + k] = 2 * sum;
    }
  }
}

/** Returns the sum over k up to i of a[i][k] b[i][k]. */
function diagonalProduct(a: Float64Array, b: Float64Array, dimensions: number, i: number): number {
  let sum = 0;
  for (let k = 0; k <= i; k++) sum += (a[i * dimensions + k] ?? 0) * (b[i * dimensions + k] ?? 0);
  return sum;
}

/**
 * Returns the parameters of a positive definite correlation matrix: the entries below the diagonal of its Cholesky
 * factor, divided by the diagonal entry of their row, in the order of the pairs; `null` when the matrix is not positive
 * definite.
 */
function rowParameters(matrix: Float64Array, dimensions: number): Float64Array | null {
  const factor = cholesky(matrix, dimensions);
  if (factor === null) return null;
  const parameters: number[] = [];
  for (let i = 1; i < dimensions; i++) {
    for (let j = 0; j < i; j++) parameters.push((factor[i * dimensions + j] ?? 0) / (factor[i * dimensions + i] ?? 1));
  }
  return Float64Array.from(parameters);
}

/** Returns the identity matrix, by rows. */
export function identity(dimensions: number): Float64Array {
  const matrix = new Float64Array(dimensions * dimensions);
  for (let i = 0; i < dimensions; i++) matrix[i * dimensions + i] = 1;
  return matrix;
}

/** Returns tr(a b) of two `size` by `size` matrices. */
function traceOfProduct(a: Float64Array, b: Float64Array, size: number): number {
  let sum = 0;
  for (let i = 0; i < size; i++) for (let k = 0; k < size; k++) sum += (a[i * size + k] ?? 0) * (b[k * size + i] ?? 0);
  return sum;
}

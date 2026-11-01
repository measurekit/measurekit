import { bfgs, cholesky } from "@measurekit/core";
import { invertPositiveDefinite } from "../math/linear-algebra.js";
import { identity } from "../math/matrix.js";

import { type Grid, normalLogWeights } from "./grid.js";

/** The gradient norm, per person, at which the M-step of a latent distribution has converged. */
const CORRELATION_TOLERANCE = 1e-10;
const CORRELATION_MAX_ITERATIONS = 100;

/** Which entries of a latent distribution the M-step estimates. */
export interface LatentFree {
  /** Whether the correlations between the dimensions are estimated. */
  readonly correlations: boolean;
  /** Whether the standard deviations of the dimensions are estimated; otherwise they stay as they are set. */
  readonly variances: boolean;
  /** Whether the means of the dimensions are estimated; otherwise they stay as they are set. */
  readonly means: boolean;
}

/** The entries of a distribution whose dimensions have mean 0 and variance 1, and whose correlations are estimated. */
const FREE_CORRELATIONS: LatentFree = { correlations: true, variances: false, means: false };

/**
 * The distribution of the dimensions in a calibration: the normal distribution with a mean and a covariance matrix,
 * weighted on the nodes of a grid, whose estimated entries an M-step can maximize.
 *
 * @remarks
 * The estimated entries, on their own scale, are the correlations in the order of `pairs`, then the logarithms of the
 * standard deviations, then the means, each group only when it is free. `size` counts them.
 */
export interface Latent {
  /** The covariance matrix by rows: the correlation matrix while the variances are 1. */
  readonly matrix: Float64Array;
  /** The mean of each dimension. */
  readonly mean: Float64Array;
  /** The standard deviation of each dimension. */
  readonly standardDeviations: Float64Array;
  /** The log weights of the nodes, normalized so that the weights sum to 1. */
  readonly logWeights: Float64Array;
  /** The number of dimensions. */
  readonly dimensions: number;
  /** The pairs of dimensions below the diagonal, in the order of the correlations: (1, 0), (2, 0), (2, 1), ... */
  readonly pairs: readonly (readonly [number, number])[];
  /** The entries the M-step estimates. */
  readonly free: LatentFree;
  /** The number of estimated entries. */
  readonly size: number;
  /**
   * Sets the covariance matrix, and the mean when given, and the weights. Returns `false`, leaving them unchanged,
   * when the matrix is not positive definite.
   */
  set(matrix: Float64Array, mean?: ArrayLike<number>): boolean;
  /**
   * Writes the gradient of the expected negative complete-data log-likelihood with respect to the estimated entries
   * into `out`, given the expected number of persons at each node.
   */
  gradient(nodeCounts: Float64Array, out: Float64Array): void;
  /** Sets the estimated entries that maximize the expected complete-data log-likelihood. */
  maximize(nodeCounts: Float64Array): void;
  /**
   * Returns the parameters of the distribution, which any values keep a distribution when the correlations are positive
   * definite: for the correlations the entries below the diagonal of the Cholesky factor over the diagonal entry of
   * their row, in the order of the pairs, then the logarithms of the standard deviations and the means.
   */
  parameters(): Float64Array;
  /** Sets the distribution and the weights from parameters. Returns `false`, leaving both unchanged, when singular. */
  assign(parameters: Float64Array): boolean;
  /** Returns the estimated entries on their own scale. */
  entries(): Float64Array;
  /** Sets the estimated entries from their own scale. Returns `false`, leaving them unchanged, when singular. */
  assignEntries(entries: Float64Array): boolean;
  /** Writes the derivative of ln w_q with respect to each estimated entry into `out`, by entry, then node. */
  scores(out: Float64Array): void;
}

/** Returns the pairs of dimensions below the diagonal in the order of the correlations. */
export function pairsOf(dimensions: number): [number, number][] {
  const pairs: [number, number][] = [];
  for (let i = 1; i < dimensions; i++) for (let j = 0; j < i; j++) pairs.push([i, j]);
  return pairs;
}

/**
 * Returns the latent distribution on a grid, standard normal with uncorrelated dimensions at first.
 *
 * @param free - The entries the M-step estimates. The others stay as they are.
 */
export function latentDistribution(grid: Grid, free: LatentFree = FREE_CORRELATIONS): Latent {
  const state = stateOf(grid, free);
  install(state, state.correlation, state.deviations, state.mean);
  return {
    matrix: state.matrix,
    mean: state.mean,
    standardDeviations: state.deviations,
    logWeights: state.logWeights,
    dimensions: grid.dimensions,
    pairs: state.pairs,
    free,
    size: state.size,
    set: (next, location = state.mean) => setCovariance(state, next, location),
    gradient(nodeCounts, out) {
      const persons = observe(state, nodeCounts);
      entryGradient(state, out);
      for (let e = 0; e < state.size; e++) out[e] = persons * (out[e] ?? 0);
    },
    maximize(nodeCounts) {
      const start = state.size === 0 ? null : parametersOf(state);
      if (start === null || observe(state, nodeCounts) === 0) return;
      const { x } = bfgs((parameters, gradient) => objective(state, parameters, gradient), start, {
        tolerance: CORRELATION_TOLERANCE,
        maxIterations: CORRELATION_MAX_ITERATIONS,
      });
      fromParameters(state, x);
    },
    parameters: () => parametersOf(state) ?? new Float64Array(state.size),
    assign: (parameters) => fromParameters(state, parameters),
    entries: () => entriesOf(state),
    assignEntries: (entries) => fromEntries(state, entries),
    scores: (out) => nodeScores(state, out),
  };
}

/** A latent distribution during calibration: its moments, the weights they give the nodes, and scratch space. */
interface State {
  readonly grid: Grid;
  readonly free: LatentFree;
  readonly pairs: readonly (readonly [number, number])[];
  /** The number of estimated correlations; the standard deviations and the means follow them in the entries. */
  readonly correlationCount: number;
  /** The number of estimated entries. */
  readonly size: number;
  readonly matrix: Float64Array;
  readonly correlation: Float64Array;
  readonly mean: Float64Array;
  readonly deviations: Float64Array;
  readonly logWeights: Float64Array;
  readonly weights: Float64Array;
  /** The inverse of the covariance matrix. */
  readonly precision: Float64Array;
  /** The log of the sum of the unnormalized weights. */
  logNormalizer: number;
  readonly scratch: Scratch;
}

/** The scratch space of the distribution's computations. */
interface Scratch {
  readonly candidate: Float64Array;
  readonly ratio: Float64Array;
  readonly spread: Float64Array;
  readonly location: Float64Array;
  readonly residual: Float64Array;
  readonly covector: Float64Array;
  /** The derivative of -Q / 2 with respect to each estimated entry, by node, then entry. */
  readonly terms: Float64Array;
  /** The quadratic form (θ - μ)ᵀ Ω (θ - μ) at each node. */
  readonly quadratic: Float64Array;
  /** The share of the expected persons at each node. */
  readonly fractions: Float64Array;
  readonly natural: Float64Array;
  readonly full: Float64Array;
  readonly factor: Factor;
}

function stateOf(grid: Grid, free: LatentFree): State {
  const { dimensions, count } = grid;
  const size = dimensions * dimensions;
  const pairs = pairsOf(dimensions);
  const correlationCount = free.correlations ? pairs.length : 0;
  const entryCount = correlationCount + (free.variances ? dimensions : 0) + (free.means ? dimensions : 0);
  return {
    grid,
    free,
    pairs,
    correlationCount,
    size: entryCount,
    matrix: new Float64Array(size),
    correlation: identity(dimensions),
    mean: new Float64Array(dimensions),
    deviations: new Float64Array(dimensions).fill(1),
    logWeights: new Float64Array(count),
    weights: new Float64Array(count),
    precision: new Float64Array(size),
    logNormalizer: 0,
    scratch: {
      candidate: new Float64Array(size),
      ratio: new Float64Array(size),
      spread: new Float64Array(dimensions),
      location: new Float64Array(dimensions),
      residual: new Float64Array(dimensions),
      covector: new Float64Array(dimensions),
      terms: new Float64Array(count * entryCount),
      quadratic: new Float64Array(count),
      fractions: new Float64Array(count),
      natural: new Float64Array(entryCount),
      full: new Float64Array(size),
      factor: { lower: new Float64Array(size), norms: new Float64Array(dimensions), gradient: new Float64Array(size) },
    },
  };
}

/** Installs the distribution of correlations `r`, standard deviations `sd` and mean `location`, unless singular. */
function install(state: State, r: Float64Array, sd: ArrayLike<number>, location: ArrayLike<number>): boolean {
  const { dimensions, count } = state.grid;
  const { candidate } = state.scratch;
  for (let i = 0; i < dimensions; i++) {
    for (let j = 0; j < dimensions; j++) {
      // The diagonal of a correlation matrix is 1, whatever rounding the factor that made it left.
      candidate[i * dimensions + j] = (sd[i] ?? 1) * (sd[j] ?? 1) * (i === j ? 1 : (r[i * dimensions + j] ?? 0));
    }
  }
  const inverse = invertPositiveDefinite(candidate, dimensions);
  if (inverse === null) return false;
  state.precision.set(inverse);
  state.matrix.set(candidate);
  state.correlation.set(r);
  for (let i = 0; i < dimensions; i++) state.correlation[i * dimensions + i] = 1;
  state.deviations.set(sd);
  state.mean.set(location);
  state.logNormalizer = normalLogWeights(state.grid, state.mean, state.precision, state.logWeights);
  for (let q = 0; q < count; q++) state.weights[q] = Math.exp(state.logWeights[q] ?? 0);
  return true;
}

/** Sets the covariance matrix and the mean; `false`, leaving the distribution unchanged, when it is not positive definite. */
function setCovariance(state: State, next: Float64Array, location: ArrayLike<number>): boolean {
  const { dimensions } = state.grid;
  const { ratio, spread } = state.scratch;
  for (let i = 0; i < dimensions; i++) {
    const variance = next[i * dimensions + i] ?? 0;
    if (!(variance > 0)) return false;
    spread[i] = Math.sqrt(variance);
  }
  for (let i = 0; i < dimensions; i++) {
    for (let j = 0; j < dimensions; j++) {
      ratio[i * dimensions + j] = i === j ? 1 : (next[i * dimensions + j] ?? 0) / ((spread[i] ?? 1) * (spread[j] ?? 1));
    }
  }
  return install(state, ratio, spread, location);
}

/** Sets the fractions of persons at the nodes from their expected numbers; returns the number of persons. */
function observe({ grid, scratch }: State, nodeCounts: Float64Array): number {
  const persons = nodeCounts.reduce((sum, value) => sum + value, 0);
  for (let q = 0; q < grid.count; q++) scratch.fractions[q] = persons > 0 ? (nodeCounts[q] ?? 0) / persons : 0;
  return persons;
}

/**
 * Writes, for each node, the quadratic form (θ - μ)ᵀ Ω (θ - μ) and the derivative of -Q / 2 with respect to each
 * estimated entry: σ_i σ_j c_i c_j for the correlation of i and j, c_i (θ_i - μ_i) for the log standard deviation of i
 * and c_i for the mean of i, with c = Ω (θ - μ).
 */
function evaluate(state: State): void {
  const { dimensions, count, nodes } = state.grid;
  const { residual, covector, quadratic } = state.scratch;
  for (let q = 0; q < count; q++) {
    for (let i = 0; i < dimensions; i++) residual[i] = (nodes[q * dimensions + i] ?? 0) - (state.mean[i] ?? 0);
    let form = 0;
    for (let i = 0; i < dimensions; i++) {
      let sum = 0;
      for (let j = 0; j < dimensions; j++) sum += (state.precision[i * dimensions + j] ?? 0) * (residual[j] ?? 0);
      covector[i] = sum;
      form += (residual[i] ?? 0) * sum;
    }
    quadratic[q] = form;
    entryTerms(state, q * state.size);
  }
}

/** Writes the derivatives of -Q / 2 at the node whose residual and covector are in the scratch, from `start` on. */
function entryTerms(state: State, start: number): void {
  const { dimensions } = state.grid;
  const { covector, residual, terms } = state.scratch;
  let entry = state.free.correlations ? correlationTerms(state, start) : start;
  if (state.free.variances)
    for (let i = 0; i < dimensions; i++) terms[entry++] = (covector[i] ?? 0) * (residual[i] ?? 0);
  if (state.free.means) for (let i = 0; i < dimensions; i++) terms[entry++] = covector[i] ?? 0;
}

/** Writes the derivatives of -Q / 2 with respect to the correlations from `start` on; returns the next index. */
function correlationTerms(state: State, start: number): number {
  const { covector, terms } = state.scratch;
  let entry = start;
  for (const [i, j] of state.pairs) {
    terms[entry++] = (state.deviations[i] ?? 1) * (state.deviations[j] ?? 1) * (covector[i] ?? 0) * (covector[j] ?? 0);
  }
  return entry;
}

/** Writes the gradient of the negative expected log-likelihood per person at the fractions of persons into `out`. */
function entryGradient(state: State, out: Float64Array): void {
  evaluate(state);
  out.fill(0);
  const { fractions, terms } = state.scratch;
  for (let q = 0; q < state.grid.count; q++) {
    const excess = (fractions[q] ?? 0) - (state.weights[q] ?? 0);
    for (let e = 0; e < state.size; e++) out[e] = (out[e] ?? 0) - excess * (terms[q * state.size + e] ?? 0);
  }
}

/** Writes the derivative of ln w_q with respect to each estimated entry into `out`, by entry, then node. */
function nodeScores(state: State, out: Float64Array): void {
  evaluate(state);
  const { count } = state.grid;
  const { terms } = state.scratch;
  for (let e = 0; e < state.size; e++) {
    let expected = 0;
    for (let q = 0; q < count; q++) expected += (state.weights[q] ?? 0) * (terms[q * state.size + e] ?? 0);
    for (let q = 0; q < count; q++) out[e * count + q] = (terms[q * state.size + e] ?? 0) - expected;
  }
}

/** Reads the entries after the correlations: the standard deviations from their logarithms, then the means. */
function readTail(state: State, entries: Float64Array, spread: Float64Array, location: Float64Array): void {
  const { dimensions } = state.grid;
  let slot = state.correlationCount;
  if (state.free.variances) for (let i = 0; i < dimensions; i++) spread[i] = Math.exp(entries[slot++] ?? 0);
  if (state.free.means) for (let i = 0; i < dimensions; i++) location[i] = entries[slot++] ?? 0;
}

/** Writes the entries after the correlations: the logarithms of the standard deviations, then the means. */
function writeTail(state: State, out: Float64Array): void {
  const { dimensions } = state.grid;
  let slot = state.correlationCount;
  if (state.free.variances) for (let i = 0; i < dimensions; i++) out[slot++] = Math.log(state.deviations[i] ?? 1);
  if (state.free.means) for (let i = 0; i < dimensions; i++) out[slot++] = state.mean[i] ?? 0;
}

/** Installs the distribution of the parameters, whose entries that are not free stay as they are. */
function fromParameters(state: State, parameters: Float64Array): boolean {
  const { ratio, spread, location, factor } = state.scratch;
  if (state.free.correlations) correlationMatrix(parameters, state.grid.dimensions, factor, ratio);
  else ratio.set(state.correlation);
  spread.set(state.deviations);
  location.set(state.mean);
  readTail(state, parameters, spread, location);
  return install(state, ratio, spread, location);
}

/** Returns the parameters of the current distribution, or `null` when its correlations are not positive definite. */
function parametersOf(state: State): Float64Array | null {
  const rows = state.free.correlations ? rowParameters(state.correlation, state.grid.dimensions) : new Float64Array(0);
  if (rows === null) return null;
  const out = new Float64Array(state.size);
  out.set(rows);
  writeTail(state, out);
  return out;
}

function entriesOf(state: State): Float64Array {
  const { dimensions } = state.grid;
  const out = new Float64Array(state.size);
  if (state.free.correlations) {
    state.pairs.forEach(([i, j], p) => {
      out[p] = state.correlation[i * dimensions + j] ?? 0;
    });
  }
  writeTail(state, out);
  return out;
}

/** Installs the distribution of the entries on their own scale, whose entries that are not free stay as they are. */
function fromEntries(state: State, entries: Float64Array): boolean {
  const { dimensions } = state.grid;
  const { ratio, spread, location } = state.scratch;
  ratio.set(state.correlation);
  if (state.free.correlations) {
    state.pairs.forEach(([i, j], p) => {
      ratio[i * dimensions + j] = entries[p] ?? 0;
      ratio[j * dimensions + i] = entries[p] ?? 0;
    });
  }
  spread.set(state.deviations);
  location.set(state.mean);
  readTail(state, entries, spread, location);
  return install(state, ratio, spread, location);
}

/** Returns the negative expected log-likelihood per person at the parameters, writing its gradient. */
function objective(state: State, parameters: Float64Array, gradient: Float64Array): number {
  // The parameters give a positive definite matrix, which can still be singular to working precision.
  if (!fromParameters(state, parameters)) {
    gradient.fill(0);
    return Number.POSITIVE_INFINITY;
  }
  const { natural, full, factor, fractions, quadratic } = state.scratch;
  entryGradient(state, natural);
  let form = 0;
  for (let q = 0; q < state.grid.count; q++) form += (fractions[q] ?? 0) * (quadratic[q] ?? 0);
  if (state.free.correlations) {
    const { dimensions } = state.grid;
    full.fill(0);
    state.pairs.forEach(([i, j], p) => {
      full[i * dimensions + j] = (natural[p] ?? 0) / 2;
      full[j * dimensions + i] = (natural[p] ?? 0) / 2;
    });
    parametersGradient(full, dimensions, factor, gradient);
  }
  for (let e = state.correlationCount; e < state.size; e++) gradient[e] = natural[e] ?? 0;
  return 0.5 * form + state.logNormalizer;
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

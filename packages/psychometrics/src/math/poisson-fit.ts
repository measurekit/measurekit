import { newton } from "./optimize.js";

/** The maximum likelihood fit of a Poisson regression. */
export interface PoissonFit {
  /** The fitted mean of each cell, in the order of the counts. */
  readonly distribution: readonly number[];
  /** Whether Newton's method reached the maximum of the likelihood within its iterations. */
  readonly converged: boolean;
  /** Newton iterations run. */
  readonly iterations: number;
}

const MAX_ITERATIONS = 200;
const STEP_TOLERANCE = 1e-6;

/**
 * Returns the maximum likelihood fit of the Poisson regression ln m = Q β to the counts, `basis` being the matrix Q by
 * cell, then term, whose first column is the constant.
 */
export function poissonFit(counts: Float64Array, basis: Float64Array, terms: number, total: number): PoissonFit {
  const fitted = new Float64Array(counts.length);
  const start = new Float64Array(terms);
  start[0] = Math.log(total / counts.length) * Math.sqrt(counts.length);
  const result = newton(
    (beta, gradient, hessian) => poissonObjective(counts, basis, beta, fitted, gradient, hessian),
    start,
    // Newton's method converges quadratically: a step below 1e-6 leaves an error near 1e-12, below the rounding of
    // the likelihood's value, which the line search would need to see.
    { tolerance: 0, stepTolerance: STEP_TOLERANCE, maxIterations: MAX_ITERATIONS },
  );
  return { distribution: Array.from(fitted), converged: result.converged, iterations: result.iterations };
}

/**
 * Returns the negative Poisson log-likelihood Σ (m_x - f_x ln m_x), up to a constant, of the counts f at the fitted
 * means m = exp(Q β), which it writes into `fitted`, with its gradient Qᵀ (m - f) and Hessian Qᵀ diag(m) Q.
 */
function poissonObjective(
  counts: Float64Array,
  basis: Float64Array,
  beta: Float64Array,
  fitted: Float64Array,
  gradient: Float64Array,
  hessian: Float64Array,
): number {
  const terms = beta.length;
  gradient.fill(0);
  hessian.fill(0);
  let value = 0;
  for (let cell = 0; cell < counts.length; cell++) {
    const offset = cell * terms;
    let eta = 0;
    for (let j = 0; j < terms; j++) eta += (basis[offset + j] ?? 0) * (beta[j] ?? 0);
    const mean = Math.exp(eta);
    const count = counts[cell] ?? 0;
    fitted[cell] = mean;
    value += mean - count * eta;
    addCell(basis, offset, mean - count, mean, gradient, hessian);
  }
  for (let j = 0; j < terms; j++) for (let k = 0; k < j; k++) hessian[k * terms + j] = hessian[j * terms + k] ?? 0;
  return value;
}

/**
 * Adds the terms of the cell whose basis row starts at `offset` to the gradient, Q_j (m - f), and to the lower triangle
 * of the Hessian, m Q_j Q_k.
 */
function addCell(
  basis: Float64Array,
  offset: number,
  residual: number,
  mean: number,
  gradient: Float64Array,
  hessian: Float64Array,
): void {
  const terms = gradient.length;
  for (let j = 0; j < terms; j++) {
    const termJ = basis[offset + j] ?? 0;
    gradient[j] = (gradient[j] ?? 0) + residual * termJ;
    for (let k = 0; k <= j; k++) {
      hessian[j * terms + k] = (hessian[j * terms + k] ?? 0) + mean * termJ * (basis[offset + k] ?? 0);
    }
  }
}

import { backtrack, dot, logistic, logLogistic, maxAbs } from "@measurekit/core";
import { choleskySolve, squares } from "./linear-algebra.js";
import type { SparseMatrix } from "./sparse-matrix.js";

/**
 * A logistic regression with an intercept per group of rows, its rows ordered by group, and the work arrays of the
 * passes over it. Without groups, all rows form one block without an intercept.
 */
export interface Problem {
  readonly design: SparseMatrix;
  readonly outcomes: Float64Array;
  /** The number of group intercepts: 0 without groups. */
  readonly groupCount: number;
  /** The rows in group order: block g has rows `order[blockStarts[g]]` to `order[blockStarts[g + 1] - 1]`. */
  readonly order: Int32Array;
  readonly blockStarts: Int32Array;
  /** Per group, the columns its rows use and their sums B_g = Σ w x, at positions `touchedStarts[g]` onward. */
  readonly touchedStarts: Int32Array;
  readonly touchedColumns: Int32Array;
  readonly touchedSums: Float64Array;
  /** A dense scratch vector over the columns, and the group that last wrote each column. */
  readonly scratch: Float64Array;
  readonly stamp: Int32Array;
  /** x β of each row, for passes in which β is fixed. */
  readonly linear: Float64Array;
}

/** Returns the problem of a design with `groupCount` groups, `groups` giving each row's group. */
export function createProblem(
  design: SparseMatrix,
  outcomes: Float64Array,
  groups: Int32Array | undefined,
  groupCount: number,
): Problem {
  const rows = outcomes.length;
  const blocks = Math.max(groupCount, 1);
  const blockStarts = new Int32Array(blocks + 1);
  for (let row = 0; row < rows; row++) {
    const next = (groups?.[row] ?? 0) + 1;
    blockStarts[next] = (blockStarts[next] ?? 0) + 1;
  }
  for (let g = 0; g < blocks; g++) blockStarts[g + 1] = (blockStarts[g + 1] ?? 0) + (blockStarts[g] ?? 0);
  const order = new Int32Array(rows);
  const next = blockStarts.slice(0, blocks);
  for (let row = 0; row < rows; row++) {
    const g = groups?.[row] ?? 0;
    order[next[g] ?? 0] = row;
    next[g] = (next[g] ?? 0) + 1;
  }
  return {
    design,
    outcomes,
    groupCount,
    order,
    blockStarts,
    touchedStarts: new Int32Array(blocks + 1),
    touchedColumns: new Int32Array(design.columns.length),
    touchedSums: new Float64Array(design.columns.length),
    scratch: new Float64Array(design.columnCount),
    stamp: new Int32Array(design.columnCount),
    linear: new Float64Array(rows),
  };
}

/** Returns x β for a row. */
export function linear(design: SparseMatrix, row: number, beta: Float64Array): number {
  let eta = 0;
  for (let k = design.rowStarts[row] ?? 0; k < (design.rowStarts[row + 1] ?? 0); k++) {
    eta += (design.values[k] ?? 0) * (beta[design.columns[k] ?? 0] ?? 0);
  }
  return eta;
}

/** Returns ln(1 + exp(η)) - y η, a row's negative log-likelihood, without overflow. */
export function rowLoss(eta: number, y: number): number {
  return -logLogistic(-eta) - y * eta;
}

/** Returns Σ y ln p + (1 - y) ln(1 - p) at the coefficients and group intercepts. */
export function logLikelihood(problem: Problem, beta: Float64Array, intercepts: Float64Array): number {
  const { design, outcomes, order, blockStarts } = problem;
  let sum = 0;
  for (let g = 0; g < blockStarts.length - 1; g++) {
    const u = intercepts[g] ?? 0;
    for (let k = blockStarts[g] ?? 0; k < (blockStarts[g + 1] ?? 0); k++) {
      const row = order[k] ?? 0;
      sum -= rowLoss(u + linear(design, row, beta), outcomes[row] ?? 0);
    }
  }
  return sum;
}

/** The Newton system: the gradients, the group block's diagonal, and the coefficients' Schur complement. */
export interface System {
  readonly gradientBeta: Float64Array;
  readonly gradientIntercepts: Float64Array;
  /** The diagonal a_g of the group block of the Hessian. */
  readonly diagonal: Float64Array;
  /** The Hessian of the coefficients, then its Schur complement C - Bᵀ A⁻¹ B, by rows. */
  readonly schur: Float64Array;
  /** Bᵀ A⁻¹ g_u, which with -g_β is the right-hand side of the coefficients' step. */
  readonly reduced: Float64Array;
}

/** Returns an empty Newton system for `p` coefficients and `n` group intercepts. */
export function createSystem(p: number, n: number): System {
  return {
    gradientBeta: new Float64Array(p),
    gradientIntercepts: new Float64Array(n),
    diagonal: new Float64Array(n),
    schur: new Float64Array(p * p),
    reduced: new Float64Array(p),
  };
}

/**
 * Assembles the gradient and the Newton system of the negative log-likelihood + λ / 2 |β|² + μ / 2 |u|², one group at
 * a time: each group adds w x xᵀ of its rows to the coefficients' block, and subtracts its coupling B_g B_gᵀ /
 * a_g, with B_g = Σ w x and a_g = Σ w + μ over its rows.
 */
export function assemble(
  problem: Problem,
  beta: Float64Array,
  intercepts: Float64Array,
  ridge: number,
  groupPenalty: number,
  system: System,
): void {
  system.gradientBeta.fill(0);
  system.schur.fill(0);
  system.reduced.fill(0);
  problem.stamp.fill(-1);
  let touched = 0;
  const blocks = problem.blockStarts.length - 1;
  for (let g = 0; g < blocks; g++) {
    problem.touchedStarts[g] = touched;
    touched = assembleBlock(problem, system, beta, g, intercepts[g] ?? 0, groupPenalty, touched);
  }
  problem.touchedStarts[blocks] = touched;
  const p = beta.length;
  for (let j = 0; j < p; j++) {
    system.gradientBeta[j] = (system.gradientBeta[j] ?? 0) + ridge * (beta[j] ?? 0);
    // A column without entries, such as a feature that is 0 on every row, carries no information: its gradient is 0,
    // and a unit diagonal keeps its coefficient where it is.
    const diagonal = (system.schur[j * p + j] ?? 0) + ridge;
    system.schur[j * p + j] = diagonal === 0 ? 1 : diagonal;
  }
}

/**
 * Assembles the rows of block g, with its intercept u and its coupling with the coefficients; returns the new count of
 * touched columns.
 */
function assembleBlock(
  problem: Problem,
  system: System,
  beta: Float64Array,
  g: number,
  u: number,
  groupPenalty: number,
  touched: number,
): number {
  let count = touched;
  let a = groupPenalty;
  let gu = groupPenalty * u;
  for (let k = problem.blockStarts[g] ?? 0; k < (problem.blockStarts[g + 1] ?? 0); k++) {
    const row = problem.order[k] ?? 0;
    const p1 = logistic(u + linear(problem.design, row, beta));
    const w = p1 * (1 - p1);
    const residual = p1 - (problem.outcomes[row] ?? 0);
    a += w;
    gu += residual;
    addRow(problem, system, row, w, residual);
    if (problem.groupCount > 0) count = addGroupSums(problem, row, w, g, count);
  }
  if (problem.groupCount > 0) {
    system.diagonal[g] = a;
    system.gradientIntercepts[g] = gu;
    subtractCoupling(problem, system, g, count, a, gu);
  }
  return count;
}

/** Adds a row's (p - y) x to the coefficients' gradient and w x xᵀ to their Hessian. */
function addRow(problem: Problem, system: System, row: number, w: number, residual: number): void {
  const { design } = problem;
  const p = system.reduced.length;
  const start = design.rowStarts[row] ?? 0;
  const end = design.rowStarts[row + 1] ?? 0;
  for (let e = start; e < end; e++) {
    const column = design.columns[e] ?? 0;
    const x = design.values[e] ?? 0;
    system.gradientBeta[column] = (system.gradientBeta[column] ?? 0) + residual * x;
    for (let f = start; f < end; f++) {
      const cell = column * p + (design.columns[f] ?? 0);
      system.schur[cell] = (system.schur[cell] ?? 0) + w * x * (design.values[f] ?? 0);
    }
  }
}

/** Adds a row's w x to its group's sums B_g, recording each column the group touches first; returns the new count. */
function addGroupSums(problem: Problem, row: number, w: number, g: number, touched: number): number {
  const { design, stamp, scratch, touchedColumns } = problem;
  let count = touched;
  for (let e = design.rowStarts[row] ?? 0; e < (design.rowStarts[row + 1] ?? 0); e++) {
    const column = design.columns[e] ?? 0;
    if (stamp[column] !== g) {
      stamp[column] = g;
      scratch[column] = 0;
      touchedColumns[count++] = column;
    }
    scratch[column] = (scratch[column] ?? 0) + w * (design.values[e] ?? 0);
  }
  return count;
}

/** Subtracts a group's B_g B_gᵀ / a_g from the Schur complement and adds B_g g_u / a_g to the right-hand side. */
function subtractCoupling(problem: Problem, system: System, g: number, touched: number, a: number, gu: number): void {
  const { scratch, touchedColumns, touchedSums } = problem;
  const p = system.reduced.length;
  const start = problem.touchedStarts[g] ?? 0;
  for (let i = start; i < touched; i++) {
    const ci = touchedColumns[i] ?? 0;
    const bi = scratch[ci] ?? 0;
    touchedSums[i] = bi;
    system.reduced[ci] = (system.reduced[ci] ?? 0) + (bi * gu) / a;
    for (let j = start; j < touched; j++) {
      const cell = ci * p + (touchedColumns[j] ?? 0);
      system.schur[cell] = (system.schur[cell] ?? 0) - (bi * (scratch[touchedColumns[j] ?? 0] ?? 0)) / a;
    }
  }
}

/**
 * Solves the Newton system for the steps of the coefficients and the group intercepts; returns `false` when the Schur
 * complement is not positive definite. Along each flat direction f the complement is singular; adding f fᵀ / |f|²,
 * for each, makes it regular and picks the step with no component along any.
 */
function solveStep(
  problem: Problem,
  system: System,
  flats: readonly FlatDirection[],
  stepBeta: Float64Array,
  stepIntercepts: Float64Array,
): boolean {
  const { gradientBeta, gradientIntercepts, diagonal, schur, reduced } = system;
  for (const { coefficients } of flats) addOuter(schur, coefficients, 1 / squares(coefficients));
  const rhs = reduced.map((value, j) => value - (gradientBeta[j] ?? 0));
  if (!solveDamped(schur, stepBeta.length, rhs, stepBeta)) return false;
  const { touchedStarts, touchedColumns, touchedSums } = problem;
  for (let g = 0; g < stepIntercepts.length; g++) {
    let coupling = 0;
    for (let i = touchedStarts[g] ?? 0; i < (touchedStarts[g + 1] ?? 0); i++) {
      coupling += (touchedSums[i] ?? 0) * (stepBeta[touchedColumns[i] ?? 0] ?? 0);
    }
    stepIntercepts[g] = -((gradientIntercepts[g] ?? 0) + coupling) / (diagonal[g] ?? 1);
  }
  return true;
}

/** The damping of a singular Schur complement's first retry, relative to its largest diagonal entry. */
const FIRST_DAMPING = 1e-10;
/** The factor by which the damping grows between retries. */
const DAMPING_GROWTH = 100;

/**
 * Solves the Schur system S x = b, retrying with S + δ I where S is singular, as columns that are linear
 * combinations of others, such as successes and failures with opportunities, make it: δ grows from 1e-10 of the
 * largest diagonal entry by 100 a try, up to that entry (Levenberg-Marquardt). The gradient has no component along
 * a direction the likelihood is flat in, so the step along it stays near 0. Adds the damping to S in place.
 */
function solveDamped(schur: Float64Array, p: number, rhs: Float64Array, out: Float64Array): boolean {
  if (choleskySolve(schur, p, rhs, out)) return true;
  let largest = 0;
  for (let j = 0; j < p; j++) largest = Math.max(largest, Math.abs(schur[j * p + j] ?? 0));
  // An infinite diagonal, from a feature whose square overflows, would keep the damping infinite and the loop going.
  if (!(largest > 0 && largest < Number.POSITIVE_INFINITY)) return false;
  let added = 0;
  for (let damping = FIRST_DAMPING * largest; damping <= largest; damping *= DAMPING_GROWTH) {
    for (let j = 0; j < p; j++) schur[j * p + j] = (schur[j * p + j] ?? 0) + damping - added;
    added = damping;
    if (choleskySolve(schur, p, rhs, out)) return true;
  }
  return false;
}

/** Adds scale f fᵀ to a square matrix stored by rows, over the nonzero entries of f. */
function addOuter(matrix: Float64Array, f: Float64Array, scale: number): void {
  const p = f.length;
  const nonzero: number[] = [];
  f.forEach((value, i) => {
    if (value !== 0) nonzero.push(i);
  });
  for (const i of nonzero) {
    for (const j of nonzero) matrix[i * p + j] = (matrix[i * p + j] ?? 0) + scale * (f[i] ?? 0) * (f[j] ?? 0);
  }
}

/**
 * A direction along which the likelihood is flat: raising the intercepts of `groups` by c and the coefficients by c f
 * changes no row's prediction, as x f = -1 on those groups' rows and 0 elsewhere.
 */
export interface FlatDirection {
  /** f, over the coefficients. */
  readonly coefficients: Float64Array;
  /** The groups whose intercepts move. */
  readonly groups: Int32Array;
}

/** Options of {@link penalizedNewton}. */
export interface PenalizedNewtonOptions {
  /** λ, the penalty on the coefficients. */
  readonly ridge: number;
  /** μ, the penalty on the group intercepts: the ridge for fixed ones, 1 / σ² for random ones at σ. */
  readonly groupPenalty: number;
  /** The flat directions, with fixed intercepts and no penalty; see `GroupIntercepts.flats`. */
  readonly flats: readonly FlatDirection[];
  readonly tolerance: number;
  readonly maxIterations: number;
}

/**
 * Minimizes the negative log-likelihood + λ / 2 |β|² + μ / 2 |u|² over the coefficients β and the group intercepts
 * u, in place, by Newton steps with a backtracking line search; where the Schur complement is not positive
 * definite, the step follows the negative gradient.
 */
export function penalizedNewton(
  problem: Problem,
  beta: Float64Array,
  intercepts: Float64Array,
  options: PenalizedNewtonOptions,
): { iterations: number; converged: boolean } {
  const { ridge, groupPenalty, flats, tolerance, maxIterations } = options;
  const system = createSystem(beta.length, intercepts.length);
  const step = { beta: new Float64Array(beta.length), intercepts: new Float64Array(intercepts.length) };
  const objective = (b: Float64Array, u: Float64Array): number =>
    -logLikelihood(problem, b, u) + (ridge / 2) * squares(b) + (groupPenalty / 2) * squares(u);
  let value = objective(beta, intercepts);
  for (let iteration = 0; iteration < maxIterations; iteration++) {
    assemble(problem, beta, intercepts, ridge, groupPenalty, system);
    if (largestGradient(system) <= tolerance) return { iterations: iteration, converged: true };
    let slope = Number.NaN;
    if (solveStep(problem, system, flats, step.beta, step.intercepts)) {
      slope = dot(system.gradientBeta, step.beta) + dot(system.gradientIntercepts, step.intercepts);
    }
    if (!(slope < 0)) {
      step.beta.set(system.gradientBeta.map((g) => -g));
      step.intercepts.set(system.gradientIntercepts.map((g) => -g));
      slope = -(squares(step.beta) + squares(step.intercepts));
    }
    const trial = searchLine(beta, intercepts, step, value, slope, objective);
    if (trial === null) return { iterations: iteration, converged: false };
    value = trial;
    for (const flat of flats) center(beta, intercepts, flat);
  }
  assemble(problem, beta, intercepts, ridge, groupPenalty, system);
  return { iterations: maxIterations, converged: largestGradient(system) <= tolerance };
}

/**
 * The rounding error of a sum of row losses, relative to it: near the optimum a Newton step lowers the objective by
 * less than that, so a step within it is taken, and the gradient decides convergence.
 */
const ROUNDING = 1e-12;

/**
 * Halves the step from (β, u) until the objective decreases enough (the Armijo condition), or rises by no more than
 * its rounding error, moving β and u there in place; returns the objective there, or `null` when no step does.
 */
function searchLine(
  beta: Float64Array,
  intercepts: Float64Array,
  step: { readonly beta: Float64Array; readonly intercepts: Float64Array },
  value: number,
  slope: number,
  objective: (b: Float64Array, u: Float64Array) => number,
): number | null {
  const trialBeta = new Float64Array(beta.length);
  const trialIntercepts = new Float64Array(intercepts.length);
  let trial = value;
  const length = backtrack(
    (fraction) => {
      for (let j = 0; j < beta.length; j++) trialBeta[j] = (beta[j] ?? 0) + fraction * (step.beta[j] ?? 0);
      for (let g = 0; g < intercepts.length; g++) {
        trialIntercepts[g] = (intercepts[g] ?? 0) + fraction * (step.intercepts[g] ?? 0);
      }
      trial = objective(trialBeta, trialIntercepts);
      return trial;
    },
    value,
    slope,
    { slack: ROUNDING * (1 + Math.abs(value)) },
  );
  if (length === null) return null;
  beta.set(trialBeta);
  intercepts.set(trialIntercepts);
  return trial;
}

/** Moves along a flat direction so that the intercepts of its groups have mean 0; the likelihood is unchanged. */
function center(beta: Float64Array, intercepts: Float64Array, flat: FlatDirection): void {
  let mean = 0;
  for (const g of flat.groups) mean += intercepts[g] ?? 0;
  mean /= flat.groups.length;
  for (const g of flat.groups) intercepts[g] = (intercepts[g] ?? 0) - mean;
  for (let j = 0; j < beta.length; j++) beta[j] = (beta[j] ?? 0) - mean * (flat.coefficients[j] ?? 0);
}

function largestGradient(system: System): number {
  return Math.max(maxAbs(system.gradientBeta), maxAbs(system.gradientIntercepts));
}

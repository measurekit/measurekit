import { dot, logistic, maxAbs } from "@measurekit/core";
import {
  ARMIJO,
  assemble,
  createSystem,
  linear,
  type Problem,
  penalizedNewton,
  rowLoss,
  SHORTEST_STEP,
  type System,
} from "./arrow-system.js";
import { choleskySolve, squares } from "./linear-algebra.js";

/** Random group intercepts fitted by the Laplace approximation. */
export interface RandomIntercepts {
  readonly coefficients: Float64Array;
  /** The conditional modes of the group intercepts. */
  readonly modes: Float64Array;
  readonly sd: number;
  /** The Laplace approximation of the marginal log-likelihood at the estimates. */
  readonly logLikelihood: number;
  readonly iterations: number;
  readonly converged: boolean;
}

/** The step in τ = ln σ of the numerical second derivatives of L. */
const TAU_STEP = 1e-5;

/** The most Newton steps to a group's conditional mode. */
const MAX_MODE_ITERATIONS = 200;

/** The step, relative to the mode, below which a group's conditional mode has converged. */
const MODE_TOLERANCE = 1e-14;

/** The bound on the first Newton step to a conditional mode, on the logit scale. */
const INITIAL_MODE_BOUND = 2;

/**
 * Fits random group intercepts u_g ~ N(0, σ²): the penalized fit at σ = 1 as the start, then quasi-Newton steps that
 * maximize the Laplace approximation L(β, σ) of the marginal log-likelihood, less the ridge, over β and τ = ln σ.
 *
 * @remarks
 * The curvature of β is the Schur complement of the penalized Hessian at the conditional modes; the curvature of τ and
 * its coupling with β come from the exact gradient at τ + δ.
 */
export function randomIntercepts(
  problem: Problem,
  ridge: number,
  tolerance: number,
  maxIterations: number,
): RandomIntercepts {
  const p = problem.design.columnCount;
  const beta = new Float64Array(p);
  const modes = new Float64Array(problem.groupCount);
  penalizedNewton(problem, beta, modes, { ridge, groupPenalty: 1, flats: [], tolerance, maxIterations });
  const systems = { current: createSystem(p, modes.length), trial: createSystem(p, modes.length) };
  const shiftedSystem = createSystem(p, modes.length);
  let tau = 0;
  let current = laplace(problem, beta, Math.exp(tau), modes, ridge, systems.current);
  const result = (iterations: number, converged: boolean): RandomIntercepts => ({
    coefficients: beta,
    modes,
    sd: Math.exp(tau),
    logLikelihood: current.logLikelihood,
    iterations,
    converged,
  });
  for (let iteration = 0; iteration < maxIterations; iteration++) {
    if (largest(current) <= tolerance) return result(iteration, true);
    const trialModes = Float64Array.from(modes);
    const shifted = laplace(problem, beta, Math.exp(tau + TAU_STEP), trialModes, ridge, shiftedSystem);
    const step = quasiNewtonStep(systems.current, current, shifted);
    const accepted = takeStep(problem, { beta, modes, tau }, step, current, systems, ridge);
    if (accepted === null) return result(iteration, false);
    current = accepted.laplace;
    tau = accepted.tau;
  }
  return result(maxIterations, largest(current) <= tolerance);
}

/**
 * Returns the ascent step over (β, τ): the curvature's inverse times the gradient. The curvature is the Schur
 * complement for β, and differences of the gradient at τ + δ for τ and its coupling with β; the gradient where the
 * curvature is not positive definite.
 */
function quasiNewtonStep(system: System, current: Laplace, shifted: Laplace): Float64Array {
  const p = current.gradientBeta.length;
  const size = p + 1;
  const curvature = new Float64Array(size * size);
  for (let i = 0; i < p; i++) {
    for (let j = 0; j < p; j++) curvature[i * size + j] = system.schur[i * p + j] ?? 0;
    const coupling = -((shifted.gradientBeta[i] ?? 0) - (current.gradientBeta[i] ?? 0)) / TAU_STEP;
    curvature[i * size + p] = coupling;
    curvature[p * size + i] = coupling;
  }
  curvature[size * size - 1] = -(shifted.gradientTau - current.gradientTau) / TAU_STEP;
  const gradient = gradientOf(current);
  const step = new Float64Array(size);
  return choleskySolve(curvature, size, gradient, step) ? step : gradient;
}

/**
 * Takes the full step when L rises enough or the gradient shrinks, otherwise halves it until L rises enough: near the
 * maximum, L changes by less than the rounding of its sum over every row, and only the gradient still shows progress.
 * Moves β and the modes in place; returns the new τ and the approximation there, or `null` when no step is taken.
 */
function takeStep(
  problem: Problem,
  point: { readonly beta: Float64Array; readonly modes: Float64Array; readonly tau: number },
  step: Float64Array,
  current: Laplace,
  systems: { current: System; trial: System },
  ridge: number,
): { tau: number; laplace: Laplace } | null {
  const { beta, modes, tau } = point;
  const p = beta.length;
  const slope = dot(gradientOf(current), step);
  const trialBeta = new Float64Array(p);
  const trialModes = new Float64Array(modes.length);
  const move = (length: number): number => {
    for (let j = 0; j < p; j++) trialBeta[j] = (beta[j] ?? 0) + length * (step[j] ?? 0);
    trialModes.set(modes);
    return tau + length * (step[p] ?? 0);
  };
  const accept = (trialTau: number, value: Laplace): { tau: number; laplace: Laplace } => {
    beta.set(trialBeta);
    modes.set(trialModes);
    return { tau: trialTau, laplace: value };
  };
  const fullTau = move(1);
  const full = laplace(problem, trialBeta, Math.exp(fullTau), trialModes, ridge, systems.trial);
  if (full.value >= current.value + ARMIJO * slope || largest(full) < largest(current)) {
    [systems.current, systems.trial] = [systems.trial, systems.current];
    return accept(fullTau, full);
  }
  for (let length = 0.5; length >= SHORTEST_STEP; length /= 2) {
    const trialTau = move(length);
    const value = laplace(problem, trialBeta, Math.exp(trialTau), trialModes, ridge, null).value;
    if (value >= current.value + ARMIJO * length * slope) {
      return accept(trialTau, laplace(problem, trialBeta, Math.exp(trialTau), trialModes, ridge, systems.current));
    }
  }
  return null;
}

/** Returns the gradient over β and τ as one vector, τ last. */
function gradientOf(state: Laplace): Float64Array {
  const gradient = new Float64Array(state.gradientBeta.length + 1);
  gradient.set(state.gradientBeta);
  gradient[state.gradientBeta.length] = state.gradientTau;
  return gradient;
}

function largest(state: Laplace): number {
  return Math.max(maxAbs(state.gradientBeta), Math.abs(state.gradientTau));
}

/** The Laplace approximation at β and σ, with its gradient over β and τ = ln σ. */
interface Laplace {
  /** L less the ridge penalty: the objective. */
  readonly value: number;
  /** L, the Laplace approximation of the marginal log-likelihood. */
  readonly logLikelihood: number;
  readonly gradientBeta: Float64Array;
  readonly gradientTau: number;
}

/**
 * Computes the conditional modes b̂_g of the group intercepts in place, from their current values, and the Laplace
 * approximation L = Σ ℓ(β, b̂) - Σ b̂² / (2σ²) - ½ Σ ln(1 + σ² h_g), h_g = Σ w over the group's rows. With a
 * system, also assembles the penalized Hessian there, whose Schur complement is the curvature of β, and the
 * gradient over β and τ.
 */
function laplace(
  problem: Problem,
  beta: Float64Array,
  sd: number,
  modes: Float64Array,
  ridge: number,
  system: System | null,
): Laplace {
  const { design, outcomes, order, blockStarts, linear: eta0 } = problem;
  const variance = sd * sd;
  for (let row = 0; row < outcomes.length; row++) eta0[row] = linear(design, row, beta);
  let logLikelihood = 0;
  for (let g = 0; g < modes.length; g++) {
    const start = blockStarts[g] ?? 0;
    const end = blockStarts[g + 1] ?? 0;
    const b = conditionalMode(problem, start, end, modes[g] ?? 0, 1 / variance);
    modes[g] = b;
    let h = 0;
    for (let k = start; k < end; k++) {
      const row = order[k] ?? 0;
      const eta = b + (eta0[row] ?? 0);
      const p1 = logistic(eta);
      h += p1 * (1 - p1);
      logLikelihood -= rowLoss(eta, outcomes[row] ?? 0);
    }
    logLikelihood -= (b * b) / (2 * variance) + Math.log1p(variance * h) / 2;
  }
  const value = logLikelihood - (ridge / 2) * squares(beta);
  if (system === null) return { value, logLikelihood, gradientBeta: new Float64Array(0), gradientTau: 0 };
  assemble(problem, beta, modes, ridge, 1 / variance, system);
  return { value, logLikelihood, ...gradients(problem, modes, variance, system) };
}

/**
 * Returns the gradient of L, the modes moving with β and σ through the log-determinant:
 * ∂L/∂β = Σ (y - p) x - ½ Σ κ_g (D_g - c_g B_g / a_g) and
 * ∂L/∂τ = Σ (b̂² / σ² - σ² h_g / (1 + σ² h_g) - κ_g c_g b̂ / (σ² a_g)), with a_g = h_g + 1/σ², κ_g = σ² / (1 + σ² h_g),
 * c_g = Σ w (1 - 2p), D_g = Σ w (1 - 2p) x and B_g = Σ w x over the group's rows, from the assembled system.
 */
function gradients(
  problem: Problem,
  modes: Float64Array,
  variance: number,
  system: System,
): { gradientBeta: Float64Array; gradientTau: number } {
  problem.scratch.fill(0);
  let gradientTau = 0;
  for (let g = 0; g < modes.length; g++) gradientTau += addGroupGradient(problem, system, g, modes[g] ?? 0, variance);
  const gradientBeta = system.gradientBeta.map((g, j) => -g - (problem.scratch[j] ?? 0) / 2);
  return { gradientBeta, gradientTau };
}

/** Adds group g's κ_g (D_g - c_g B_g / a_g) to `problem.scratch` and returns its term of ∂L/∂τ. */
function addGroupGradient(problem: Problem, system: System, g: number, b: number, variance: number): number {
  const { design, order, blockStarts, linear: eta0, touchedStarts, touchedColumns, touchedSums, scratch } = problem;
  const a = system.diagonal[g] ?? 1;
  const h = a - 1 / variance;
  const kappa = variance / (1 + variance * h);
  let c = 0;
  for (let k = blockStarts[g] ?? 0; k < (blockStarts[g + 1] ?? 0); k++) {
    const row = order[k] ?? 0;
    const p1 = logistic(b + (eta0[row] ?? 0));
    const skew = p1 * (1 - p1) * (1 - 2 * p1);
    c += skew;
    addScaledRow(design, row, kappa * skew, scratch);
  }
  for (let i = touchedStarts[g] ?? 0; i < (touchedStarts[g + 1] ?? 0); i++) {
    const column = touchedColumns[i] ?? 0;
    scratch[column] = (scratch[column] ?? 0) - (kappa * c * (touchedSums[i] ?? 0)) / a;
  }
  return (b * b) / variance - (variance * h) / (1 + variance * h) - (kappa * c * b) / (variance * a);
}

/** Adds scale x of a row to `out`. */
function addScaledRow(design: Problem["design"], row: number, scale: number, out: Float64Array): void {
  for (let e = design.rowStarts[row] ?? 0; e < (design.rowStarts[row + 1] ?? 0); e++) {
    const column = design.columns[e] ?? 0;
    out[column] = (out[column] ?? 0) + scale * (design.values[e] ?? 0);
  }
}

/**
 * Returns the mode of a group's intercept given β and σ: the maximum of Σ ℓ(x β + b) - b² precision / 2 over its rows,
 * from `initial`, with x β in `problem.linear`. Newton steps are bounded, and the bound halves whenever a
 * step overshoots the maximum by more than the step before, so the iteration converges on this concave function.
 */
function conditionalMode(problem: Problem, start: number, end: number, initial: number, precision: number): number {
  const { outcomes, order, linear: eta0 } = problem;
  let b = initial;
  let bound = INITIAL_MODE_BOUND;
  let previous = Number.POSITIVE_INFINITY;
  for (let iteration = 0; iteration < MAX_MODE_ITERATIONS; iteration++) {
    let slope = -precision * b;
    let curvature = precision;
    for (let k = start; k < end; k++) {
      const row = order[k] ?? 0;
      const p1 = logistic(b + (eta0[row] ?? 0));
      slope += (outcomes[row] ?? 0) - p1;
      curvature += p1 * (1 - p1);
    }
    if (Math.abs(slope) > previous) bound /= 2;
    previous = Math.abs(slope);
    const step = Math.max(-bound, Math.min(bound, slope / curvature));
    b += step;
    if (Math.abs(step) <= MODE_TOLERANCE * (1 + Math.abs(b))) break;
  }
  return b;
}

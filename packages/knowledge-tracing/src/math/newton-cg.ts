import { backtrack, dot, maxAbs } from "@measurekit/core";

/** A smooth function with a positive definite Hessian, which Newton-CG minimizes without forming the matrix. */
export interface NewtonObjective {
  /** Returns the objective at x. */
  value(x: Float64Array): number;
  /** Writes the gradient and the diagonal of the Hessian at x, and prepares {@link hessianVector} for x. */
  linearize(x: Float64Array, gradient: Float64Array, diagonal: Float64Array): void;
  /** Writes H v into `out`, for the Hessian at the point of the last {@link linearize}. */
  hessianVector(v: Float64Array, out: Float64Array): void;
}

/** Options of {@link newtonCg}. */
export interface NewtonCgOptions {
  /** The largest absolute gradient component at which the minimization has converged. */
  readonly tolerance: number;
  /** The most Newton iterations. */
  readonly maxIterations: number;
}

/**
 * The rounding error of a sum of terms, relative to it: near the optimum a Newton step lowers the objective by less than
 * that, so a step within it is taken, and the gradient decides convergence.
 */
export const OBJECTIVE_ROUNDING = 1e-12;

/** The most conjugate gradient steps of one Newton iteration. */
const MAX_CG_STEPS = 1_000;

/** The largest share of the gradient's norm that the residual of a Newton step may keep: a step that is at least half exact. */
const LARGEST_FORCING = 0.5;

/**
 * Minimizes a smooth objective in place from `x` by Newton steps whose linear system is solved approximately by
 * conjugate gradients on Hessian-vector products, with the diagonal of the Hessian as the preconditioner and a
 * backtracking line search.
 *
 * @remarks
 * Memory and time per iteration grow with the work of one Hessian-vector product, not with the square of the number of
 * variables. The conjugate gradients stop when the residual is under min(0.5, √|g|) of the gradient's norm, the forcing
 * sequence that keeps the convergence superlinear.
 *
 * @see {@link https://doi.org/10.1007/978-0-387-40065-5 | Nocedal and Wright (2006), chapter 7}
 */
export function newtonCg(
  objective: NewtonObjective,
  x: Float64Array,
  options: NewtonCgOptions,
): { iterations: number; converged: boolean } {
  const { tolerance, maxIterations } = options;
  const size = x.length;
  const gradient = new Float64Array(size);
  const diagonal = new Float64Array(size);
  const step = new Float64Array(size);
  const trial = new Float64Array(size);
  const solver = conjugateGradients(objective, size);
  let value = objective.value(x);
  for (let iteration = 0; iteration < maxIterations; iteration++) {
    objective.linearize(x, gradient, diagonal);
    if (maxAbs(gradient) <= tolerance) return { iterations: iteration, converged: true };
    solver(gradient, diagonal, step);
    let trialValue = value;
    const length = backtrack(
      (fraction) => {
        for (let j = 0; j < size; j++) trial[j] = (x[j] ?? 0) + fraction * (step[j] ?? 0);
        trialValue = objective.value(trial);
        return trialValue;
      },
      value,
      dot(gradient, step),
      { slack: OBJECTIVE_ROUNDING * (1 + Math.abs(value)) },
    );
    if (length === null) return { iterations: iteration, converged: false };
    x.set(trial);
    value = trialValue;
  }
  objective.linearize(x, gradient, diagonal);
  return { iterations: maxIterations, converged: maxAbs(gradient) <= tolerance };
}

/** The work arrays of the conjugate gradients. */
interface CgWork {
  readonly residual: Float64Array;
  /** The residual over the diagonal of the Hessian. */
  readonly scaled: Float64Array;
  readonly direction: Float64Array;
  /** H times the direction. */
  readonly product: Float64Array;
}

/**
 * Returns the solver of H s = -g by preconditioned conjugate gradients from s = 0, with its work arrays: it writes
 * s, a descent direction, for a gradient and the diagonal of the Hessian.
 */
function conjugateGradients(
  objective: NewtonObjective,
  size: number,
): (gradient: Float64Array, diagonal: Float64Array, step: Float64Array) => void {
  const work: CgWork = {
    residual: new Float64Array(size),
    scaled: new Float64Array(size),
    direction: new Float64Array(size),
    product: new Float64Array(size),
  };
  const limit = Math.min(size, MAX_CG_STEPS);
  return (gradient, diagonal, step) => {
    step.fill(0);
    for (let j = 0; j < size; j++) work.residual[j] = -(gradient[j] ?? 0);
    const norm = Math.sqrt(dot(gradient, gradient));
    divide(work.scaled, work.residual, diagonal);
    work.direction.set(work.scaled);
    descend(objective, work, diagonal, step, { target: Math.min(LARGEST_FORCING, Math.sqrt(norm)) * norm, limit });
    // The first direction is the preconditioned negative gradient, so a break before any step still leaves a descent.
    if (maxAbs(step) === 0) step.set(work.scaled);
  };
}

/** Runs conjugate gradients steps until the residual is within the target or the limit of steps is reached. */
function descend(
  objective: NewtonObjective,
  work: CgWork,
  diagonal: Float64Array,
  step: Float64Array,
  stop: { readonly target: number; readonly limit: number },
): void {
  const { residual, scaled, direction, product } = work;
  let weight = dot(residual, scaled);
  for (let k = 0; k < stop.limit; k++) {
    objective.hessianVector(direction, product);
    const curvature = dot(direction, product);
    // A direction without curvature is flat: the step so far is the best that the system offers.
    if (!(curvature > 0)) return;
    const length = weight / curvature;
    addScaled(step, length, direction);
    addScaled(residual, -length, product);
    if (Math.sqrt(dot(residual, residual)) <= stop.target) return;
    divide(scaled, residual, diagonal);
    const next = dot(residual, scaled);
    for (let j = 0; j < step.length; j++) direction[j] = (scaled[j] ?? 0) + (next / weight) * (direction[j] ?? 0);
    weight = next;
  }
}

/** Adds scale times x to y. */
function addScaled(y: Float64Array, scale: number, x: Float64Array): void {
  for (let j = 0; j < y.length; j++) y[j] = (y[j] ?? 0) + scale * (x[j] ?? 0);
}

/** Writes the entries of the numerator divided by those of the denominator into `out`. */
function divide(out: Float64Array, numerator: Float64Array, denominator: Float64Array): void {
  for (let j = 0; j < out.length; j++) out[j] = (numerator[j] ?? 0) / (denominator[j] ?? 1);
}

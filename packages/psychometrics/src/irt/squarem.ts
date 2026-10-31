import type { Convergence, Stopping } from "@measurekit/core";

/** One EM iteration of a state that can be read and set as a vector in unbounded coordinates. */
export interface Iteration {
  /**
   * Runs the E-step and the M-step from the current state and returns the log-likelihood at the state it started from
   * and the largest change of a parameter in the step.
   */
  step(): { readonly logLikelihood: number; readonly change: number };
  /** Returns the current state. */
  get(): Float64Array;
  /** Sets the current state. */
  set(state: Float64Array): void;
}

/** The α of the plain EM step to x₂, which an extrapolation is never shorter than. */
const PLAIN_STEP = -1;
/** The factor by which the longest α grows after a jump to it is kept, and shrinks after one is not. */
const LIMIT_FACTOR = 4;
/** The longest α of the first cycle. */
const FIRST_LIMIT = 4;

/** The iterations run so far and the end that they are run to. */
interface Run extends Stopping {
  readonly iteration: Iteration;
  iterations: number;
  /** The longest jump, as a multiple of the step from x₀ to x₁, that a cycle takes. */
  limit: number;
}

/** Runs an EM step, counts it and returns its log-likelihood and whether it changed no parameter by more than the tolerance. */
function advance(run: Run): { readonly logLikelihood: number; readonly converged: boolean } {
  run.iterations++;
  const { logLikelihood, change } = run.iteration.step();
  return { logLikelihood, converged: change <= run.tolerance };
}

/**
 * Runs EM with the squared extrapolation SQUAREM (scheme S3), until a plain EM step changes no parameter by more than
 * `tolerance` or `maxIterations` EM iterations have run.
 *
 * @remarks
 * A cycle is two plain EM steps, from x₀ to x₁ and x₂, then a jump to x₀ - 2αr + α²v, with r = x₁ - x₀, v = x₂ - 2x₁ + x₀
 * and α = -‖r‖ / ‖v‖, which is where the linear convergence of EM would end if its rate were constant, and a third EM
 * step from there. The jump is kept when the log-likelihood there is at least that at x₁, so that the log-likelihood
 * never falls below that of the first iteration; otherwise the cycle goes to x₂ instead, whose log-likelihood is at least
 * that of x₁. The jump is at most `limit` times the plain step, which starts at 4 and is multiplied by 4 after a jump to
 * it is kept and divided by 4 after one is not. Each EM step of a cycle counts as an iteration.
 *
 * @see {@link https://doi.org/10.1111/j.1467-9469.2007.00585.x | Varadhan and Roland (2008)}
 */
export function squarem(iteration: Iteration, stopping: Stopping): Convergence {
  const run: Run = { ...stopping, iteration, iterations: 0, limit: FIRST_LIMIT };
  while (run.iterations < run.maxIterations) {
    const start = iteration.get();
    if (advance(run).converged) return { converged: true, iterations: run.iterations };
    const x1 = iteration.get();
    if (run.iterations >= run.maxIterations) break;
    const second = advance(run);
    if (second.converged) return { converged: true, iterations: run.iterations };
    if (run.iterations >= run.maxIterations) break;
    if (jump(run, [start, x1, iteration.get()], second.logLikelihood))
      return { converged: true, iterations: run.iterations };
  }
  return { converged: false, iterations: run.iterations };
}

/**
 * Extrapolates from the two plain steps, takes an EM step from there and returns whether it changed no parameter by more
 * than the tolerance. A jump that lowers the log-likelihood below `floor` is replaced by the plain step to x₂.
 */
function jump(run: Run, [start, x1, x2]: readonly [Float64Array, Float64Array, Float64Array], floor: number): boolean {
  const r = x1.map((x, j) => x - (start[j] ?? 0));
  const v = x2.map((x, j) => x - 2 * (x1[j] ?? 0) + (start[j] ?? 0));
  const alpha = norm(v) > 0 ? Math.max(-run.limit, Math.min(PLAIN_STEP, -norm(r) / norm(v))) : PLAIN_STEP;
  run.iteration.set(start.map((x, j) => x - 2 * alpha * (r[j] ?? 0) + alpha * alpha * (v[j] ?? 0)));
  const jumped = advance(run);
  if (jumped.converged) return true;
  if (jumped.logLikelihood >= floor) {
    if (alpha === -run.limit) run.limit *= LIMIT_FACTOR;
    return false;
  }
  run.limit = Math.max(-PLAIN_STEP, run.limit / LIMIT_FACTOR);
  if (run.iterations >= run.maxIterations) {
    run.iteration.set(x2);
    return false;
  }
  // The jump overshot: the plain step, to x₂, does not lower the log-likelihood.
  run.iteration.set(x2);
  return advance(run).converged;
}

function norm(values: Float64Array): number {
  let sum = 0;
  for (const value of values) sum += value * value;
  return Math.sqrt(sum);
}

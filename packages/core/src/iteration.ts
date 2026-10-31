/** The options that stop an iterative method. */
export interface Stopping {
  /** The largest change of one cycle at which the method has converged. */
  readonly tolerance: number;
  /** The most cycles to run: at least 1. */
  readonly maxIterations: number;
}

/** How an iterative method ended. */
export interface Convergence {
  /** Whether the last cycle changed by at most the tolerance. */
  readonly converged: boolean;
  /** The cycles run. */
  readonly iterations: number;
}

/**
 * Runs `cycle` until it reports a change of at most `tolerance`, or `maxIterations` cycles have run.
 *
 * @remarks
 * What a change is, such as the largest change of a parameter, of the log-likelihood or of a histogram, is the
 * cycle's own test: it returns the size of the change it made. A change that is `NaN` never converges.
 *
 * @param cycle - Runs one cycle and returns its change.
 * @param stopping - The tolerance and the most cycles, as {@link checkStopping} checks them.
 * @returns The cycles run, and whether the last one converged.
 */
export function iterateToConvergence(cycle: () => number, { tolerance, maxIterations }: Stopping): Convergence {
  let iterations = 0;
  let converged = false;
  while (!converged && iterations < maxIterations) {
    iterations++;
    converged = cycle() <= tolerance;
  }
  return { converged, iterations };
}

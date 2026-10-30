/** The outcome of one EM iteration from some parameters. */
export interface ExpectationMaximizationStep<TParameters> {
  /** The log-likelihood at the parameters the step started from: the E-step's. */
  readonly logLikelihood: number;
  /** The parameters that maximize the expected complete-data log-likelihood: the M-step's. */
  readonly parameters: TParameters;
}

/** Stopping rules of expectation maximization. */
export interface ExpectationMaximizationOptions {
  /** The largest change of the log-likelihood between iterations at which EM has converged. */
  readonly tolerance: number;
  /** The most iterations to run: at least 1. */
  readonly maxIterations: number;
}

/** The outcome of expectation maximization. */
export interface ExpectationMaximization<TParameters> {
  /** The last parameters whose log-likelihood was computed. */
  readonly parameters: TParameters;
  /** The log-likelihood at `parameters`. */
  readonly logLikelihood: number;
  /** Iterations run: E-steps computed. */
  readonly iterations: number;
  /** Whether the log-likelihood changed by at most the tolerance in the last iteration. */
  readonly converged: boolean;
}

/**
 * Runs expectation maximization from `start`. Each iteration calls `step`, which returns the log-likelihood at its
 * parameters and the next parameters. EM stops when the log-likelihood changes by at most the tolerance, and returns
 * the parameters at which it was computed, so that `logLikelihood` belongs to `parameters`.
 */
export function expectationMaximization<TParameters>(
  start: TParameters,
  step: (parameters: TParameters) => ExpectationMaximizationStep<TParameters>,
  { tolerance, maxIterations }: ExpectationMaximizationOptions,
): ExpectationMaximization<TParameters> {
  let parameters = start;
  let previous = Number.NEGATIVE_INFINITY;
  for (let iteration = 1; ; iteration++) {
    const { logLikelihood, parameters: next } = step(parameters);
    const converged = Math.abs(logLikelihood - previous) <= tolerance;
    if (converged || iteration >= maxIterations) return { parameters, logLikelihood, iterations: iteration, converged };
    previous = logLikelihood;
    parameters = next;
  }
}

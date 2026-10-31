import { type Convergence, iterateToConvergence, type Stopping } from "@measurekit/core";

/** The outcome of one EM iteration from some parameters. */
export interface ExpectationMaximizationStep<TParameters> {
  /** The log-likelihood at the parameters the step started from: the E-step's. */
  readonly logLikelihood: number;
  /** The parameters that maximize the expected complete-data log-likelihood: the M-step's. */
  readonly parameters: TParameters;
}

/** The outcome of expectation maximization: the iterations are the E-steps computed. */
export interface ExpectationMaximization<TParameters> extends Convergence {
  /** The last parameters whose log-likelihood was computed. */
  readonly parameters: TParameters;
  /** The log-likelihood at `parameters`. */
  readonly logLikelihood: number;
}

/**
 * Runs expectation maximization from `start`. Each iteration calls `step`, which returns the log-likelihood at its
 * parameters and the next parameters. EM stops when the log-likelihood changes by at most the tolerance, and returns
 * the parameters at which it was computed, so that `logLikelihood` belongs to `parameters`.
 */
export function expectationMaximization<TParameters>(
  start: TParameters,
  step: (parameters: TParameters) => ExpectationMaximizationStep<TParameters>,
  stopping: Stopping,
): ExpectationMaximization<TParameters> {
  let next = start;
  let parameters = start;
  let logLikelihood = Number.NEGATIVE_INFINITY;
  const { converged, iterations } = iterateToConvergence(() => {
    parameters = next;
    const previous = logLikelihood;
    ({ logLikelihood, parameters: next } = step(parameters));
    return Math.abs(logLikelihood - previous);
  }, stopping);
  return { parameters, logLikelihood, iterations, converged };
}

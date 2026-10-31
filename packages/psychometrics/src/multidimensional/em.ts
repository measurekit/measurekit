import { DataError } from "@measurekit/core";
import { type Fit, parameterCount } from "./standard-errors.js";

/** The most dimensions that EM calibrates: its grid grows as the number of points to the power of the dimensions. */
const MAX_EM_DIMENSIONS = 3;

/**
 * Checks that EM can calibrate the number of dimensions.
 *
 * @throws `DataError` when it is above {@link MAX_EM_DIMENSIONS}.
 */
export function checkEmDimensions(dimensions: number): void {
  if (dimensions > MAX_EM_DIMENSIONS) {
    throw new DataError(`EM calibrates up to ${MAX_EM_DIMENSIONS} dimensions, found ${dimensions}.`);
  }
}

/** Runs EM cycles of the E-step and M-step until no parameter changes by more than `tolerance` or `maxIterations`. */
export function expectationMaximization<TParameters>(
  fit: Fit<TParameters>,
  tolerance: number,
  maxIterations: number,
): { readonly converged: boolean; readonly iterations: number } {
  const { items, latent, correlated, maximization, eStep } = fit;
  const current = new Float64Array(parameterCount(fit));
  const previous = new Float64Array(current.length);
  let iterations = 0;
  let converged = false;
  while (!converged && iterations < maxIterations) {
    iterations++;
    eStep.run();
    estimatesOf(fit, previous);
    for (const item of items) maximization.maximize(item);
    if (correlated) latent.maximize(eStep.nodeCounts);
    estimatesOf(fit, current);
    converged = largestChange(previous, current) <= tolerance;
  }
  return { converged, iterations };
}

/** Writes the estimated parameters, on their own scale, into `out`: the items' estimated entries, then the correlations. */
function estimatesOf<TParameters>({ items, latent, correlated }: Fit<TParameters>, out: Float64Array): void {
  let slot = 0;
  for (const item of items) for (const j of item.free) out[slot++] = item.vector[j] ?? 0;
  if (correlated) for (const [i, j] of latent.pairs) out[slot++] = latent.matrix[i * latent.dimensions + j] ?? 0;
}

function largestChange(previous: Float64Array, current: Float64Array): number {
  let largest = 0;
  for (let j = 0; j < current.length; j++)
    largest = Math.max(largest, Math.abs((current[j] ?? 0) - (previous[j] ?? 0)));
  return largest;
}

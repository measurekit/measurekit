// How well calibrateMultidimensional recovers simulated five-dimensional items by MH-RM, which takes minutes: MH-RM on five
// dimensions takes seconds even for a few hundred persons, so the tests check it against EM and against mirt. 30 M2PL items, 6
// on each of 5 dimensions correlated 0.4 to the power of their distance, with slopes from 0.8 to 1.3, are answered by 2,000
// simulated persons; each seed is calibrated by MH-RM with the same seed and Hessian standard errors. The thresholds, set
// before the first run of the test this study replaces: every slope, intercept and correlation within four standard errors
// of the true value, and a root mean square error of the slopes below 0.2.
//
// It writes results/multidimensional-mhrm-recovery.json beside it as each fit finishes: the seeds and sizes, the
// thresholds, each fit's largest errors in standard errors, its root mean square error of the slopes and its seconds, and
// the worst of them over the seeds. It takes about 2 minutes, and exits with 1 when a fit misses a threshold. Run after
// `pnpm build`: node validation/psychometrics/multidimensional-mhrm-recovery.ts

import { cholesky, mean } from "@measurekit/core";
import { createRandom } from "@measurekit/psychometrics";
import {
  calibrateMultidimensional,
  type MultidimensionalItem,
  type MultidimensionalLogisticParameters,
  multidimensionalTwoParameterLogistic,
} from "@measurekit/psychometrics/multidimensional";
import { simulateMultidimensionalResponses } from "@measurekit/psychometrics/sim";
import { timed, writeResults } from "../study.ts";

const SEEDS = [20261029, 20261030, 20261031];
const DIMENSIONS = 5;
const PERSONS = 2000;
const STANDARD_ERRORS = 4;
const RMSE_LIMIT = 0.2;

const results = {
  source:
    "calibrateMultidimensional by MH-RM, createRandom(seed) and the same seed for MH-RM: 30 M2PL items, 6 on each of 5 dimensions correlated 0.4 to the power of their distance, 2,000 simulated persons, Hessian standard errors. The errors are in standard errors of the estimate.",
  seeds: SEEDS,
  sizes: { persons: PERSONS, dimensions: DIMENSIONS, items: 30 },
  thresholds: { standardErrors: STANDARD_ERRORS, slopeRmse: RMSE_LIMIT },
  worst: { standardErrors: 0, slopeRmse: 0 },
  fits: [] as object[],
};
let missed = false;

for (const seed of SEEDS) {
  const random = createRandom(seed);
  const correlations = Array.from({ length: DIMENSIONS }, (_, i) =>
    Array.from({ length: DIMENSIONS }, (__, j) => 0.4 ** Math.abs(i - j)),
  );
  const factor = cholesky(Float64Array.from(correlations.flat()), DIMENSIONS);
  if (factor === null) throw new Error("The correlations are not positive definite.");
  const items: MultidimensionalItem<MultidimensionalLogisticParameters>[] = Array.from({ length: 30 }, (_, i) => ({
    model: multidimensionalTwoParameterLogistic,
    parameters: {
      a: Array.from({ length: DIMENSIONS }, (__, d) =>
        d === i % DIMENSIONS ? 0.8 + 0.1 * Math.floor(i / DIMENSIONS) : 0,
      ),
      intercept: ((i % 7) - 3) * 0.4,
      c: 0,
      d: 1,
    },
  }));
  const thetas = Array.from({ length: PERSONS }, () => {
    const z = Array.from({ length: DIMENSIONS }, () => random.normal());
    return Array.from({ length: DIMENSIONS }, (_, d) =>
      z.reduce((sum, value, k) => sum + (factor[d * DIMENSIONS + k] ?? 0) * value, 0),
    );
  });
  const matrix = simulateMultidimensionalResponses(items, thetas, random);
  const { value: fit, seconds } = timed(() =>
    calibrateMultidimensional(matrix, multidimensionalTwoParameterLogistic, {
      loadings: Object.fromEntries(items.map((_, i) => [`item${i + 1}`, [`D${(i % DIMENSIONS) + 1}`]])),
      method: "mhrm",
      seed,
      standardErrors: "hessian",
    }),
  );
  const largest = { slope: 0, intercept: 0, correlation: 0 };
  const slopeErrors: number[] = [];
  fit.items.forEach((estimate, j) => {
    const truth = items[j]?.parameters;
    const errors = estimate.standardErrors;
    if (truth === undefined || errors === null) throw new Error(`Item ${j + 1} has no standard errors.`);
    const d = j % DIMENSIONS;
    const slopeError = (estimate.parameters.a[d] ?? Number.NaN) - (truth.a[d] ?? Number.NaN);
    slopeErrors.push(slopeError);
    largest.slope = Math.max(largest.slope, Math.abs(slopeError) / (errors[`a${d + 1}`] ?? Number.NaN));
    largest.intercept = Math.max(
      largest.intercept,
      Math.abs(estimate.parameters.intercept - truth.intercept) / (errors.intercept ?? Number.NaN),
    );
  });
  for (let i = 1; i < DIMENSIONS; i++) {
    for (let j = 0; j < i; j++) {
      const error = (fit.correlations[i]?.[j] ?? Number.NaN) - (correlations[i]?.[j] ?? Number.NaN);
      const standardError = fit.correlationStandardErrors?.[i]?.[j] ?? Number.NaN;
      largest.correlation = Math.max(largest.correlation, Math.abs(error) / standardError);
    }
  }
  const slopeRmse = Math.sqrt(mean(slopeErrors.map((error) => error * error)));
  const within =
    fit.converged && Object.values(largest).every((value) => value < STANDARD_ERRORS) && slopeRmse < RMSE_LIMIT;
  missed ||= !within;
  results.fits.push({
    seed,
    ...largest,
    slopeRmse,
    within,
    iterations: fit.iterations,
    converged: fit.converged,
    seconds,
  });
  results.worst = {
    standardErrors: Math.max(results.worst.standardErrors, ...Object.values(largest)),
    slopeRmse: Math.max(results.worst.slopeRmse, slopeRmse),
  };
  writeResults(import.meta.url, results);
}

if (missed) {
  console.error("A fit missed a threshold: see results/multidimensional-mhrm-recovery.json.");
  process.exitCode = 1;
}

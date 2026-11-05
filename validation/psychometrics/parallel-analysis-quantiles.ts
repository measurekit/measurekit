// The random eigenvalues of parallelAnalysis against those of base R's permutations of the same data, on psych::bfi, at the
// replications of the reference (2000 permuted data sets of base R) and 1000 of this library, on the seeds 1 to 20. The
// error of a mean or a quantile is its distance from base R's in Monte Carlo standard errors of both sets of replications,
// as the test measures it: the means of the components and the factors, and their 0.95 quantiles. The test holds the means
// and the quantiles of the components at 300 replications; the quantiles of the factors have a heavy upper tail, from the
// permuted data sets whose first loading is at its bound, which 300 replications do not hold within 6 errors on every seed.
//
// It writes results/parallel-analysis-quantiles.json beside it, takes about 2 minutes, and exits with 1 when an error
// passes its bound. Run after `pnpm build`: node validation/psychometrics/parallel-analysis-quantiles.ts

import { readFileSync } from "node:fs";
import { createRandom } from "@measurekit/psychometrics";
import { buildResponseMatrix } from "@measurekit/psychometrics/data";
import { parallelAnalysis } from "@measurekit/psychometrics/dimensionality";
import { writeResults } from "../study.ts";

const REPLICATIONS = 1000;
const SEEDS = 20;
/** The bounds, in Monte Carlo standard errors: the means, and the quantiles of the components and of the factors. */
const BOUNDS = { mean: 4, componentQuantile: 6, factorQuantile: 6 } as const;

interface Reference {
  readonly iterations: number;
  readonly components: { readonly mean: number[]; readonly quantile: number[]; readonly sd: number[] };
  readonly factors: { readonly mean: number[]; readonly quantile: number[]; readonly sd: number[] };
}

const fixture = JSON.parse(
  readFileSync(
    new URL("../../packages/psychometrics/test/fixtures/dimensionality-parallel.json", import.meta.url),
    "utf8",
  ),
) as { cases: { responses: number[][]; itemIds: string[]; permutation: Reference }[] };
const bfi = fixture.cases[0];
if (bfi === undefined) throw new Error("The fixture has no case.");
const matrix = buildResponseMatrix(bfi.responses, { itemIds: bfi.itemIds });
const reference = bfi.permutation;

/** The standard error of the 0.95 quantile of a normal variable over n draws, in its standard deviations. */
const quantileError = (n: number): number => Math.sqrt((0.95 * 0.05) / n) / 0.10314;

/** The largest distance of the values from the reference's, in Monte Carlo standard errors. */
function worst(
  actual: readonly number[],
  expected: readonly number[],
  sd: readonly number[],
  error: (n: number) => number,
): number {
  return Math.max(
    ...expected.map(
      (value, k) =>
        Math.abs((actual[k] ?? Number.NaN) - value) /
        ((sd[k] ?? Number.NaN) * Math.hypot(error(reference.iterations), error(REPLICATIONS))),
    ),
  );
}
const meanError = (n: number): number => Math.sqrt(1 / n);

const runs = Array.from({ length: SEEDS }, (_, k) => {
  const { components, factors } = parallelAnalysis(matrix, createRandom(k + 1), {
    correlation: "pearson",
    replications: REPLICATIONS,
  });
  return {
    seed: k + 1,
    componentMean: worst(components.randomMean, reference.components.mean, reference.components.sd, meanError),
    factorMean: worst(factors.randomMean, reference.factors.mean, reference.factors.sd, meanError),
    componentQuantile: worst(
      components.randomQuantile,
      reference.components.quantile,
      reference.components.sd,
      quantileError,
    ),
    factorQuantile: worst(factors.randomQuantile, reference.factors.quantile, reference.factors.sd, quantileError),
  };
});
const largest = {
  mean: Math.max(...runs.flatMap((run) => [run.componentMean, run.factorMean])),
  componentQuantile: Math.max(...runs.map((run) => run.componentQuantile)),
  factorQuantile: Math.max(...runs.map((run) => run.factorQuantile)),
};
const within = (["mean", "componentQuantile", "factorQuantile"] as const).every((k) => largest[k] <= BOUNDS[k]);

writeResults(import.meta.url, {
  source:
    "parallelAnalysis on psych::bfi, Pearson correlations, 1,000 replications on the seeds 1 to 20, against base R's 2,000 permuted data sets: the distance of the random eigenvalues' means and 0.95 quantiles from base R's, in Monte Carlo standard errors",
  replications: REPLICATIONS,
  referenceReplications: reference.iterations,
  bounds: BOUNDS,
  largest,
  within,
  runs,
});
process.exitCode = within ? 0 : 1;

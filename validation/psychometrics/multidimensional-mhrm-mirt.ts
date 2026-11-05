// How closely calibrateMultidimensional's MH-RM agrees with mirt's on five dimensions, which takes minutes: the tests hold
// MH-RM against EM on two dimensions instead. mirt's MH-RM ran with seeds 1 to 4 on the first 800 complete cases of the 25
// items of psych::bfi under the graded model (multidimensional-mhrm.json, by multidimensional.R); its estimates differ from
// their mean over the seeds by up to 0.076, and its standard errors by 20% from seed to seed. Each seed here is compared
// with mirt's mean over its seeds, within the tolerances that the test of seed 1 held: every slope and intercept within 0.3,
// every correlation within 0.08, and of more than 100 standard errors, the median ratio to mirt's mean within 0.1 of 1 and
// every ratio from 0.6 to 1.6.
//
// It writes results/multidimensional-mhrm-mirt.json beside it as each fit finishes: the seeds and sizes, the tolerances,
// each fit's largest differences, its ratios of standard errors and its seconds, and the worst over the seeds. It takes
// about 2 minutes, and exits with 1 when a fit misses a tolerance. Run after `pnpm build`:
// node validation/psychometrics/multidimensional-mhrm-mirt.ts

import { readFileSync } from "node:fs";
import { buildResponseMatrix } from "@measurekit/psychometrics/data";
import {
  calibrateMultidimensional,
  type MultidimensionalGradedParameters,
  multidimensionalGradedResponse,
} from "@measurekit/psychometrics/multidimensional";
import { timed, writeResults } from "../study.ts";

interface MirtRun {
  readonly items: readonly {
    readonly parameters: { readonly a: readonly number[]; readonly intercepts: readonly number[] };
    readonly standardErrors: Readonly<Record<string, number | null>>;
  }[];
  readonly correlations: readonly (readonly number[])[];
}
interface MirtMhrm {
  readonly itemIds: readonly string[];
  readonly responses: readonly (readonly number[])[];
  readonly loadings: Readonly<Record<string, readonly string[]>>;
  readonly runs: readonly MirtRun[];
}
const mirt = JSON.parse(
  readFileSync(
    new URL("../../packages/psychometrics/test/fixtures/multidimensional-mhrm.json", import.meta.url),
    "utf8",
  ),
) as MirtMhrm;

const SEEDS = [1, 2, 3, 4];
const TOLERANCES = { parameters: 0.3, correlations: 0.08, medianRatio: 0.1, smallestRatio: 0.6, largestRatio: 1.6 };

const average = (values: readonly number[]): number => values.reduce((sum, value) => sum + value, 0) / values.length;
const slopes = (a: readonly number[]): number[] => a.filter((value) => value !== 0);
/** mirt's mean over its seeds of each item's slopes and intercepts, in that order. */
const meanParameters = mirt.itemIds.map((_, i) => {
  const vectors = mirt.runs.map(({ items }) => {
    const parameters = items[i]?.parameters;
    if (parameters === undefined) throw new Error(`mirt has no item ${i + 1}.`);
    return [...slopes(parameters.a), ...parameters.intercepts];
  });
  return (vectors[0] ?? []).map((_, k) => average(vectors.map((vector) => vector[k] ?? Number.NaN)));
});
/** mirt's mean over its seeds of each standard error that every seed of mirt gives, by item and name. */
const meanErrors = mirt.itemIds.map((_, i) => {
  const names = Object.keys(mirt.runs[0]?.items[i]?.standardErrors ?? {});
  return Object.fromEntries(
    names.flatMap((name) => {
      const values = mirt.runs.map(({ items }) => items[i]?.standardErrors[name]);
      return values.every((value) => typeof value === "number") ? [[name, average(values as number[])]] : [];
    }),
  );
});

const results = {
  source:
    "calibrateMultidimensional by MH-RM with Hessian standard errors, on the 25 graded items of psych::bfi, the first 800 complete cases, 5 correlated dimensions, against the mean of mirt's MH-RM over seeds 1 to 4.",
  seeds: SEEDS,
  sizes: { persons: mirt.responses.length, items: mirt.itemIds.length, dimensions: 5 },
  tolerances: TOLERANCES,
  worst: { parameters: 0, correlations: 0, medianRatio: 0, smallestRatio: Number.POSITIVE_INFINITY, largestRatio: 0 },
  fits: [] as object[],
};
const data = buildResponseMatrix(mirt.responses, { itemIds: [...mirt.itemIds] });
let missed = false;

for (const seed of SEEDS) {
  const { value: fit, seconds } = timed(() =>
    calibrateMultidimensional(data, multidimensionalGradedResponse, {
      loadings: mirt.loadings,
      method: "mhrm",
      seed,
      standardErrors: "hessian",
    }),
  );
  let parameters = 0;
  fit.items.forEach(({ parameters: found }, i) => {
    const { a, intercepts } = found as MultidimensionalGradedParameters;
    [...slopes(a), ...intercepts].forEach((value, k) => {
      parameters = Math.max(parameters, Math.abs(value - (meanParameters[i]?.[k] ?? Number.NaN)));
    });
  });
  let correlations = 0;
  fit.correlations.forEach((row, i) => {
    row.forEach((value, j) => {
      const reference = average(mirt.runs.map((run) => run.correlations[i]?.[j] ?? Number.NaN));
      correlations = Math.max(correlations, Math.abs(value - reference));
    });
  });
  const ratios = fit.items
    .flatMap(({ standardErrors }, i) =>
      Object.entries(meanErrors[i] ?? {}).map(([name, mean]) => (standardErrors?.[name] ?? Number.NaN) / mean),
    )
    .sort((x, y) => x - y);
  const ratio = {
    count: ratios.length,
    median: ratios[Math.floor(ratios.length / 2)] ?? Number.NaN,
    smallest: ratios[0] ?? Number.NaN,
    largest: ratios.at(-1) ?? Number.NaN,
  };
  const within =
    fit.converged &&
    parameters <= TOLERANCES.parameters &&
    correlations <= TOLERANCES.correlations &&
    ratio.count > 100 &&
    Math.abs(ratio.median - 1) <= TOLERANCES.medianRatio &&
    ratio.smallest > TOLERANCES.smallestRatio &&
    ratio.largest < TOLERANCES.largestRatio;
  missed ||= !within;
  results.worst = {
    parameters: Math.max(results.worst.parameters, parameters),
    correlations: Math.max(results.worst.correlations, correlations),
    medianRatio: Math.max(results.worst.medianRatio, Math.abs(ratio.median - 1)),
    smallestRatio: Math.min(results.worst.smallestRatio, ratio.smallest),
    largestRatio: Math.max(results.worst.largestRatio, ratio.largest),
  };
  results.fits.push({ seed, parameters, correlations, ratio, within, iterations: fit.iterations, seconds });
  writeResults(import.meta.url, results);
}

if (missed) {
  console.error("A fit missed a tolerance: see results/multidimensional-mhrm-mirt.json.");
  process.exitCode = 1;
}

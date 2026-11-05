// Which standard errors of five-dimensional MH-RM are right, this library's or mirt's, which differ by a median of 12%
// (multidimensional-mhrm-mirt.ts). The mean of mirt's estimates over its seeds 1 to 10 on psych::bfi's 25 graded items
// (multidimensional-mhrm.json) is taken as the truth; 50 data sets of 800 persons are simulated from it, each is
// calibrated by MH-RM with Hessian standard errors, and the standard deviation of each estimate over the data sets, less
// the variance of MH-RM over ten seeds on one data set, is the standard error that both should give. Each parameter's
// standard error, as this library and mirt average it over the ten seeds on bfi and as this library gives it on the
// simulated data, is divided by that deviation. This library's are held to it, the median within 0.1 of 1 and every ratio
// from 0.6 to 1.6, the bounds that mirt's held them to before; mirt's ratios are recorded, as they run small.
//
// It reads results/multidimensional-mhrm-mirt.json, which that study writes, and writes
// results/multidimensional-mhrm-errors.json beside it as each fit finishes: the sizes and seeds, each fit's estimates,
// standard errors, iterations and seconds, and at the end the ratios by parameter with their medians, quartiles and
// ranges. It takes about 25 minutes, and exits with 1 when this library's ratios miss a bound. Run after `pnpm build` and
// multidimensional-mhrm-mirt.ts: node validation/psychometrics/multidimensional-mhrm-errors.ts

import { readFileSync } from "node:fs";
import { cholesky } from "@measurekit/core";
import { createRandom } from "@measurekit/psychometrics";
import { buildResponseMatrix } from "@measurekit/psychometrics/data";
import {
  calibrateMultidimensional,
  type MultidimensionalGradedParameters,
  multidimensionalGradedResponse,
} from "@measurekit/psychometrics/multidimensional";
import { simulateMultidimensionalResponses } from "@measurekit/psychometrics/sim";
import { timed, writeResults } from "../study.ts";

const DATA_SETS = 50;
const BOUNDS = { median: 0.1, smallest: 0.6, largest: 1.6 };
const MONTE_CARLO_SEEDS = 10;
const PERSONS = 800;
const DIMENSIONS = 5;

interface MirtRun {
  readonly items: readonly {
    readonly parameters: { readonly a: readonly number[]; readonly intercepts: readonly number[] };
  }[];
  readonly correlations: readonly (readonly number[])[];
}
const mirt = JSON.parse(
  readFileSync(
    new URL("../../packages/psychometrics/test/fixtures/multidimensional-mhrm.json", import.meta.url),
    "utf8",
  ),
) as {
  readonly itemIds: readonly string[];
  readonly loadings: Readonly<Record<string, readonly string[]>>;
  readonly runs: readonly MirtRun[];
};
const averagedErrors = (
  JSON.parse(readFileSync(new URL("results/multidimensional-mhrm-mirt.json", import.meta.url), "utf8")) as {
    readonly averagedErrors: {
      readonly errors: readonly {
        readonly item: string;
        readonly name: string;
        readonly here: number;
        readonly mirt: number;
      }[];
    };
  }
).averagedErrors.errors;

const average = (values: readonly number[]): number => values.reduce((sum, value) => sum + value, 0) / values.length;
const variance = (values: readonly number[]): number => {
  const center = average(values);
  return values.reduce((sum, value) => sum + (value - center) ** 2, 0) / (values.length - 1);
};
const runs = mirt.runs;
/** The truth: the mean of mirt's estimates over its seeds. */
const truth = mirt.itemIds.map((_, i) => {
  const mean = (pick: (run: MirtRun) => readonly number[]): number[] =>
    (pick(runs[0] as MirtRun) ?? []).map((_, k) => average(runs.map((run) => pick(run)[k] ?? Number.NaN)));
  return {
    a: mean((run) => run.items[i]?.parameters.a ?? []),
    intercepts: mean((run) => run.items[i]?.parameters.intercepts ?? []),
  };
});
const correlations = Array.from({ length: DIMENSIONS }, (_, i) =>
  Array.from({ length: DIMENSIONS }, (__, j) => average(runs.map((run) => run.correlations[i]?.[j] ?? Number.NaN))),
);
const factor = cholesky(Float64Array.from(correlations.flat()), DIMENSIONS);
if (factor === null) throw new Error("mirt's mean correlations are not positive definite.");
const items = truth.map((parameters) => ({ model: multidimensionalGradedResponse, parameters }));

/** The names of the estimated entries, as the standard errors name them, with their values in a calibration. */
const entries = truth.flatMap(({ a, intercepts }, i) => [
  ...a.flatMap((slope, d) =>
    slope === 0
      ? []
      : [{ item: i, name: `a${d + 1}`, value: (p: MultidimensionalGradedParameters) => p.a[d] ?? Number.NaN }],
  ),
  ...intercepts.map((_, k) => ({
    item: i,
    name: `intercept${k + 1}`,
    value: (p: MultidimensionalGradedParameters) => p.intercepts[k] ?? Number.NaN,
  })),
]);

/** Simulates a data set of 800 persons from the truth. */
function simulate(seed: number) {
  const random = createRandom(seed);
  const thetas = Array.from({ length: PERSONS }, () => {
    const z = Array.from({ length: DIMENSIONS }, () => random.normal());
    return Array.from({ length: DIMENSIONS }, (_, d) =>
      z.reduce((sum, value, k) => sum + (factor?.[d * DIMENSIONS + k] ?? 0) * value, 0),
    );
  });
  const matrix = simulateMultidimensionalResponses(items, thetas, random);
  return buildResponseMatrix(matrix.responses, { itemIds: [...mirt.itemIds] });
}

/** Calibrates a data set by MH-RM and returns its estimates and standard errors in the order of `entries`. */
function calibrate(data: ReturnType<typeof simulate>, seed: number) {
  const { value: fit, seconds } = timed(() =>
    calibrateMultidimensional(data, multidimensionalGradedResponse, {
      loadings: mirt.loadings,
      method: "mhrm",
      seed,
      standardErrors: "hessian",
    }),
  );
  const estimates = entries.map(({ item, value }) =>
    value(fit.items[item]?.parameters as MultidimensionalGradedParameters),
  );
  const errors = entries.map(({ item, name }) => fit.items[item]?.standardErrors?.[name] ?? Number.NaN);
  // A dimension turned against the truth would make its slopes' spread meaningless.
  const reflected = fit.dimensions.filter((_, d) =>
    entries.some(
      ({ item, name }, k) => name === `a${d + 1}` && Math.sign(estimates[k] ?? 0) !== Math.sign(truth[item]?.a[d] ?? 0),
    ),
  );
  return { estimates, errors, reflected, converged: fit.converged, iterations: fit.iterations, seconds };
}

const results = {
  source:
    "calibrateMultidimensional by MH-RM with Hessian standard errors on data sets of 800 persons simulated with createRandom(seed) from the mean of mirt's MH-RM over seeds 1 to 10 on psych::bfi's 25 graded items and 5 correlated dimensions. The empirical standard error of an estimate is its standard deviation over the data sets, less the variance of MH-RM over seeds 1 to 10 on data set 1.",
  sizes: {
    persons: PERSONS,
    items: mirt.itemIds.length,
    dimensions: DIMENSIONS,
    dataSets: DATA_SETS,
    monteCarloSeeds: MONTE_CARLO_SEEDS,
  },
  entries: entries.map(({ item, name }) => `${mirt.itemIds[item]} ${name}`),
  monteCarlo: [] as object[],
  fits: [] as object[],
  comparison: {} as object,
};

const first = simulate(1);
const monteCarlo: ReturnType<typeof calibrate>[] = [];
for (let seed = 1; seed <= MONTE_CARLO_SEEDS; seed++) {
  const fit = calibrate(first, seed);
  monteCarlo.push(fit);
  results.monteCarlo.push({ seed, ...fit });
  writeResults(import.meta.url, results);
}
const fits: ReturnType<typeof calibrate>[] = [];
for (let set = 1; set <= DATA_SETS; set++) {
  try {
    // Data set 1 with seed 1 is the first fit of the Monte Carlo runs.
    const fit = set === 1 && monteCarlo[0] !== undefined ? monteCarlo[0] : calibrate(simulate(set), set);
    fits.push(fit);
    results.fits.push({ dataSet: set, seed: set, ...fit });
  } catch (error) {
    // A simulated data set can leave a category of an item without a response, which calibration refuses.
    results.fits.push({ dataSet: set, seed: set, error: error instanceof Error ? error.message : String(error) });
  }
  writeResults(import.meta.url, results);
}

/** The median, quartiles and range of ratios. */
function summary(ratios: readonly number[]) {
  const sorted = [...ratios].sort((x, y) => x - y);
  const at = (q: number): number => sorted[Math.floor(q * (sorted.length - 1))] ?? Number.NaN;
  return { count: sorted.length, median: at(0.5), quartiles: [at(0.25), at(0.75)], range: [sorted[0], sorted.at(-1)] };
}
const usable = fits.filter(
  ({ converged, reflected, estimates, errors }) =>
    converged && reflected.length === 0 && estimates.every(Number.isFinite) && errors.every(Number.isFinite),
);
const empirical = entries.map((_, k) => {
  const spread = variance(usable.map(({ estimates }) => estimates[k] ?? Number.NaN));
  const monteCarloVariance = variance(monteCarlo.map(({ estimates }) => estimates[k] ?? Number.NaN));
  return { raw: Math.sqrt(spread), corrected: Math.sqrt(Math.max(0, spread - monteCarloVariance)) };
});
const byParameter = averagedErrors.flatMap(({ item, name, here, mirt: theirs }) => {
  const k = entries.findIndex((entry) => mirt.itemIds[entry.item] === item && entry.name === name);
  const sd = empirical[k];
  if (k < 0 || sd === undefined) return [];
  const simulated = average(usable.map(({ errors }) => errors[k] ?? Number.NaN));
  return [{ item, name, empirical: sd.corrected, raw: sd.raw, here, mirt: theirs, simulated }];
});
const ratios = (pick: (row: (typeof byParameter)[number]) => number, sd: "empirical" | "raw") =>
  summary(byParameter.map((row) => pick(row) / row[sd]));
/** Whether ratios meet the bounds: the median within 0.1 of 1, every ratio from 0.6 to 1.6. */
const meets = ({ count, median, range }: ReturnType<typeof summary>): boolean =>
  count > 100 &&
  Math.abs(median - 1) <= BOUNDS.median &&
  (range[0] ?? 0) >= BOUNDS.smallest &&
  (range[1] ?? Number.POSITIVE_INFINITY) <= BOUNDS.largest;
const here = ratios(({ here: value }) => value, "empirical");
const hereOnSimulations = ratios(({ simulated }) => simulated, "empirical");
const within = usable.length >= 40 && meets(here) && meets(hereOnSimulations);
results.comparison = {
  bounds: BOUNDS,
  within,
  dataSetsUsed: usable.length,
  here,
  mirt: ratios(({ mirt: theirs }) => theirs, "empirical"),
  hereOnSimulations,
  againstTheRawDeviation: {
    here: ratios(({ here }) => here, "raw"),
    mirt: ratios(({ mirt: theirs }) => theirs, "raw"),
  },
  byParameter,
};
writeResults(import.meta.url, results);

if (!within) {
  console.error("This library's standard errors missed a bound: see results/multidimensional-mhrm-errors.json.");
  process.exitCode = 1;
}

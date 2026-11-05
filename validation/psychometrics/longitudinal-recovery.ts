// How well calibrateLongitudinal recovers the moments of the occasions at the sizes of the guide, which the tests hold at a
// size that runs in seconds. Persons answer the same 2PL items on 2 or 3 occasions, with means 0, 0.3 and 0.6, standard
// deviations 1, 1.15 and 1.3 and an autoregressive correlation of 0.85; the error of a fit is the largest distance of an
// estimate from the moments of the simulated abilities, as longitudinal-recovery.R measures mirt's. Three studies:
//
// - recovery: 5,000 persons, 10 and 40 items per occasion, 2 and 3 occasions, seeds 1 to 3, at the default nodes, held to
//   1.5 times the worst error of mirt on 10 items and the same design (the fixture of longitudinal-recovery.R), rounded up
//   to a multiple of 0.005;
// - nodes: 40 items on 3 occasions at 15 nodes, which misses those thresholds, why the default is 21;
// - grid: 2,000 persons, 40 items, 3 occasions, seeds 101 and 102, the largest distance of a moment between 15 and 31
//   nodes and between 21 and 31.
//
// It writes results/longitudinal-recovery.json beside it as each fit finishes: the seeds, sizes and nodes of each fit, its
// errors and seconds, and for each design the thresholds and the worst errors over its seeds. It takes about 5 minutes,
// and exits with 1 when a recovery misses its threshold. Run after `pnpm build`:
// node validation/psychometrics/longitudinal-recovery.ts

import { readFileSync } from "node:fs";
import { logistic, mean } from "@measurekit/core";
import { createRandom } from "@measurekit/psychometrics";
import { buildResponseMatrix, type ResponseMatrix } from "@measurekit/psychometrics/data";
import { calibrateLongitudinal, type LongitudinalCalibration } from "@measurekit/psychometrics/multidimensional";
import { timed, writeResults } from "../study.ts";

/** The default nodes per occasion of calibrateLongitudinal, by the number of occasions. */
const DEFAULT_POINTS: Readonly<Record<number, number>> = { 2: 31, 3: 21 };
const SEEDS = [1, 2, 3];
const MOMENTS = { means: [0, 0.3, 0.6], sds: [1, 1.15, 1.3], rho: 0.85 } as const;
/** The thresholds are rounded up to a multiple of this. */
const STEP = 0.005;

interface MirtRecovery {
  readonly recovery: readonly {
    readonly occasions: number;
    readonly worst: { readonly mean: number; readonly sd: number; readonly correlation: number };
  }[];
}
const mirt = JSON.parse(
  readFileSync(
    new URL("../../packages/psychometrics/test/fixtures/longitudinal-recovery.json", import.meta.url),
    "utf8",
  ),
) as MirtRecovery;

/** 1.5 times the worst error of mirt on the design of `occasions` occasions, rounded up to a multiple of 0.005. */
function thresholds(occasions: number) {
  const row = mirt.recovery.find((each) => each.occasions === occasions);
  if (row === undefined) throw new Error(`No recovery of mirt on ${occasions} occasions.`);
  const limit = (error: number): number => Math.ceil((1.5 * error) / STEP - 1e-9) * STEP;
  return { mean: limit(row.worst.mean), sd: limit(row.worst.sd), correlation: limit(row.worst.correlation) };
}

/**
 * Draws the abilities of persons on the first `occasions` occasions of {@link MOMENTS} and their responses to the same 2PL
 * items on each, with slopes from 0.8 to 1.8 and difficulties of standard deviation 0.8: the simulation of the tests.
 */
function simulate(seed: number, persons: number, itemCount: number, occasions: number) {
  const random = createRandom(seed);
  const slopes = Array.from({ length: itemCount }, () => 0.8 + random.uniform());
  const difficulties = Array.from({ length: itemCount }, () => 0.8 * random.normal());
  const ids = Array.from({ length: itemCount }, (_, i) => `q${i + 1}`);
  const means = MOMENTS.means.slice(0, occasions);
  const thetas = Array.from({ length: persons }, () => {
    let state = random.normal();
    return means.map((value, t) => {
      if (t > 0) state = MOMENTS.rho * state + Math.sqrt(1 - MOMENTS.rho ** 2) * random.normal();
      return value + (MOMENTS.sds[t] ?? 1) * state;
    });
  });
  const matrices: ResponseMatrix[] = means.map((_, t) =>
    buildResponseMatrix(
      thetas.map((theta) =>
        ids.map((__, i) =>
          random.uniform() < logistic((slopes[i] ?? 0) * ((theta[t] ?? 0) - (difficulties[i] ?? 0))) ? 1 : 0,
        ),
      ),
      { itemIds: ids },
    ),
  );
  return { matrices, thetas };
}

const column = (thetas: readonly (readonly number[])[], t: number): number[] => thetas.map((theta) => theta[t] ?? 0);
const spread = (values: readonly number[]): number => {
  const center = mean(values);
  return Math.sqrt(values.reduce((sum, value) => sum + (value - center) ** 2, 0) / (values.length - 1));
};
const correlate = (x: readonly number[], y: readonly number[]): number => {
  const [cx, cy] = [mean(x), mean(y)];
  const cross = x.reduce((sum, value, i) => sum + (value - cx) * ((y[i] ?? 0) - cy), 0);
  return cross / Math.sqrt(x.reduce((s, v) => s + (v - cx) ** 2, 0) * y.reduce((s, v) => s + (v - cy) ** 2, 0));
};

/**
 * The largest distance of the estimates from the moments of the abilities that the persons were given: the means and
 * standard deviations of the occasions after the first, in units of the first occasion's sample standard deviation and
 * from its sample mean, and the correlations.
 */
function errorsOf(fit: LongitudinalCalibration, thetas: readonly (readonly number[])[]) {
  const first = column(thetas, 0);
  const errors = { mean: 0, sd: 0, correlation: 0 };
  fit.occasions.forEach(({ mean: estimated, sd }, t) => {
    const values = column(thetas, t);
    if (t > 0) {
      errors.mean = Math.max(errors.mean, Math.abs(estimated - (mean(values) - mean(first)) / spread(first)));
      errors.sd = Math.max(errors.sd, Math.abs(sd - spread(values) / spread(first)));
    }
    for (let u = 0; u < t; u++) {
      const estimate = fit.correlations[t]?.[u] ?? Number.NaN;
      errors.correlation = Math.max(errors.correlation, Math.abs(estimate - correlate(values, column(thetas, u))));
    }
  });
  return errors;
}

/** The largest difference between the means, standard deviations and correlations of two calibrations. */
function momentDistance(a: LongitudinalCalibration, b: LongitudinalCalibration): number {
  const differences = a.occasions.flatMap(({ mean: m, sd }, t) => [
    Math.abs(m - (b.occasions[t]?.mean ?? Number.NaN)),
    Math.abs(sd - (b.occasions[t]?.sd ?? Number.NaN)),
  ]);
  for (const [i, row] of a.correlations.entries()) {
    for (const [j, value] of row.entries()) differences.push(Math.abs(value - (b.correlations[i]?.[j] ?? Number.NaN)));
  }
  return Math.max(...differences);
}

/** Calibrates and times one fit, at the default nodes or at `points`. */
function fitTimed(matrices: readonly ResponseMatrix[], points?: number) {
  const { value: fit, seconds } = timed(() =>
    calibrateLongitudinal(matrices, { ...(points === undefined ? {} : { points }), maxIterations: 1000 }),
  );
  return { fit, seconds };
}

const results = {
  source:
    "calibrateLongitudinal, createRandom(seed): persons answer the same 2PL items on 2 or 3 occasions, abilities with means 0, 0.3 and 0.6, standard deviations 1, 1.15 and 1.3 and an autoregressive correlation of 0.85. The error is the largest distance of an estimate from the mean, standard deviation or correlation of the simulated abilities; a threshold is 1.5 times the worst error of mirt on 10 items and the same design, rounded up to a multiple of 0.005.",
  recovery: [] as object[],
  nodes: [] as object[],
  grid: [] as object[],
};
const save = (): void => writeResults(import.meta.url, results);
let missed = false;

for (const [occasions, items] of [
  [2, 10],
  [2, 40],
  [3, 10],
  [3, 40],
] as const) {
  const limits = thresholds(occasions);
  const fits = SEEDS.map((seed) => {
    const { matrices, thetas } = simulate(seed, 5000, items, occasions);
    const { fit, seconds } = fitTimed(matrices);
    const errors = errorsOf(fit, thetas);
    const within = fit.converged && (["mean", "sd", "correlation"] as const).every((k) => errors[k] <= limits[k]);
    missed ||= !within;
    return { seed, ...errors, iterations: fit.iterations, converged: fit.converged, within, seconds };
  });
  const worst = {
    mean: Math.max(...fits.map((each) => each.mean)),
    sd: Math.max(...fits.map((each) => each.sd)),
    correlation: Math.max(...fits.map((each) => each.correlation)),
  };
  results.recovery.push({
    persons: 5000,
    itemsPerOccasion: items,
    occasions,
    points: DEFAULT_POINTS[occasions],
    seeds: SEEDS,
    thresholds: limits,
    worst,
    fits,
  });
  save();
}

{
  const { matrices, thetas } = simulate(1, 5000, 40, 3);
  const { fit, seconds } = fitTimed(matrices, 15);
  results.nodes.push({
    persons: 5000,
    itemsPerOccasion: 40,
    occasions: 3,
    points: 15,
    seed: 1,
    ...errorsOf(fit, thetas),
    thresholds: thresholds(3),
    seconds,
  });
  save();
}

for (const seed of [101, 102]) {
  const { matrices } = simulate(seed, 2000, 40, 3);
  const [coarse, usual, fine] = [15, 21, 31].map((points) => fitTimed(matrices, points));
  if (coarse === undefined || usual === undefined || fine === undefined) throw new Error("Three grids.");
  results.grid.push({
    persons: 2000,
    itemsPerOccasion: 40,
    occasions: 3,
    points: [15, 21, 31],
    seed,
    fifteenAgainstThirtyOne: momentDistance(coarse.fit, fine.fit),
    twentyOneAgainstThirtyOne: momentDistance(usual.fit, fine.fit),
    converged: [coarse.fit.converged, usual.fit.converged, fine.fit.converged],
    seconds: [coarse.seconds, usual.seconds, fine.seconds],
  });
  save();
}

if (missed) {
  console.error("A recovery missed its threshold: see results/longitudinal-recovery.json.");
  process.exitCode = 1;
}

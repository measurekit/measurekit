// How well calibrateMultidimensional recovers the means and covariances of several groups at the size of the guide's
// advice, which the tests hold at a size that runs in seconds. Three groups of 700, 600 and 600 persons with their own
// means and covariances answer 40 M2PL items on each of 3 dimensions; each seed is calibrated at 21 nodes per dimension and
// at the default 15, and the error of a fit is the largest distance of a group's mean or covariance from the values that
// generated it. The guide advises 21 nodes above 20 items per dimension from these errors.
//
// It writes results/multidimensional-groups-recovery.json beside it as each fit finishes: the seeds and sizes, the limits,
// each fit's errors and seconds, and the worst errors over the seeds at each number of nodes. It takes about 10 minutes, and
// exits with 1 when a fit at 21 nodes misses MEAN_LIMIT or COVARIANCE_LIMIT. Run after `pnpm build`:
// node validation/psychometrics/multidimensional-groups-recovery.ts

import { cholesky } from "@measurekit/core";
import { createRandom } from "@measurekit/psychometrics";
import {
  calibrateMultidimensional,
  type MultidimensionalLogisticParameters,
  multidimensionalTwoParameterLogistic,
} from "@measurekit/psychometrics/multidimensional";
import { simulateMultidimensionalResponses } from "@measurekit/psychometrics/sim";
import { timed, writeResults } from "../study.ts";

const SEEDS = [20261031, 20261032, 20261033];
const DIMENSIONS = 3;
const PER_DIMENSION = 40;
/** The largest errors of a mean and of a covariance at 21 nodes: the tolerances of the test of seed 20261031 it replaces. */
const MEAN_LIMIT = 0.18;
const COVARIANCE_LIMIT = 0.25;
const TRUTH = [
  {
    group: "A",
    means: [0, 0, 0],
    persons: 700,
    covariance: [
      [1, 0.3, 0.2],
      [0.3, 1, 0.4],
      [0.2, 0.4, 1],
    ],
  },
  {
    group: "B",
    means: [0.4, -0.3, 0.5],
    persons: 600,
    covariance: [
      [1.4, 0.4, 0.3],
      [0.4, 0.7, 0.2],
      [0.3, 0.2, 1.2],
    ],
  },
  {
    group: "C",
    means: [-0.5, 0.3, 0.2],
    persons: 600,
    covariance: [
      [0.8, 0.2, 0.1],
      [0.2, 1.3, 0.5],
      [0.1, 0.5, 0.9],
    ],
  },
];

/** Simulates the responses of the three groups to 40 items on each of 3 dimensions, with slopes from 0.8 to 1.8. */
function simulate(seed: number) {
  const random = createRandom(seed);
  const items = Array.from({ length: DIMENSIONS * PER_DIMENSION }, (_, i) => ({
    model: multidimensionalTwoParameterLogistic,
    parameters: {
      a: Array.from({ length: DIMENSIONS }, (__, d) =>
        d === Math.floor(i / PER_DIMENSION) ? 0.8 + random.uniform() : 0,
      ),
      intercept: random.normal() * 0.7,
      c: 0,
      d: 1,
    } satisfies MultidimensionalLogisticParameters,
  }));
  const thetas: number[][] = [];
  const labels: string[] = [];
  for (const { group, means, persons, covariance } of TRUTH) {
    const factor = cholesky(Float64Array.from(covariance.flat()), DIMENSIONS);
    if (factor === null) throw new Error(`The covariance of group ${group} is not positive definite.`);
    for (let p = 0; p < persons; p++) {
      const z = Array.from({ length: DIMENSIONS }, () => random.normal());
      thetas.push(
        means.map(
          (mean, i) =>
            mean + z.reduce((sum, value, j) => sum + (j <= i ? (factor[i * DIMENSIONS + j] ?? 0) * value : 0), 0),
        ),
      );
      labels.push(group);
    }
  }
  return { matrix: simulateMultidimensionalResponses(items, thetas, random), labels };
}

const results = {
  source:
    "calibrateMultidimensional, createRandom(seed): three groups of 700, 600 and 600 persons, 40 M2PL items on each of 3 dimensions with slopes from 0.8 to 1.8, at 21 and 15 nodes per dimension. The error is the largest distance of a group's mean or covariance from the value that generated it.",
  seeds: SEEDS,
  sizes: { persons: TRUTH.map(({ persons }) => persons), dimensions: DIMENSIONS, itemsPerDimension: PER_DIMENSION },
  limits: { mean: MEAN_LIMIT, covariance: COVARIANCE_LIMIT },
  worst: {} as Record<string, { mean: number; covariance: number }>,
  fits: [] as {
    readonly seed: number;
    readonly points: number;
    readonly mean: number;
    readonly covariance: number;
    readonly within: boolean;
    readonly iterations: number;
    readonly converged: boolean;
    readonly seconds: number;
  }[],
};
let missed = false;

for (const seed of SEEDS) {
  const { matrix, labels } = simulate(seed);
  for (const points of [21, 15]) {
    const { value: fit, seconds } = timed(() =>
      calibrateMultidimensional(matrix, multidimensionalTwoParameterLogistic, {
        loadings: Object.fromEntries(matrix.itemIds.map((id, i) => [id, [`D${Math.floor(i / PER_DIMENSION) + 1}`]])),
        groups: labels,
        points,
        maxIterations: 1000,
      }),
    );
    const errors = { mean: 0, covariance: 0 };
    TRUTH.forEach(({ group, means, covariance }, g) => {
      const found = fit.groups?.[g];
      if (found?.group !== group) throw new Error(`Group ${g + 1} is not ${group}.`);
      means.forEach((value, d) => {
        errors.mean = Math.max(errors.mean, Math.abs((found.means[d] ?? Number.NaN) - value));
      });
      covariance.forEach((row, i) => {
        row.forEach((value, j) => {
          errors.covariance = Math.max(errors.covariance, Math.abs((found.covariance[i]?.[j] ?? Number.NaN) - value));
        });
      });
    });
    const within = fit.converged && errors.mean <= MEAN_LIMIT && errors.covariance <= COVARIANCE_LIMIT;
    if (points === 21) missed ||= !within;
    results.fits.push({
      seed,
      points,
      ...errors,
      within,
      iterations: fit.iterations,
      converged: fit.converged,
      seconds,
    });
    const atPoints = results.fits.filter((each) => each.points === points);
    results.worst[`${points} nodes`] = {
      mean: Math.max(...atPoints.map((each) => each.mean)),
      covariance: Math.max(...atPoints.map((each) => each.covariance)),
    };
    writeResults(import.meta.url, results);
  }
}

if (missed) {
  console.error("A fit at 21 nodes missed its limits: see results/multidimensional-groups-recovery.json.");
  process.exitCode = 1;
}

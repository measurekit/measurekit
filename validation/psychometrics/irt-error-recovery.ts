// Two studies of the standard errors of `calibrate`, at the sizes of the guide, which the tests hold at sizes that run in
// seconds:
//
// - bootstrap: the recipe of irt.md, 400 calibrations of 1,000 persons drawn with replacement from 8 2PL items, on seeds 1 to
//   20: the standard deviation of each item's a and b against the Hessian's standard error, the largest distance of a ratio
//   from 1 and the root mean square of the ratios over the 16 parameters;
// - coverage: a pretest of 4 items and 1,000 persons under 6 fixed items calibrated before, on 150 other persons (the
//   fixed items' error matters) or on the same 1,000, 20 replications on each of the seeds 7000 to 7019: the share of the 95%
//   intervals of the pretest items' a and b, and of the mean of theta, that cover the truth, with the fixed items taken as
//   known and with their error carried by `fixedCovariance`.
//
// It writes results/irt-error-recovery.json beside it, takes about 20 minutes, and exits with 1 when a share misses the bound
// it states. Run after `pnpm build`: node validation/psychometrics/irt-error-recovery.ts

import { mean, normalQuantile } from "@measurekit/core";
import { createRandom } from "@measurekit/psychometrics";
import { buildResponseMatrix } from "@measurekit/psychometrics/data";
import {
  type Calibration,
  calibrate,
  calibrateWithCovariance,
  type LogisticItem,
  twoParameterLogistic,
} from "@measurekit/psychometrics/irt";
import { simulateResponses, simulateThetas } from "@measurekit/psychometrics/sim";
import { writeResults } from "../study.ts";

const SEEDS = 20;
const BOOTSTRAP_REPLICATES = 400;
const COVERAGE_REPLICATIONS = 20;
const COVERAGE_FIRST_SEED = 7000;

/** The least share of the intervals with the error carried that must cover, over all the seeds of a design. */
const CARRIED_FLOOR = 0.85;
/** The largest share with the fixed items taken as known that may cover, where their error is large: b and the mean. */
const KNOWN_CEILING = 0.85;
/** The largest distance of the root mean square of the ratios of the bootstrap's errors to the Hessian's from 1. */
const BOOTSTRAP_POOLED = 0.1;

const spread = (values: readonly number[]): number => {
  const center = mean(values);
  return Math.sqrt(values.reduce((sum, value) => sum + (value - center) ** 2, 0) / (values.length - 1));
};

/** The bootstrap of the recipe on one seed. */
function bootstrapStudy(seed: number) {
  const random = createRandom(seed);
  const items = Array.from({ length: 8 }, (_, j) => ({
    model: twoParameterLogistic,
    parameters: { a: 0.8 + 0.1 * j, b: -1.4 + 0.4 * j, c: 0, d: 1 },
  }));
  const matrix = simulateResponses(items, simulateThetas(1000, random), random);
  const fit = calibrate(matrix, twoParameterLogistic, { standardErrors: "hessian" });
  const persons = matrix.responses.length;
  const replicates = Array.from({ length: BOOTSTRAP_REPLICATES }, () => {
    const rows = Array.from({ length: persons }, () => matrix.responses[Math.floor(random.uniform() * persons)] ?? []);
    return calibrate(buildResponseMatrix(rows, { itemIds: matrix.itemIds }), twoParameterLogistic);
  });
  const ratios = fit.items.flatMap((item, j) =>
    (["a", "b"] as const).map(
      (name) =>
        spread(replicates.map((replicate) => replicate.items[j]?.parameters[name] ?? Number.NaN)) /
        (item.standardErrors?.[name] ?? Number.NaN),
    ),
  );
  return {
    seed,
    worst: Math.max(...ratios.map((ratio) => Math.abs(ratio - 1))),
    pooled: Math.sqrt(mean(ratios.map((ratio) => ratio ** 2))) - 1,
    converged: replicates.every((replicate) => replicate.converged),
  };
}

/**
 * Replicates a pretest under fixed items calibrated before and counts the 95% intervals of the pretest items' a and b and
 * of the mean of theta that cover the truth, with the fixed items known and with their error carried.
 */
function coverageStudy(samePersons: boolean, seed: number) {
  const random = createRandom(seed);
  const operational = 6;
  const items = Array.from({ length: operational + 4 }, (_, j) => ({
    id: `q${String(j + 1).padStart(2, "0")}`,
    model: twoParameterLogistic,
    parameters: { a: 0.8 + 1.2 * random.uniform(), b: -1.5 + 3 * random.uniform(), c: 0, d: 1 },
  }));
  const itemIds = items.map((item) => item.id);
  const z = normalQuantile(0.975);
  const counts = { known: { a: 0, b: 0, mean: 0 }, carried: { a: 0, b: 0, mean: 0 }, items: 0, means: 0 };
  for (let replication = 0; replication < COVERAGE_REPLICATIONS; replication++) {
    const persons = (count: number, center: number, sd: number) =>
      simulateThetas(count, random).map((theta) => center + sd * theta);
    const thetas = samePersons ? persons(1000, 0, 1) : persons(1000, 0.3, 1.1);
    const rows = simulateResponses(items, thetas, random).responses;
    const first = samePersons
      ? rows.map((row) => row.slice(0, operational))
      : simulateResponses(items.slice(0, operational), persons(150, 0, 1), random).responses;
    const calibration = calibrateWithCovariance(
      buildResponseMatrix(first, { itemIds: itemIds.slice(0, operational) }),
      twoParameterLogistic,
      { standardErrors: "hessian" },
    );
    const covariance = calibration.covariance;
    if (covariance === null) throw new Error("The calibration of the fixed items has no covariance.");
    const base = {
      fixed: Object.fromEntries(calibration.calibration.items.map((item) => [item.id, item.parameters])),
      ...(samePersons ? {} : { latent: "normal" as const }),
      standardErrors: "hessian" as const,
    };
    const matrix = buildResponseMatrix(rows, { itemIds });
    const fits = {
      known: () => calibrate(matrix, twoParameterLogistic, base),
      carried: () => calibrate(matrix, twoParameterLogistic, { ...base, fixedCovariance: covariance }),
    };
    for (const name of ["known", "carried"] as const) {
      const fit: Calibration<LogisticItem> = fits[name]();
      items.slice(operational).forEach((truth, k) => {
        const estimate = fit.items[operational + k];
        const errors = estimate?.standardErrors;
        const parameters = estimate?.parameters as LogisticItem | undefined;
        if (parameters === undefined || errors === null || errors === undefined) throw new Error("No errors.");
        if (Math.abs(parameters.a - truth.parameters.a) <= z * (errors.a ?? Number.NaN)) counts[name].a++;
        if (Math.abs(parameters.b - truth.parameters.b) <= z * (errors.b ?? Number.NaN)) counts[name].b++;
      });
      const group = fit.groups[0];
      if (
        !samePersons &&
        group?.standardErrors != null &&
        Math.abs(group.mean - 0.3) <= z * group.standardErrors.mean
      ) {
        counts[name].mean++;
      }
    }
    counts.items += 4;
    counts.means += 1;
  }
  return { seed, ...counts };
}

const results = {
  source:
    "calibrate: the recipe of irt.md (400 bootstrap calibrations of 1,000 persons and 8 items, seeds 1 to 20) and the coverage of the pretest items' 95% intervals under fixed items calibrated on 150 other persons or on the same 1,000 (20 replications, seeds 7000 to 7019)",
  bootstrap: { replicates: BOOTSTRAP_REPLICATES, seeds: [] as object[], worst: 0, pooledAbsolute: 0 },
  coverage: [] as object[],
};
const save = (): void => writeResults(import.meta.url, results);
let missed = false;

for (let seed = 1; seed <= SEEDS; seed++) {
  const row = bootstrapStudy(seed);
  results.bootstrap.seeds.push(row);
  results.bootstrap.worst = Math.max(results.bootstrap.worst, row.worst);
  results.bootstrap.pooledAbsolute = Math.max(results.bootstrap.pooledAbsolute, Math.abs(row.pooled));
  missed ||= !row.converged || Math.abs(row.pooled) > BOOTSTRAP_POOLED;
  save();
}

for (const samePersons of [false, true]) {
  const rows = Array.from({ length: SEEDS }, (_, k) => coverageStudy(samePersons, COVERAGE_FIRST_SEED + k));
  const total = (pick: (row: (typeof rows)[number]) => number, of: "items" | "means"): number =>
    rows.reduce((sum, row) => sum + pick(row), 0) / rows.reduce((sum, row) => sum + row[of], 0);
  const shares = {
    carried: {
      a: total((row) => row.carried.a, "items"),
      b: total((row) => row.carried.b, "items"),
      mean: total((row) => row.carried.mean, "means"),
    },
    known: {
      a: total((row) => row.known.a, "items"),
      b: total((row) => row.known.b, "items"),
      mean: total((row) => row.known.mean, "means"),
    },
  };
  const carried = samePersons ? [shares.carried.a, shares.carried.b] : Object.values(shares.carried);
  missed ||= carried.some((share) => share < CARRIED_FLOOR);
  if (!samePersons) missed ||= shares.known.b > KNOWN_CEILING || shares.known.mean > KNOWN_CEILING;
  results.coverage.push({
    fixedItemsCalibratedOn: samePersons ? "the same 1,000 persons" : "150 other persons",
    replications: COVERAGE_REPLICATIONS,
    seeds: [COVERAGE_FIRST_SEED, COVERAGE_FIRST_SEED + SEEDS - 1],
    shares,
    bySeed: rows,
  });
  save();
}

process.exitCode = missed ? 1 : 0;

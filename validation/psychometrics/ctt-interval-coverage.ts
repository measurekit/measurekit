// The coverage of the percentile intervals of alpha and omega, the recipe of ctt.md, at the size of its sample: 200 persons
// answer 8 items of five categories that load 0.4 to 0.8 on one factor, the bootstrap draws the persons with replacement
// 1,000 times, and 500 seeded samples say how often the 95% interval holds the population value, the statistic of 200,000
// persons. The test holds the rule at 40 samples and 100 resamples.
//
// It writes results/ctt-interval-coverage.json beside it, takes about 2 minutes, and exits with 1 when a coverage is
// outside 0.92 to 0.98. Run after `pnpm build`: node validation/psychometrics/ctt-interval-coverage.ts

import { createRandom } from "@measurekit/psychometrics";
import { cronbachAlpha, mcdonaldOmega } from "@measurekit/psychometrics/ctt";
import { buildResponseMatrix, type ResponseMatrix } from "@measurekit/psychometrics/data";
import { writeResults } from "../study.ts";

const LOADINGS = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8];
const CUTS = [-1.2, -0.4, 0.4, 1.2];
const SAMPLES = 500;
const PERSONS = 200;
const RESAMPLES = 1000;
/** The range of the coverage that holds. */
const COVERAGE = [0.92, 0.98] as const;

type Random = ReturnType<typeof createRandom>;

/** Persons answering items of five categories that load on one factor, as in ctt.md. */
function sample(persons: number, random: Random): ResponseMatrix {
  const rows = Array.from({ length: persons }, () => {
    const factor = random.normal();
    return LOADINGS.map((loading) => {
      const score = loading * factor + Math.sqrt(1 - loading ** 2) * random.normal();
      return CUTS.filter((cut) => score > cut).length;
    });
  });
  return buildResponseMatrix(rows);
}

/** The recipe of ctt.md: the percentile interval of a statistic over bootstrap samples of the persons. */
function percentileInterval(
  data: ResponseMatrix,
  statistic: (resample: ResponseMatrix) => number,
  random: Random,
): [number, number] {
  const persons = data.responses.length;
  const values = Array.from({ length: RESAMPLES }, () => {
    const drawn = Array.from({ length: persons }, () => data.responses[Math.floor(random.uniform() * persons)] ?? []);
    return statistic(buildResponseMatrix(drawn, { itemIds: data.itemIds }));
  }).sort((a, b) => a - b);
  return [values[Math.floor(0.025 * RESAMPLES)] ?? 0, values[Math.ceil(0.975 * RESAMPLES) - 1] ?? 0];
}

const omegaOf = (resample: ResponseMatrix): number => mcdonaldOmega(resample).omega;
const population = sample(200_000, createRandom(1));
const truth = { alpha: cronbachAlpha(population), omega: omegaOf(population) };
const random = createRandom(2);
const covered = { alpha: 0, omega: 0 };
for (let set = 0; set < SAMPLES; set++) {
  const data = sample(PERSONS, random);
  const [alphaLow, alphaHigh] = percentileInterval(data, cronbachAlpha, random);
  const [omegaLow, omegaHigh] = percentileInterval(data, omegaOf, random);
  if (alphaLow <= truth.alpha && truth.alpha <= alphaHigh) covered.alpha++;
  if (omegaLow <= truth.omega && truth.omega <= omegaHigh) covered.omega++;
}
const coverage = { alpha: covered.alpha / SAMPLES, omega: covered.omega / SAMPLES };
const within = [coverage.alpha, coverage.omega].every((share) => COVERAGE[0] <= share && share <= COVERAGE[1]);

writeResults(import.meta.url, {
  source:
    "percentile intervals of cronbachAlpha and mcdonaldOmega over 1,000 bootstrap samples of the persons, on 500 seeded samples of 200 persons answering 8 items of five categories that load 0.4 to 0.8 on one factor; the population value is the statistic of 200,000 persons, createRandom(1); the samples and the bootstrap draw from createRandom(2)",
  samples: SAMPLES,
  persons: PERSONS,
  resamples: RESAMPLES,
  truth,
  coverage,
  bound: COVERAGE,
  within,
});
process.exitCode = within ? 0 : 1;

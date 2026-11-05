// How well calibrateBifactor recovers a two-tier model of the size of the guide, which the tests hold at 16 items and 600
// persons, where it is compared with mirt. The responses and the truth are those of bifactor.R's case `recovery`: 3,000
// persons answer 80 items, 40 on each of two primary dimensions correlated 0.5, in 16 specific dimensions of 5 items, on 21
// nodes for each dimension. The error of a fit is the root mean square error of the slopes on the primary dimensions and on
// the specific ones, and the distance of the correlation from 0.5; the bounds are the thresholds of the study when it
// moved here from the tests.
//
// It writes results/bifactor-recovery.json beside it, takes about 10 seconds, and exits with 1 when an error passes its bound.
// Run after `pnpm build`: node validation/psychometrics/bifactor-recovery.ts

import { readFileSync } from "node:fs";
import { buildResponseMatrix } from "@measurekit/psychometrics/data";
import { calibrateBifactor } from "@measurekit/psychometrics/multidimensional";
import { timed, writeResults } from "../study.ts";

/** The largest error of the recovery: the primary slopes, the specific slopes and the correlation of the primary dimensions. */
const BOUNDS = { primary: 0.1, specific: 0.14, correlation: 0.04 } as const;
/** The tolerance of EM: a recovery needs the estimates to a few digits, not mirt's optimum. */
const TOLERANCE = 1e-3;

interface RecoveryFixture {
  readonly itemIds: string[];
  readonly responses: number[][];
  readonly specific: Record<string, string>;
  readonly primary: Record<string, string[]>;
  readonly truth: { readonly slopes: number[][]; readonly intercepts: number[]; readonly correlation: number };
}

const fixture = JSON.parse(
  readFileSync(new URL("../../packages/psychometrics/test/fixtures/bifactor-recovery.json", import.meta.url), "utf8"),
) as RecoveryFixture;

const { value: fit, seconds } = timed(() =>
  calibrateBifactor(buildResponseMatrix(fixture.responses, { itemIds: fixture.itemIds }), fixture.specific, {
    primary: fixture.primary,
    tolerance: TOLERANCE,
  }),
);

/** The root mean square error of the slopes on the dimensions `from` to `to`, over the items that load on them. */
function slopeError(from: number, to: number): number {
  let sum = 0;
  let count = 0;
  fit.items.forEach((item, i) => {
    for (let d = from; d < to; d++) {
      const expected = fixture.truth.slopes[i]?.[d] ?? 0;
      if (expected === 0) continue;
      sum += ((item.parameters.a[d] ?? Number.NaN) - expected) ** 2;
      count++;
    }
  });
  return Math.sqrt(sum / count);
}

const errors = {
  primary: slopeError(0, 2),
  specific: slopeError(2, 18),
  correlation: Math.abs((fit.correlations[1]?.[0] ?? Number.NaN) - fixture.truth.correlation),
};
const within = fit.converged && (["primary", "specific", "correlation"] as const).every((k) => errors[k] <= BOUNDS[k]);

writeResults(import.meta.url, {
  source:
    "calibrateBifactor on the responses of bifactor.R's case recovery: a two-tier 2PL of 3,000 persons and 80 items, 40 on each of two primary dimensions correlated 0.5, in 16 specific dimensions of 5 items, on 21 nodes",
  persons: fixture.responses.length,
  items: fixture.itemIds.length,
  tolerance: TOLERANCE,
  bounds: BOUNDS,
  errors,
  iterations: fit.iterations,
  converged: fit.converged,
  within,
  seconds,
});
console.log(JSON.stringify({ errors, bounds: BOUNDS, iterations: fit.iterations, converged: fit.converged, seconds }));
process.exitCode = within ? 0 : 1;

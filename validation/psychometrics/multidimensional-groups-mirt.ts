// How closely calibrateMultidimensional agrees with mirt's multipleGroup for two groups on three correlated dimensions,
// which takes seconds outside the tests and many times as long under coverage: each cycle of EM estimates the 3 × 3
// covariances of both groups over 15 ** 3 nodes. The case is `three` of multidimensional-groups.json, by
// multidimensional-groups.R: 600 and 500 persons, 18 items, 6 on each dimension. The calibration runs to a tolerance of
// 1e-6, and is held to the tolerances of the other groups: every slope, intercept and correlation within 2e-4, every mean
// and covariance within 5e-4, the log-likelihood, AIC and BIC within 5e-6. The tests calibrate three dimensions in groups
// on a small grid instead.
//
// It writes results/multidimensional-groups-mirt.json: the sizes, the tolerances, the largest differences and the seconds.
// It takes about 10 seconds, and exits with 1 when a difference misses its tolerance. Run after `pnpm build`:
// node validation/psychometrics/multidimensional-groups-mirt.ts

import { readFileSync } from "node:fs";
import { buildResponseMatrix } from "@measurekit/psychometrics/data";
import {
  calibrateMultidimensional,
  type MultidimensionalLogisticParameters,
  multidimensionalTwoParameterLogistic,
} from "@measurekit/psychometrics/multidimensional";
import { timed, writeResults } from "../study.ts";

interface Moments {
  readonly group: string;
  readonly means: readonly number[];
  readonly covariance: readonly (readonly number[])[];
}
interface Case {
  readonly points: number;
  readonly itemIds: readonly string[];
  readonly responses: readonly (readonly number[])[];
  readonly personGroups: readonly string[];
  readonly loadings: Readonly<Record<string, readonly string[]>>;
  readonly items: readonly { readonly parameters: { readonly a: readonly number[]; readonly intercept: number } }[];
  readonly groupMoments: readonly Moments[];
  readonly correlations: readonly (readonly number[])[];
  readonly logLikelihood: number;
  readonly aic: number;
  readonly bic: number;
}
const { three } = JSON.parse(
  readFileSync(
    new URL("../../packages/psychometrics/test/fixtures/multidimensional-groups.json", import.meta.url),
    "utf8",
  ),
) as { readonly three: Case };

const TOLERANCES = { parameters: 2e-4, moments: 5e-4, likelihood: 5e-6 };

const { value: fit, seconds } = timed(() =>
  calibrateMultidimensional(
    buildResponseMatrix(three.responses, { itemIds: [...three.itemIds] }),
    multidimensionalTwoParameterLogistic,
    {
      loadings: three.loadings,
      points: three.points,
      groups: [...three.personGroups],
      tolerance: 1e-6,
      maxIterations: 2000,
    },
  ),
);

const differences = { parameters: 0, moments: 0, likelihood: 0 };
three.items.forEach(({ parameters: expected }, i) => {
  const found = fit.items[i]?.parameters as MultidimensionalLogisticParameters | undefined;
  expected.a.forEach((slope, d) => {
    differences.parameters = Math.max(differences.parameters, Math.abs((found?.a[d] ?? Number.NaN) - slope));
  });
  differences.parameters = Math.max(
    differences.parameters,
    Math.abs((found?.intercept ?? Number.NaN) - expected.intercept),
  );
});
three.correlations.forEach((row, i) => {
  row.forEach((value, j) => {
    differences.parameters = Math.max(
      differences.parameters,
      Math.abs((fit.correlations[i]?.[j] ?? Number.NaN) - value),
    );
  });
});
three.groupMoments.forEach(({ group, means, covariance }, g) => {
  const found = fit.groups?.[g];
  if (found?.group !== group) throw new Error(`Group ${g + 1} is not ${group}.`);
  means.forEach((value, d) => {
    differences.moments = Math.max(differences.moments, Math.abs((found.means[d] ?? Number.NaN) - value));
  });
  covariance.forEach((row, i) => {
    row.forEach((value, j) => {
      differences.moments = Math.max(differences.moments, Math.abs((found.covariance[i]?.[j] ?? Number.NaN) - value));
    });
  });
});
differences.likelihood = Math.max(
  Math.abs(fit.logLikelihood - three.logLikelihood),
  Math.abs(fit.aic - three.aic),
  Math.abs(fit.bic - three.bic),
);
const within =
  fit.converged &&
  differences.parameters <= TOLERANCES.parameters &&
  differences.moments <= TOLERANCES.moments &&
  differences.likelihood <= TOLERANCES.likelihood;

writeResults(import.meta.url, {
  source:
    "calibrateMultidimensional of the case three of multidimensional-groups.json, two groups on three correlated dimensions, against mirt's multipleGroup at the same 15 nodes per dimension.",
  sizes: { persons: three.responses.length, items: three.itemIds.length, dimensions: 3, points: three.points },
  tolerances: TOLERANCES,
  differences,
  within,
  converged: fit.converged,
  iterations: fit.iterations,
  seconds,
});

if (!within) {
  console.error("A difference missed its tolerance: see results/multidimensional-groups-mirt.json.");
  process.exitCode = 1;
}

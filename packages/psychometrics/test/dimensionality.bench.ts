import { createRandom } from "@measurekit/core";
import { test } from "vitest";
import { SLOW_OPTIONS } from "../../../scripts/bench-harness.js";
import {
  buildResponseMatrix,
  exploratoryFactorAnalysis,
  minimumAveragePartial,
  omegaHierarchical,
  parallelAnalysis,
  polychoricCorrelations,
  smoothCorrelations,
} from "../src/index.js";
import { defined } from "./assertions.js";
import { benchmark } from "./budgets.js";
import { readFixture } from "./fixtures.js";

interface BenchDimensionalityFixture {
  itemIds: string[];
  responses: (number | null)[][];
  replications: number;
}
const fixture = readFixture<BenchDimensionalityFixture>("bench-dimensionality.json");

const matrix = buildResponseMatrix(fixture.responses, { itemIds: fixture.itemIds });

interface BenchSmoothingFixture {
  loadings: number[][];
}
const smoothing = readFixture<BenchSmoothingFixture>("bench-dimensionality-smoothing.json");

/**
 * Returns the correlations of the first `items` items of the five-factor loadings of the fixture, by rows: the sums of
 * the products of their loadings, which are exact, as R builds them, with 1 on the diagonal.
 */
function factorCorrelations(items: number): Float64Array {
  const r = new Float64Array(items * items);
  for (let i = 0; i < items; i++) {
    for (let j = 0; j < items; j++) {
      const row = defined(smoothing.loadings[i]);
      const column = defined(smoothing.loadings[j]);
      r[i * items + j] = i === j ? 1 : row.reduce((sum, loading, k) => sum + loading * defined(column[k]), 0);
    }
  }
  return r;
}

test("correlates 25 items of 6 categories answered by 2,800 persons", async (context) => {
  await benchmark(context, "dimensionality.polychoricCorrelations", () => polychoricCorrelations(matrix), SLOW_OPTIONS);
  await benchmark(
    context,
    "dimensionality.parallelAnalysis",
    () => parallelAnalysis(matrix, createRandom(1), { replications: fixture.replications }),
    SLOW_OPTIONS,
  );
});

test("fits five factors to 25 items answered by 2,800 persons and tests how many components they have", async (context) => {
  await benchmark(context, "dimensionality.exploratoryFactorAnalysis", () =>
    exploratoryFactorAnalysis(matrix, { factors: 5 }),
  );
  await benchmark(context, "dimensionality.minimumAveragePartial", () =>
    minimumAveragePartial(matrix, { maxFactors: 8 }),
  );
});

test("fits five factors to 25 items answered by 2,800 persons by maximum likelihood, and a general factor above them", async (context) => {
  await benchmark(context, "dimensionality.maximumLikelihoodFactors", () =>
    exploratoryFactorAnalysis(matrix, { factors: 5, extraction: "maximum-likelihood" }),
  );
  await benchmark(context, "dimensionality.omegaHierarchical", () => omegaHierarchical(matrix, { factors: 5 }));
});

test("smooths positive definite correlations of 25, 50, 100, 200 and 1,000 items, the most, by their eigendecomposition", async (context) => {
  const sizes = [
    [25, "dimensionality.smoothCorrelations25"],
    [50, "dimensionality.smoothCorrelations50"],
    [100, "dimensionality.smoothCorrelations100"],
    [200, "dimensionality.smoothCorrelations200"],
    [1000, "dimensionality.smoothCorrelations1000"],
  ] as const;
  for (const [items, name] of sizes) {
    const r = factorCorrelations(items);
    await benchmark(context, name, () => smoothCorrelations(r, items), items === 1000 ? SLOW_OPTIONS : undefined);
  }
});

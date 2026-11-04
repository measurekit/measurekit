import { createRandom } from "@measurekit/core";
import { test } from "vitest";
import { SLOW_OPTIONS } from "../../../bench/harness.js";
import {
  buildResponseMatrix,
  exploratoryFactorAnalysis,
  minimumAveragePartial,
  omegaHierarchical,
  parallelAnalysis,
  polychoricCorrelations,
} from "../src/index.js";
import { benchmark } from "./budgets.js";
import { readFixture } from "./fixtures.js";

interface BenchDimensionalityFixture {
  itemIds: string[];
  responses: (number | null)[][];
  replications: number;
}
const fixture = readFixture<BenchDimensionalityFixture>("bench-dimensionality.json");

const matrix = buildResponseMatrix(fixture.responses, { itemIds: fixture.itemIds });

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

import { createRandom } from "@measurekit/core";
import { test } from "vitest";
import { SLOW_OPTIONS } from "../../../bench/harness.js";
import { parallelAnalysis, polychoricCorrelations, responseMatrix } from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-dimensionality.json" with { type: "json" };

const matrix = responseMatrix(fixture.responses, { itemIds: fixture.itemIds });

test("correlates 25 items of 6 categories answered by 2,800 persons", async (context) => {
  await benchmark(context, "dimensionality.polychoricCorrelations", () => polychoricCorrelations(matrix), SLOW_OPTIONS);
  await benchmark(
    context,
    "dimensionality.parallelAnalysis",
    () => parallelAnalysis(matrix, createRandom(1), { replications: fixture.replications }),
    SLOW_OPTIONS,
  );
});

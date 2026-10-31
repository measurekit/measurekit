import { test } from "vitest";
import { SLOW_OPTIONS } from "../../../bench/harness.js";
import {
  automatedItemSelection,
  invariantItemOrdering,
  monotonicity,
  responseMatrix,
  scalability,
} from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-mokken.json" with { type: "json" };

const matrix = responseMatrix(fixture.responses, { itemIds: fixture.itemIds });

test("scales 20 items of 5 categories answered by the persons of the acl data", async (context) => {
  await benchmark(context, "mokken.scalability", () => scalability(matrix));
  await benchmark(context, "mokken.automatedItemSelection", () => automatedItemSelection(matrix));
  await benchmark(context, "mokken.monotonicity", () => monotonicity(matrix));
  await benchmark(context, "mokken.invariantItemOrdering", () => invariantItemOrdering(matrix), SLOW_OPTIONS);
});

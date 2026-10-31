import { test } from "vitest";
import { auc, metrics } from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-evaluate.json" with { type: "json" };

test("scores 100,000 predictions", async (context) => {
  await benchmark(context, "evaluate.auc", () => auc(fixture.predictions, fixture.outcomes));
  await benchmark(context, "evaluate.metrics", () => metrics(fixture.predictions, fixture.outcomes));
});

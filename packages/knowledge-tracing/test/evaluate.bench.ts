import { test } from "vitest";
import { auc, metrics, metricsByOpportunity } from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-evaluate.json" with { type: "json" };
import opportunity from "./fixtures/bench-evaluate-opportunity.json" with { type: "json" };
import { columnsLog } from "./logs.js";

test("scores 100,000 predictions", async (context) => {
  await benchmark(context, "evaluate.auc", () => auc(fixture.predictions, fixture.outcomes));
  await benchmark(context, "evaluate.metrics", () => metrics(fixture.predictions, fixture.outcomes));
});

test("scores 50,000 predictions by opportunity, over 5 skills", async (context) => {
  const log = columnsLog({ ...opportunity, item: opportunity.learner, skill: opportunity.skill });
  await benchmark(context, "evaluate.metricsByOpportunity", () => metricsByOpportunity(log, opportunity.prediction));
});

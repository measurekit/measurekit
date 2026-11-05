import { test } from "vitest";
import { auc, metrics, metricsByOpportunity } from "../src/evaluate/index.js";
import { benchmark } from "./budgets.js";
import { readFixture } from "./fixtures.js";
import { columnsLog } from "./logs.js";

interface BenchEvaluateFixture {
  predictions: number[];
  outcomes: number[];
}
const fixture = readFixture<BenchEvaluateFixture>("bench-evaluate.json");

interface BenchEvaluateOpportunityFixture {
  learner: string[];
  skill: string[];
  time: number[];
  outcome: (number | null)[];
  prediction: (number | null)[];
}
const opportunity = readFixture<BenchEvaluateOpportunityFixture>("bench-evaluate-opportunity.json");

test("scores 100,000 predictions", async (context) => {
  await benchmark(context, "evaluate.auc", () => auc(fixture.predictions, fixture.outcomes));
  await benchmark(context, "evaluate.metrics", () => metrics(fixture.predictions, fixture.outcomes));
});

test("scores 50,000 predictions by opportunity, over 5 skills", async (context) => {
  const log = columnsLog({ ...opportunity, item: opportunity.learner, skill: opportunity.skill });
  await benchmark(context, "evaluate.metricsByOpportunity", () => metricsByOpportunity(log, opportunity.prediction));
});

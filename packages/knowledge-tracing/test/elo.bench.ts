import { test } from "vitest";
import { elo } from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-elo.json" with { type: "json" };
import { columnsLog, online } from "./helpers.js";

/** 2,000 learners with 50 attempts each at 400 items: 100,000 attempts. */
const log = columnsLog(fixture);

test("replays 100,000 attempts through the learner-item Elo model", async (context) => {
  const model = elo();
  await benchmark(context, "elo.replay", () => online(model, log));
});

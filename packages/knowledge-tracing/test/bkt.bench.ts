import { test } from "vitest";
import { SLOW_OPTIONS } from "../../../bench/harness.js";
import { bkt } from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-bkt.json" with { type: "json" };
import { columnsLog, online } from "./helpers.js";

/** 200 learners at 5 skills, 20 attempts at each: 20,000 attempts. */
const log = columnsLog(fixture);

test("fits the 5 skills of 20,000 attempts by EM from five starts", async (context) => {
  await benchmark(context, "bkt.fit", () => bkt().fit(log), SLOW_OPTIONS);
});

test("replays 20,000 attempts through the fitted model", async (context) => {
  const { model } = bkt().fit(log);
  await benchmark(context, "bkt.replay", () => online(model, log));
});

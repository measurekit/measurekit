import { test } from "vitest";
import { eventLog } from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-elo.json" with { type: "json" };

/** The 100,000 events of the Elo benchmark, as input to `eventLog`. */
const events = fixture.learner.map((learner, i) => ({
  learner,
  item: fixture.item[i] ?? "",
  skills: [],
  time: fixture.time[i] ?? 0,
  outcome: fixture.outcome[i] ?? 0,
}));

test("builds and checks a log of 100,000 events", async (context) => {
  await benchmark(context, "data.eventLog", () => eventLog(events));
});

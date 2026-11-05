import { test } from "vitest";
import { buildEventLog } from "../src/data/index.js";
import { benchmark } from "./budgets.js";
import { type BenchEloFixture, readFixture } from "./fixtures.js";

const fixture = readFixture<BenchEloFixture>("bench-elo.json");

/** The 100,000 events of the Elo benchmark, as input to `buildEventLog`. */
const events = fixture.learner.map((learner, i) => ({
  learner,
  item: fixture.item[i] ?? "",
  skills: [],
  time: fixture.time[i] ?? 0,
  outcome: fixture.outcome[i] ?? 0,
}));

test("builds and checks a log of 100,000 events", async (context) => {
  await benchmark(context, "data.buildEventLog", () => buildEventLog(events));
});

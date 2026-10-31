import { test } from "vitest";
import { eventLog, irt } from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-irt.json" with { type: "json" };
import { defined, online } from "./helpers.js";

/** 2,000 learners answer 40 items in turn: 80,000 responses. */
const events = fixture.responses.flatMap((row, learner) =>
  row.map((outcome, item) => ({ learner: `l${learner}`, item: `item${item}`, skills: [], time: item, outcome })),
);
const log = eventLog(events);
const items = Object.fromEntries(fixture.items.a.map((a, i) => [`item${i}`, { a, b: defined(fixture.items.b[i]) }]));

test("updates the grid posterior of the ability of 2,000 learners after each of 40 responses", async (context) => {
  const model = irt({ parameters: { items } });
  await benchmark(context, "irt.replay", () => online(model, log));
});

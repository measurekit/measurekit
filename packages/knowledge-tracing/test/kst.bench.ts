import { test } from "vitest";
import { SLOW_OPTIONS } from "../../../bench/harness.js";
import { eventLog, iita, kst, surmiseStructure } from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-kst.json" with { type: "json" };

/** 1,000 persons answer 12 items: 12,000 responses. */
const log = eventLog(
  fixture.responses.flatMap((row, person) =>
    row.map((outcome, item) => ({
      learner: `p${person}`,
      item: fixture.itemIds[item] ?? "",
      skills: [],
      time: item,
      outcome,
    })),
  ),
);
/** A chain of prerequisites: each item needs the one before it, so the structure has 13 states. */
const structure = surmiseStructure(
  Object.fromEntries(fixture.itemIds.map((id, i) => [id, i === 0 ? [] : [fixture.itemIds[i - 1] ?? ""]])),
);

test("fits the basic local independence model of a chain of 12 items", async (context) => {
  const model = kst({ structure, over: "items" });
  await benchmark(context, "kst.blim", () => model.fit(log), SLOW_OPTIONS);
});

test("runs the inductive item tree analysis of the responses of 1,000 persons to 12 items", async (context) => {
  await benchmark(context, "kst.iita", () => iita(log));
});

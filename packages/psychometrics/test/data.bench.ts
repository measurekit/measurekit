import { test } from "vitest";
import { scoreResponses } from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-data.json" with { type: "json" };

test("scores the choices of 2,000 persons on 50 items against a key", async (context) => {
  await benchmark(context, "data.scoreResponses", () => scoreResponses(fixture.choices, fixture.key));
});

import { test } from "vitest";
import { generalizabilityStudy } from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-generalizability.json" with { type: "json" };

test("estimates the variance components of persons, items and raters crossed, 8,000 scores", async (context) => {
  await benchmark(context, "generalizability.generalizabilityStudy", () =>
    generalizabilityStudy(fixture.scores, { facets: ["item", "rater"] }),
  );
});

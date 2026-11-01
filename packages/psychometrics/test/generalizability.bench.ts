import { test } from "vitest";
import { generalizabilityStudy } from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-generalizability.json" with { type: "json" };
import unbalancedFixture from "./fixtures/bench-generalizability-unbalanced.json" with { type: "json" };

test("estimates the variance components of persons, items and raters crossed, 8,000 scores", async (context) => {
  await benchmark(context, "generalizability.generalizabilityStudy", () =>
    generalizabilityStudy(fixture.scores, { facets: ["item", "rater"] }),
  );
});

test("estimates the variance components of persons, items and raters crossed with a quarter of the scores missing, 6,058 scores", async (context) => {
  await benchmark(context, "generalizability.generalizabilityStudyUnbalanced", () =>
    generalizabilityStudy(unbalancedFixture.scores, { facets: ["item", "rater"], unbalanced: true }),
  );
});

import { test } from "vitest";
import { generalizabilityStudy } from "../src/index.js";
import { benchmark } from "./budgets.js";
import { readFixture } from "./fixtures.js";

interface BenchGeneralizabilityFixture {
  scores: { levels: { person: string; item: string; rater: string }; score: number }[];
}
const fixture = readFixture<BenchGeneralizabilityFixture>("bench-generalizability.json");

interface BenchGeneralizabilityUnbalancedFixture {
  scores: { levels: { person: string; item: string; rater: string }; score: number }[];
}
const unbalancedFixture = readFixture<BenchGeneralizabilityUnbalancedFixture>("bench-generalizability-unbalanced.json");

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

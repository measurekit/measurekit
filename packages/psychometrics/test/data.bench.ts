import { test } from "vitest";
import { scoreResponses } from "../src/index.js";
import { benchmark } from "./budgets.js";
import { readFixture } from "./fixtures.js";

interface BenchDataFixture {
  key: string[];
  choices: string[][];
}
const fixture = readFixture<BenchDataFixture>("bench-data.json");

test("scores the choices of 2,000 persons on 50 items against a key", async (context) => {
  await benchmark(context, "data.scoreResponses", () => scoreResponses(fixture.choices, fixture.key));
});

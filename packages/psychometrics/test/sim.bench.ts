import { createRandom } from "@measurekit/core";
import { test } from "vitest";
import { type IrtItem, simulateResponses, simulateThetas, twoParameterLogistic } from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-irt.json" with { type: "json" };
import { defined } from "./helpers.js";

const PERSONS = 2000;
const SEED = 20261029;
/** The 2PL items that mirt estimated for the IRT benchmarks. */
const items: IrtItem[] = fixture.items.a.map((a, i) => ({
  model: twoParameterLogistic,
  parameters: { a, b: defined(fixture.items.b[i]), c: 0, d: 1 },
}));

test("simulates the abilities of 2,000 persons and their responses to 40 items", async (context) => {
  await benchmark(context, "sim.simulateResponses", () => {
    const random = createRandom(SEED);
    return simulateResponses(items, simulateThetas(PERSONS, random), random);
  });
});

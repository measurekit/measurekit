import { test } from "vitest";
import { createRandom } from "../src/index.js";
import { benchmark } from "./budgets.js";

/** The draws of each distribution: a million. */
const DRAWS = 1_000_000;
const SEED = 20261029;

test("draws a million uniform numbers", async (context) => {
  const random = createRandom(SEED);
  await benchmark(context, "random.uniform", () => {
    let sum = 0;
    for (let i = 0; i < DRAWS; i++) sum += random.uniform();
    return sum;
  });
});

test("draws a million normal numbers", async (context) => {
  const random = createRandom(SEED);
  await benchmark(context, "random.normal", () => {
    let sum = 0;
    for (let i = 0; i < DRAWS; i++) sum += random.normal();
    return sum;
  });
});

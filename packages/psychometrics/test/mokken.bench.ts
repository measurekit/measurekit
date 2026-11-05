import { createRandom } from "@measurekit/core";
import { test } from "vitest";
import { SLOW_OPTIONS } from "../../../scripts/bench-harness.js";
import { buildResponseMatrix } from "../src/data/index.js";
import {
  automatedItemSelection,
  conditionalAssociation,
  guttmanErrors,
  invariantItemOrdering,
  monotonicity,
  personScalability,
  scalability,
  twoWayImputation,
} from "../src/mokken/index.js";
import { benchmark } from "./budgets.js";
import { type BenchIrtFixture, readFixture } from "./fixtures.js";

const irtFixture = readFixture<BenchIrtFixture>("bench-irt.json");

interface BenchMokkenFixture {
  itemIds: string[];
  responses: number[][];
}
const fixture = readFixture<BenchMokkenFixture>("bench-mokken.json");

interface BenchMokkenStepsFixture {
  itemIds: string[];
  responses: (number | null)[][];
}
const stepsFixture = readFixture<BenchMokkenStepsFixture>("bench-mokken-steps.json");

const matrix = buildResponseMatrix(fixture.responses, { itemIds: fixture.itemIds });

test("scales 20 items of 5 categories answered by the persons of the acl data", async (context) => {
  await benchmark(context, "mokken.scalability", () => scalability(matrix));
  await benchmark(context, "mokken.automatedItemSelection", () => automatedItemSelection(matrix));
  await benchmark(context, "mokken.monotonicity", () => monotonicity(matrix));
  await benchmark(context, "mokken.invariantItemOrdering", () => invariantItemOrdering(matrix), SLOW_OPTIONS);
});

test("finds the Guttman errors and the conditional association of the 20 items of the acl data, and imputes them with one response in ten missing", async (context) => {
  const missing = buildResponseMatrix(stepsFixture.responses, { itemIds: stepsFixture.itemIds });
  await benchmark(context, "mokken.guttmanErrors", () => guttmanErrors(matrix));
  await benchmark(context, "mokken.conditionalAssociation", () => conditionalAssociation(matrix));
  await benchmark(context, "mokken.twoWayImputation", () => twoWayImputation(missing, createRandom(1)));
});

test("finds the normed Guttman errors, U3 and Ht of 2,000 persons on 40 dichotomous items", async (context) => {
  const binary = buildResponseMatrix(irtFixture.binary.responses, { itemIds: irtFixture.binary.itemIds });
  await benchmark(context, "mokken.personScalability", () => personScalability(binary));
});

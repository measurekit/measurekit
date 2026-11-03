import { createRandom } from "@measurekit/core";
import { test } from "vitest";
import { SLOW_OPTIONS } from "../../../bench/harness.js";
import {
  automatedItemSelection,
  conditionalAssociation,
  guttmanErrors,
  invariantItemOrdering,
  monotonicity,
  personScalability,
  responseMatrix,
  scalability,
  twoWayImputation,
} from "../src/index.js";
import { benchmark } from "./budgets.js";
import irtFixture from "./fixtures/bench-irt.json" with { type: "json" };
import fixture from "./fixtures/bench-mokken.json" with { type: "json" };
import stepsFixture from "./fixtures/bench-mokken-steps.json" with { type: "json" };

const matrix = responseMatrix(fixture.responses, { itemIds: fixture.itemIds });

test("scales 20 items of 5 categories answered by the persons of the acl data", async (context) => {
  await benchmark(context, "mokken.scalability", () => scalability(matrix));
  await benchmark(context, "mokken.automatedItemSelection", () => automatedItemSelection(matrix));
  await benchmark(context, "mokken.monotonicity", () => monotonicity(matrix));
  await benchmark(context, "mokken.invariantItemOrdering", () => invariantItemOrdering(matrix), SLOW_OPTIONS);
});

test("finds the Guttman errors and the conditional association of the 20 items of the acl data, and imputes them with one response in ten missing", async (context) => {
  const missing = responseMatrix(stepsFixture.responses, { itemIds: stepsFixture.itemIds });
  await benchmark(context, "mokken.guttmanErrors", () => guttmanErrors(matrix));
  await benchmark(context, "mokken.conditionalAssociation", () => conditionalAssociation(matrix));
  await benchmark(context, "mokken.twoWayImputation", () => twoWayImputation(missing, createRandom(1)));
});

test("finds the normed Guttman errors, U3 and Ht of 2,000 persons on 40 dichotomous items", async (context) => {
  const binary = responseMatrix(irtFixture.binary.responses, { itemIds: irtFixture.binary.itemIds });
  await benchmark(context, "mokken.personScalability", () => personScalability(binary));
});

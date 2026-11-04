import { expect, test } from "vitest";
import { SLOW_OPTIONS } from "../../../bench/harness.js";
import { buildResponseMatrix, calibrateBifactor } from "../src/index.js";
import { benchmark } from "./budgets.js";
import { readFixture } from "./fixtures.js";

interface BenchBifactorFixture {
  items12: { itemIds: string[]; responses: number[][]; specific: Record<string, string> };
  items40: { itemIds: string[]; responses: number[][]; specific: Record<string, string> };
}
const fixture = readFixture<BenchBifactorFixture>("bench-bifactor.json");

const twelve = buildResponseMatrix(fixture.items12.responses, { itemIds: fixture.items12.itemIds });
const forty = buildResponseMatrix(fixture.items40.responses, { itemIds: fixture.items40.itemIds });

test("calibrates a bifactor 2PL of 2,000 persons and 12 items in 3 specific dimensions", async (context) => {
  const options = { specific: fixture.items12.specific };
  expect(calibrateBifactor(twelve, options).converged).toBe(true);
  await benchmark(
    context,
    "multidimensional.calibrateBifactor12",
    () => calibrateBifactor(twelve, options),
    SLOW_OPTIONS,
  );
});

test("calibrates a bifactor 2PL of 2,000 persons and 40 items in 4 specific dimensions", async (context) => {
  const options = { specific: fixture.items40.specific };
  expect(calibrateBifactor(forty, options).converged).toBe(true);
  await benchmark(
    context,
    "multidimensional.calibrateBifactor40",
    () => calibrateBifactor(forty, options),
    SLOW_OPTIONS,
  );
});

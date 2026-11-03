import { test } from "vitest";
import {
  type CalibratedItem,
  itemFeatureRegression,
  linearChange,
  lltm,
  responseMatrix,
  twoParameterLogistic,
} from "../src/index.js";
import { defined } from "./assertions.js";
import { benchmark } from "./budgets.js";
import { readFixture } from "./fixtures.js";

interface BenchExplanatoryFixture {
  itemIds: string[];
  difficulties: number[];
  standardErrors: number[];
  features: number[][];
  lltm: {
    itemIds: string[];
    responses: number[][];
    features: Record<string, { steps: number; density: number; open: number }>;
  };
}
const fixture = readFixture<BenchExplanatoryFixture>("bench-explanatory.json");

interface BenchExplanatoryChangeFixture {
  itemIds: string[];
  groups: string[];
  occasions: number[][][];
}
const changeFixture = readFixture<BenchExplanatoryChangeFixture>("bench-explanatory-change.json");

const items: CalibratedItem<unknown>[] = fixture.itemIds.map((id, i) => ({
  id,
  model: twoParameterLogistic,
  parameters: { a: 1, b: defined(fixture.difficulties[i]), c: 0, d: 1 },
  standardErrors: { a: 0.1, b: defined(fixture.standardErrors[i]) },
}));
const features = Object.fromEntries(
  fixture.itemIds.map((id, i) => {
    const [f1, f2, f3] = defined(fixture.features[i]);
    return [id, { f1: defined(f1), f2: defined(f2), f3: defined(f3) }];
  }),
);

test("regresses the difficulties of 300 items on three features", async (context) => {
  await benchmark(context, "explanatory.itemFeatureRegression", () => itemFeatureRegression(items, features));
});

test("estimates the effects of three features on the difficulty of 20 items answered by 1,000 persons", async (context) => {
  const matrix = responseMatrix(fixture.lltm.responses, { itemIds: fixture.lltm.itemIds });
  await benchmark(context, "explanatory.lltm", () => lltm(matrix, fixture.lltm.features));
});

test("estimates the change of 8 items between two occasions of 2,000 persons in two groups", async (context) => {
  const occasions = changeFixture.occasions.map((rows) => responseMatrix(rows, { itemIds: changeFixture.itemIds }));
  await benchmark(context, "explanatory.linearChange", () =>
    linearChange(occasions, { groups: changeFixture.groups, baseline: "ctl" }),
  );
});

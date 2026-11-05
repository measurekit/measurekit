import { test } from "vitest";
import { type TestItem, twoParameterLogistic } from "../src/irt/index.js";
import {
  bookmarkCutScore,
  classificationAccuracy,
  livingstonLewisClassification,
} from "../src/standard-setting/index.js";
import { defined } from "./assertions.js";
import { benchmark } from "./budgets.js";
import { type BenchIrtFixture, readFixture } from "./fixtures.js";

const fixture = readFixture<BenchIrtFixture>("bench-irt.json");

interface BenchStandardSettingFixture {
  scores: number[];
  reliability: number;
  maximum: number;
  cut: number;
}
const classificationFixture = readFixture<BenchStandardSettingFixture>("bench-standard-setting.json");

const items: TestItem[] = fixture.items.a.map((a, i) => ({
  id: defined(fixture.binary.itemIds[i]),
  model: twoParameterLogistic,
  parameters: { a, b: defined(fixture.items.b[i]), c: 0, d: 1 },
}));
const PERSONS = 2000;
const estimates = Array.from({ length: PERSONS }, (_, i) => ({ theta: ((i % 400) - 200) / 100, standardError: 0.3 }));

test("finds the cut score of a bookmark in a booklet of 40 items", async (context) => {
  await benchmark(context, "standardSetting.bookmarkCutScore", () => bookmarkCutScore(items, 20));
});

test("measures the accuracy and consistency of the classification of 2,000 persons by two cuts", async (context) => {
  await benchmark(context, "standardSetting.classificationAccuracy", () =>
    classificationAccuracy(estimates, [-0.5, 0.8]),
  );
});

test("measures the accuracy and consistency of the classification of 3,000 total scores of 40 items by one cut", async (context) => {
  const { scores, reliability, maximum, cut } = classificationFixture;
  await benchmark(context, "standardSetting.livingstonLewis", () =>
    livingstonLewisClassification(scores, [cut], reliability, maximum),
  );
});

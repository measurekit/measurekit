import { expect, test } from "vitest";
import { buildResponseMatrix } from "../src/data/index.js";
import { type IrtItem, nominalResponse } from "../src/irt/index.js";
import {
  generalizedBinomialTest,
  kIndex,
  m4Index,
  omegaIndex,
  type PersonPair,
  selectPairs,
} from "../src/similarity/index.js";
import { defined } from "./assertions.js";
import { benchmark } from "./budgets.js";
import { readFixture } from "./fixtures.js";

interface BenchSimilarityFixture {
  itemIds: string[];
  responses: number[][];
  items: { a: number[]; c: number[] }[];
  thetas: number[];
  key: number[];
  rooms: string[];
  pairs: [number, number][];
}
const fixture = readFixture<BenchSimilarityFixture>("bench-similarity.json");

const matrix = buildResponseMatrix(fixture.responses, { itemIds: fixture.itemIds });
const items = fixture.items.map((parameters): IrtItem => ({ model: nominalResponse, parameters }));
const pairs = fixture.pairs.map(([a, b]): PersonPair => [defined(matrix.personIds[a]), defined(matrix.personIds[b])]);
const { key, thetas } = fixture;

test("selects the 12,000 pairs of 1,000 persons in rooms of 25", async (context) => {
  expect(selectPairs(matrix.personIds, { room: fixture.rooms }).pairs).toHaveLength(12_000);
  await benchmark(context, "similarity.selectPairs", () => selectPairs(matrix.personIds, { room: fixture.rooms }));
});

test("computes omega of 1,200 pairs on 40 items of 5 options", async (context) => {
  await benchmark(context, "similarity.omegaIndex", () => omegaIndex(matrix, items, thetas, pairs));
});

test("computes the generalized binomial test of 1,200 pairs on 40 items of 5 options", async (context) => {
  await benchmark(context, "similarity.generalizedBinomialTest", () =>
    generalizedBinomialTest(matrix, items, thetas, pairs),
  );
});

test("computes M4 of 1,200 pairs on 40 items of 5 options", async (context) => {
  await benchmark(context, "similarity.m4Index", () => m4Index(matrix, items, thetas, pairs, { key }));
});

test("computes the K index and its variants of 1,200 pairs among 1,000 persons on 40 items of 5 options", async (context) => {
  await benchmark(context, "similarity.kIndex", () => kIndex(matrix, pairs, { key }));
});

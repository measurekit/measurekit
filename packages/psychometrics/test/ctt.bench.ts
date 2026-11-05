import { test } from "vitest";
import { cronbachAlpha, itemStatistics, mcdonaldOmega, subscoreValue } from "../src/ctt/index.js";
import { buildResponseMatrix } from "../src/data/index.js";
import { benchmark } from "./budgets.js";
import { readFixture } from "./fixtures.js";

interface BenchCttFixture {
  itemIds: string[];
  responses: number[][];
  sectionSize: number;
  omegaItems: number;
}
const fixture = readFixture<BenchCttFixture>("bench-ctt.json");

const matrix = buildResponseMatrix(fixture.responses, { itemIds: fixture.itemIds });
const omegaMatrix = buildResponseMatrix(
  fixture.responses.map((row) => row.slice(0, fixture.omegaItems)),
  { itemIds: fixture.itemIds.slice(0, fixture.omegaItems) },
);
/** Five sections of ten consecutive items. */
const sections = Object.fromEntries(
  Array.from({ length: fixture.itemIds.length / fixture.sectionSize }, (_, s) => [
    `section${s + 1}`,
    fixture.itemIds.slice(s * fixture.sectionSize, (s + 1) * fixture.sectionSize),
  ]),
);

test("analyzes 3,000 persons and 50 items", async (context) => {
  await benchmark(context, "ctt.itemStatistics", () => itemStatistics(matrix));
  await benchmark(context, "ctt.cronbachAlpha", () => cronbachAlpha(matrix));
  await benchmark(context, "ctt.mcdonaldOmega", () => mcdonaldOmega(omegaMatrix));
  await benchmark(context, "ctt.subscoreValue", () => subscoreValue(matrix, sections));
});

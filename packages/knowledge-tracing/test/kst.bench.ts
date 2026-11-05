import { test } from "vitest";
import { SLOW_OPTIONS } from "../../../scripts/bench-harness.js";
import {
  buildEventLog,
  clauseStructure,
  diffTest,
  iita,
  kst,
  structureBase,
  structureValidity,
  surmiseStructure,
  unionClosure,
} from "../src/index.js";
import { benchmark } from "./budgets.js";
import { readFixture } from "./fixtures.js";

interface BenchKstFixture {
  itemIds: string[];
  responses: number[][];
}
const fixture = readFixture<BenchKstFixture>("bench-kst.json");

interface BenchKstStructureFixture {
  domain: string[];
  states: string[][];
  clauses: Record<string, string[][]>;
}
const structureFixture = readFixture<BenchKstStructureFixture>("bench-kst-structure.json");

/** 1,000 persons answer 12 items: 12,000 responses. */
const log = buildEventLog(
  fixture.responses.flatMap((row, person) =>
    row.map((outcome, item) => ({
      learner: `p${person}`,
      item: fixture.itemIds[item] ?? "",
      skills: [],
      time: item,
      outcome,
    })),
  ),
);
/** A chain of prerequisites: each item needs the one before it, so the structure has 13 states. */
const structure = surmiseStructure(
  Object.fromEntries(fixture.itemIds.map((id, i) => [id, i === 0 ? [] : [fixture.itemIds[i - 1] ?? ""]])),
);

test("fits the basic local independence model of a chain of 12 items", async (context) => {
  const model = kst(structure, { over: "items" });
  await benchmark(context, "kst.blim", () => model.fit(log), SLOW_OPTIONS);
});

test("fits the basic local independence model of a chain of 12 items by minimum discrepancy maximum likelihood", async (context) => {
  const model = kst(structure, { over: "items", method: "mdml" });
  await benchmark(context, "kst.mdml", () => model.fit(log), SLOW_OPTIONS);
});

test("finds the local identifiability of the model of a chain of 12 items and 13 states", async (context) => {
  const model = kst(structure, { over: "items" });
  await benchmark(context, "kst.identifiability", () => model.identifiability());
});

test("measures the validity of a chain of 12 items against the responses of 1,000 persons", async (context) => {
  await benchmark(context, "kst.validity", () => structureValidity(structure, log));
});

/** The first 6 items of the same persons: DAKS's variance sums over all 2 ^ items patterns, so its test is sized there. */
const testItems = fixture.itemIds.slice(0, 6);
const testLog = buildEventLog(log.filter((event) => testItems.includes(event.item)));
const testPrerequisites = Object.fromEntries(testItems.map((id, i) => [id, i === 0 ? [] : [testItems[i - 1] ?? ""]]));

test("tests the fit of a quasi order of 6 items to the responses of 1,000 persons", async (context) => {
  await benchmark(context, "kst.diffTest", () => diffTest(testLog, testPrerequisites));
});

/** The 12 items of the same persons with 3 responses in 10 left out, and the chain without its edge from item 6 to 7. */
const sparseLog = buildEventLog(
  log.filter((event) => (Number(event.learner.slice(1)) * 7 + fixture.itemIds.indexOf(event.item) * 3) % 10 >= 3),
);
const chain = Object.fromEntries(fixture.itemIds.map((id, i) => [id, i === 0 ? [] : [fixture.itemIds[i - 1] ?? ""]]));
const broken = { ...chain, [fixture.itemIds[6] ?? ""]: [] };

test("tests one edge of a chain of 12 items on the incomplete responses of 1,000 persons", async (context) => {
  await benchmark(context, "kst.diffTestIncomplete", () =>
    diffTest(sparseLog, chain, { alternative: broken, tail: "greater" }),
  );
});

test("runs the inductive item tree analysis of the responses of 1,000 persons to 12 items", async (context) => {
  await benchmark(context, "kst.iita", () => iita(log));
});

/** Three chains of four elements: 125 states over 12 elements. */
const space = unionClosure(structureFixture.domain, structureFixture.states);
const atoms = structureBase(space);

test("closes the 12 atoms of a space of 125 states under union", async (context) => {
  await benchmark(context, "kst.unionClosure", () => unionClosure(structureFixture.domain, atoms));
});

test("finds the base of a space of 12 elements and 125 states", async (context) => {
  await benchmark(context, "kst.structureBase", () => structureBase(space));
});

/**
 * A cycle of 1,000 elements, the most a surmise relation takes, each requiring the next: one class, whose closure
 * reaches every element from each.
 */
const cycle = Object.fromEntries(Array.from({ length: 1000 }, (_, e) => [`e${e}`, [`e${(e + 1) % 1000}`]]));

test("builds the space of a cycle of prerequisites of 1,000 elements", async (context) => {
  await benchmark(context, "kst.surmiseStructure", () => surmiseStructure(cycle));
});

test("builds the space of 12 elements with clauses", async (context) => {
  await benchmark(context, "kst.clauseStructure", () => clauseStructure(structureFixture.clauses));
});

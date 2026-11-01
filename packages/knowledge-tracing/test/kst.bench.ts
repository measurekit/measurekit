import { test } from "vitest";
import { SLOW_OPTIONS } from "../../../bench/harness.js";
import { clauseStructure, eventLog, iita, kst, structureBase, surmiseStructure, unionClosure } from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-kst.json" with { type: "json" };
import structureFixture from "./fixtures/bench-kst-structure.json" with { type: "json" };

/** 1,000 persons answer 12 items: 12,000 responses. */
const log = eventLog(
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
  const model = kst({ structure, over: "items" });
  await benchmark(context, "kst.blim", () => model.fit(log), SLOW_OPTIONS);
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

test("builds the space of 12 elements with clauses", async (context) => {
  await benchmark(context, "kst.clauseStructure", () => clauseStructure(structureFixture.clauses));
});

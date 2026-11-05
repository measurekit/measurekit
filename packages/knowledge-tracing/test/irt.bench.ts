import { test } from "vitest";
import { buildEventLog, irt } from "../src/index.js";
import { defined } from "./assertions.js";
import { benchmark } from "./budgets.js";
import { readFixture } from "./fixtures.js";
import { online } from "./logs.js";

interface BenchIrtFixture {
  responses: number[][];
  items: { a: number[]; b: number[] };
  nominal: { responses: number[][]; a: number[][]; c: number[][] };
}
const fixture = readFixture<BenchIrtFixture>("bench-irt.json");

/** 2,000 learners answer 40 items in turn: 80,000 responses. */
const DRIFT_LEARNERS = 200;
const events = fixture.responses.flatMap((row, learner) =>
  row.map((outcome, item) => ({ learner: `l${learner}`, item: `item${item}`, skills: [], time: item, outcome })),
);
const log = buildEventLog(events);
/** The first 200 learners, their responses a minute apart: 8,000 responses. */
const driftLog = buildEventLog(events.filter(({ learner }) => Number(learner.slice(1)) < DRIFT_LEARNERS));
const items = Object.fromEntries(fixture.items.a.map((a, i) => [`item${i}`, { a, b: defined(fixture.items.b[i]) }]));
/** The same 2,000 learners and 40 items, nominal of 4 categories: category k is the outcome k / 3. */
const nominalLog = buildEventLog(
  fixture.nominal.responses.flatMap((row, learner) =>
    row.map((k, item) => ({ learner: `l${learner}`, item: `item${item}`, skills: [], time: item, outcome: k / 3 })),
  ),
);
const nominalItems = Object.fromEntries(
  fixture.nominal.a.map((a, i) => [`item${i}`, { model: "nominal" as const, a, c: defined(fixture.nominal.c[i]) }]),
);

test("updates the grid posterior of the ability of 2,000 learners after each of 40 responses", async (context) => {
  const model = irt({ items });
  await benchmark(context, "irt.replay", () => online(model, log));
});

test("updates the grid posterior of the ability of 2,000 learners after each of 40 responses to nominal items", async (context) => {
  const model = irt({ items: nominalItems });
  await benchmark(context, "irt.nominal", () => online(model, nominalLog));
});

test("updates the grid posterior of 200 learners with drift on 241 points after each of 40 responses", async (context) => {
  // The times are minutes, so that θ drifts by a variance of 0.02 a week.
  const model = irt({ items }, { points: 241, drift: { variance: 0.02, per: 10_080 } });
  await benchmark(context, "irt.drift", () => online(model, driftLog));
});

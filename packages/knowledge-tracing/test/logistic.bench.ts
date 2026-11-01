import { test } from "vitest";
import { SLOW_OPTIONS } from "../../../bench/harness.js";
import { afm, type Event, eventLog, logistic, searchRates } from "../src/index.js";
import { checkTerms } from "../src/logistic/terms.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-bkt.json" with { type: "json" };
import itemsFixture from "./fixtures/bench-logistic-items.json" with { type: "json" };
import lktFeatures from "./fixtures/logistic-lkt-features.json" with { type: "json" };
import ratesFixture from "./fixtures/logistic-rates.json" with { type: "json" };
import { columnsLog, online } from "./helpers.js";

/** The log of the BKT benchmark: 20,000 attempts of 200 learners at 5 skills. */
const log = columnsLog(fixture);

test("fits the additive factors model of 20,000 attempts", async (context) => {
  await benchmark(context, "logistic.afm", () => afm().fit(log));
});

/** The log of the validation of LKT's features: 1,941 events of 17 learners, with the features that LKT computes. */
const lktEvents: readonly Event[] = JSON.parse(JSON.stringify(lktFeatures.events));
const lktSpecs = lktFeatures.features
  .filter(({ lkt }) => lkt !== null)
  .map(({ feature }) => JSON.parse(JSON.stringify(feature)));

test("replays 1,941 events with the 16 features of LKT's other names", async (context) => {
  const skills = Object.fromEntries(
    lktEvents.flatMap(({ skills: ids }) => ids).map((skill) => [skill, { intercept: 0, coefficients: {} }]),
  );
  const labels = checkTerms(lktSpecs).map(({ label }) => label);
  const parameters = {
    skills: Object.fromEntries(
      Object.keys(skills).map((skill) => [
        skill,
        { intercept: 0, coefficients: Object.fromEntries(labels.map((label) => [label, 0])) },
      ]),
    ),
  };
  const model = logistic({ features: lktSpecs, parameters });
  await benchmark(context, "logistic.features", () => online(model, eventLog(lktEvents)));
});

/** The model of Best-LR: intercepts of skills and items, a proficiency for each learner, and log-counts at each skill and overall. */
const bestLr = logistic({
  proficiency: "fixed",
  intercepts: ["skill", "item"],
  ridge: 1,
  features: [
    "logSuccesses",
    "logFailures",
    { feature: "logSuccesses", over: "learner" },
    { feature: "logFailures", over: "learner" },
  ],
});

/** The log of a benchmark fixture whose learners, items and skills are codes, the events in time order. */
function codedLog(columns: {
  readonly learner: readonly number[];
  readonly item: readonly number[];
  readonly skill: readonly number[];
  readonly outcome: readonly number[];
}) {
  return eventLog(
    columns.learner.map((learner, i) => ({
      learner: `l${learner}`,
      item: `i${columns.item[i]}`,
      skills: [`s${columns.skill[i]}`],
      time: i,
      outcome: columns.outcome[i] ?? 0,
    })),
  );
}

test("fits 20,000 events at 1,000 items with intercepts of skills and items, a proficiency for each learner and a ridge", async (context) => {
  const log = codedLog(itemsFixture.small);
  await benchmark(context, "logistic.fitItems", () => bestLr.fit(log), SLOW_OPTIONS);
});

test("fits 150,000 events at 49,546 items with intercepts of skills and items, a proficiency for each learner and a ridge", async (context) => {
  const log = codedLog(itemsFixture.large);
  await benchmark(context, "logistic.fitManyItems", () => bestLr.fit(log), SLOW_OPTIONS);
});

test("searches the decay of a proportion of successes and the exponent of a recency over 3,600 events", async (context) => {
  const log = eventLog(JSON.parse(JSON.stringify(ratesFixture.events)));
  const { lower, upper, start } = ratesFixture.bounds;
  const build = (rates: readonly number[]) =>
    logistic({ features: [{ decayedProportion: rates[0] ?? 0 }, { recency: rates[1] ?? 0 }] });
  await benchmark(
    context,
    "logistic.searchRates",
    () => searchRates(build, log, { start, lower, upper }),
    SLOW_OPTIONS,
  );
});

import { test } from "vitest";
import { afm, type Event, eventLog, logistic } from "../src/index.js";
import { checkFeatures } from "../src/logistic/features.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-bkt.json" with { type: "json" };
import lktFeatures from "./fixtures/logistic-lkt-features.json" with { type: "json" };
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
  const labels = checkFeatures(lktSpecs).map(({ label }) => label);
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

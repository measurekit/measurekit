import { test } from "vitest";
import { afm } from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-bkt.json" with { type: "json" };
import { columnsLog } from "./helpers.js";

/** The log of the BKT benchmark: 20,000 attempts of 200 learners at 5 skills. */
const log = columnsLog(fixture);

test("fits the additive factors model of 20,000 attempts", async (context) => {
  await benchmark(context, "logistic.afm", () => afm().fit(log));
});

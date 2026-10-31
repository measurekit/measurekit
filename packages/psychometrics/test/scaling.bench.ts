import { test } from "vitest";
import { conversionTable, type IrtItem, twoParameterLogistic } from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-irt.json" with { type: "json" };
import { defined } from "./helpers.js";

/** The reporting scale: 15 points per logit around 150, from 100 to 200 in whole points. */
const SCALE = { slope: 15, intercept: 150, lowest: 100, highest: 200, step: 1 };
const items: IrtItem[] = fixture.items.a.map((a, i) => ({
  model: twoParameterLogistic,
  parameters: { a, b: defined(fixture.items.b[i]), c: 0, d: 1 },
}));

test("converts each summed score of 40 items to a scale score", async (context) => {
  await benchmark(context, "scaling.conversionTableEapSum", () => conversionTable(items, SCALE, { method: "eap-sum" }));
  await benchmark(context, "scaling.conversionTableTrueScore", () =>
    conversionTable(items, SCALE, { method: "true-score" }),
  );
});

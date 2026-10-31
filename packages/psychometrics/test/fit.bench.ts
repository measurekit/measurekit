import { test } from "vitest";
import {
  abilityEstimates,
  itemFit,
  personFit,
  posteriorMeanEstimate,
  responseMatrix,
  summedScoreFit,
  type TestItem,
  twoParameterLogistic,
  yenQ3,
} from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-irt.json" with { type: "json" };
import { defined } from "./helpers.js";

/** The data and the 2PL items of the IRT benchmarks: 2,000 persons and 40 items, with the posterior means of the abilities. */
const matrix = responseMatrix(fixture.binary.responses, { itemIds: fixture.binary.itemIds });
const items: TestItem[] = fixture.items.a.map((a, i) => ({
  id: defined(fixture.binary.itemIds[i]),
  model: twoParameterLogistic,
  parameters: { a, b: defined(fixture.items.b[i]), c: 0, d: 1 },
}));
const thetas = abilityEstimates(matrix, items, posteriorMeanEstimate).map(({ theta }) => defined(theta));

test("computes the fit of 40 items to 2,000 persons", async (context) => {
  await benchmark(context, "fit.itemFit", () => itemFit(matrix, items, thetas));
  await benchmark(context, "fit.personFit", () => personFit(matrix, items, thetas));
  await benchmark(context, "fit.summedScoreFit", () => summedScoreFit(matrix, items));
  await benchmark(context, "fit.yenQ3", () => yenQ3(matrix, items, thetas));
});

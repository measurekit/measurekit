import { test } from "vitest";
import { SLOW_OPTIONS } from "../../../bench/harness.js";
import {
  adjacentAgreement,
  cohenKappa,
  exactAgreement,
  fleissKappa,
  intraclassCorrelation,
  manyFacetRasch,
  standardizedMeanDifference,
} from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-rater.json" with { type: "json" };

test("measures the agreement of raters", async (context) => {
  await benchmark(context, "rater.cohenKappa", () => cohenKappa(fixture.pair));
  await benchmark(context, "rater.fleissKappa", () => fleissKappa(fixture.panel));
  await benchmark(context, "rater.exactAgreement", () => exactAgreement(fixture.panel));
  await benchmark(context, "rater.adjacentAgreement", () => adjacentAgreement(fixture.panel));
  await benchmark(context, "rater.meanDifference", () => standardizedMeanDifference(fixture.smd));
  await benchmark(context, "rater.intraclassCorrelation", () => intraclassCorrelation(fixture.continuous));
});

test("calibrates the many-facet Rasch model of 200 persons, 10 items and 4 raters", async (context) => {
  await benchmark(
    context,
    "rater.manyFacetRasch",
    () => manyFacetRasch(fixture.scores, { facets: ["item", "rater"] }),
    SLOW_OPTIONS,
  );
});

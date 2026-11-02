import { test } from "vitest";
import { SLOW_OPTIONS } from "../../../bench/harness.js";
import {
  type DifGroup,
  irtLikelihoodRatioDif,
  logisticRegressionDif,
  lordChiSquare,
  mantelHaenszel,
  ordinalLogisticRegressionDif,
  rajuAreaDif,
  responseMatrix,
  sibtest,
  twoParameterLogistic,
} from "../src/index.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-dif.json" with { type: "json" };
import irtFixture from "./fixtures/bench-dif-irt.json" with { type: "json" };

const groupsOf = (focal: readonly number[]): DifGroup[] => focal.map((flag) => (flag === 1 ? "focal" : "reference"));

const dichotomous = responseMatrix(fixture.dichotomous.responses, { itemIds: fixture.dichotomous.itemIds });
const groups = groupsOf(fixture.dichotomous.focal);
const polytomous = responseMatrix(fixture.polytomous.responses, { itemIds: fixture.polytomous.itemIds });
const polytomousGroups = groupsOf(fixture.polytomous.focal);

test("tests 30 items of 2,000 persons in two groups for differential functioning", async (context) => {
  await benchmark(context, "dif.mantelHaenszel", () => mantelHaenszel(dichotomous, groups));
  await benchmark(context, "dif.logisticRegression", () => logisticRegressionDif(dichotomous, groups));
  await benchmark(
    context,
    "dif.lordChiSquare",
    () => lordChiSquare(dichotomous, groups, { model: twoParameterLogistic }),
    SLOW_OPTIONS,
  );
  await benchmark(
    context,
    "dif.lordChiSquarePurified",
    () => lordChiSquare(dichotomous, groups, { model: twoParameterLogistic, purify: true }),
    SLOW_OPTIONS,
  );
});

test("tests 10 items of 4 categories of 1,000 persons in two groups by ordinal logistic regression", async (context) => {
  await benchmark(context, "dif.ordinalLogistic", () => ordinalLogisticRegressionDif(polytomous, polytomousGroups));
});

test("tests 15 items of 2,000 persons in two groups by IRT and by SIBTEST", async (context) => {
  const matrix = responseMatrix(irtFixture.responses, { itemIds: irtFixture.itemIds });
  const irtGroups = groupsOf(irtFixture.focal);
  const options = { model: twoParameterLogistic };
  await benchmark(
    context,
    "dif.irtLikelihoodRatio",
    () => irtLikelihoodRatioDif(matrix, irtGroups, options),
    SLOW_OPTIONS,
  );
  await benchmark(context, "dif.rajuArea", () => rajuAreaDif(matrix, irtGroups, options), SLOW_OPTIONS);
  await benchmark(context, "dif.sibtest", () => sibtest(matrix, irtGroups));
});

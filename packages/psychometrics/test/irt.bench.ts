import { expect, test } from "vitest";
import { SLOW_OPTIONS } from "../../../scripts/bench-harness.js";
import {
  abilityEstimates,
  buildResponseMatrix,
  calibrate,
  calibrateUnfolding,
  conditionalMaximumLikelihood,
  generalizedPartialCredit,
  gradedResponse,
  jointMaximumLikelihood,
  maximumLikelihoodEstimate,
  oneParameterLogistic,
  posteriorMeanEstimate,
  posteriorModeEstimate,
  summedScoreTable,
  type TestItem,
  testInformation,
  threeParameterLogistic,
  twoParameterLogistic,
  weightedLikelihoodEstimate,
} from "../src/index.js";
import { defined } from "./assertions.js";
import { benchmark } from "./budgets.js";
import { type BenchIrtConditionalFixture, type BenchIrtFixture, readFixture } from "./fixtures.js";

const fixture = readFixture<BenchIrtFixture>("bench-irt.json");

const conditionalFixture = readFixture<BenchIrtConditionalFixture>("bench-irt-conditional.json");

interface BenchIrtLatentRegressionFixture {
  itemIds: string[];
  responses: number[][];
  covariate: number[];
}
const regressionFixture = readFixture<BenchIrtLatentRegressionFixture>("bench-irt-latent-regression.json");

interface BenchIrtUnfoldingFixture {
  itemIds: string[];
  responses: number[][];
}
const unfoldingFixture = readFixture<BenchIrtUnfoldingFixture>("bench-irt-unfolding.json");

const binary = buildResponseMatrix(fixture.binary.responses, { itemIds: fixture.binary.itemIds });
const three = buildResponseMatrix(fixture.three.responses, { itemIds: fixture.three.itemIds });
const graded = buildResponseMatrix(fixture.graded.responses, { itemIds: fixture.graded.itemIds });
const testlets = buildResponseMatrix(fixture.testlet.responses, { itemIds: fixture.testlet.itemIds });
/** The size of the calibration with standard errors from the Hessian: the first persons and items of the binary data. */
const PILOT_PERSONS = 500;
const PILOT_ITEMS = 20;
/** The 2PL items that mirt estimated on the binary data. */
const items: TestItem[] = fixture.items.a.map((a, i) => ({
  id: defined(fixture.binary.itemIds[i]),
  model: twoParameterLogistic,
  parameters: { a, b: defined(fixture.items.b[i]), c: 0, d: 1 },
}));

test("calibrates the one-parameter model of 2,000 persons and 40 items", async (context) => {
  expect(calibrate(binary, { model: oneParameterLogistic }).converged).toBe(true);
  await benchmark(context, "irt.calibrate1pl", () => calibrate(binary, { model: oneParameterLogistic }), SLOW_OPTIONS);
});

test("calibrates the two-parameter model of 2,000 persons and 40 items", async (context) => {
  expect(calibrate(binary, { model: twoParameterLogistic }).converged).toBe(true);
  await benchmark(context, "irt.calibrate2pl", () => calibrate(binary, { model: twoParameterLogistic }), SLOW_OPTIONS);
});

test("calibrates the three-parameter model of 2,000 persons and 40 items", async (context) => {
  expect(calibrate(three, { model: threeParameterLogistic }).converged).toBe(true);
  await benchmark(context, "irt.calibrate3pl", () => calibrate(three, { model: threeParameterLogistic }), SLOW_OPTIONS);
});

test("calibrates the two-parameter model with cross-product standard errors", async (context) => {
  const options = { model: twoParameterLogistic, standardErrors: "cross-product" } as const;
  expect(calibrate(binary, options).converged).toBe(true);
  await benchmark(context, "irt.calibrate2plCrossProduct", () => calibrate(binary, options), SLOW_OPTIONS);
});

test("calibrates the two-parameter model of 500 persons and 20 items with standard errors from the Hessian", async (context) => {
  const pilot = buildResponseMatrix(
    fixture.binary.responses.slice(0, PILOT_PERSONS).map((row) => row.slice(0, PILOT_ITEMS)),
    { itemIds: fixture.binary.itemIds.slice(0, PILOT_ITEMS) },
  );
  const options = { model: twoParameterLogistic, standardErrors: "hessian" } as const;
  expect(calibrate(pilot, options).converged).toBe(true);
  await benchmark(context, "irt.calibrate2plHessian", () => calibrate(pilot, options), SLOW_OPTIONS);
});

test("calibrates the two-parameter model of 2,000 persons and 40 items with a latent regression", async (context) => {
  const regression = buildResponseMatrix(regressionFixture.responses, { itemIds: regressionFixture.itemIds });
  const options = { model: twoParameterLogistic, covariates: { x: regressionFixture.covariate } } as const;
  expect(calibrate(regression, options).converged).toBe(true);
  await benchmark(context, "irt.calibrate2plLatentRegression", () => calibrate(regression, options), SLOW_OPTIONS);
});

test("calibrates the graded response model of 1,500 persons and 20 items of 5 categories", async (context) => {
  expect(calibrate(graded, { model: gradedResponse }).converged).toBe(true);
  await benchmark(context, "irt.calibrateGraded", () => calibrate(graded, { model: gradedResponse }), SLOW_OPTIONS);
});

test("calibrates the generalized partial credit model of 1,500 persons and 20 items of 5 categories", async (context) => {
  expect(calibrate(graded, { model: generalizedPartialCredit }).converged).toBe(true);
  await benchmark(
    context,
    "irt.calibrateGpcm",
    () => calibrate(graded, { model: generalizedPartialCredit }),
    SLOW_OPTIONS,
  );
});

test("calibrates the testlet model of 1,000 persons and 6 testlets of 5 items", async (context) => {
  const options = {
    model: twoParameterLogistic,
    testlets: fixture.testlet.testlets,
    quadrature: { points: fixture.testlet.points },
  };
  expect(calibrate(testlets, options).converged).toBe(true);
  await benchmark(context, "irt.calibrateTestlets", () => calibrate(testlets, options), SLOW_OPTIONS);
});

test("estimates the ability of 2,000 persons on 40 items", async (context) => {
  await benchmark(context, "irt.abilityPosteriorMean", () =>
    abilityEstimates(binary, items, { estimator: posteriorMeanEstimate }),
  );
  await benchmark(context, "irt.abilityPosteriorMode", () =>
    abilityEstimates(binary, items, { estimator: posteriorModeEstimate }),
  );
  await benchmark(context, "irt.abilityMaximumLikelihood", () =>
    abilityEstimates(binary, items, { estimator: maximumLikelihoodEstimate }),
  );
  await benchmark(context, "irt.abilityWeightedLikelihood", () =>
    abilityEstimates(binary, items, { estimator: weightedLikelihoodEstimate }),
  );
});

test("calibrates the Rasch model of 2,000 persons and 40 items by joint maximum likelihood", async (context) => {
  expect(jointMaximumLikelihood(binary).converged).toBe(true);
  await benchmark(context, "irt.jointMaximumLikelihood", () => jointMaximumLikelihood(binary), SLOW_OPTIONS);
});

test("tabulates the posterior of each summed score of 40 items", async (context) => {
  await benchmark(context, "irt.summedScoreTable", () => summedScoreTable(items));
});

test("sums the information of 40 items at 121 values of theta", async (context) => {
  const thetas = Array.from({ length: 121 }, (_, i) => -6 + i / 10);
  await benchmark(context, "irt.testInformation", () => {
    let sum = 0;
    for (const theta of thetas) sum += testInformation(items, theta);
    return sum;
  });
});

test("calibrates the Rasch model of 2,000 persons and 15 items, and the partial credit and rating scale models of 1,500 persons and 8 items, by conditional maximum likelihood", async (context) => {
  const rasch = buildResponseMatrix(conditionalFixture.rasch.responses, { itemIds: conditionalFixture.rasch.itemIds });
  const credit = buildResponseMatrix(conditionalFixture.partialCredit.responses, {
    itemIds: conditionalFixture.partialCredit.itemIds,
  });
  await benchmark(context, "irt.conditionalRasch", () => conditionalMaximumLikelihood(rasch));
  await benchmark(context, "irt.conditionalPartialCredit", () =>
    conditionalMaximumLikelihood(credit, { model: "partial-credit" }),
  );
  await benchmark(context, "irt.conditionalRatingScale", () =>
    conditionalMaximumLikelihood(credit, { model: "rating-scale" }),
  );
});

test("calibrates the generalized graded unfolding model of 2,000 persons and 15 items of 4 categories", async (context) => {
  const unfolding = buildResponseMatrix(unfoldingFixture.responses, { itemIds: unfoldingFixture.itemIds });
  expect(calibrateUnfolding(unfolding).converged).toBe(true);
  await benchmark(context, "irt.calibrateUnfolding", () => calibrateUnfolding(unfolding), SLOW_OPTIONS);
});

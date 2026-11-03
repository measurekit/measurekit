import { test } from "vitest";
import {
  abilityEstimates,
  andersenTest,
  calibrate,
  correctedPersonFit,
  itemFit,
  kernelItemCurves,
  limitedInformationFit,
  localDependenceChiSquare,
  martinLofTest,
  personFit,
  posteriorMeanEstimate,
  posteriorModeEstimate,
  raschWaldTest,
  responseMatrix,
  rootMeanSquareDeviation,
  summedScoreFit,
  type TestItem,
  twoParameterLogistic,
  yenQ3,
} from "../src/index.js";
import { defined } from "./assertions.js";
import { benchmark } from "./budgets.js";
import fixture from "./fixtures/bench-irt.json" with { type: "json" };
import conditionalFixture from "./fixtures/bench-irt-conditional.json" with { type: "json" };

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

test("computes lz*, kernel curves, M2*, the RMSD and local dependence of 40 items and 2,000 persons", async (context) => {
  const modes = abilityEstimates(matrix, items, posteriorModeEstimate).map(({ theta }) => defined(theta));
  const calibration = calibrate(matrix, { model: twoParameterLogistic });
  await benchmark(context, "fit.correctedPersonFit", () =>
    correctedPersonFit(matrix, items, modes, { estimator: "posterior-mode" }),
  );
  await benchmark(context, "fit.kernelItemCurves", () => kernelItemCurves(matrix));
  await benchmark(context, "fit.limitedInformationFit", () => limitedInformationFit(matrix, calibration));
  await benchmark(context, "fit.rootMeanSquareDeviation", () => rootMeanSquareDeviation(matrix, items));
  await benchmark(context, "fit.localDependenceChiSquare", () => localDependenceChiSquare(matrix, items));
});

test("tests the Rasch model of 15 items and 2,000 persons by Andersen's, Martin-Löf's and Wald's tests", async (context) => {
  const rasch = responseMatrix(conditionalFixture.rasch.responses, { itemIds: conditionalFixture.rasch.itemIds });
  await benchmark(context, "fit.andersenTest", () => andersenTest(rasch));
  await benchmark(context, "fit.martinLofTest", () => martinLofTest(rasch));
  await benchmark(context, "fit.raschWaldTest", () => raschWaldTest(rasch));
});

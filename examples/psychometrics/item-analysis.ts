// A test of 20 items, answered by 800 simulated persons in two groups, analyzed the way a test developer would: classical
// item statistics and reliability, a 2PL calibration with the fit of its items, ability estimates, a check for items that
// function differently for the groups, and an adaptive test run on the calibrated items.
// Run after `pnpm build`: node examples/psychometrics/item-analysis.ts
import {
  abilityEstimates,
  calibrate,
  createRandom,
  cronbachAlpha,
  type DifGroup,
  itemFit,
  itemStatistics,
  mantelHaenszel,
  simulateItems,
  simulateResponses,
  simulateThetas,
  twoParameterLogistic,
} from "@measurekit/psychometrics";
import { anyOf, createAdaptiveTest, standardErrorThreshold, testLength } from "@measurekit/psychometrics/cat";
import { buildResponseMatrix } from "@measurekit/psychometrics/data";

// The data: item 7 is one point harder for the second group.
const random = createRandom(2026);
const items = simulateItems(twoParameterLogistic, 20, random);
const shifted = items.map((item, j) =>
  j === 6 ? { ...item, parameters: { ...item.parameters, b: item.parameters.b + 1 } } : item,
);
const first = simulateResponses(items, simulateThetas(400, random), random);
const second = simulateResponses(shifted, simulateThetas(400, random), random);
const matrix = buildResponseMatrix([...first.responses, ...second.responses]);
const groups: DifGroup[] = matrix.personIds.map((_, person) => (person < 400 ? "reference" : "focal"));

// Classical test theory: the easiest and the hardest item, and the reliability of the total score.
const statistics = itemStatistics(matrix);
const byMean = [...statistics].sort((a, b) => (a.mean ?? 0) - (b.mean ?? 0));
console.log("Hardest item:", byMean[0]?.id, "p =", byMean[0]?.mean?.toFixed(2));
console.log("Easiest item:", byMean.at(-1)?.id, "p =", byMean.at(-1)?.mean?.toFixed(2));
console.log("Cronbach's alpha:", cronbachAlpha(matrix).toFixed(3));

// Item response theory: calibrate, then check the fit of the items to the estimated abilities.
const irt = calibrate(matrix, twoParameterLogistic, { standardErrors: "hessian" });
console.log("Converged in", irt.iterations, "iterations; log-likelihood", irt.logLikelihood.toFixed(1));
const estimates = abilityEstimates(matrix, irt.items);
const thetas = estimates.map((estimate) => estimate.theta ?? 0);
const worst = [...itemFit(matrix, irt.items, thetas)].sort(
  (a, b) => (b.outfitMeanSquare ?? 0) - (a.outfitMeanSquare ?? 0),
)[0];
console.log("Worst outfit:", worst?.id, worst?.outfitMeanSquare?.toFixed(2));
console.log(
  "First person: theta",
  estimates[0]?.theta?.toFixed(2),
  "with standard error",
  estimates[0]?.standardError?.toFixed(2),
);

// Differential item functioning: Mantel-Haenszel flags item 7 for the second group.
const flagged = mantelHaenszel(matrix, groups).items.filter((item) => item.etsCategory !== "A");
console.log("Items with DIF:", flagged.map((item) => `${item.id} (${item.etsCategory})`).join(", "));

// An adaptive test of the first person on the calibrated items: each next item is the most informative one.
const test = createAdaptiveTest(irt.items, anyOf(testLength(12), standardErrorThreshold(0.4)));
const answers = Object.fromEntries(matrix.itemIds.map((id, j) => [id, matrix.responses[0]?.[j] ?? 0]));
let state = test.start(random);
while (state.next !== null) state = test.answer(state, answers[state.next] ?? 0, random);
console.log(
  "Adaptive test:",
  state.items.length,
  "items, theta",
  state.estimate.theta.toFixed(2),
  "against",
  estimates[0]?.theta?.toFixed(2),
);

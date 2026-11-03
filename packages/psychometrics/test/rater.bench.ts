import { test } from "vitest";
import { SLOW_OPTIONS } from "../../../bench/harness.js";
import {
  adjacentAgreement,
  cohenKappa,
  exactAgreement,
  fleissKappa,
  intraclassCorrelation,
  manyFacetRasch,
  type PopulationDeviations,
  ratingCorrelation,
  standardizedMeanDifference,
} from "../src/index.js";
import { benchmark } from "./budgets.js";
import { type RaterAutomatedScoringFixture, readFixture } from "./fixtures.js";

interface BenchRaterFixture {
  pair: number[][];
  panel: number[][];
  continuous: number[][];
  smd: number[][];
  scores: { levels: { item: string; rater: string }; score: number }[];
}
const fixture = readFixture<BenchRaterFixture>("bench-rater.json");

interface BenchRaterInteractionsFixture {
  scores: { levels: { item: string; rater: string }; score: number }[];
}
const interactionsFixture = readFixture<BenchRaterInteractionsFixture>("bench-rater-interactions.json");

const scoringFixture = readFixture<RaterAutomatedScoringFixture>("rater-automated-scoring.json");

/** The metrics whose degradation from human-human to machine-human RSMTool reports. */
const DEGRADED = ["correlation", "kappa", "quadraticKappa", "exact", "adjacent", "smd"] as const;

/** The standard deviation of scores, on n - 1. */
function standardDeviation(scores: readonly number[]): number {
  const mean = scores.reduce((sum, score) => sum + score, 0) / scores.length;
  return Math.sqrt(scores.reduce((sum, score) => sum + (score - mean) ** 2, 0) / (scores.length - 1));
}

/** The recipe of rater.md: machine scores rounded and trimmed to the scale. */
function toScale(scores: readonly number[], [low = 0, high = 0]: readonly number[]): number[] {
  return scores.map((score) => Math.min(high, Math.max(low, Math.round(score))));
}

/** The recipe of rater.md: every metric of a second rating against the first, the human, as rows [second, first]. */
function evaluate(human: readonly number[], machine: readonly number[], population?: PopulationDeviations) {
  const pairs = machine.map((score, i) => [score, human[i] ?? null]);
  const squares = machine.reduce((sum, score, i) => sum + (score - (human[i] ?? 0)) ** 2, 0);
  const humanMean = human.reduce((sum, score) => sum + score, 0) / human.length;
  const spread = human.reduce((sum, score) => sum + (score - humanMean) ** 2, 0);
  return {
    kappa: cohenKappa(pairs),
    quadraticKappa: cohenKappa(pairs, { weights: "quadratic" }),
    exact: exactAgreement(pairs),
    adjacent: adjacentAgreement(pairs),
    smd: standardizedMeanDifference(pairs, population === undefined ? {} : { population }),
    correlation: ratingCorrelation(pairs),
    rSquared: 1 - squares / spread,
    rmse: Math.sqrt(squares / human.length),
    machineSd: standardDeviation(machine),
    humanSd: standardDeviation(human),
  };
}

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

test("calibrates the many-facet Rasch model of 150 persons, 8 items and 3 raters with the interaction of items and raters", async (context) => {
  await benchmark(context, "rater.manyFacetRaschInteractions", () =>
    manyFacetRasch(interactionsFixture.scores, { facets: ["item", "rater"], interactions: [["item", "rater"]] }),
  );
});

test("evaluates the automated scores of the 8,859 essays of ELLIPSE against two human raters", async (context) => {
  const { first, second, machine, scale } = scoringFixture.cases.ELLIPSE;
  await benchmark(context, "rater.automatedScoring", () => {
    const system = evaluate(first, toScale(machine, scale));
    const humans = evaluate(first, second);
    return {
      system,
      humans,
      degradation: DEGRADED.map((name) => (system[name] ?? 0) - (humans[name] ?? 0)),
      disattenuated: (system.correlation ?? 0) / Math.sqrt(humans.correlation ?? 0),
    };
  });
});

test("evaluates the automated scores of the 1,818 essays of PERSUADE 2.0 overall and in each of 6 groups", async (context) => {
  const { human, machine, scale, groups } = scoringFixture.cases["PERSUADE 2.0"];
  const names = [...new Set(groups)].sort();
  await benchmark(context, "rater.automatedScoringGroups", () => {
    const scaled = toScale(machine, scale);
    const population = { first: standardDeviation(scaled), second: standardDeviation(human) };
    return {
      overall: evaluate(human, scaled),
      groups: names.map((name) => {
        const rows = groups.flatMap((group, i) => (group === name ? [i] : []));
        return evaluate(
          rows.map((i) => human[i] ?? 0),
          rows.map((i) => scaled[i] ?? 0),
          population,
        );
      }),
    };
  });
});

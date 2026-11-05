import { test } from "vitest";
import { SLOW_OPTIONS } from "../../../scripts/bench-harness.js";
import {
  adjacentAgreement,
  automatedScoreAgreement,
  cohenKappa,
  exactAgreement,
  fleissKappa,
  intraclassCorrelation,
  manyFacetRasch,
  routingCurve,
  scoreMapping,
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

interface RaterScoringFixture {
  routing: { confidence: number[] };
}
const scoringReferences = readFixture<RaterScoringFixture>("rater-scoring.json");

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
    () => manyFacetRasch(fixture.scores, ["item", "rater"]),
    SLOW_OPTIONS,
  );
});

test("calibrates the many-facet Rasch model of 150 persons, 8 items and 3 raters with the interaction of items and raters", async (context) => {
  await benchmark(context, "rater.manyFacetRaschInteractions", () =>
    manyFacetRasch(interactionsFixture.scores, ["item", "rater"], { interactions: [["item", "rater"]] }),
  );
});

const ellipse = scoringFixture.cases.ELLIPSE;
const persuade = scoringFixture.cases["PERSUADE 2.0"];
const ellipseScores = ellipse.machine.map((machine, i) => ({
  machine,
  human: ellipse.first[i] ?? 0,
  second: ellipse.second[i] ?? null,
  confidence: scoringReferences.routing.confidence[i] ?? null,
}));
const persuadeScores = persuade.machine.map((machine, i) => ({
  machine,
  human: persuade.human[i] ?? 0,
  group: persuade.groups[i] ?? null,
}));

test("evaluates the automated scores of the 8,859 essays of ELLIPSE against two human raters", async (context) => {
  await benchmark(context, "rater.automatedScoring", () =>
    automatedScoreAgreement(ellipseScores, { scale: { lowest: 1, highest: 5 } }),
  );
});

test("evaluates the automated scores of the 1,818 essays of PERSUADE 2.0 overall and in each of 6 groups", async (context) => {
  await benchmark(context, "rater.automatedScoringGroups", () =>
    automatedScoreAgreement(persuadeScores, { scale: { lowest: 1, highest: 6 } }),
  );
});

test("traces the routing curve of the 8,859 essays of ELLIPSE over 89 confidences", async (context) => {
  await benchmark(context, "rater.routingCurve", () =>
    routingCurve(ellipseScores, { scale: { lowest: 1, highest: 5 } }),
  );
});

test("fits the isotonic mapping of the 8,859 essays of ELLIPSE from machine to human scores", async (context) => {
  await benchmark(context, "rater.scoreMappingIsotonic", () => scoreMapping(ellipseScores, { method: "isotonic" }));
});

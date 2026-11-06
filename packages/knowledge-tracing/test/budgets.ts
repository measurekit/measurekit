import { benchmarksOf } from "../../../scripts/bench-harness.js";

/**
 * The most that the fastest sample of each benchmark may take, in milliseconds: five times the fastest sample of the CI
 * runner over five runs of `bench` of 2026-11-05 on the code of commit 59d3e2cc, rounded up to one significant digit, the
 * one rule of the family that validation/README.md explains.
 */
const BUDGETS = {
  "bkt.fit": 500,
  "bkt.fitSkills": 3_000,
  "bkt.fitLessons": 600,
  "bkt.fitPairs": 1_000,
  "bkt.replay": 70,
  "elo.replay": 200,
  "logistic.afm": 100,
  "logistic.features": 60,
  "logistic.fitItems": 400,
  "logistic.fitManyItems": 3_000,
  "logistic.searchRates": 2_000,
  "irt.replay": 3_000,
  "irt.nominal": 5_000,
  "irt.drift": 2_000,
  "kst.blim": 200,
  "kst.mdml": 400,
  "kst.identifiability": 70,
  "kst.validity": 20,
  "kst.diffTest": 5,
  "kst.diffTestIncomplete": 8,
  "kst.iita": 10,
  "kst.unionClosure": 0.7,
  "kst.structureBase": 0.5,
  "kst.clauseStructure": 3,
  "kst.surmiseStructure": 50,
  "evaluate.auc": 60,
  "evaluate.metrics": 90,
  "evaluate.metricsByOpportunity": 200,
  "practice.recommend": 0.5,
  "data.buildEventLog": 200,
} as const;

/** Runs a benchmark against its budget. */
export const benchmark = benchmarksOf(BUDGETS);

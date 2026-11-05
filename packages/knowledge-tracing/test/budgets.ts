import { benchmarksOf } from "../../../scripts/bench-harness.js";

/**
 * The most that the fastest sample of each benchmark may take, in milliseconds: five times the fastest sample of the CI
 * runner over the 13 green runs of `bench` of 2026-11-05 from 00:30 UTC, after the changes to the code they time that day,
 * rounded up to one significant digit, the one rule of STANDARDS.md section 11 that validation/README.md explains.
 */
const BUDGETS = {
  "bkt.fit": 500,
  "bkt.fitSkills": 3_000,
  "bkt.fitLessons": 600,
  "bkt.fitPairs": 1_000,
  "bkt.replay": 100,
  "elo.replay": 200,
  "logistic.afm": 200,
  "logistic.features": 500,
  "logistic.fitItems": 400,
  "logistic.fitManyItems": 4_000,
  "logistic.searchRates": 2_000,
  "irt.replay": 4_000,
  "irt.nominal": 5_000,
  "irt.drift": 2_000,
  "kst.blim": 200,
  "kst.mdml": 400,
  "kst.identifiability": 80,
  "kst.validity": 20,
  "kst.diffTest": 8,
  "kst.diffTestIncomplete": 20,
  "kst.iita": 20,
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

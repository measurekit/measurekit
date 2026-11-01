import { benchmarksOf } from "../../../bench/harness.js";

/**
 * The most that the fastest sample of each benchmark may take, in milliseconds: five times the measurement, rounded up to
 * one significant digit, as docs/performance.md explains.
 */
const BUDGETS = {
  "bkt.fit": 700,
  "bkt.fitLessons": 1_000,
  "bkt.fitPairs": 3_000,
  "bkt.replay": 80,
  "elo.replay": 200,
  "logistic.afm": 200,
  "logistic.features": 300,
  "irt.replay": 2_000,
  "irt.drift": 1_000,
  "kst.blim": 200,
  "kst.mdml": 600,
  "kst.identifiability": 70,
  "kst.validity": 6,
  "kst.diffTest": 3,
  "kst.iita": 9,
  "kst.unionClosure": 0.5,
  "kst.structureBase": 0.3,
  "kst.clauseStructure": 2,
  "evaluate.auc": 200,
  "evaluate.metrics": 200,
  "evaluate.metricsByOpportunity": 200,
  "practice.recommend": 0.6,
  "data.eventLog": 90,
} as const;

/** Runs a benchmark against its budget. */
export const benchmark = benchmarksOf(BUDGETS);

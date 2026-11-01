import { benchmarksOf } from "../../../bench/harness.js";

/**
 * The most that the fastest sample of each benchmark may take, in milliseconds: five times the median of the fastest samples
 * that the CI runner took, rounded up to one significant digit, as docs/performance.md explains.
 */
const BUDGETS = {
  "bkt.fit": 3_000,
  "bkt.fitLessons": 4_000,
  "bkt.fitPairs": 7_000,
  "bkt.replay": 200,
  "elo.replay": 300,
  "logistic.afm": 200,
  "logistic.features": 600,
  "logistic.fitItems": 600,
  "logistic.fitManyItems": 5_000,
  "logistic.searchRates": 3_000,
  "irt.replay": 5_000,
  "irt.drift": 3_000,
  "kst.blim": 300,
  "kst.mdml": 600,
  "kst.identifiability": 200,
  "kst.validity": 20,
  "kst.diffTest": 6,
  "kst.iita": 20,
  "kst.unionClosure": 1,
  "kst.structureBase": 0.6,
  "kst.clauseStructure": 5,
  "evaluate.auc": 200,
  "evaluate.metrics": 300,
  "evaluate.metricsByOpportunity": 300,
  "practice.recommend": 0.6,
  "data.eventLog": 200,
} as const;

/** Runs a benchmark against its budget. */
export const benchmark = benchmarksOf(BUDGETS);

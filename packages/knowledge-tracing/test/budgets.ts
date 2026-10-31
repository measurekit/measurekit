import { benchmarksOf } from "../../../bench/harness.js";

/**
 * The most that the fastest sample of each benchmark may take, in milliseconds: five times the measurement, rounded up to
 * one significant digit, as docs/performance.md explains.
 */
const BUDGETS = {
  "bkt.fit": 700,
  "bkt.replay": 80,
  "elo.replay": 200,
  "logistic.afm": 200,
  "irt.replay": 2_000,
  "kst.blim": 200,
  "kst.iita": 9,
  "evaluate.auc": 200,
  "evaluate.metrics": 200,
  "practice.recommend": 0.6,
  "data.eventLog": 90,
} as const;

/** Runs a benchmark against its budget. */
export const benchmark = benchmarksOf(BUDGETS);

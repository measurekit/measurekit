import { benchmarksOf } from "../../../scripts/bench-harness.js";

/**
 * The most that the fastest sample of each benchmark may take, in milliseconds: five times the fastest sample that the CI
 * runner took, its median over the green runs of `bench` of 2026-11-03 and 2026-11-04, rounded up to one significant digit, as
 * validation/README.md explains. A budget marked "laptop" is for a benchmark that the runner has not taken, or took only before
 * its code changed: five times the laptop's fastest sample times the median ratio of the runner's to the laptop's in its
 * module, until the runner's samples replace it.
 */
const BUDGETS = {
  "bkt.fit": 2_000,
  "bkt.fitSkills": 6_000,
  "bkt.fitLessons": 3_000,
  "bkt.fitPairs": 4_000,
  "bkt.replay": 200,
  "elo.replay": 300,
  "logistic.afm": 200,
  "logistic.features": 700,
  "logistic.fitItems": 600,
  "logistic.fitManyItems": 5_000,
  "logistic.searchRates": 3_000, // laptop
  "irt.replay": 6_000,
  "irt.nominal": 9_000, // laptop
  "irt.drift": 4_000,
  "kst.blim": 400,
  "kst.mdml": 700,
  "kst.identifiability": 200,
  "kst.validity": 20,
  "kst.diffTest": 7,
  "kst.diffTestIncomplete": 20, // laptop
  "kst.iita": 20,
  "kst.unionClosure": 2,
  "kst.structureBase": 0.7,
  "kst.clauseStructure": 5,
  "kst.surmiseStructure": 70,
  "evaluate.auc": 80,
  "evaluate.metrics": 200,
  "evaluate.metricsByOpportunity": 200,
  "practice.recommend": 0.6,
  "data.buildEventLog": 200,
} as const;

/** Runs a benchmark against its budget. */
export const benchmark = benchmarksOf(BUDGETS);

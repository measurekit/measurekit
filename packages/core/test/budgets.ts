import { benchmarksOf } from "../../../bench/harness.js";

/**
 * The most that the fastest sample of each benchmark may take, in milliseconds: five times the measurement, rounded up to
 * one significant digit, as docs/performance.md explains.
 */
const BUDGETS = {
  "numerics.normalQuantile": 200,
  "numerics.logistic": 200,
  "numerics.cholesky": 4,
  "random.uniform": 20,
  "random.normal": 90,
} as const;

/** Runs a benchmark against its budget. */
export const benchmark = benchmarksOf(BUDGETS);

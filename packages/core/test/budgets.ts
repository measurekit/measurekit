import { benchmarksOf } from "../../../bench/harness.js";

/**
 * The most that the fastest sample of each benchmark may take, in milliseconds: five times the median of the fastest samples
 * that the CI runner took, rounded up to one significant digit, as docs/performance.md explains.
 */
const BUDGETS = {
  "numerics.normalQuantile": 100,
  "numerics.logistic": 70,
  "numerics.cholesky": 4,
  "random.uniform": 20,
  "random.normal": 200,
} as const;

/** Runs a benchmark against its budget. */
export const benchmark = benchmarksOf(BUDGETS);

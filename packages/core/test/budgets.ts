import { benchmarksOf } from "../../../scripts/bench-harness.js";

/**
 * The most that the fastest sample of each benchmark may take, in milliseconds: five times the fastest sample that the CI
 * runner took, its median over the green runs of `bench` of 2026-11-03 and 2026-11-04, rounded up to one significant digit, as
 * validation/README.md explains.
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

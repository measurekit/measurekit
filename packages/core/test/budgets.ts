import { benchmarksOf } from "../../../scripts/bench-harness.js";

/**
 * The most that the fastest sample of each benchmark may take, in milliseconds: five times the fastest sample of the CI
 * runner over five runs of `bench` of 2026-11-05 on the code of commit 59d3e2cc, rounded up to one significant digit, the
 * one rule of the family that validation/README.md explains.
 */
const BUDGETS = {
  "numerics.normalQuantile": 40,
  "numerics.logistic": 40,
  "numerics.cholesky": 3,
  "random.uniform": 10,
  "random.normal": 200,
} as const;

/** Runs a benchmark against its budget. */
export const benchmark = benchmarksOf(BUDGETS);

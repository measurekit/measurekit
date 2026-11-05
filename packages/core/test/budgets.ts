import { benchmarksOf } from "../../../scripts/bench-harness.js";

/**
 * The most that the fastest sample of each benchmark may take, in milliseconds: five times the fastest sample of the CI
 * runner over the 13 green runs of `bench` of 2026-11-05 from 00:30 UTC, after the changes to the code they time that day,
 * rounded up to one significant digit, the one rule of STANDARDS.md section 11 that validation/README.md explains.
 */
const BUDGETS = {
  "numerics.normalQuantile": 50,
  "numerics.logistic": 40,
  "numerics.cholesky": 3,
  "random.uniform": 20,
  "random.normal": 200,
} as const;

/** Runs a benchmark against its budget. */
export const benchmark = benchmarksOf(BUDGETS);

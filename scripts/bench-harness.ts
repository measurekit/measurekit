import { expect, type TestContext } from "vitest";

/** The run options of tinybench: the time and the number of samples a benchmark takes, and its warm-up. */
export interface RunOptions {
  readonly time: number;
  readonly iterations: number;
  readonly warmupTime: number;
  readonly warmupIterations: number;
}

/** The run options of a benchmark that takes milliseconds: at least 10 samples over half a second. */
const BENCH_OPTIONS: RunOptions = { time: 500, iterations: 10, warmupTime: 100, warmupIterations: 3 };

/** The run options of a benchmark that takes a second or more: 5 samples, after one run to warm up. */
export const SLOW_OPTIONS: RunOptions = { time: 0, iterations: 5, warmupTime: 0, warmupIterations: 1 };

/** The part of a test's context that a benchmark uses. */
type BenchmarkContext = Pick<TestContext, "annotate" | "bench">;

/**
 * Returns the function that runs a benchmark against the budget of its name. The benchmark reports its fastest sample and
 * its median with the budget, and fails when the fastest sample is above the budget. The fastest sample is the least
 * affected by the other work of a shared machine. A budget is in milliseconds.
 */
export function benchmarksOf<const TBudgets extends Readonly<Record<string, number>>>(budgets: TBudgets) {
  return async (
    { annotate, bench }: BenchmarkContext,
    name: keyof TBudgets & string,
    fn: () => unknown,
    options: RunOptions = BENCH_OPTIONS,
  ): Promise<void> => {
    const { latency } = await bench(name, fn).run(options);
    const budget = budgets[name] ?? Number.NaN;
    await annotate(
      `${name}: fastest ${latency.min.toFixed(3)} ms, median ${latency.p50.toFixed(3)} ms, budget ${budget} ms`,
    );
    expect(latency.min, `The fastest sample of "${name}" in milliseconds`).toBeLessThan(budget);
  };
}

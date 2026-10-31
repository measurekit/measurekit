import { expect, test } from "vitest";
import { cholesky, logistic, normalQuantile } from "../src/index.js";
import { benchmark } from "./budgets.js";

/** The evaluations of each function: a million points. */
const POINTS = 1_000_000;
/** The order of the matrix that is factored. */
const ORDER = 200;
/** The correlation between two neighbors of the matrix that is factored. */
const CORRELATION = 0.9;

// The loops index the arrays and call a function bound here: Vitest turns an imported name into a getter, which a million
// calls would time, and the iterator of a typed array costs several times as much as an index.
const quantile = normalQuantile;
const logisticFunction = logistic;

test("evaluates the normal quantile function at a million probabilities", async (context) => {
  const probabilities = Float64Array.from({ length: POINTS }, (_, i) => (i + 0.5) / POINTS);
  await benchmark(context, "numerics.normalQuantile", () => {
    let sum = 0;
    for (let i = 0; i < POINTS; i++) sum += quantile(probabilities[i] ?? 0);
    return sum;
  });
});

test("evaluates the logistic function at a million points", async (context) => {
  const points = Float64Array.from({ length: POINTS }, (_, i) => (12 * i) / POINTS - 6);
  await benchmark(context, "numerics.logistic", () => {
    let sum = 0;
    for (let i = 0; i < POINTS; i++) sum += logisticFunction(points[i] ?? 0);
    return sum;
  });
});

test("factors a correlation matrix of order 200", async (context) => {
  const matrix = new Float64Array(ORDER * ORDER);
  for (let i = 0; i < ORDER; i++)
    for (let j = 0; j < ORDER; j++) matrix[i * ORDER + j] = CORRELATION ** Math.abs(i - j);
  expect(cholesky(matrix, ORDER)).not.toBeNull();
  await benchmark(context, "numerics.cholesky", () => cholesky(matrix, ORDER));
});

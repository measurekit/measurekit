import { expect } from "vitest";

/** Expects `actual` within `tolerance` of `expected`, or `null` when `expected` is `null`. */
export function expectClose(actual: number | null, expected: number | null, tolerance: number, label = ""): void {
  if (expected === null) {
    expect(actual, label).toBeNull();
    return;
  }
  expect(actual, label).not.toBeNull();
  expect(Math.abs((actual ?? Number.NaN) - expected), label).toBeLessThanOrEqual(tolerance);
}

/** Expects two lists of numbers to have equal lengths and entries within `tolerance`. */
export function expectAllClose(
  actual: ArrayLike<number>,
  expected: ArrayLike<number>,
  tolerance: number,
  label = "",
): void {
  expect(actual.length, label).toBe(expected.length);
  for (let i = 0; i < expected.length; i++) {
    expectClose(actual[i] ?? null, expected[i] ?? null, tolerance, `${label}[${i}]`);
  }
}

/** Returns the predictions of a fit whose events all have an outcome, failing the test on a `null`. */
export function allPredicted(predictions: readonly (number | null)[]): number[] {
  return predictions.map((prediction) => defined(prediction));
}

/** Returns the value, failing the test when it is `undefined` or `null`. */
export function defined<T>(value: T | null | undefined): T {
  if (value === undefined || value === null) throw new Error(`Expected a value, found ${value}.`);
  return value;
}

/** Returns the Pearson correlation of paired values. */
export function correlation(x: readonly number[], y: readonly number[]): number {
  const mean = (values: readonly number[]): number => values.reduce((sum, value) => sum + value, 0) / values.length;
  const mx = mean(x);
  const my = mean(y);
  let xy = 0;
  let xx = 0;
  let yy = 0;
  x.forEach((value, i) => {
    const dx = value - mx;
    const dy = defined(y[i]) - my;
    xy += dx * dy;
    xx += dx * dx;
    yy += dy * dy;
  });
  return xy / Math.sqrt(xx * yy);
}

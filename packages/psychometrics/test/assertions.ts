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

/** Returns the value, failing the test when it is `undefined` or `null`. */
export function defined<T>(value: T | null | undefined): T {
  if (value === undefined || value === null) throw new Error(`Expected a value, found ${value}.`);
  return value;
}

/**
 * Expects `actual` to have the names of `expected`, in order, with each value, a number or a list, within `tolerance`
 * times `scale(expected value)` of the expected one.
 */
export function expectParameters(
  actual: unknown,
  expected: unknown,
  tolerance: number,
  label: string,
  scale: (value: number) => number = () => 1,
): void {
  const actualEntries = numberLists(actual);
  const expectedEntries = numberLists(expected);
  expect(
    actualEntries.map(([name]) => name),
    label,
  ).toEqual(expectedEntries.map(([name]) => name));
  expectedEntries.forEach(([name, values], index) => {
    const found = actualEntries[index]?.[1] ?? [];
    expect(found.length, `${label} ${name}`).toBe(values.length);
    values.forEach((value, j) => {
      const number = found[j];
      const target = typeof value === "number" ? value : null;
      const allowed = tolerance * scale(target ?? 0);
      expectClose(typeof number === "number" ? number : null, target, allowed, `${label} ${name}[${j}]`);
    });
  });
}

/** Returns an object's entries with each value as a list: a number as a list of one. */
function numberLists(value: unknown): [string, unknown[]][] {
  if (typeof value !== "object" || value === null) return [];
  return Object.entries(value).map(([name, entry]) => [name, Array.isArray(entry) ? entry : [entry]]);
}

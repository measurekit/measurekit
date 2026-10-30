import fc from "fast-check";
import { expect } from "vitest";
import {
  fourParameterLogistic,
  generalizedPartialCredit,
  gradedResponse,
  type IrtItem,
  type IrtModel,
  nominalResponse,
  oneParameterLogistic,
  partialCredit,
  ratingScale,
  type TestItem,
  threeParameterLogistic,
  twoParameterLogistic,
} from "../src/index.js";

/** Expects `actual` within `tolerance` of `expected`, or `null` when `expected` is `null`. */
export function expectClose(actual: number | null, expected: number | null, tolerance: number, label = ""): void {
  if (expected === null) {
    expect(actual, label).toBeNull();
    return;
  }
  expect(actual, label).not.toBeNull();
  expect(Math.abs((actual ?? Number.NaN) - expected), label).toBeLessThanOrEqual(tolerance);
}

/** Returns the mean of at least one value. */
export function mean(values: readonly number[]): number {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

/** Returns the value, failing the test when it is `undefined` or `null`. */
export function defined<T>(value: T | null | undefined): T {
  if (value === undefined || value === null) throw new Error(`Expected a value, found ${value}.`);
  return value;
}

/**
 * An item's parameters in a fixture: numbers and lists of numbers by name. TypeScript types the parameters of items of
 * different models in one JSON array with the names of the others as optional, hence `undefined`.
 */
export type FixtureParameters = Readonly<Record<string, number | readonly number[] | undefined>>;

/** The built-in models by the names that fixtures give them. */
export const MODELS: Readonly<Record<string, IrtModel>> = {
  "1pl": oneParameterLogistic,
  "2pl": twoParameterLogistic,
  "3pl": threeParameterLogistic,
  "4pl": fourParameterLogistic,
  grm: gradedResponse,
  gpcm: generalizedPartialCredit,
  pcm: partialCredit,
  rsm: ratingScale,
  nrm: nominalResponse,
};

/** Items of a fixture: a model name and the package's parameters. */
export type FixtureItems = readonly { readonly model: string; readonly parameters: FixtureParameters }[];

/** Returns the items of a fixture. */
export function fixtureItems(rows: FixtureItems): IrtItem[] {
  return rows.map(({ model, parameters }) => ({ model: defined(MODELS[model]), parameters }));
}

/** Returns a 2PL item with the id, `a` and `b`. */
export function item2pl(id: string, a: number, b: number): TestItem {
  return { id, model: twoParameterLogistic, parameters: { a, b, c: 0, d: 1 } };
}

/** Returns the entries of a record whose values are defined: JSON imports type each record with every key. */
export function definedEntries<T>(record: Readonly<Record<string, T | undefined>>): Record<string, T> {
  return Object.fromEntries(
    Object.entries(record).flatMap(([key, value]) => (value === undefined ? [] : [[key, value]])),
  );
}

/** Standard errors in a fixture by parameter name. */
export type FixtureErrors = Readonly<Record<string, number | undefined>>;

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

/** A slope of any sign, including 0. */
export const arbitrarySlope = fc.double({ min: -3, max: 4, noNaN: true });
/** A difficulty, threshold or intercept. */
export const arbitraryLocation = fc.double({ min: -4, max: 4, noNaN: true });
/** One to four thresholds in any order. */
export const arbitraryLocations = fc.array(arbitraryLocation, { minLength: 1, maxLength: 4 });

/** Items of every model, with parameters in range: the 4PL, GRM, GPCM, PCM, RSM and NRM. */
export const arbitraryItem: fc.Arbitrary<IrtItem> = fc.oneof(
  fc
    .record({
      a: arbitrarySlope,
      b: arbitraryLocation,
      c: fc.double({ min: 0, max: 0.4, noNaN: true }),
      d: fc.double({ min: 0.6, max: 1, noNaN: true }),
    })
    .map((parameters) => ({ model: fourParameterLogistic, parameters })),
  fc
    .record({
      a: fc.double({ min: 0.2, max: 3, noNaN: true }),
      positive: fc.boolean(),
      first: arbitraryLocation,
      gaps: fc.array(fc.double({ min: 0.05, max: 2, noNaN: true }), { maxLength: 3 }),
    })
    .map(({ a, positive, first, gaps }) => {
      const b = [first];
      for (const gap of gaps) b.push((b.at(-1) ?? 0) + gap);
      return { model: gradedResponse, parameters: positive ? { a, b } : { a: -a, b: b.reverse() } };
    }),
  fc
    .record({ a: arbitrarySlope, b: arbitraryLocations })
    .map((parameters) => ({ model: generalizedPartialCredit, parameters })),
  fc.record({ a: arbitrarySlope, b: arbitraryLocations }).map((parameters) => ({ model: partialCredit, parameters })),
  fc
    .record({ a: arbitrarySlope, b: arbitraryLocation, steps: arbitraryLocations })
    .map((parameters) => ({ model: ratingScale, parameters })),
  fc
    .integer({ min: 2, max: 5 })
    .chain((count) =>
      fc.record({
        a: fc.array(arbitrarySlope, { minLength: count, maxLength: count }),
        c: fc.array(arbitraryLocation, { minLength: count, maxLength: count }),
      }),
    )
    .map((parameters) => ({ model: nominalResponse, parameters })),
);

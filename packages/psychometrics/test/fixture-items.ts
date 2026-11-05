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
} from "../src/irt/index.js";
import { defined } from "./assertions.js";

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

/** Items of a fixture: a model name and the item parameters. */
export type FixtureItems = readonly { readonly model: string; readonly parameters: FixtureParameters }[];

/** Returns the items of a fixture. */
export function fixtureItems(rows: FixtureItems): IrtItem[] {
  return rows.map(({ model, parameters }) => ({ model: defined(MODELS[model]), parameters }));
}

/** Returns a 2PL item with the id, `a` and `b`. */
export function item2pl(id: string, a: number, b: number): TestItem {
  return { id, model: twoParameterLogistic, parameters: { a, b, c: 0, d: 1 } };
}

/**
 * Returns the entries of a record whose values are defined: a fixture's type makes a key optional that only some of
 * its cases hold.
 */
export function definedEntries<T>(record: Readonly<Record<string, T | undefined>>): Record<string, T> {
  return Object.fromEntries(
    Object.entries(record).flatMap(([key, value]) => (value === undefined ? [] : [[key, value]])),
  );
}

/** Standard errors in a fixture by parameter name. */
export type FixtureErrors = Readonly<Record<string, number | undefined>>;

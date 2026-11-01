import { checkRecord, DataError, describe, isNonEmptyString, isRecord, isUnitInterval } from "@measurekit/core";
import type { Blim, Constraints, RateConstraints } from "./estimation.js";

/** Rates of items held at given values, by item. */
export interface FixedRates {
  /** The lucky guess of each item, a probability from 0 to 1. */
  readonly guess?: Readonly<Record<string, number>>;
  /** The careless error of each item, a probability from 0 to 1. */
  readonly slip?: Readonly<Record<string, number>>;
}

/** Groups of items that share one estimate of a rate. */
export interface EqualRates {
  /** Groups of at least 2 items that share a lucky guess. */
  readonly guess?: readonly (readonly string[])[];
  /** Groups of at least 2 items that share a careless error. */
  readonly slip?: readonly (readonly string[])[];
}

type RateName = "guess" | "slip";
const RATE_NAMES: readonly RateName[] = ["guess", "slip"];

/** The restrictions of one kind of rate, checked. */
interface RateRestrictions {
  readonly fixed: ReadonlyMap<string, number>;
  readonly groups: readonly (readonly string[])[];
}

/** The restrictions on the rates of a model's items, checked, by item id. */
export type Restrictions = Readonly<Record<RateName, RateRestrictions>>;

/**
 * Checks the options `fixed` and `equal`, which may come from JavaScript or JSON, and returns them.
 *
 * @throws `DataError` when a rate is not a probability, a group has fewer than 2 distinct items, or an item has its
 * rate fixed and shared, or is in two groups.
 */
export function checkRestrictions(fixed: FixedRates, equal: EqualRates): Restrictions {
  checkRecord(fixed, "The option fixed must be an object of fixed rates by guess and slip");
  checkRecord(equal, "The option equal must be an object of groups of items by guess and slip");
  const { guess: fixedGuess = {}, slip: fixedSlip = {} } = fixed;
  const { guess: equalGuess = [], slip: equalSlip = [] } = equal;
  return { guess: restrictionsOf("guess", fixedGuess, equalGuess), slip: restrictionsOf("slip", fixedSlip, equalSlip) };
}

function restrictionsOf(name: RateName, rates: unknown, groups: unknown): RateRestrictions {
  const fixed = fixedOf(name, rates);
  const equal = groupsOf(name, groups);
  const shared = new Set<string>();
  for (const id of equal.flat()) {
    if (shared.has(id)) throw new DataError(`Item "${id}" is in two groups of the option equal.${name}.`);
    if (fixed.has(id))
      throw new DataError(`Item "${id}" has a fixed ${name} and is in a group of the option equal.${name}.`);
    shared.add(id);
  }
  return { fixed, groups: equal };
}

function fixedOf(name: RateName, rates: unknown): Map<string, number> {
  if (!isRecord(rates)) {
    throw new DataError(`The option fixed.${name} must be an object of rates by item, found ${describe(rates)}.`);
  }
  const held = new Map<string, number>();
  for (const [id, rate] of Object.entries(rates)) {
    if (!isUnitInterval(rate)) {
      throw new DataError(
        `The option fixed.${name} needs a probability from 0 to 1 for item "${id}", found ${describe(rate)}.`,
      );
    }
    held.set(id, rate);
  }
  return held;
}

function groupsOf(name: RateName, groups: unknown): string[][] {
  if (!Array.isArray(groups)) {
    throw new DataError(`The option equal.${name} must be a list of groups of items, found ${describe(groups)}.`);
  }
  return groups.map((group: unknown) => {
    if (
      !(
        Array.isArray(group) &&
        group.length >= 2 &&
        group.every(isNonEmptyString) &&
        new Set(group).size === group.length
      )
    ) {
      throw new DataError(
        `Each group of the option equal.${name} needs at least 2 distinct item ids, found ${describe(group)}.`,
      );
    }
    return [...group];
  });
}

/**
 * Returns the constraints of a fit over `items`, and writes the fixed rates into `start`. `unknown` says why an item
 * that the restrictions name is not one of them, for the error.
 *
 * @throws `DataError` when a restriction names an item that is not among `items`.
 */
export function constraintsOf(
  restrictions: Restrictions,
  items: readonly string[],
  start: Blim,
  unknown: string,
): Constraints {
  const index = new Map(items.map((id, i) => [id, i]));
  const constraints = (name: RateName): RateConstraints => {
    const held = new Uint8Array(items.length);
    const group = Int32Array.from(items, (_, i) => i);
    const position = (id: string, option: string): number => {
      const i = index.get(id);
      if (i === undefined) throw new DataError(`The option ${option}.${name} names item "${id}", which ${unknown}.`);
      return i;
    };
    const values = start[name];
    for (const [id, rate] of restrictions[name].fixed) {
      const i = position(id, "fixed");
      held[i] = 1;
      values[i] = rate;
    }
    for (const members of restrictions[name].groups) {
      const positions = members.map((id) => position(id, "equal"));
      const first = Math.min(...positions);
      for (const i of positions) group[i] = first;
    }
    return { held, group };
  };
  return { guess: constraints("guess"), slip: constraints("slip") };
}

/** Returns whether any rate is shared, which the local identifiability of a model has no reference for. */
export function sharesRates(restrictions: Restrictions): boolean {
  return RATE_NAMES.some((name) => restrictions[name].groups.length > 0);
}

import {
  checkArray,
  checkFinite,
  checkIntegerFrom,
  checkOptions,
  checkPositive,
  checkPositiveInteger,
  checkRandom,
  checkRecord,
  DataError,
  describe,
  type Random,
  shuffle,
} from "@measurekit/core";
import { indexItems } from "../irt/internal.js";
import { type ContentConstraint, poolAreas, readConstraint } from "./content-constraints.js";
import { enemyIndices } from "./enemies.js";
import type { PoolItem } from "./items.js";
import { poolTestlets } from "./testlets.js";

/**
 * Two projected deviations that differ by less than this share of the smaller are one, so that rounding in the sums does
 * not decide which of two units a form takes; the first in pool order does.
 */
const DEVIATION_TIE = 1e-12;

/** A point of a form's target information function. */
export interface InformationTarget {
  /** The theta. */
  readonly theta: number;
  /** The test information the form should have at `theta`. */
  readonly information: number;
  /**
   * The weight of a deviation of one unit of information from the target.
   *
   * @defaultValue `1`
   */
  readonly weight?: number;
}

/** Options of {@link assembleForms}. */
export interface FormAssemblyOptions {
  /** The item pool. Items that share a `testlet` go on a form together. */
  readonly pool: readonly PoolItem[];
  /** The number of items of each form. */
  readonly length: number;
  /**
   * Content constraints, on the pool items' `contentAreas`.
   *
   * @defaultValue `[]`
   */
  readonly content?: readonly ContentConstraint[];
  /**
   * The target information function, at one or more thetas.
   *
   * @defaultValue `[]`
   */
  readonly target?: readonly InformationTarget[];
  /**
   * Enemy sets: ids of items of which a form holds at most one.
   *
   * @defaultValue none
   */
  readonly enemies?: readonly (readonly string[])[];
  /**
   * The number of forms, from 1.
   *
   * @defaultValue `1`
   */
  readonly forms?: number;
  /**
   * The most items two forms may share, from 0.
   *
   * @defaultValue `0`
   */
  readonly overlap?: number;
  /**
   * Orders the pool at random before assembly, which settles ties.
   *
   * @defaultValue none: ties go to the first in pool order
   */
  readonly random?: Random;
}

/** A form assembled from a pool. */
export interface AssembledForm {
  /** The form's items, in the order of selection. A form is short when it has fewer items than `length`. */
  readonly items: readonly PoolItem[];
  /** The number of the form's items in each constrained content area. */
  readonly contentCounts: Readonly<Record<string, number>>;
  /** The form's test information at the theta of each target point, in their order. */
  readonly information: readonly number[];
}

/** A property the method projects: the values of the items or the units, the bounds and the weight of a deviation from them. */
interface Property {
  readonly values: Float64Array;
  readonly lower: number;
  readonly upper: number;
  readonly weight: number;
}

/** The pool as the units that a form takes whole: an item alone, or a testlet. */
interface Units {
  /** The pool indices of each unit's items, in pool order. */
  readonly members: readonly (readonly number[])[];
  /** Whether two items of the unit are enemies, so that a form cannot take it. */
  readonly divided: readonly boolean[];
}

/** What a form has so far. */
interface FormState {
  /** The pool indices of its items, in the order of selection. */
  readonly chosen: number[];
  /** Each property's sum over its items. */
  readonly totals: Float64Array;
  /** 1 for each pool item on the form. */
  readonly held: Uint8Array;
  /** For each pool item, the number of its enemies on the form. */
  readonly enemiesHeld: Int32Array;
  /** Whether the form can take no more units. */
  done: boolean;
}

/** The forms in assembly and what they share. */
interface Assembly {
  readonly forms: readonly FormState[];
  /** The number of items that each pair of forms shares, by `first * forms + second`. */
  readonly shared: Int32Array;
  /** The forms that hold each pool item. */
  readonly holders: number[][];
}

/**
 * Assembles fixed forms from a pool by the weighted deviations method of Swanson and Stocking, a heuristic without a
 * solver, one form or several that share at most `overlap` items.
 *
 * @remarks
 * A unit is an item, or the items of a testlet, which a form takes whole. The forms take units in rotation, one unit at a
 * time: a round lets each form take one in order, the next round in reverse, so that no form always chooses first. A unit
 * is available to a form when it fits the places left, none of its items is on the form or an enemy of one on the form,
 * and the form would share at most `overlap` items with any other form. A form chooses, among the units available to it,
 * the one with the smallest weighted sum of the deviations of its projected properties from their bounds, the first in pool
 * order on ties, which are deviations that differ by less than a share of 1e-12. It projects a property, a content area's count or the information at a target theta, as the value of the
 * units chosen, plus the unit's, plus the mean value per item of the other units available times the places left. A
 * target's bounds are its information; a content area's value is 1 for each item with the area among its `contentAreas`.
 *
 * A form is short, with fewer than `length` items, where the pool, the enemies or `overlap` leave no unit available to it.
 * Constraints that cannot all be met give the forms with the smallest deviations the method finds; a result reports
 * the counts and the information to check, and the method cannot prove that a blueprint is infeasible. Where a form
 * cannot be assembled as a solver would, check the counts against the bounds.
 *
 * @param options - The pool, the length, the content bounds and target, the enemy sets, the number of forms and the
 *   overlap; each option has its default.
 * @returns The forms, in the order of the rotation's first round.
 * @throws `DataError` when an item's id or parameters are invalid, ids repeat, the length is not an integer from 1 to
 * the pool size, `forms` is not a positive integer, `overlap` is not an integer from 0, a constraint has bounds that are
 * not finite numbers with minimum ≤ maximum or a weight that is not positive, an item's `contentAreas` is not an array of
 * distinct non-empty strings, a target has a theta or information that is not finite or a weight that is not positive,
 * an enemy set does not name two or more distinct pool items, or `random` is not a random number generator.
 * @see {@link https://doi.org/10.1177/014662169301700205 | Swanson and Stocking (1993)}
 */
export function assembleForms(options: FormAssemblyOptions): readonly AssembledForm[] {
  checkOptions(options);
  const { pool, length, content = [], target = [], enemies = [], forms = 1, overlap = 0, random } = options;
  const indices = indexItems(pool);
  if (!(Number.isSafeInteger(length) && length >= 1 && length <= pool.length)) {
    throw new DataError(
      `The option length must be an integer from 1 to the pool size ${pool.length}, found ${describe(length)}.`,
    );
  }
  checkArray(content, "The option content must be an array of content constraints");
  checkArray(target, "The option target must be an array of information targets");
  checkPositiveInteger(forms, "The option forms");
  checkIntegerFrom(overlap, "The option overlap", 0);
  if (random !== undefined) checkRandom(random);
  const conflicts = enemyIndices(pool, indices, enemies);
  const itemAreas = poolAreas(content.map(({ area }) => area))(pool);
  const properties = [
    ...content.map((constraint, k) => contentProperty(itemAreas, k, constraint)),
    ...target.map((point) => informationProperty(pool, point)),
  ];
  const units = poolUnits(pool, conflicts, random);
  const byUnit = properties.map((property) => unitProperty(property, units));
  const assembly = startAssembly(pool.length, forms, properties.length);
  const sizes = units.members.map((members) => members.length);
  const free = new Int32Array(forms);
  const step = (form: number): boolean =>
    assembleStep({ units, sizes, properties: byUnit, conflicts, length, overlap, assembly, free }, form);
  rotate(forms, step);
  return assembly.forms.map(({ chosen, totals }) => ({
    items: chosen.flatMap((index) => pool[index] ?? []),
    contentCounts: Object.fromEntries(content.map(({ area }, k) => [area, totals[k] ?? 0])),
    information: target.map((_, t) => totals[content.length + t] ?? 0),
  }));
}

/** Calls `step` for each form in turn, the forms in order and then in reverse, until a round lets none take a unit. */
function rotate(forms: number, step: (form: number) => boolean): void {
  for (let round = 0; ; round++) {
    const order = Array.from({ length: forms }, (_, k) => (round % 2 === 0 ? k : forms - 1 - k));
    let taken = false;
    for (const form of order) taken = step(form) || taken;
    if (!taken) return;
  }
}

function startAssembly(poolSize: number, forms: number, propertyCount: number): Assembly {
  return {
    forms: Array.from({ length: forms }, () => ({
      chosen: [],
      totals: new Float64Array(propertyCount),
      held: new Uint8Array(poolSize),
      enemiesHeld: new Int32Array(poolSize),
      done: false,
    })),
    shared: new Int32Array(forms * forms),
    holders: Array.from({ length: poolSize }, () => []),
  };
}

/**
 * Returns the units of a pool in the order that settles ties: pool order, or a random permutation of the items, a unit
 * taking the place of its first item to come. The items of a testlet stay in pool order.
 */
function poolUnits(
  pool: readonly PoolItem[],
  conflicts: readonly (readonly number[])[],
  random: Random | undefined,
): Units {
  const order = Int32Array.from(pool.keys());
  if (random !== undefined) shuffle(order, random);
  const testlets = poolTestlets(pool);
  const seen = new Uint8Array(testlets?.members.length ?? 0);
  const members: (readonly number[])[] = [];
  for (const index of order) {
    const testlet = testlets?.of[index] ?? -1;
    if (testlet < 0) {
      members.push([index]);
    } else if (seen[testlet] !== 1) {
      seen[testlet] = 1;
      members.push(testlets?.members[testlet] ?? []);
    }
  }
  const divided = members.map((items) =>
    items.some((index) => (conflicts[index] ?? []).some((enemy) => items.includes(enemy))),
  );
  return { members, divided };
}

/** Returns a property over the units: each one's value is the sum of its items'. */
function unitProperty(property: Property, { members }: Units): Property {
  const values = Float64Array.from(members, (items) =>
    items.reduce((sum, index) => sum + (property.values[index] ?? 0), 0),
  );
  return { ...property, values };
}

/** Everything a step of the assembly reads. */
interface StepContext {
  readonly units: Units;
  readonly sizes: readonly number[];
  readonly properties: readonly Property[];
  readonly conflicts: readonly (readonly number[])[];
  readonly length: number;
  readonly overlap: number;
  readonly assembly: Assembly;
  /** A scratch array of the forms, for counting the items that a unit would share with each. */
  readonly free: Int32Array;
}

/** Lets a form take the unit with the smallest projected deviation, if one is available, and returns whether it did. */
function assembleStep(context: StepContext, formIndex: number): boolean {
  const { units, sizes, properties, assembly, length } = context;
  const form = assembly.forms[formIndex];
  if (form === undefined || form.done) return false;
  const places = length - form.chosen.length;
  const candidates: number[] = [];
  for (let unit = 0; unit < units.members.length; unit++) {
    if (isAvailable(context, formIndex, unit, places)) candidates.push(unit);
  }
  if (candidates.length === 0) {
    form.done = true;
    return false;
  }
  const sums = new Float64Array(properties.length);
  let remaining = 0;
  for (const unit of candidates) {
    remaining += sizes[unit] ?? 0;
    properties.forEach(({ values }, p) => {
      sums[p] = (sums[p] ?? 0) + (values[unit] ?? 0);
    });
  }
  let best = -1;
  let bestDeviation = Number.POSITIVE_INFINITY;
  for (const unit of candidates) {
    const size = sizes[unit] ?? 0;
    const deviation = projectedDeviation(
      properties,
      { sums, totals: form.totals, remaining, left: places - size },
      unit,
      size,
    );
    if (deviation < bestDeviation * (1 - DEVIATION_TIE)) {
      best = unit;
      bestDeviation = deviation;
    }
  }
  take(context, formIndex, best);
  return true;
}

/** Returns whether a unit is available to a form: it fits, none of its items is held or in conflict, and overlap is kept. */
function isAvailable(context: StepContext, formIndex: number, unit: number, places: number): boolean {
  const { units, sizes, assembly, overlap } = context;
  const form = assembly.forms[formIndex];
  const members = units.members[unit] ?? [];
  if (form === undefined || (sizes[unit] ?? 0) > places || units.divided[unit] === true) return false;
  for (const index of members) {
    if (form.held[index] === 1 || (form.enemiesHeld[index] ?? 0) > 0) return false;
  }
  return sharesWithin(context, formIndex, members, overlap);
}

/** Returns whether a form that took the items would share at most `overlap` items with each other form. */
function sharesWithin(context: StepContext, formIndex: number, members: readonly number[], overlap: number): boolean {
  const { assembly, free } = context;
  const count = assembly.forms.length;
  for (const index of members) {
    for (const other of assembly.holders[index] ?? []) free[other] = (free[other] ?? 0) + 1;
  }
  let within = true;
  for (let other = 0; other < count; other++) {
    within = within && (free[other] ?? 0) + (assembly.shared[formIndex * count + other] ?? 0) <= overlap;
    free[other] = 0;
  }
  return within;
}

/** Puts a unit on a form: its items, their properties, the enemies they close and the items now shared. */
function take(context: StepContext, formIndex: number, unit: number): void {
  const { units, properties, conflicts, assembly } = context;
  const form = assembly.forms[formIndex];
  if (form === undefined) return;
  const count = assembly.forms.length;
  for (const index of units.members[unit] ?? []) {
    form.chosen.push(index);
    form.held[index] = 1;
    for (const enemy of conflicts[index] ?? []) form.enemiesHeld[enemy] = (form.enemiesHeld[enemy] ?? 0) + 1;
    for (const other of assembly.holders[index] ?? []) {
      assembly.shared[formIndex * count + other] = (assembly.shared[formIndex * count + other] ?? 0) + 1;
      assembly.shared[other * count + formIndex] = (assembly.shared[other * count + formIndex] ?? 0) + 1;
    }
    assembly.holders[index]?.push(formIndex);
  }
  properties.forEach(({ values }, p) => {
    form.totals[p] = (form.totals[p] ?? 0) + (values[unit] ?? 0);
  });
}

/** The state of an assembly before a unit is taken. */
interface Progress {
  /** Each property's sum over the units available. */
  readonly sums: Float64Array;
  /** Each property's sum over the items on the form. */
  readonly totals: Float64Array;
  /** The number of items in the units available. */
  readonly remaining: number;
  /** The places left after the unit. */
  readonly left: number;
}

/** Returns the weighted sum of the deviations of the form's projected properties with a unit of `size` items. */
function projectedDeviation(properties: readonly Property[], progress: Progress, unit: number, size: number): number {
  const { sums, totals, remaining, left } = progress;
  let deviation = 0;
  properties.forEach(({ values, lower, upper, weight }, p) => {
    const value = values[unit] ?? 0;
    const others = remaining > size ? ((sums[p] ?? 0) - value) / (remaining - size) : 0;
    const projection = (totals[p] ?? 0) + value + left * others;
    deviation += weight * (Math.max(0, lower - projection) + Math.max(0, projection - upper));
  });
  return deviation;
}

function contentProperty(
  itemAreas: readonly (readonly number[])[],
  area: number,
  constraint: ContentConstraint,
): Property {
  const { minimum, maximum, weight } = readConstraint(constraint);
  return {
    values: Float64Array.from(itemAreas, (areas) => (areas.includes(area) ? 1 : 0)),
    lower: minimum,
    upper: maximum,
    weight,
  };
}

function informationProperty(pool: readonly PoolItem[], point: InformationTarget): Property {
  checkRecord(point, "An information target must be an object { theta, information }");
  const { theta, information, weight = 1 } = point;
  checkFinite(theta, "The theta of an information target");
  checkFinite(information, `The information of the target at theta ${theta}`);
  checkPositive(weight, `The weight of the target at theta ${theta}`);
  return {
    values: Float64Array.from(pool, (item) => item.model.information(item.parameters, theta)),
    lower: information,
    upper: information,
    weight,
  };
}

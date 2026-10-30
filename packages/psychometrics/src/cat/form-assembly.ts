import {
  checkArray,
  checkFinite,
  checkOptions,
  checkPositive,
  checkRecord,
  DataError,
  describe,
} from "@measurekit/core";
import { indexItems } from "../irt/items.js";
import type { PoolItem } from "./items.js";

/** A content constraint of a form: the number of its items in a content area. */
export interface ContentConstraint {
  /** The content area, as the pool items' `contentArea`. */
  readonly area: string;
  /** The fewest items of the area. */
  readonly minimum: number;
  /** The most items of the area. */
  readonly maximum: number;
  /**
   * The weight of a deviation of one item from the bounds.
   *
   * @defaultValue `1`
   */
  readonly weight?: number;
}

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

/** Options of {@link assembleForm}. */
export interface FormAssemblyOptions {
  /** The item pool. */
  readonly pool: readonly PoolItem[];
  /** The number of items of the form. */
  readonly length: number;
  /**
   * Content constraints.
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
}

/** A form assembled from a pool. */
export interface AssembledForm {
  /** The form's items, in the order of selection. */
  readonly items: readonly PoolItem[];
  /** The number of the form's items in each constrained content area. */
  readonly contentCounts: Readonly<Record<string, number>>;
  /** The form's test information at the theta of each target point, in their order. */
  readonly information: readonly number[];
}

/** A property the method projects: the items' values, the bounds and the weight of a deviation from them. */
interface Property {
  readonly values: Float64Array;
  readonly lower: number;
  readonly upper: number;
  readonly weight: number;
}

/**
 * Assembles a fixed form by the weighted deviations method of Swanson and Stocking, a heuristic without a solver.
 *
 * @remarks
 * It selects the items one at a time. For each candidate it projects every property of the form, a content area's count
 * or the information at a target theta, as the value of the items selected, plus the candidate's, plus the mean of the
 * other remaining items times the places left; and it selects the candidate whose projection has the smallest weighted
 * sum of deviations from the bounds, the first in pool order on ties. A target's bounds are its information.
 * Constraints that cannot all be met give the form with the smallest deviations the method finds; the result reports
 * the counts and the information to check.
 *
 * @throws `DataError` when an item's id or parameters are invalid, ids repeat, the length is not an integer from 1 to
 * the pool size, a constraint has bounds that are not finite numbers with minimum ≤ maximum or a weight that is not
 * positive, or a target has a theta or information that is not finite or a weight that is not positive.
 * @see {@link https://doi.org/10.1177/014662169301700205 | Swanson and Stocking (1993)}
 */
export function assembleForm(options: FormAssemblyOptions): AssembledForm {
  checkOptions(options);
  const { pool, length, content = [], target = [] } = options;
  indexItems(pool);
  if (!(Number.isSafeInteger(length) && length >= 1 && length <= pool.length)) {
    throw new DataError(
      `The option length must be an integer from 1 to the pool size ${pool.length}, found ${describe(length)}.`,
    );
  }
  checkArray(content, "The option content must be an array of content constraints");
  checkArray(target, "The option target must be an array of information targets");
  const properties = [
    ...content.map((constraint) => contentProperty(pool, constraint)),
    ...target.map((point) => informationProperty(pool, point)),
  ];
  const sums = properties.map(({ values }) => values.reduce((sum, value) => sum + value, 0));
  const totals = properties.map(() => 0);
  const available = new Uint8Array(pool.length).fill(1);
  const selected: number[] = [];
  for (let place = 1; place <= length; place++) {
    const remaining = pool.length - selected.length;
    const left = length - place;
    let best = -1;
    let bestDeviation = Number.POSITIVE_INFINITY;
    for (let i = 0; i < pool.length; i++) {
      if (available[i] !== 1) continue;
      const deviation = projectedDeviation(properties, { sums, totals, remaining, left }, i);
      if (deviation < bestDeviation) {
        best = i;
        bestDeviation = deviation;
      }
    }
    selected.push(best);
    available[best] = 0;
    properties.forEach(({ values }, p) => {
      sums[p] = (sums[p] ?? 0) - (values[best] ?? 0);
      totals[p] = (totals[p] ?? 0) + (values[best] ?? 0);
    });
  }
  return {
    items: selected.flatMap((index) => pool[index] ?? []),
    contentCounts: Object.fromEntries(content.map(({ area }, p) => [area, totals[p] ?? 0])),
    information: target.map((_, t) => totals[content.length + t] ?? 0),
  };
}

/** The state of an assembly before a place is filled. */
interface Progress {
  /** Each property's sum over the items not selected. */
  readonly sums: readonly number[];
  /** Each property's sum over the items selected. */
  readonly totals: readonly number[];
  /** The number of items not selected. */
  readonly remaining: number;
  /** The places left after this one. */
  readonly left: number;
}

/** Returns the weighted sum of the deviations of the form's projected properties with candidate i. */
function projectedDeviation(properties: readonly Property[], progress: Progress, i: number): number {
  const { sums, totals, remaining, left } = progress;
  let deviation = 0;
  properties.forEach(({ values, lower, upper, weight }, p) => {
    const value = values[i] ?? 0;
    const others = remaining > 1 ? ((sums[p] ?? 0) - value) / (remaining - 1) : 0;
    const projection = (totals[p] ?? 0) + value + left * others;
    deviation += weight * (Math.max(0, lower - projection) + Math.max(0, projection - upper));
  });
  return deviation;
}

function contentProperty(pool: readonly PoolItem[], constraint: ContentConstraint): Property {
  checkRecord(constraint, "A content constraint must be an object { area, minimum, maximum }");
  const { area, minimum, maximum, weight = 1 } = constraint;
  if (typeof area !== "string") {
    throw new DataError(`A content constraint needs a string area, found ${describe(area)}.`);
  }
  checkFinite(minimum, `The minimum of the constraint on "${area}"`);
  checkFinite(maximum, `The maximum of the constraint on "${area}"`);
  if (!(minimum <= maximum)) {
    throw new DataError(
      `The constraint on "${area}" needs the minimum at most the maximum, found ${minimum} and ${maximum}.`,
    );
  }
  checkPositive(weight, `The weight of the constraint on "${area}"`);
  return {
    values: Float64Array.from(pool, (item) => (item.contentArea === area ? 1 : 0)),
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

import { checkMethod, checkPositive, checkPositiveInteger, DataError } from "@measurekit/core";
import type { AbilityEstimate } from "../irt/internal.js";
import { type ContentConstraint, countAreas, poolAreas, readBounds } from "./content-constraints.js";
import type { PoolControls, PoolItem } from "./items.js";
import type { AdaptiveTestContext } from "./selection.js";

/**
 * A rule that ends an adaptive test. The test checks it before each item, also the first; `context.eligible` marks the
 * items that may still be administered. Built-in rules implement this interface as a custom rule does, and
 * {@link allOf} and {@link anyOf} combine rules.
 *
 * @typeParam TItem - The items of the pool.
 * @typeParam TEstimate - The estimate of what the test measures.
 */
export interface StoppingRule<TItem extends PoolControls = PoolItem, TEstimate = AbilityEstimate> {
  /** Returns whether the test ends. */
  stops(context: AdaptiveTestContext<TItem, TEstimate>): boolean;
}

/**
 * An estimate with the precision {@link standardErrorThreshold} reads: a `standardError`, as an {@link AbilityEstimate}
 * has, or a `standardErrors` for each dimension. The rule never stops a test whose estimate has neither.
 */
export interface PrecisionEstimate {
  /** The standard error of a one-dimensional estimate. */
  readonly standardError?: number;
  /** The standard error of each dimension of a multidimensional estimate. */
  readonly standardErrors?: readonly number[];
}

/**
 * Returns a rule that stops once `count` items have been administered, catR's `length`: alone or in {@link anyOf} a
 * maximum length, in {@link allOf} a minimum length.
 *
 * @param count - The number of items, a positive integer.
 * @returns The stopping rule.
 * @throws `DataError` when `count` is not a positive integer.
 */
export function testLength<TItem extends PoolControls, TEstimate>(count: number): StoppingRule<TItem, TEstimate> {
  checkPositiveInteger(count, "A test length");
  return { stops: ({ administered }) => administered.length >= count };
}

/**
 * Returns a rule that stops once the standard error of the estimate is at most `threshold`, catR's `precision`: of an
 * estimate with `standardErrors`, once every one is.
 *
 * @param threshold - The largest standard error, above 0.
 * @returns The stopping rule.
 * @throws `DataError` when `threshold` is not a positive finite number.
 */
export function standardErrorThreshold<TItem extends PoolControls, TEstimate extends PrecisionEstimate>(
  threshold: number,
): StoppingRule<TItem, TEstimate> {
  checkPositive(threshold, "A standard error threshold");
  return {
    stops: ({ estimate: { standardError, standardErrors } }) =>
      standardErrors === undefined
        ? (standardError ?? Number.POSITIVE_INFINITY) <= threshold
        : standardErrors.every((error) => error <= threshold),
  };
}

/**
 * Returns a rule that stops once no eligible item has information above `threshold` at the current estimate, catR's
 * `minInfo`; also when no item is eligible.
 *
 * @param threshold - The information below which no item is worth giving, above 0.
 * @returns The stopping rule.
 * @throws `DataError` when `threshold` is not a positive finite number.
 */
export function informationThreshold(threshold: number): StoppingRule {
  checkPositive(threshold, "An information threshold");
  return {
    stops: ({ pool, eligible, estimate }) =>
      pool.every(
        (item, index) => eligible[index] !== 1 || item.model.information(item.parameters, estimate.theta) <= threshold,
      ),
  };
}

/**
 * Returns a rule that stops once the administered items meet every minimum of the content constraints, so that a stop
 * on precision, composed with {@link allOf}, never comes before the blueprint's minimums are met. Alone it stops a test
 * whose constraints have no minimum at once.
 *
 * @remarks
 * Areas are counted by the pool array, so a pool must not change after its first use.
 *
 * @param constraints - The areas with their minimums and maximums.
 * @returns The stopping rule.
 * @throws `DataError` when the constraints are invalid or on one area twice, or a bound is not an integer from 0; from
 * the rule, when an item's `contentAreas` is not an array of distinct non-empty strings.
 */
export function contentMinimumsMet<TItem extends PoolControls, TEstimate>(
  constraints: readonly ContentConstraint[],
): StoppingRule<TItem, TEstimate> {
  const { areas, minimums } = readBounds(constraints);
  const itemAreasOf = poolAreas(areas);
  return {
    stops: ({ pool, administered }) => {
      const counts = countAreas(administered, itemAreasOf(pool), areas.length);
      return minimums.every((minimum, k) => (counts[k] ?? 0) >= minimum);
    },
  };
}

/**
 * Returns a rule that stops when every one of `rules` stops.
 *
 * @param rules - The rules, at least one.
 * @returns The stopping rule.
 * @throws `DataError` without rules or when one is not a stopping rule.
 */
export function allOf<TItem extends PoolControls, TEstimate>(
  ...rules: readonly StoppingRule<TItem, TEstimate>[]
): StoppingRule<TItem, TEstimate> {
  checkRules(rules, "allOf");
  return { stops: (context) => rules.every((rule) => rule.stops(context)) };
}

/**
 * Returns a rule that stops when any one of `rules` stops.
 *
 * @param rules - The rules, at least one.
 * @returns The stopping rule.
 * @throws `DataError` without rules or when one is not a stopping rule.
 */
export function anyOf<TItem extends PoolControls, TEstimate>(
  ...rules: readonly StoppingRule<TItem, TEstimate>[]
): StoppingRule<TItem, TEstimate> {
  checkRules(rules, "anyOf");
  return { stops: (context) => rules.some((rule) => rule.stops(context)) };
}

function checkRules(rules: readonly unknown[], name: string): void {
  if (rules.length === 0) throw new DataError(`${name} needs at least one stopping rule.`);
  for (const rule of rules) checkMethod(rule, "stops", `${name} needs stopping rules, such as testLength(20)`);
}

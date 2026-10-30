import { checkMethod, checkPositive, checkPositiveInteger, DataError } from "@measurekit/core";
import type { AdaptiveTestContext } from "./selection.js";

/**
 * A rule that ends an adaptive test. The test checks it before each item, also the first; `context.eligible` marks the
 * items that may still be administered. Built-in rules implement this interface as a custom rule does, and
 * {@link allOf} and {@link anyOf} combine rules.
 */
export interface StoppingRule {
  /** Returns whether the test ends. */
  stops(context: AdaptiveTestContext): boolean;
}

/**
 * Returns a rule that stops once `count` items have been administered, catR's `length`: alone or in {@link anyOf} a
 * maximum length, in {@link allOf} a minimum length.
 *
 * @throws `DataError` when `count` is not a positive integer.
 */
export function testLength(count: number): StoppingRule {
  checkPositiveInteger(count, "A test length");
  return { stops: ({ administered }) => administered.length >= count };
}

/**
 * Returns a rule that stops once the standard error of the estimate is at most `threshold`, catR's `precision`.
 *
 * @throws `DataError` when `threshold` is not a positive finite number.
 */
export function standardErrorThreshold(threshold: number): StoppingRule {
  checkPositive(threshold, "A standard error threshold");
  return { stops: ({ estimate }) => estimate.standardError <= threshold };
}

/**
 * Returns a rule that stops once no eligible item has information above `threshold` at the current estimate, catR's
 * `minInfo`; also when no item is eligible.
 *
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
 * Returns a rule that stops when every one of `rules` stops.
 *
 * @throws `DataError` without rules or when one is not a stopping rule.
 */
export function allOf(...rules: readonly StoppingRule[]): StoppingRule {
  checkRules(rules, "allOf");
  return { stops: (context) => rules.every((rule) => rule.stops(context)) };
}

/**
 * Returns a rule that stops when any one of `rules` stops.
 *
 * @throws `DataError` without rules or when one is not a stopping rule.
 */
export function anyOf(...rules: readonly StoppingRule[]): StoppingRule {
  checkRules(rules, "anyOf");
  return { stops: (context) => rules.some((rule) => rule.stops(context)) };
}

function checkRules(rules: readonly StoppingRule[], name: string): void {
  if (rules.length === 0) throw new DataError(`${name} needs at least one stopping rule.`);
  for (const rule of rules) checkMethod(rule, "stops", `${name} needs stopping rules, such as testLength(20)`);
}

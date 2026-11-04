import {
  checkNonNegative,
  checkPositive,
  checkPositiveInteger,
  DataError,
  describe,
  type FieldNames,
} from "@measurekit/core";

/** The result of a check of one item: how often its response function or its order against other items is violated. */
export interface ItemCheck {
  /** The item id. */
  readonly id: string;
  /** The number of comparisons in which a violation could happen. */
  readonly activeComparisons: number;
  /** The number of violations: decreases or reversals of more than the minimum violation. */
  readonly violations: number;
  /** The largest violation, in the units of the scores' proportions or means; 0 where there is none. */
  readonly maximumViolation: number;
  /** The sum of the violations. */
  readonly violationSum: number;
  /** The largest test statistic of a violation; `null` where there is no violation. */
  readonly maximumZ: number | null;
  /** The number of violations that the tests find significant. */
  readonly significantViolations: number;
  /** Molenaar and Sijtsma's criterion, rounded down: below 40 an item's violations are small, above 80 serious. */
  readonly criterion: number;
}

/** The counts that the criterion of an item is made of. */
export interface CriterionTerms {
  /** The item's scalability coefficient H_i; `null` where undefined. */
  readonly itemCoefficient: number | null;
  readonly activeComparisons: number;
  readonly violations: number;
  readonly maximumViolation: number;
  readonly violationSum: number;
  /** The largest test statistic, 0 where there is none. */
  readonly maximumZ: number;
  readonly significantViolations: number;
}

/**
 * Returns the criterion of Molenaar and Sijtsma, as mokken's summaries: 50 (0.3 - H_i) + √#vi + 100 #vi / #ac + 100
 * maxvi + 10 √sum + 1000 sum / #ac + 5 zmax + 10 √#zsig + 100 #zsig / #ac, 0 for an item without violations, rounded
 * down.
 */
export function criterionOf(terms: CriterionTerms): number {
  if (terms.violations === 0) return 0;
  const {
    activeComparisons: active,
    violations,
    maximumViolation,
    violationSum,
    maximumZ,
    significantViolations,
  } = terms;
  const value =
    50 * (0.3 - (terms.itemCoefficient ?? 0)) +
    Math.sqrt(violations) +
    (100 * violations) / active +
    100 * maximumViolation +
    10 * Math.sqrt(violationSum) +
    (1000 * violationSum) / active +
    5 * maximumZ +
    10 * Math.sqrt(significantViolations) +
    (100 * significantViolations) / active;
  return Math.floor(value);
}

/** The options of the checks of Mokken's assumptions that monotonicity and invariant item ordering share. */
export interface MonotonicityOptions {
  /**
   * The smallest decrease of the proportion of persons who score at least a step, from one rest score group to a higher
   * one, that counts as a violation; the invariant item ordering takes the smallest difference of two items' means.
   *
   * @defaultValue `0.03`; for the invariant item ordering, `0.03` times the highest score
   */
  readonly minimumViolation?: number;
  /**
   * The fewest persons in a group of one rest score or similar rest scores.
   *
   * @defaultValue a tenth of the persons from 500, a fifth up to 499, a third up to 250 and 50 below 150
   */
  readonly minimumGroupSize?: number;
  /**
   * The level of the one-sided tests of violations, a number above 0 and below 1.
   *
   * @defaultValue `0.05`
   */
  readonly alpha?: number;
}

export const MONOTONICITY_OPTIONS: FieldNames<MonotonicityOptions> = {
  minimumViolation: true,
  minimumGroupSize: true,
  alpha: true,
};

/**
 * Checks a level: a number above 0 and below 1.
 *
 * @throws `DataError` when it is not.
 */
export function checkLevel(value: unknown, label: string): asserts value is number {
  checkPositive(value, label);
  if (value >= 1) throw new DataError(`${label} must be a number above 0 and below 1, found ${describe(value)}.`);
}

/** The options of a check, checked, with the minimum group size and violation still to default. */
export interface CheckedOptions {
  readonly minimumViolation: number | undefined;
  readonly minimumGroupSize: number | undefined;
  readonly alpha: number;
}

/**
 * Checks the options of a check of Mokken's assumptions, which its caller has checked to be an object of its options.
 *
 * @throws `DataError` when a minimum is invalid or the level is not above 0 and below 1.
 */
export function checkMonotonicityOptions(options: MonotonicityOptions): CheckedOptions {
  const { minimumViolation, minimumGroupSize, alpha = 0.05 } = options;
  if (options.minimumViolation !== undefined) checkNonNegative(minimumViolation, "The option minimumViolation");
  if (options.minimumGroupSize !== undefined) checkPositiveInteger(minimumGroupSize, "The option minimumGroupSize");
  checkLevel(alpha, "The option alpha");
  return { minimumViolation, minimumGroupSize, alpha };
}

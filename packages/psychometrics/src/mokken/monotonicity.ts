import { DataError, normalQuantile } from "@measurekit/core";
import type { ResponseMatrix } from "../data/response-matrix.js";
import { defaultGroupSize, restScoreGroups } from "./groups.js";
import { checkMonotonicityOptions, criterionOf, type ItemCheck, type MonotonicityOptions } from "./item-check.js";
import { itemScalability, usableItems } from "./scalability.js";
import { covariances, type MokkenData, mokkenData, scoreOf } from "./scores.js";

/** The default smallest decrease of a proportion that counts as a violation, mokken's `minvi`. */
const DEFAULT_MINIMUM_VIOLATION = 0.03;
/** The proportions at or below which a group has none of the persons above a step, and at or above which all. */
const NONE = 1e-10;
const ALL = 0.999999999999;

/** The violations of one item's response function over its steps. */
interface Violations {
  activeComparisons: number;
  violations: number;
  maximumViolation: number;
  violationSum: number;
  maximumZ: number;
  significantViolations: number;
}

/**
 * Checks monotonicity: that the probability of scoring at least each step of an item does not decrease with the rest
 * score, the sum of the other items.
 *
 * @remarks
 * For each item, the persons are split into groups of similar rest scores (at least `minimumGroupSize` persons, with
 * persons of one rest score in one group). For each step, the proportion of a group scoring at least the step is
 * compared with those of the groups with lower rest scores: a decrease of more than `minimumViolation` is a violation,
 * and one of a pair of groups is tested with Molenaar and Sijtsma's z, one-sided at level `alpha`. A comparison is
 * active when the lower group's proportion is above 0 and the higher group's below 1. The criterion combines the
 * counts: below 40, an item's violations are negligible. It follows the mokken package's `check.monotonicity`, over the
 * persons who answered every item.
 *
 * @throws `DataError` when the matrix is invalid, has fewer than two items, fewer than two persons answered every
 * item, the scores span more than 10 categories, an option is invalid, or the minimum group size is more than half of
 * the persons or than all of them.
 * @see {@link https://doi.org/10.18637/jss.v048.i05 | van der Ark (2012)}
 * @see {@link https://doi.org/10.4135/9781412984676 | Sijtsma and Molenaar (2002)}
 */
export function monotonicity(matrix: ResponseMatrix, options: MonotonicityOptions = {}): ItemCheck[] {
  const checked = checkMonotonicityOptions(options);
  const data = mokkenData(matrix, "The monotonicity check");
  const minimumViolation = checked.minimumViolation ?? DEFAULT_MINIMUM_VIOLATION;
  const minimumGroupSize = checked.minimumGroupSize ?? defaultGroupSize(data.persons);
  if (data.persons < minimumGroupSize) {
    throw new DataError(
      `The monotonicity check needs at least ${minimumGroupSize} persons, the minimum group size, who answered every item, found ${data.persons}.`,
    );
  }
  if (minimumGroupSize > data.persons / 2) {
    throw new DataError(
      `The minimum group size ${minimumGroupSize} is more than half of the ${data.persons} persons who answered every item.`,
    );
  }
  const moments = covariances(data);
  const itemCoefficients = itemScalability(moments, data.items, usableItems(moments, data.items));
  const critical = normalQuantile(1 - checked.alpha);
  return Array.from({ length: data.items }, (_, item) => {
    const found = itemViolations(data, item, minimumGroupSize, minimumViolation, critical);
    return {
      id: data.itemIds[item] ?? "",
      activeComparisons: found.activeComparisons,
      violations: found.violations,
      maximumViolation: found.maximumViolation,
      violationSum: found.violationSum,
      maximumZ: found.violations > 0 ? found.maximumZ : null,
      significantViolations: found.significantViolations,
      criterion: Math.max(0, criterionOf({ ...found, itemCoefficient: itemCoefficients[item] ?? null })),
    };
  });
}

/** Returns the violations of monotonicity of an item. */
function itemViolations(
  data: MokkenData,
  item: number,
  minimumGroupSize: number,
  minimumViolation: number,
  critical: number,
): Violations {
  const { persons, categories } = data;
  const rest = new Float64Array(persons);
  for (let person = 0; person < persons; person++) {
    for (let other = 0; other < data.items; other++) {
      if (other !== item) rest[person] = (rest[person] ?? 0) + scoreOf(data, person, other);
    }
  }
  const groups = restScoreGroups(rest, minimumGroupSize);
  const counts = new Float64Array(groups.count * categories);
  const sizes = new Float64Array(groups.count);
  for (let person = 0; person < persons; person++) {
    const group = groups.members[person] ?? 0;
    const at = group * categories + scoreOf(data, person, item);
    counts[at] = (counts[at] ?? 0) + 1;
    sizes[group] = (sizes[group] ?? 0) + 1;
  }
  const total: Violations = {
    activeComparisons: 0,
    violations: 0,
    maximumViolation: 0,
    violationSum: 0,
    maximumZ: 0,
    significantViolations: 0,
  };
  for (let step = 1; step < categories; step++) {
    const above = Float64Array.from({ length: groups.count }, (_, group) => {
      let reached = 0;
      for (let score = step; score < categories; score++) reached += counts[group * categories + score] ?? 0;
      return reached;
    });
    addStep(total, above, sizes, minimumViolation, critical);
  }
  return total;
}

/** Adds the comparisons of one step's proportions across the rest score groups to the totals. */
function addStep(
  total: Violations,
  above: Float64Array,
  sizes: Float64Array,
  minimumViolation: number,
  critical: number,
): void {
  const groups = above.length;
  const proportion = (group: number): number => (above[group] ?? 0) / (sizes[group] ?? 1);
  for (let lower = 0; lower < groups; lower++) {
    for (let higher = lower + 1; higher < groups; higher++) {
      if (proportion(lower) > NONE && proportion(higher) < ALL) total.activeComparisons += 1;
      // A violation is a group with a higher rest score in which fewer persons reach the step.
      const decrease = proportion(lower) - proportion(higher);
      if (!(decrease > minimumViolation)) continue;
      total.violations += 1;
      total.violationSum += decrease;
      total.maximumViolation = Math.max(total.maximumViolation, decrease);
      const z = violationZ(above, sizes, lower, higher);
      total.maximumZ = Math.max(total.maximumZ, z);
      if (z > critical) total.significantViolations += 1;
    }
  }
}

/**
 * Returns Molenaar and Sijtsma's z of a violation, |2 (√((f₂ + 1)(g₁ + 1)) - √(f₁ g₂))| / √(n + m - 1), with f₁ and f₂ the
 * persons of the higher group below and at or above the step, g₁ and g₂ those of the lower, and n and m the groups'
 * sizes.
 */
function violationZ(above: Float64Array, sizes: Float64Array, lower: number, higher: number): number {
  const reachedHigher = above[higher] ?? 0;
  const reachedLower = above[lower] ?? 0;
  const missedHigher = (sizes[higher] ?? 0) - reachedHigher;
  const missedLower = (sizes[lower] ?? 0) - reachedLower;
  const statistic =
    (2 * (Math.sqrt((reachedHigher + 1) * (missedLower + 1)) - Math.sqrt(missedHigher * reachedLower))) /
    Math.sqrt((sizes[higher] ?? 0) + (sizes[lower] ?? 0) - 1);
  return Math.abs(statistic);
}

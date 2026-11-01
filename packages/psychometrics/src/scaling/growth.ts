import {
  checkArray,
  checkFinite,
  checkPositive,
  checkRecord,
  DataError,
  describe,
  isNonEmptyString,
} from "@measurekit/core";

/** The distribution of scale scores in one group, such as a grade, on a scale the groups share. */
export interface GroupScale {
  /** The group's label, such as `"grade 4"`. */
  readonly group: string;
  /** The mean of the group's scale scores. */
  readonly mean: number;
  /** The standard deviation of the group's scale scores, above 0. */
  readonly sd: number;
}

/** The growth from one group to the next on a shared scale. */
export interface ScaleGrowth {
  /** The label of the earlier group. */
  readonly from: string;
  /** The label of the later group. */
  readonly to: string;
  /** The later mean less the earlier one, in scale points. */
  readonly meanChange: number;
  /** The later standard deviation over the earlier one: above 1 when the groups spread out. */
  readonly sdRatio: number;
  /** The mean change in units of the pooled standard deviation of the two groups: (μ_to - μ_from) / √((σ²_from + σ²_to) / 2). */
  readonly effectSize: number;
}

/**
 * Returns the growth between each group and the next on a scale they share, such as the grades of a vertical scale.
 *
 * @remarks
 * The groups are in the order of the scale, the earliest first, so `n` groups give `n - 1` rows. The mean change is in
 * scale points, the ratio of standard deviations says whether the groups spread out or close in, and the effect size
 * relates the change to the spread of the two groups, so that grades on scales of different units compare. The groups'
 * means and standard deviations come from a calibration with `groups`, or from a separate calibration of each group
 * chained to a base scale.
 *
 * @throws `DataError` when `groups` is not an array of at least two groups, a group is not an object with a non-empty
 * `group` label, a finite `mean` and a positive `sd`, or two groups have the same label.
 * @see {@link https://doi.org/10.1111/j.1745-3984.1986.tb00252.x | Yen (1986)}
 */
export function scaleGrowth(groups: readonly GroupScale[]): readonly ScaleGrowth[] {
  checkArray(groups, "The groups must be an array");
  if (groups.length < 2) {
    throw new DataError(`Growth needs at least two groups on the shared scale, found ${groups.length}.`);
  }
  const labels = new Set<string>();
  groups.forEach((scale, i) => {
    const label = `group ${i + 1}`;
    checkRecord(scale, `The ${label} must be an object { group, mean, sd }`);
    if (!isNonEmptyString(scale.group)) {
      throw new DataError(`The ${label} needs a label that is a non-empty string, found ${describe(scale.group)}.`);
    }
    if (labels.has(scale.group)) throw new DataError(`The groups need different labels, found "${scale.group}" twice.`);
    labels.add(scale.group);
    checkFinite(scale.mean, `The mean of ${label}`);
    checkPositive(scale.sd, `The standard deviation of ${label}`);
  });
  return groups.slice(1).map((to, i) => {
    const from = groups[i] ?? to;
    const meanChange = to.mean - from.mean;
    return {
      from: from.group,
      to: to.group,
      meanChange,
      sdRatio: to.sd / from.sd,
      effectSize: meanChange / (Math.hypot(from.sd, to.sd) / Math.SQRT2),
    };
  });
}

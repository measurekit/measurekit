import { DataError } from "@measurekit/core";
import { type CompressedResponses, personGroups } from "../irt/internal.js";
import type { ItemState } from "./calibration-setup.js";
import { type Latent, latentDistribution } from "./correlation.js";
import type { Design } from "./design.js";
import type { Grid } from "./grid.js";

/** The groups of a calibration: each person's group and the distribution of theta in each. */
export interface Population {
  /** The labels of the groups in order of first appearance; `null` without the option `groups`. */
  readonly labels: readonly string[] | null;
  /** The index of each person's group, in the order of the persons of the responses. */
  readonly groupOf: Int32Array;
  /** The distribution of each group on the grid. */
  readonly latents: readonly Latent[];
}

/** The options of a calibration that set its groups. */
interface PopulationOptions {
  readonly groups?: readonly string[] | undefined;
}

/**
 * Checks the groups against the data and the anchors, and builds the distribution of each group.
 *
 * @remarks
 * The first group has means 0 and variances 1 unless anchors set the scale; then every group's means and variances are
 * estimated, as are those of every group but the first without anchors. The correlations are estimated in every group
 * when the design has them.
 *
 * @throws `DataError` when `groups` does not give each person a string label, a group has no person with a response or,
 * with correlated dimensions, no more persons with a response than dimensions, the groups share no item with the first
 * group, directly or through other groups, or anchors leave a dimension of some set of groups without a scale.
 */
export function population<TParameters>(
  options: PopulationOptions,
  responses: CompressedResponses,
  items: readonly ItemState<TParameters>[],
  design: Design,
  grid: Grid,
): Population {
  const persons = responses.offsets.length - 1;
  const { labels, groupOf } = personGroups(options.groups, persons);
  const names: readonly (string | null)[] = labels ?? [null];
  const answered = answeredItems(responses, groupOf, names.length);
  names.forEach((name, g) => {
    if (!answered[g]?.some(Boolean)) throw new DataError(`Group "${name}" has no person with a response.`);
  });
  if (design.correlated) {
    checkCorrelatedGroups(names, respondentCounts(responses, groupOf, names.length), design.dimensionNames.length);
  }
  const anchored = items.some((item) => item.fixed);
  checkScale(names, answered, items, design, anchored);
  const latents = names.map((_, g) => {
    const estimated = anchored || g > 0;
    return latentDistribution(grid, { correlations: design.correlated, variances: estimated, means: estimated });
  });
  return { labels, groupOf, latents };
}

/** Returns the number of persons with a response in each group. */
function respondentCounts(responses: CompressedResponses, groupOf: Int32Array, groupCount: number): Int32Array {
  const counts = new Int32Array(groupCount);
  groupOf.forEach((group, person) => {
    if ((responses.offsets[person + 1] ?? 0) > (responses.offsets[person] ?? 0))
      counts[group] = (counts[group] ?? 0) + 1;
  });
  return counts;
}

/**
 * Checks that every group whose correlations are estimated has more persons than correlated dimensions: the thetas of
 * no more persons than dimensions span no more than a hyperplane, so that the correlations of the group run to ±1,
 * where the likelihood has no maximum.
 *
 * @param names - The label of each group, `null` for the persons of a calibration without groups.
 * @param counts - The number of persons with a response in each group.
 * @param dimensions - The number of dimensions whose correlations are estimated.
 * @throws `DataError` when a group has no more persons with a response than those dimensions.
 */
export function checkCorrelatedGroups(
  names: readonly (string | null)[],
  counts: ArrayLike<number>,
  dimensions: number,
): void {
  names.forEach((name, g) => {
    const persons = counts[g] ?? 0;
    if (persons > dimensions) return;
    const who = name === null ? "The calibration" : `Group "${name}"`;
    throw new DataError(
      `${who} has ${persons} ${persons === 1 ? "person" : "persons"} with a response, no more than its ${dimensions} correlated dimensions, whose correlations then run to ±1: estimate the correlations from more persons, or set correlated to false.`,
    );
  });
}

/** Returns, for each group, whether some person of it answered each item. */
function answeredItems(responses: CompressedResponses, groupOf: Int32Array, groupCount: number): boolean[][] {
  const answered = Array.from({ length: groupCount }, () => new Array<boolean>(responses.itemIds.length).fill(false));
  groupOf.forEach((group, person) => {
    for (let r = responses.offsets[person] ?? 0; r < (responses.offsets[person + 1] ?? 0); r++) {
      const row = answered[group];
      if (row !== undefined) row[responses.items[r] ?? 0] = true;
    }
  });
  return answered;
}

/**
 * Checks that the groups lie on one scale: they are linked by items, and, with anchors, every set of linked groups has
 * an anchor on each dimension.
 */
function checkScale<TParameters>(
  names: readonly (string | null)[],
  answered: readonly (readonly boolean[])[],
  items: readonly ItemState<TParameters>[],
  design: Design,
  anchored: boolean,
): void {
  const component = linkedGroups(answered);
  const label = (g: number): string => `group "${names[g]}"`;
  if (!anchored) {
    const apart = component.findIndex((c) => c !== component[0]);
    if (apart >= 0) {
      throw new DataError(
        `The ${label(apart)} shares no item with the ${label(0)}, directly or through other groups, so their distributions cannot be put on one scale; give them common items or fix some items.`,
      );
    }
    return;
  }
  for (const set of new Set(component)) {
    const first = component.indexOf(set);
    design.dimensionNames.forEach((dimension, d) => {
      const anchor = items.some(
        (item, i) =>
          item.fixed &&
          (item.vector[d] ?? 0) !== 0 &&
          answered.some((row, g) => component[g] === set && row[i] === true),
      );
      if (!anchor) {
        const who = names[first] === null ? "the persons" : `the ${label(first)} or the groups linked to it`;
        throw new DataError(
          `No fixed item with a slope on dimension "${dimension}" was answered by ${who}, so the fixed items do not set its scale.`,
        );
      }
    });
  }
}

/** Returns, for each group, the index of the lowest group it is linked to through items that groups answer in common. */
export function linkedGroups(answered: readonly (readonly boolean[])[]): number[] {
  const component = answered.map((_, g) => g);
  const root = (g: number): number => {
    let at = g;
    while (component[at] !== at) at = component[at] ?? at;
    return at;
  };
  const itemCount = answered[0]?.length ?? 0;
  for (let i = 0; i < itemCount; i++) {
    let first = -1;
    answered.forEach((row, g) => {
      if (row[i] !== true) return;
      if (first < 0) first = g;
      else component[Math.max(root(g), root(first))] = Math.min(root(g), root(first));
    });
  }
  return component.map((_, g) => root(g));
}

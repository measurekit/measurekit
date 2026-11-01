import { DataError, mean } from "@measurekit/core";
import type { ResponseMatrix } from "../data/response-matrix.js";
import { variance } from "../math/statistics.js";
import { type MokkenData, mokkenData, scoreOf } from "./scores.js";

/**
 * The item steps with a proportion strictly between 0 and 1, ordered by their proportion, and the proportions of two
 * steps of different items passed together.
 */
interface StepMatrix {
  /** The number of item steps. */
  readonly size: number;
  /** The proportion of persons who pass each step, in ascending order; steps of equal counts keep their order. */
  readonly proportion: Float64Array;
  /** The proportion of persons who pass both steps, by step, then step; 0 where `observed` is 0. */
  readonly joint: Float64Array;
  /** Whether the joint proportion is observed: 0 for two steps of one item, which no person can be said to pass together. */
  readonly observed: Uint8Array;
  /** The group of steps with an equal proportion that each step belongs to. */
  readonly group: Int32Array;
  /** The steps of each group. */
  readonly members: readonly (readonly number[])[];
}

/**
 * Returns the Molenaar-Sijtsma reliability of the total score, the coefficient MS, which assumes that the items fit the
 * double monotonicity model.
 *
 * @remarks
 * Under local independence the variance of the true score is the sum over all pairs of item steps of P(pass both) - P(pass
 * one) P(pass the other), the pairs of steps of one item included, and the reliability is that variance over the
 * variance of the total score. The pairs of steps of one item have no observed joint proportion, so method MS
 * estimates each from the pairs of that step with the steps next to it in the order of the steps' proportions, which
 * the model of double monotonicity fixes for every person: the items' response functions do not cross. The estimate is
 * the mean of eight values from the nearest steps with an observed proportion on each side, bounded below by the
 * product of the two proportions, which independence gives, and above by the smaller of them, which perfect
 * dependence gives. Steps that every person or no person passes are left out. Steps passed by equal numbers of
 * persons are tied, and a tied step takes the mean over its group, as `check.reliability` of the mokken package does.
 *
 * The statistic uses the persons who answered every item, with the lowest score of all items taken as 0. It is not
 * limited to 1: with data that do not fit the model, it can exceed it.
 *
 * @throws `DataError` when the matrix is invalid, has fewer than two items, fewer than two persons answered every item,
 * the scores span more than 10 categories, or the total scores have no variance.
 * @see {@link https://doi.org/10.1007/BF02293957 | Sijtsma and Molenaar (1987)}
 * @see {@link https://doi.org/10.1111/bmsp.12078 | Sijtsma and van der Ark (2017)}
 */
export function molenaarSijtsmaReliability(matrix: ResponseMatrix): number {
  const data = mokkenData(matrix, "Molenaar-Sijtsma reliability");
  const totals = Float64Array.from({ length: data.persons }, (_, person) => {
    let total = 0;
    for (let item = 0; item < data.items; item++) total += scoreOf(data, person, item);
    return total;
  });
  const totalVariance = (variance(totals) * (data.persons - 1)) / data.persons;
  if (totalVariance === 0) throw new DataError("Molenaar-Sijtsma reliability needs total scores with variance.");
  return excessOverIndependence(stepMatrix(data)) / totalVariance;
}

/** Returns the sum over all pairs of steps of the joint proportion less the product of the proportions. */
function excessOverIndependence(steps: StepMatrix): number {
  const { size, proportion } = steps;
  let excess = 0;
  for (let a = 0; a < size; a++) {
    for (let b = 0; b < size; b++) {
      excess += jointProportion(steps, a, b) - (proportion[a] ?? 0) * (proportion[b] ?? 0);
    }
  }
  return excess;
}

/** Returns the steps of the items with their proportions and joint proportions, ordered by proportion. */
function stepMatrix(data: MokkenData): StepMatrix {
  const counts = stepCounts(data);
  const order = Array.from(counts.keys())
    .filter((id) => (counts[id] ?? 0) > 0 && (counts[id] ?? 0) < data.persons)
    .sort((a, b) => (counts[a] ?? 0) - (counts[b] ?? 0) || a - b);
  const size = order.length;
  const position = new Int32Array(counts.length).fill(-1);
  order.forEach((id, rank) => {
    position[id] = rank;
  });
  const joint = new Float64Array(size * size);
  const observed = new Uint8Array(size * size);
  for (let i = 0; i < data.items; i++) {
    for (let j = i + 1; j < data.items; j++) {
      recordPairs(data, [i, j], position, { size, joint, observed });
    }
  }
  const proportion = Float64Array.from(order, (id) => (counts[id] ?? 0) / data.persons);
  const { group, members } = equalProportions(order.map((id) => counts[id] ?? 0));
  return { size, proportion, joint, observed, group, members };
}

/** Records the joint proportions of the steps of two items that are in the order of steps, `position` by step. */
function recordPairs(
  data: MokkenData,
  [i, j]: readonly [number, number],
  position: Int32Array,
  target: { readonly size: number; readonly joint: Float64Array; readonly observed: Uint8Array },
): void {
  const { persons, categories } = data;
  const passed = passesTogether(data, i, j);
  const levels = categories - 1;
  for (let x = 1; x < categories; x++) {
    for (let y = 1; y < categories; y++) {
      const a = position[i * levels + x - 1] ?? -1;
      const b = position[j * levels + y - 1] ?? -1;
      if (a < 0 || b < 0) continue;
      target.joint[a * target.size + b] = target.joint[b * target.size + a] =
        (passed[x * (categories + 1) + y] ?? 0) / persons;
      target.observed[a * target.size + b] = target.observed[b * target.size + a] = 1;
    }
  }
}

/** Returns the number of persons who pass each step, by item, then level from 1: the score is at least the level. */
function stepCounts(data: MokkenData): Uint32Array {
  const levels = data.categories - 1;
  const counts = new Uint32Array(data.items * levels);
  for (let person = 0; person < data.persons; person++) {
    for (let item = 0; item < data.items; item++) {
      for (let level = 1; level <= scoreOf(data, person, item); level++) {
        const step = item * levels + level - 1;
        counts[step] = (counts[step] ?? 0) + 1;
      }
    }
  }
  return counts;
}

/**
 * Returns the number of persons whose scores on items i and j are at least x and y, by x, then y, with a row and a
 * column of zeros after the highest scores so that the sums need no edge cases.
 */
function passesTogether(data: MokkenData, i: number, j: number): Uint32Array {
  const { persons, categories } = data;
  const width = categories + 1;
  const passed = new Uint32Array(width * width);
  for (let person = 0; person < persons; person++) {
    const cell = scoreOf(data, person, i) * width + scoreOf(data, person, j);
    passed[cell] = (passed[cell] ?? 0) + 1;
  }
  for (let x = categories - 1; x >= 0; x--) {
    for (let y = categories - 1; y >= 0; y--) {
      const at = x * width + y;
      passed[at] =
        (passed[at] ?? 0) + (passed[at + 1] ?? 0) + (passed[at + width] ?? 0) - (passed[at + width + 1] ?? 0);
    }
  }
  return passed;
}

/** Groups the positions of ascending `counts` that hold the same count. */
function equalProportions(counts: readonly number[]): { group: Int32Array; members: number[][] } {
  const group = new Int32Array(counts.length);
  const members: number[][] = [];
  counts.forEach((count, step) => {
    if (step === 0 || count !== counts[step - 1]) members.push([]);
    group[step] = members.length - 1;
    members[members.length - 1]?.push(step);
  });
  return { group, members };
}

/** Returns the joint proportion of steps a and b: observed, or estimated from the steps next to them. */
function jointProportion(steps: StepMatrix, a: number, b: number): number {
  if (steps.observed[a * steps.size + b] === 1) return steps.joint[a * steps.size + b] ?? 0;
  const estimate = estimatedJoint(steps, a, b);
  const lower = (steps.proportion[a] ?? 0) * (steps.proportion[b] ?? 0);
  const upper = Math.min(steps.proportion[a] ?? 0, steps.proportion[b] ?? 0);
  return Math.min(Math.max(estimate, lower), upper);
}

/** Returns the mean of the observed joint proportions of two groups of steps, or `null` where none is observed. */
function groupMean(steps: StepMatrix, rows: readonly number[], columns: readonly number[]): number | null {
  let sum = 0;
  let count = 0;
  for (const row of rows) {
    for (const column of columns) {
      if (steps.observed[row * steps.size + column] !== 1) continue;
      sum += steps.joint[row * steps.size + column] ?? 0;
      count++;
    }
  }
  return count === 0 ? null : sum / count;
}

/** Which way to look from an unobserved pair of steps for the nearest step that has an observed joint proportion. */
type Direction = "after" | "before";

/**
 * Returns the estimate of the joint proportion of unobserved steps a and b: the mean of the values that the nearest
 * steps with an observed proportion give, by equations 17 and 21 of Molenaar and Sijtsma (1988), or the mean of the
 * observed proportions of the steps with the same proportions when both are tied.
 *
 * @remarks
 * A step tied with another is its own neighbor on the side of the tie, so that its group's mean replaces a search.
 * The nearest steps are those next to b in the row of a (to the right and to the left) and next to a in the column of
 * b (below and above), which are the steps with the next higher and lower proportions that have an observed joint
 * proportion with the other step.
 */
function estimatedJoint(steps: StepMatrix, a: number, b: number): number {
  const rows = membersOf(steps, a);
  const columns = membersOf(steps, b);
  const aTied = rows.length > 1;
  const bTied = columns.length > 1;
  if (aTied && bTied) return groupMean(steps, rows, columns) ?? 0;
  const pa = steps.proportion[a] ?? 0;
  const pb = steps.proportion[b] ?? 0;
  const acrossRow = bTied ? [b, b] : [nearestInRow(steps, a, b, "after"), nearestInRow(steps, a, b, "before")];
  const downColumn = aTied ? [a, a] : [nearestInColumn(steps, a, b, "after"), nearestInColumn(steps, a, b, "before")];
  const values = [
    ...neighborValues(steps, acrossRow, (step) => groupMean(steps, rows, membersOf(steps, step)), pb, pa),
    ...neighborValues(steps, downColumn, (step) => groupMean(steps, membersOf(steps, step), columns), pa, pb),
  ];
  return values.length === 0 ? 0 : mean(values);
}

/**
 * Returns the estimates that the neighbor `steps` give of a joint proportion, two for each neighbor with an observed
 * joint proportion: `jointOf` gives that proportion, `moved` is the proportion of the step the neighbor replaces and
 * `other` that of the step it stays paired with.
 */
function neighborValues(
  steps: StepMatrix,
  neighbors: readonly (number | null)[],
  jointOf: (step: number) => number | null,
  moved: number,
  other: number,
): number[] {
  return neighbors.flatMap((step) => {
    const joint = step === null ? null : jointOf(step);
    return step === null || joint === null ? [] : extrapolate(joint, moved, other, steps.proportion[step] ?? 0);
  });
}

/**
 * Returns the two estimates of the joint proportion of a step with proportion `moved` and a step with proportion
 * `other`, from `joint`, the observed proportion of a step with proportion `neighbor` and the other step: the first
 * scales the joint proportion by the ratio of the proportions, the second adds what lies between them.
 */
function extrapolate(joint: number, moved: number, other: number, neighbor: number): [number, number] {
  return [(joint * moved) / neighbor, (joint * (1 - moved) - other * (neighbor - moved)) / (1 - neighbor)];
}

/** Returns the steps of the group that step `step` is in. */
function membersOf(steps: StepMatrix, step: number): readonly number[] {
  return steps.members[steps.group[step] ?? 0] ?? [];
}

/** Returns the nearest step to b, from b on or back to the first, whose joint proportion with a is observed. */
function nearestInRow(steps: StepMatrix, a: number, b: number, direction: Direction): number | null {
  const stride = direction === "after" ? 1 : -1;
  for (let step = b; step >= 0 && step < steps.size; step += stride) {
    if (steps.observed[a * steps.size + step] === 1) return step;
  }
  return null;
}

/** Returns the nearest step to a, from a on or back to the first, whose joint proportion with b is observed. */
function nearestInColumn(steps: StepMatrix, a: number, b: number, direction: Direction): number | null {
  const stride = direction === "after" ? 1 : -1;
  for (let step = a; step >= 0 && step < steps.size; step += stride) {
    if (steps.observed[step * steps.size + b] === 1) return step;
  }
  return null;
}

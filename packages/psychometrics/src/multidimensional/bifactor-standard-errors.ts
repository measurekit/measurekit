import { symmetrized } from "../math/matrix.js";
import type { BifactorFit } from "./bifactor-fit.js";
import { addPersonMeans, addSubPattern, finishProducts, type GroupScores, groupScores } from "./bifactor-scores.js";
import type { Latent } from "./correlation.js";
import { covarianceOfInformation, itemInformation, STEP } from "./standard-errors.js";

/**
 * Returns the covariance matrix of the estimated parameters on their own scale, by rows, in the order of the items'
 * estimated entries and then the correlation; `null` when the information cannot be inverted or gives a variance that
 * is not positive.
 *
 * @remarks
 * The scores are those of the complete data on the unbounded scale of calibration. A person's posterior over all
 * dimensions is the posterior of the primary nodes times, given each, an independent posterior over the nodes of each
 * specific dimension. The mean of the scores of the items of a group is then a sum over the primary nodes of the
 * posterior weight times a mean per sub-pattern, and the product of the scores of items in two groups a sum of the
 * product of their means; only the product of two items of one group needs their joint nodes, and it is summed once for
 * each sub-pattern. `"hessian"` is Louis' formula: the expected complete-data information less the posterior variance
 * of the scores. `"cross-product"` is the sum over persons of the outer products of the persons' mean scores.
 * The fit is left at its estimates, and the E-step's last run is the one the counts come from.
 *
 * @see {@link https://doi.org/10.1111/j.2517-6161.1982.tb01203.x | Louis (1982)}
 */
export function bifactorCovariance<TParameters>(
  fit: BifactorFit<TParameters>,
  method: "hessian" | "cross-product",
): Float64Array | null {
  const size = parameterCount(fit);
  if (size === 0) return null;
  const louis = method === "hessian";
  const { outer, second } = scoreMoments(fit, size, louis);
  const information = louis ? completeInformation(fit, size) : outer;
  if (louis) {
    for (let k = 0; k < information.length; k++) {
      information[k] = (information[k] ?? 0) - ((second[k] ?? 0) - (outer[k] ?? 0));
    }
  }
  return covarianceOfInformation(symmetrized(information, size), fit.items, size);
}

/** Returns the number of estimated parameters: the items' estimated entries, then the correlation. */
function parameterCount<TParameters>({ items, latent, correlated }: BifactorFit<TParameters>): number {
  return items.reduce((sum, item) => sum + item.free.length, 0) + (correlated ? latent.pairs.length : 0);
}

/**
 * Returns Σ_p E_p[s] E_p[s]ᵀ over persons and, with `withSecond`, Σ_p E_p[s sᵀ], where s is the complete-data score
 * of all estimated parameters and E_p the expectation under person p's posterior.
 */
function scoreMoments<TParameters>(
  fit: BifactorFit<TParameters>,
  size: number,
  withSecond: boolean,
): { readonly outer: Float64Array; readonly second: Float64Array } {
  const groups = scoresOf(fit);
  fit.expectation.forEachSubPattern((g, u, conditional, mass) => {
    const scores = groups[g];
    const nodes = fit.groups[g];
    if (scores !== undefined && nodes !== undefined) addSubPattern(scores, nodes, u, conditional, mass, withSecond);
  });
  const outer = new Float64Array(size * size);
  const second = new Float64Array(size * size);
  const crossings = crossBlocks(groups);
  const mean = new Float64Array(size);
  fit.expectation.forEachPattern((frequency, posterior, subPatterns) => {
    mean.fill(0);
    groups.forEach((scores, g) => {
      addPersonMeans(scores, mean, subPatterns[g] ?? 0, posterior);
    });
    addOuter(outer, mean, frequency);
    if (withSecond) addCrossings(crossings, groups, subPatterns, posterior, frequency);
  });
  if (withSecond) {
    fit.groups.forEach((nodes, g) => {
      const scores = groups[g];
      if (scores !== undefined) finishProducts(scores, nodes);
    });
    for (const crossing of crossings) placeCrossing(crossing, groups, second, size);
    for (const { slots, within } of groups) placeWithin(slots, within, second, size);
  }
  return { outer, second };
}

/** Returns the scores of each group of items, and of the correlation when it is estimated. */
function scoresOf<TParameters>(fit: BifactorFit<TParameters>): GroupScores[] {
  const primaryNodes = fit.latent.logWeights.length;
  const offsets = new Int32Array(fit.items.length);
  let slot = 0;
  fit.items.forEach((item, i) => {
    offsets[i] = slot;
    slot += item.free.length;
  });
  const groups = fit.groups.map((nodes) => {
    const slots = Int32Array.from(
      nodes.patterns.group.items.flatMap((i) => Array.from(fit.items[i]?.free ?? [], (_, l) => (offsets[i] ?? 0) + l)),
    );
    return groupScores(nodes, slots, primaryNodes);
  });
  if (fit.correlated) groups.push(correlationScores(fit, slot, primaryNodes));
  return groups;
}

/** Returns the scores of the correlation, which depend on the primary node alone, and their products. */
function correlationScores<TParameters>(
  { latent, expectation }: BifactorFit<TParameters>,
  first: number,
  primaryNodes: number,
): GroupScores {
  const pairs = latent.pairs.length;
  const scores = new Float64Array(pairs * primaryNodes);
  latent.scores(scores);
  const means = new Float64Array(primaryNodes * pairs);
  const within = new Float64Array(pairs * pairs);
  for (let q = 0; q < primaryNodes; q++) {
    for (let a = 0; a < pairs; a++) means[q * pairs + a] = scores[a * primaryNodes + q] ?? 0;
    for (let a = 0; a < pairs; a++) {
      for (let b = 0; b < pairs; b++) {
        const product = (means[q * pairs + a] ?? 0) * (means[q * pairs + b] ?? 0);
        within[a * pairs + b] = (within[a * pairs + b] ?? 0) + (expectation.nodeCounts[q] ?? 0) * product;
      }
    }
  }
  const slots = Int32Array.from({ length: pairs }, (_, a) => first + a);
  return { slots, primaryNodes, means, within, items: [], pairs: [] };
}

/** Adds `frequency` times the outer product of a person's mean scores to `outer`. */
function addOuter(outer: Float64Array, mean: Float64Array, frequency: number): void {
  const size = mean.length;
  for (let a = 0; a < size; a++) {
    const left = frequency * (mean[a] ?? 0);
    for (let b = 0; b < size; b++) outer[a * size + b] = (outer[a * size + b] ?? 0) + left * (mean[b] ?? 0);
  }
}

/** Adds the products within a group to `second` at the group's slots. */
function placeWithin(slots: Int32Array, within: Float64Array, second: Float64Array, size: number): void {
  slots.forEach((a, i) => {
    slots.forEach((b, j) => {
      second[a * size + b] = (second[a * size + b] ?? 0) + (within[i * slots.length + j] ?? 0);
    });
  });
}

/** The sum over persons of the expected products of the scores of the entries of two groups, in the groups' order. */
interface Crossing {
  readonly first: number;
  readonly second: number;
  readonly values: Float64Array;
}

/** Returns an empty crossing for every two groups. */
function crossBlocks(groups: readonly GroupScores[]): Crossing[] {
  return groups.flatMap((first, g) =>
    groups.slice(g + 1).map((other, offset) => ({
      first: g,
      second: g + 1 + offset,
      values: new Float64Array(first.slots.length * other.slots.length),
    })),
  );
}

/**
 * Adds, for a person, the expected products of the scores of every two groups to the crossings: the sum over the
 * primary nodes of the posterior weight times the product of the groups' means.
 */
function addCrossings(
  crossings: readonly Crossing[],
  groups: readonly GroupScores[],
  subPatterns: Int32Array,
  posterior: Float64Array,
  frequency: number,
): void {
  for (const { first, second, values } of crossings) {
    const left = groups[first];
    const right = groups[second];
    if (left === undefined || right === undefined) continue;
    const leftBase = (subPatterns[first] ?? 0) * posterior.length * left.slots.length;
    const rightBase = (subPatterns[second] ?? 0) * posterior.length * right.slots.length;
    for (let q = 0; q < posterior.length; q++) {
      addProduct(
        values,
        [left, leftBase + q * left.slots.length],
        [right, rightBase + q * right.slots.length],
        frequency * (posterior[q] ?? 0),
      );
    }
  }
}

/** Adds `weight` times the outer product of two groups' mean scores, each at its offset, to `values`. */
function addProduct(
  values: Float64Array,
  [left, leftAt]: readonly [GroupScores, number],
  [right, rightAt]: readonly [GroupScores, number],
  weight: number,
): void {
  const leftWidth = left.slots.length;
  const rightWidth = right.slots.length;
  for (let i = 0; i < leftWidth; i++) {
    const scaled = weight * (left.means[leftAt + i] ?? 0);
    for (let j = 0; j < rightWidth; j++) {
      values[i * rightWidth + j] = (values[i * rightWidth + j] ?? 0) + scaled * (right.means[rightAt + j] ?? 0);
    }
  }
}

/** Writes a crossing to both triangles of `second` at the slots of its two groups. */
function placeCrossing(
  { first, second: other, values }: Crossing,
  groups: readonly GroupScores[],
  second: Float64Array,
  size: number,
): void {
  const left = groups[first]?.slots ?? new Int32Array(0);
  const right = groups[other]?.slots ?? new Int32Array(0);
  left.forEach((a, i) => {
    right.forEach((b, j) => {
      second[a * size + b] = values[i * right.length + j] ?? 0;
      second[b * size + a] = values[i * right.length + j] ?? 0;
    });
  });
}

/**
 * Returns the expected complete-data information of the estimated parameters on the unbounded scale: the Hessian of the
 * negative expected complete-data log-posterior of each item, and of the correlation, at the last E-step's counts, by
 * central differences of the exact gradient. It is block diagonal, as the items and the correlation share no parameters.
 */
function completeInformation<TParameters>(fit: BifactorFit<TParameters>, size: number): Float64Array {
  const { items, latent, correlated, maximization, expectation } = fit;
  const information = new Float64Array(size * size);
  const slot = itemInformation(items, maximization, information, size);
  if (correlated) correlationInformation(latent, expectation.nodeCounts, information, size, slot);
  return information;
}

/** Writes the Hessian of the negative expected log-likelihood of the correlation into the block at `slot`. */
function correlationInformation(
  latent: Latent,
  nodeCounts: Float64Array,
  information: Float64Array,
  size: number,
  slot: number,
): void {
  const pairs = latent.pairs.length;
  const start = latent.matrix.slice();
  const up = new Float64Array(pairs);
  const down = new Float64Array(pairs);
  for (let a = 0; a < pairs; a++) {
    const [i, j] = latent.pairs[a] ?? [0, 0];
    const step = STEP * Math.max(1, Math.abs(start[i * latent.dimensions + j] ?? 0));
    latent.set(shifted(start, latent.dimensions, i, j, step));
    latent.gradient(nodeCounts, up);
    latent.set(shifted(start, latent.dimensions, i, j, -step));
    latent.gradient(nodeCounts, down);
    for (let b = 0; b < pairs; b++)
      information[(slot + b) * size + slot + a] = ((up[b] ?? 0) - (down[b] ?? 0)) / (2 * step);
  }
  latent.set(start);
}

/** Returns the correlation matrix with the correlation of dimensions i and j moved by `step`. */
function shifted(matrix: Float64Array, dimensions: number, i: number, j: number, step: number): Float64Array {
  const next = matrix.slice();
  next[i * dimensions + j] = (next[i * dimensions + j] ?? 0) + step;
  next[j * dimensions + i] = (next[j * dimensions + i] ?? 0) + step;
  return next;
}

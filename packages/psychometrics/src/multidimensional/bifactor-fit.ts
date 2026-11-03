import { largestChange } from "@measurekit/core";
import { type CompressedResponses, checkStandardErrorSize, type Iteration, squarem } from "../irt/internal.js";
import { type BifactorExpectation, bifactorExpectation, type GroupNodes } from "./bifactor-expectation.js";
import { groupGrid, groupWeights, specificRule } from "./bifactor-nodes.js";
import { type GroupPatterns, patternsOf } from "./bifactor-patterns.js";
import { bifactorCovariance, bifactorParameterCount } from "./bifactor-standard-errors.js";
import type { BifactorStructure } from "./bifactor-structure.js";
import type { MultidimensionalCalibrationOptions } from "./calibration-options.js";
import { type ItemState, setUp } from "./calibration-setup.js";
import { type Latent, latentDistribution } from "./correlation.js";
import { checkTables, type Grid, productGrid } from "./grid.js";
import { type ItemMaximization, itemMaximization } from "./m-step.js";
import type { Settings, Solution } from "./solution.js";

/** Settings of the estimation of a bifactor model, after their defaults are filled in. */
export interface BifactorSettings {
  readonly tolerance: number;
  readonly maxIterations: number;
  readonly standardErrors: Settings["standardErrors"];
}

/** What EM works on: the items, the latent distribution of the primary dimensions, the M-step and the E-step. */
export interface BifactorFit<TParameters> {
  /** The items in the item order of the data. */
  readonly items: readonly ItemState<TParameters>[];
  readonly groups: readonly GroupNodes<TParameters>[];
  readonly latent: Latent;
  /** Whether the correlation of the two primary dimensions is estimated. */
  readonly correlated: boolean;
  readonly maximization: ItemMaximization<TParameters>;
  readonly expectation: BifactorExpectation;
}

/**
 * Sets up a bifactor calibration: the nodes of every group of items, the items with their starting values, the E-step
 * and the M-step.
 *
 * @param pairs - Whether the standard errors will need the persons expected at each node of a group for each pair of
 *   responses to two of its items, which Louis' formula takes: these tables count toward the limit as well.
 * @throws `DataError` when `points` makes a grid of more nodes than a calibration allows or tables of more than 2 ** 24
 * entries, or an item cannot be estimated from its responses.
 */
export function bifactorFit<TParameters>(
  responses: CompressedResponses,
  options: Pick<MultidimensionalCalibrationOptions<TParameters>, "model" | "priors">,
  structure: BifactorStructure,
  points: number,
  pairs: boolean,
): BifactorFit<TParameters> {
  const { design, primaryCount } = structure;
  const dimensions = design.dimensionNames.length;
  const primary = productGrid(primaryCount, points);
  const rule = specificRule(points);
  const table = patternsOf(responses, structure.groups);
  const grids = table.groups.map(({ group }) => groupGrid(primary, rule, dimensions, group.dimension));
  const nodesOfItem = new Int32Array(responses.itemIds.length);
  table.groups.forEach(({ group }, s) => {
    for (const item of group.items) nodesOfItem[item] = grids[s]?.count ?? 0;
  });
  const unplaced = setUp(responses, options, design, 0);
  checkTables(tableEntries(unplaced, table.groups, grids, pairs));
  const items = unplaced.map((item, i) => ({
    ...item,
    counts: new Float64Array((nodesOfItem[i] ?? 0) * item.categoryCount),
  }));
  const groups = table.groups.map((patterns, s): GroupNodes<TParameters> => {
    const { group } = patterns;
    return {
      patterns,
      grid: grids[s] ?? primary,
      items: group.items.flatMap((i) => (items[i] === undefined ? [] : [items[i]])),
      weights: groupWeights(rule, group.dimension),
    };
  });
  const latent = latentDistribution(primary);
  return {
    items,
    groups,
    latent,
    correlated: design.correlated,
    maximization: groupMaximization(groups),
    expectation: bifactorExpectation(table, groups, latent.logWeights),
  };
}

/**
 * Returns the entries of the tables over the nodes of the groups: each item's categories at the nodes of its group and,
 * with `pairs`, the products of the categories of each two items of a group.
 */
function tableEntries<TParameters>(
  items: readonly ItemState<TParameters>[],
  groups: readonly GroupPatterns[],
  grids: readonly Grid[],
  pairs: boolean,
): number {
  return groups.reduce((sum, { group }, s) => {
    const categories = group.items.map((i) => items[i]?.categoryCount ?? 0);
    let entries = categories.reduce((total, count) => total + count, 0);
    if (pairs) {
      categories.forEach((count, j) => {
        for (const other of categories.slice(j + 1)) entries += count * other;
      });
    }
    return sum + entries * (grids[s]?.count ?? 0);
  }, 0);
}

/** Returns the M-step of the items, each over the nodes of its group. */
function groupMaximization<TParameters>(groups: readonly GroupNodes<TParameters>[]): ItemMaximization<TParameters> {
  const steps = new Map<ItemState<TParameters>, ItemMaximization<TParameters>>();
  for (const { items, grid } of groups) {
    const step = itemMaximization(items, grid);
    for (const item of items) steps.set(item, step);
  }
  return {
    assign: (item, eta) => steps.get(item)?.assign(item, eta),
    objective: (item, eta, gradient, hessian) => steps.get(item)?.objective(item, eta, gradient, hessian) ?? 0,
    maximize: (item) => steps.get(item)?.maximize(item),
    coordinates: (item) => steps.get(item)?.coordinates(item) ?? new Float64Array(0),
  };
}

/**
 * Runs EM with SQUAREM until no parameter changes by more than the tolerance, and returns the solution with the
 * log-likelihood at the estimates.
 */
export function solveBifactor<TParameters>(
  fit: BifactorFit<TParameters>,
  dimensions: number,
  settings: BifactorSettings,
): Solution<TParameters> {
  const { standardErrors } = settings;
  if (standardErrors !== "none") checkStandardErrorSize(bifactorParameterCount(fit));
  const { converged, iterations } = squarem(emIteration(fit), settings);
  const logLikelihood = fit.expectation.run();
  return {
    items: fit.items,
    correlations: embedded(fit.latent, dimensions),
    correlated: fit.correlated,
    latentSize: fit.latent.size,
    groups: null,
    covariance: standardErrors === "none" ? null : bifactorCovariance(fit, standardErrors),
    logLikelihood,
    converged,
    iterations,
  };
}

/** Returns the correlations of all dimensions: those of the primary dimensions, and the specific ones uncorrelated. */
function embedded(latent: Latent, dimensions: number): Float64Array {
  const matrix = new Float64Array(dimensions * dimensions);
  for (let i = 0; i < dimensions; i++) {
    for (let j = 0; j < dimensions; j++) {
      const inside = i < latent.dimensions && j < latent.dimensions;
      matrix[i * dimensions + j] = inside ? (latent.matrix[i * latent.dimensions + j] ?? 0) : Number(i === j);
    }
  }
  return matrix;
}

/**
 * Returns EM as an iteration on the unbounded coordinates of the items' estimated entries, then the parameters of the
 * correlation, which any values keep a correlation matrix when it is positive definite.
 */
function emIteration<TParameters>(fit: BifactorFit<TParameters>): Iteration {
  const { items, latent, correlated, maximization, expectation } = fit;
  const itemSize = items.reduce((sum, item) => sum + item.free.length, 0);
  const current = new Float64Array(itemSize + (correlated ? latent.pairs.length : 0));
  const previous = new Float64Array(current.length);
  return {
    step() {
      const logLikelihood = expectation.run();
      estimatesOf(fit, previous);
      for (const item of items) maximization.maximize(item);
      if (correlated) latent.maximize(expectation.nodeCounts);
      estimatesOf(fit, current);
      return { logLikelihood, change: largestChange(previous, current) };
    },
    get() {
      const state = new Float64Array(current.length);
      let slot = 0;
      for (const item of items) {
        state.set(maximization.coordinates(item), slot);
        slot += item.free.length;
      }
      if (correlated) state.set(latent.parameters(), itemSize);
      return state;
    },
    set(state) {
      // A correlation that is not positive definite stays as it was: the log-likelihood at the jump decides.
      if (correlated) latent.assign(state.subarray(itemSize));
      let slot = 0;
      for (const item of items) {
        maximization.assign(item, state.subarray(slot, slot + item.free.length));
        slot += item.free.length;
      }
    },
  };
}

/** Writes the estimated parameters, on their own scale, into `out`: the items' estimated entries, then the correlation. */
function estimatesOf<TParameters>({ items, latent, correlated }: BifactorFit<TParameters>, out: Float64Array): void {
  let slot = 0;
  for (const item of items) for (const j of item.free) out[slot++] = item.vector[j] ?? 0;
  if (correlated) for (const [i, j] of latent.pairs) out[slot++] = latent.matrix[i * latent.dimensions + j] ?? 0;
}

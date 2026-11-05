import { type CompressedResponses, checkStandardErrorSize } from "../irt/internal.js";
import { type BifactorExpectation, bifactorExpectation, type GroupNodes } from "./bifactor-expectation.js";
import { groupGrid, groupWeights, specificRule } from "./bifactor-nodes.js";
import { type GroupPatterns, patternsOf } from "./bifactor-patterns.js";
import { bifactorCovariance } from "./bifactor-standard-errors.js";
import type { BifactorStructure } from "./bifactor-structure.js";
import type { MultidimensionalCalibrationOptions } from "./calibration-options.js";
import { type ItemState, setUp } from "./calibration-setup.js";
import { type Latent, latentDistribution } from "./correlation.js";
import { expectationMaximization } from "./em.js";
import { checkTables, type Grid, productGrid } from "./grid.js";
import { type ItemMaximization, itemMaximization } from "./m-step.js";
import type { MultidimensionalModel } from "./model.js";
import type { Settings, Solution } from "./solution.js";
import { type EmModel, parameterCount } from "./standard-errors.js";

/** Settings of the estimation of a bifactor model, after their defaults are filled in. */
export interface BifactorSettings {
  readonly tolerance: number;
  readonly maxIterations: number;
  readonly standardErrors: Settings["standardErrors"];
}

/**
 * What EM works on: the items in the item order of the data, with `latents` the distribution of the primary dimensions
 * when its correlation is estimated.
 */
export interface BifactorFit<TParameters> extends EmModel<TParameters> {
  readonly groups: readonly GroupNodes<TParameters>[];
  /** The distribution of the primary dimensions. */
  readonly latent: Latent;
  /** Whether the correlation of the two primary dimensions is estimated. */
  readonly correlated: boolean;
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
  model: MultidimensionalModel<TParameters>,
  options: Pick<MultidimensionalCalibrationOptions<TParameters>, "priors">,
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
  const unplaced = setUp(responses, model, options, design, 0);
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
  const expectation = bifactorExpectation(table, groups, latent.logWeights);
  return {
    items,
    groups,
    latent,
    latents: design.correlated ? [latent] : [],
    correlated: design.correlated,
    maximization: groupMaximization(groups),
    runExpectation: () => expectation.run(),
    countsOf: () => expectation.nodeCounts,
    expectation,
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
  const { standardErrors, tolerance, maxIterations } = settings;
  if (standardErrors !== "none") checkStandardErrorSize(parameterCount(fit));
  const { converged, iterations } = expectationMaximization(fit, tolerance, maxIterations);
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

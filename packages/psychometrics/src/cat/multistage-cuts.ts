import {
  checkArray,
  checkChoice,
  checkOptions,
  checkPositive,
  DataError,
  normalQuantile,
  type ThetaDistribution,
} from "@measurekit/core";
import {
  checkRange,
  checkThetaDistribution,
  expectedScore,
  type IrtItem,
  indexItems,
  largestCategoryCount,
} from "../irt/internal.js";
import { findRoot } from "../math/internal.js";
import {
  locateStages,
  type MultistageModule,
  type MultistageRouting,
  type MultistageTestOptions,
  ROUTINGS,
} from "./multistage-test.js";

/** The stages of a panel, as {@link multistageTest} takes them. */
type Stages = MultistageTestOptions["stages"];

/**
 * How {@link multistageCuts} derives the cuts between the modules of a stage: `maximum-information`, at the theta where
 * the information of two adjacent modules is equal; `routing-shares`, at the thetas that send given shares of a
 * population to the modules.
 */
export type MultistageCutsMethod = "maximum-information" | "routing-shares";

/** Options of {@link multistageCuts}. */
export interface MultistageCutsOptions {
  /**
   * How the cuts between the modules of a stage are derived.
   *
   * @defaultValue `"maximum-information"`
   */
  readonly method?: MultistageCutsMethod;
  /**
   * The scale of the cuts: `theta`, or `number-correct`, the total score of the responses so far, as
   * {@link MultistageTestOptions.routing} names them.
   *
   * @defaultValue `"theta"`
   */
  readonly scale?: MultistageRouting;
  /**
   * The interval of theta in which the peaks and the crossings of the information curves are searched. Only
   * `maximum-information` takes it.
   *
   * @defaultValue `[-6, 6]`
   */
  readonly range?: readonly [lower: number, upper: number];
  /**
   * The distribution of theta of the persons the shares are of, with nodes in ascending order. Only `routing-shares`
   * takes it.
   *
   * @defaultValue the standard normal distribution
   */
  readonly population?: ThetaDistribution;
  /**
   * The shares of the population that each module takes, one list for each stage after the first with a positive share
   * for each of its modules, in the order of the modules; a list sums to 1 within 1e-6 and is divided by its sum. Only
   * `routing-shares` takes it.
   *
   * @defaultValue equal shares for the modules of a stage
   */
  readonly shares?: readonly (readonly number[])[];
}

const METHODS: readonly MultistageCutsMethod[] = ["maximum-information", "routing-shares"];

/** The method that reads each option that only one method reads. */
const OPTIONS_OF_METHOD: readonly (readonly [
  option: "range" | "population" | "shares",
  method: MultistageCutsMethod,
])[] = [
  ["range", "maximum-information"],
  ["population", "routing-shares"],
  ["shares", "routing-shares"],
];

/** The spacing of the grid of theta on which the peaks of the information curves are found; theta has unit SD. */
const GRID_STEP = 0.01;

/**
 * A score within this of a whole score counts as it, so that the rounding of a curve and a root does not lift a cut
 * that lies exactly at a whole score, such as the middle of a symmetric module, to the next.
 */
const SCORE_TOLERANCE = 1e-9;

/** How far the shares of a stage may be from summing to 1: the tolerance of the weights of a `ThetaDistribution`. */
const SHARE_SUM_TOLERANCE = 1e-6;

/**
 * Returns the cut points of every module of a multistage panel that routes to the next stage, in the order of the
 * panel, ready as the `cuts` of the modules in {@link multistageTest}.
 *
 * @remarks
 * A module of the stages but the last has one fewer cut than the modules of the next stage, in ascending order. The
 * cuts into a stage are the same for every module of the stage before, as a person of a given theta belongs in the same
 * module of a stage whichever module they come from; only on the scale `number-correct` do they differ, by the items
 * taken before. The `cuts` of the modules passed in are ignored.
 *
 * `maximum-information` puts the cut between two adjacent modules of a stage at the theta where their information is
 * equal, so that a person is routed to the more informative module; routes through the two modules share what comes
 * before, and compare by the modules' information alone. The modules of a stage must go from the easiest to the
 * hardest. Where the curves cross more than once the cut is the first crossing above the peak of the easier module, and
 * it must lie before the peak of the harder one: each module must be the more informative at its own peak. The peaks
 * are found on a grid of 0.01 over `range`.
 *
 * `routing-shares` puts the cuts at the quantiles of the population that send the shares to the modules, from the
 * easiest. A population of nodes spreads the weight of each node evenly over the cell between the midpoints to its
 * neighbors, the first and the last cell as wide as the one beside it, so that a quantile varies smoothly with its
 * share. The standard normal population is exact.
 *
 * On the scale `number-correct`, the cut at a theta is the smallest whole score at or above the expected score there,
 * the test characteristic curve of the items taken so far, so that a person with at least that score is routed up. The
 * items are those of the routing module and of the module of each earlier stage that routing by theta chooses at the
 * cut. A score within 1e-9 of a whole score counts as it.
 *
 * @param stages - The stages of the panel, in order, each a list of modules, as {@link multistageTest} takes them.
 * @param options - The method, the scale and what the method reads.
 * @returns The cuts of each module of the stages but the last, in ascending order.
 * @throws `DataError` when the stages, the items or an option are invalid, an option belongs to the other method, the
 * information of adjacent modules does not cross between their peaks, or the cuts of a module on the scale
 * `number-correct` do not ascend.
 * @see {@link https://doi.org/10.1207/s15324818ame1903_2 | Luecht, Brumfield and Breithaupt (2006)}
 */
export function multistageCuts(stages: Stages, options: MultistageCutsOptions = {}): readonly (readonly number[])[] {
  checkOptions(options);
  const { method = "maximum-information", scale = "theta", range, population, shares } = options;
  checkChoice(method, "The option method", METHODS);
  checkChoice(scale, "The option scale", ROUTINGS);
  for (const [option, owner] of OPTIONS_OF_METHOD) {
    if (options[option] !== undefined && owner !== method) {
      throw new DataError(`The option ${option} belongs to the method "${owner}", not to "${method}".`);
    }
  }
  locateStages(stages);
  indexItems(stages.flat().flatMap((module) => module.items));
  // The cuts into each stage by theta, the first stage with none.
  const thetaCuts =
    method === "maximum-information"
      ? informationCuts(stages, checkRange(range))
      : shareCuts(stages, population, shares);
  return stages.slice(0, -1).flatMap((modules, stage) =>
    modules.map((module) => {
      const cuts = thetaCuts[stage + 1] ?? [];
      return scale === "theta" ? [...cuts] : numberCorrectCuts(stages, thetaCuts, stage, module, cuts);
    }),
  );
}

/** A module's information on the grid, with the index of its largest value. */
interface InformationCurve {
  readonly module: MultistageModule;
  readonly values: Float64Array;
  readonly peak: number;
}

/** Returns the cuts into each stage at the crossings of the information of its adjacent modules. */
function informationCuts(stages: Stages, [lower, upper]: readonly [number, number]): number[][] {
  const steps = Math.max(1, Math.ceil((upper - lower) / GRID_STEP));
  const grid = Float64Array.from({ length: steps + 1 }, (_, i) => lower + ((upper - lower) * i) / steps);
  return stages.map((modules, stage) => {
    if (modules.length === 1) return [];
    const curves = modules.map((module): InformationCurve => {
      const values = grid.map((theta) => information(module.items, theta));
      return { module, values, peak: values.indexOf(values.reduce((most, value) => Math.max(most, value), 0)) };
    });
    return curves.slice(1).map((above, j) => crossing(stage, curves[j] ?? above, above, grid));
  });
}

function information(items: readonly IrtItem[], theta: number): number {
  let sum = 0;
  for (const item of items) sum += item.model.information(item.parameters, theta);
  return sum;
}

/**
 * Returns the first theta above the peak of `below` at which `above` is at least as informative.
 *
 * @throws `DataError` when the peaks do not ascend, or `below` is the less informative at its own peak or `above` at its.
 */
function crossing(stage: number, below: InformationCurve, above: InformationCurve, grid: Float64Array): number {
  const names = `"${below.module.id}" and "${above.module.id}" of stage ${stage + 1}`;
  const start = below.peak;
  const end = above.peak;
  if (!(start < end)) {
    throw new DataError(
      `The information of modules ${names} peaks at theta ${grid[start]} and ${grid[end]}; a stage needs its modules from the easiest to the hardest, with peaks that ascend.`,
    );
  }
  const gap = above.values.map((value, i) => value - (below.values[i] ?? 0));
  if (!((gap[start] ?? 0) <= 0 && (gap[end] ?? 0) >= 0)) {
    throw new DataError(
      `The information of modules ${names} does not cross between their peaks at theta ${grid[start]} and ${grid[end]}: each needs to be the more informative at its own peak.`,
    );
  }
  const first = gap.findIndex((value, i) => i >= start && value >= 0);
  const [lower, upper] = [grid[first - 1] ?? 0, grid[first] ?? 0];
  const [gapLower, gapUpper] = [gap[first - 1] ?? 0, gap[first] ?? 0];
  if (first === start || gapUpper === 0) return upper;
  return findRoot(
    (theta) => information(above.module.items, theta) - information(below.module.items, theta),
    lower,
    upper,
    gapLower,
    gapUpper,
  );
}

/** Returns the cuts into each stage at the quantiles of the population that send the shares to its modules. */
function shareCuts(
  stages: Stages,
  population: ThetaDistribution | undefined,
  shares: readonly (readonly number[])[] | undefined,
): number[][] {
  const quantile = populationQuantile(population);
  if (shares !== undefined) {
    checkArray(shares, "The option shares must be an array of the shares of the modules of each stage after the first");
    if (shares.length !== stages.length - 1) {
      throw new DataError(
        `The option shares needs a list for each of the ${stages.length - 1} stages after the first, found ${shares.length}.`,
      );
    }
  }
  return stages.map((modules, stage) => {
    if (stage === 0) return [];
    const stageShares =
      shares === undefined ? modules.map(() => 1) : checkShares(shares[stage - 1], stage, modules.length);
    const total = stageShares.reduce((sum, share) => sum + share, 0);
    const cuts: number[] = [];
    let below = 0;
    for (const share of stageShares.slice(0, -1)) {
      below += share;
      cuts.push(quantile(below / total));
    }
    return cuts;
  });
}

/**
 * Checks the shares of the modules of a stage.
 *
 * @throws `DataError` when they are not a list of a positive share for each module, summing to 1.
 */
function checkShares(list: readonly number[] | undefined, stage: number, count: number): readonly number[] {
  checkArray(list, `The shares of the modules of stage ${stage + 1} must be an array`);
  // The check has thrown for what is not a list; the default is for the types.
  const shares = list ?? [];
  if (shares.length !== count) {
    throw new DataError(`Stage ${stage + 1} has ${count} modules, found ${shares.length} shares.`);
  }
  for (const share of shares) checkPositive(share, `A share of a module of stage ${stage + 1}`);
  const sum = shares.reduce((total, share) => total + share, 0);
  if (!(Math.abs(sum - 1) <= SHARE_SUM_TOLERANCE)) {
    throw new DataError(`The shares of the modules of stage ${stage + 1} must sum to 1, found ${sum}.`);
  }
  return shares;
}

/**
 * Returns the quantile function of a population: the standard normal by default, else the nodes with their weights
 * spread over the cells between the midpoints of the nodes.
 *
 * @throws `DataError` when the population is invalid or its nodes are not in ascending order.
 */
function populationQuantile(population: ThetaDistribution | undefined): (probability: number) => number {
  if (population === undefined) return normalQuantile;
  const { nodes, weights } = checkThetaDistribution(population, "population");
  nodes.forEach((node, q) => {
    const previous = nodes[q - 1] ?? node;
    if (node < previous) {
      throw new DataError(`The nodes of the population must be in ascending order, found ${previous} before ${node}.`);
    }
  });
  // The end cells are as wide as their neighbors.
  const last = nodes.length - 1;
  const first = nodes[0] ?? 0;
  const final = nodes[last] ?? 0;
  const lowEnd = first - ((nodes[1] ?? first) - first) / 2;
  const highEnd = final + (final - (nodes[last - 1] ?? final)) / 2;
  const edges = Float64Array.from({ length: nodes.length + 1 }, (_, q) =>
    q === 0 ? lowEnd : q === nodes.length ? highEnd : ((nodes[q - 1] ?? 0) + (nodes[q] ?? 0)) / 2,
  );
  return (probability) => {
    let below = 0;
    let cell = 0;
    while (cell < weights.length - 1 && below + (weights[cell] ?? 0) < probability) {
      below += weights[cell] ?? 0;
      cell++;
    }
    const weight = weights[cell] ?? 0;
    const fraction = weight > 0 ? Math.min(1, (probability - below) / weight) : 0;
    const lower = edges[cell] ?? 0;
    return lower + fraction * ((edges[cell + 1] ?? 0) - lower);
  };
}

/**
 * Returns the cuts of a module on the scale of number-correct from its cuts by theta.
 *
 * @throws `DataError` when the cuts do not ascend.
 */
function numberCorrectCuts(
  stages: Stages,
  thetaCuts: readonly (readonly number[])[],
  stage: number,
  module: MultistageModule,
  cuts: readonly number[],
): number[] {
  const scores = cuts.map((theta) => {
    const route = stages.slice(0, stage).flatMap((modules, earlier) => {
      const position = (thetaCuts[earlier] ?? []).filter((cut) => cut <= theta).length;
      return modules[position]?.items ?? [];
    });
    // Math.max turns -0 into 0.
    return Math.max(0, Math.ceil(expectedTotal([...route, ...module.items], theta) - SCORE_TOLERANCE));
  });
  if (scores.some((score, j) => score < (scores[j - 1] ?? score))) {
    throw new DataError(
      `The cuts of module "${module.id}" on the scale number-correct would be [${scores.join(", ")}], which do not ascend: the routes at its cuts have expected scores that fall as theta rises. Use the scale theta, or modules whose expected scores do not fall.`,
    );
  }
  return scores;
}

/** Returns the test characteristic curve of the items: the sum of their expected scores at `theta`. */
function expectedTotal(items: readonly IrtItem[], theta: number): number {
  const probabilities = new Float64Array(largestCategoryCount(items));
  let score = 0;
  for (const item of items) score += expectedScore(item, theta, probabilities);
  return score;
}

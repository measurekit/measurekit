import { type DescentPoint, maxAbs, stepAlong } from "@measurekit/core";
import {
  type ArrowSystem,
  addDiagonal,
  addItemHessian,
  arrowDirection,
  arrowSystem,
  clearSystem,
  scaleSystem,
} from "./arrow-system.js";
import { fromUnbounded, toUnbounded, unboundedDerivative } from "./bounds.js";
import type { FreeParameter, ItemState, Setup } from "./calibration-setup.js";
import { distributionObjective, normalWeights } from "./latent.js";
import type { IrtModel } from "./model.js";
import { MIN_PROBABILITY } from "./probability.js";
import { testletObjective, testletWeights } from "./testlets.js";

/** Items whose parameters the M-step estimates together because they share some, with those free parameters. */
interface Block<TParameters> {
  readonly items: readonly ItemState<TParameters>[];
  readonly parameters: Int32Array;
  /** Whether the block includes the parameters of the normal distributions of theta. */
  readonly distributions: boolean;
}

/** The M-step of a calibration. */
export interface Maximization<TParameters> {
  /** The blocks of estimated items that the M-step maximizes separately. */
  readonly blocks: readonly Block<TParameters>[];
  /** All estimated items with all free parameters, those of the normal distributions of theta included. */
  readonly all: Block<TParameters>;
  /**
   * Sets the block's free parameters from their unbounded coordinates `eta`, and its items' parameters and its
   * distributions' weights from them.
   */
  assign(block: Block<TParameters>, eta: Float64Array): void;
  /**
   * Assigns `eta` to the block and returns the negative expected complete-data log-posterior of its items and
   * distributions, given the expected counts of the last E-step, writing its gradient with respect to `eta` into
   * `gradient`.
   */
  objective(block: Block<TParameters>, eta: Float64Array, gradient: Float64Array): number;
  /**
   * Takes one Fisher-scoring step up the block's expected complete-data log-posterior, with a line search, and assigns
   * the result: a generalized EM step, which never lowers the expected log-posterior and, from the previous iteration's
   * estimates, is close to exact. The step solves the block's arrow-shaped system by its Schur complement.
   */
  maximize(block: Block<TParameters>): void;
}

/** Returns the M-step of a calibration, with its scratch space. */
export function maximization<TParameters>(setup: Setup<TParameters>, nodes: Float64Array): Maximization<TParameters> {
  const { items, population, design, free, values } = setup;
  const estimated = items.filter((item) => item.estimated);
  const gradient = new Float64Array(values.length);
  const categories = estimated.reduce((most, item) => Math.max(most, item.categoryCount), 0);
  const entries = estimated.reduce((most, item) => Math.max(most, item.vector.length), 0);
  const probabilities = new Float64Array(categories);
  const derivatives = new Float64Array(categories * entries);
  // The Hessian of one item's free entries, by rows, which the block's system takes apart.
  const local = new Float64Array(entries * entries);
  const systems = new Map<Block<TParameters>, ArrowSystem>();

  const assign = (block: Block<TParameters>, eta: Float64Array): void => {
    for (let l = 0; l < block.parameters.length; l++) {
      const j = block.parameters[l] ?? 0;
      const { lower, upper } = free[j] ?? UNBOUNDED;
      values[j] = fromUnbounded(eta[l] ?? 0, lower, upper);
    }
    for (const item of block.items) {
      for (const p of item.freeEntries) item.vector[p] = values[item.slots[p] ?? 0] ?? 0;
      item.parameters = item.model.fromVector(item.vector);
    }
    if (block.distributions) refreshWeights();
  };

  /** Sets the weights of the normal distributions and the testlets from their parameters. */
  const refreshWeights = (): void => {
    normalWeights(population, values, nodes);
    if (design !== null) testletWeights(design, values, nodes);
  };

  // The position of each free parameter in the block being evaluated, and dx/dη at each position.
  const positions = new Int32Array(values.length);
  const scales = new Float64Array(values.length);

  /**
   * Returns -Σ count log P over an item's nodes and categories, adds its gradient at the item's slots and, when
   * `information` is set, writes the expected information Σ n P'P'ᵀ / P of the item's counts, n persons at a node, of
   * its free entries into `local`.
   */
  const itemObjective = (item: ItemState<TParameters>, information: boolean): number => {
    if (information) local.fill(0, 0, item.freeEntries.length ** 2);
    let value = 0;
    for (let q = 0; q < item.nodes.length; q++) value += nodeObjective(item, q, information);
    return value;
  };

  /** Returns the item's term of {@link itemObjective} at node q. */
  const nodeObjective = (item: ItemState<TParameters>, q: number, information: boolean): number => {
    const theta = item.nodes[q] ?? 0;
    const categoryCount = item.categoryCount;
    item.model.probabilities(item.parameters, theta, probabilities);
    item.model.parameterDerivatives(item.parameters, theta, derivatives);
    let persons = 0;
    for (let k = 0; k < categoryCount; k++) persons += item.counts[q * categoryCount + k] ?? 0;
    let value = 0;
    for (let k = 0; k < categoryCount; k++) {
      const count = item.counts[q * categoryCount + k] ?? 0;
      const probability = Math.max(probabilities[k] ?? 0, MIN_PROBABILITY);
      if (count > 0) value -= count * Math.log(probability);
      addGradient(item, k, count / probability);
      if (information) addInformation(item, k, persons / probability);
    }
    return value;
  };

  /** Subtracts `weight` times category k's derivatives from the gradient at the slots of the item's free entries. */
  const addGradient = (item: ItemState<TParameters>, k: number, weight: number): void => {
    const size = item.vector.length;
    const entries = item.freeEntries;
    for (let e = 0; e < entries.length; e++) {
      const i = entries[e] ?? 0;
      const slot = item.slots[i] ?? 0;
      gradient[slot] = (gradient[slot] ?? 0) - weight * (derivatives[k * size + i] ?? 0);
    }
  };

  /** Adds `weight` times the outer product of category k's derivatives at the item's free entries to `local`. */
  const addInformation = (item: ItemState<TParameters>, k: number, weight: number): void => {
    const size = item.vector.length;
    const entries = item.freeEntries;
    const count = entries.length;
    for (let e = 0; e < count; e++) {
      const scaled = weight * (derivatives[k * size + (entries[e] ?? 0)] ?? 0);
      for (let f = 0; f < count; f++) {
        const index = e * count + f;
        local[index] = (local[index] ?? 0) + scaled * (derivatives[k * size + (entries[f] ?? 0)] ?? 0);
      }
    }
  };

  /**
   * Returns the objective of {@link Maximization.objective}, and with `system` writes the block's Fisher-scoring
   * Hessian with respect to `eta` into it.
   */
  const evaluate = (
    block: Block<TParameters>,
    eta: Float64Array,
    out: Float64Array,
    system: ArrowSystem | null,
  ): number => {
    assign(block, eta);
    const size = block.parameters.length;
    for (let l = 0; l < size; l++) {
      const j = block.parameters[l] ?? 0;
      gradient[j] = 0;
      positions[j] = l;
    }
    if (system !== null) clearSystem(system);
    let value = 0;
    block.items.forEach((item, t) => {
      value += itemObjective(item, system !== null);
      if (system !== null) addItemHessian(system, t, local, item.freeEntries.length);
    });
    value += priorObjective(block, system);
    if (block.distributions) {
      value += distributionObjective(population, values, nodes, gradient);
      if (design !== null) value += testletObjective(design, values, nodes, gradient);
    }
    toUnboundedScale(block, out, system);
    return value;
  };

  /**
   * Returns minus the sum of the block's log priors, adding their gradient and curvature to the gradient and the
   * Hessian's diagonal on the parameters' own scale.
   */
  const priorObjective = (block: Block<TParameters>, system: ArrowSystem | null): number => {
    const size = block.parameters.length;
    let value = 0;
    for (let l = 0; l < size; l++) {
      const j = block.parameters[l] ?? 0;
      const prior = free[j]?.prior;
      if (prior === undefined) continue;
      const x = values[j] ?? 0;
      value -= prior.logDensity(x);
      gradient[j] = (gradient[j] ?? 0) - prior.logDensityDerivative(x);
      if (system !== null) addDiagonal(system, l, -prior.logDensitySecondDerivative(x));
    }
    return value;
  };

  /**
   * Writes the block's gradient on the unbounded scale into `out` and rescales the Hessian: J g and J H J, with J the
   * diagonal of dx/dη. Fisher scoring leaves out the term in the second derivative of the transformation, which
   * vanishes for unbounded parameters.
   */
  const toUnboundedScale = (block: Block<TParameters>, out: Float64Array, system: ArrowSystem | null): void => {
    const size = block.parameters.length;
    for (let l = 0; l < size; l++) {
      const j = block.parameters[l] ?? 0;
      const { lower, upper } = free[j] ?? UNBOUNDED;
      scales[l] = unboundedDerivative(values[j] ?? 0, lower, upper);
      out[l] = (gradient[j] ?? 0) * (scales[l] ?? 0);
    }
    if (system !== null) scaleSystem(system, scales);
  };

  return {
    blocks: blocksOf(estimated),
    all: { items: estimated, parameters: Int32Array.from(values.keys()), distributions: true },
    assign,
    objective: (block, eta, out) => evaluate(block, eta, out, null),
    maximize(block) {
      const size = block.parameters.length;
      const system = systems.get(block) ?? arrowSystem(block.items.map(freeSlots), block.parameters);
      systems.set(block, system);
      const point: DescentPoint = {
        x: Float64Array.from(block.parameters, (j) => {
          const { lower, upper } = free[j] ?? UNBOUNDED;
          return toUnbounded(values[j] ?? 0, lower, upper);
        }),
        gradient: new Float64Array(size),
        value: 0,
      };
      point.value = evaluate(block, point.x, point.gradient, system);
      // At a stationary point the step is 0; elsewhere it follows the line search, as one step of Newton's method.
      if (maxAbs(point.gradient) === 0) {
        assign(block, point.x);
        return;
      }
      const search = new Float64Array(size);
      arrowDirection(system, point.gradient, search);
      const trial: DescentPoint = { x: new Float64Array(size), gradient: new Float64Array(size), value: 0 };
      const moved = stepAlong((eta, out) => evaluate(block, eta, out, null), point, search, trial);
      assign(block, moved ? trial.x : point.x);
    },
  };
}

/** A free parameter without bounds or prior. */
const UNBOUNDED: FreeParameter = {
  lower: Number.NEGATIVE_INFINITY,
  upper: Number.POSITIVE_INFINITY,
  prior: undefined,
};

/** Returns the unbounded coordinates of the parameters `values`, whose bounds are in `free`. */
export function unboundedCoordinates(values: ArrayLike<number>, free: readonly FreeParameter[]): Float64Array {
  return Float64Array.from(values, (x, j) => {
    const { lower, upper } = free[j] ?? UNBOUNDED;
    return toUnbounded(x, lower, upper);
  });
}

/** Returns dx/dη at each of the parameters `values`, whose bounds are in `free`. */
export function unboundedScales(values: ArrayLike<number>, free: readonly FreeParameter[]): Float64Array {
  return Float64Array.from(values, (x, j) => {
    const { lower, upper } = free[j] ?? UNBOUNDED;
    return unboundedDerivative(x, lower, upper);
  });
}

/** Returns the free parameters behind an item's free entries. */
function freeSlots<TParameters>(item: ItemState<TParameters>): Int32Array {
  return item.freeEntries.map((p) => item.slots[p] ?? 0);
}

/**
 * Splits the estimated items into blocks the M-step can maximize separately: the items of a model with a shared
 * parameter form one block, every other item its own.
 */
function blocksOf<TParameters>(items: readonly ItemState<TParameters>[]): Block<TParameters>[] {
  const blocks: Block<TParameters>[] = [];
  const sharing = new Map<IrtModel<TParameters>, ItemState<TParameters>[]>();
  for (const item of items) {
    if (item.model.parameters(item.categoryCount).some((parameter) => parameter.shared === true)) {
      const group = sharing.get(item.model) ?? [];
      group.push(item);
      sharing.set(item.model, group);
    } else {
      blocks.push({ items: [item], parameters: freeSlots(item), distributions: false });
    }
  }
  for (const group of sharing.values()) {
    const parameters = new Set(group.flatMap((item) => [...freeSlots(item)]));
    blocks.push({ items: group, parameters: Int32Array.from(parameters), distributions: false });
  }
  return blocks;
}

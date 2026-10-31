import { fromUnbounded, toUnbounded, unboundedDerivative } from "../irt/bounds.js";
import { MIN_PROBABILITY } from "../irt/probability.js";
import { invert } from "../math/linear-algebra.js";
import { type ItemState, UNBOUNDED } from "./calibration-setup.js";
import type { Grid } from "./grid.js";

/**
 * Each M-step takes one Fisher-scoring step, halved until it does not lower the expected log-posterior: a generalized
 * EM step, which from the previous iteration's estimates is close to exact. The step is halved at most this many times.
 */
const MAX_HALVINGS = 40;

/** The M-step of the items of a calibration. */
export interface ItemMaximization<TParameters> {
  /** Sets the item's estimated entries from their unbounded coordinates `eta`, and its parameters from them. */
  assign(item: ItemState<TParameters>, eta: Float64Array): void;
  /**
   * Assigns `eta` to the item and returns the negative expected complete-data log-posterior of the item given the
   * expected counts of the last E-step, writing, when given, its gradient with respect to `eta` into `gradient` and its
   * Fisher-scoring Hessian into `hessian`.
   */
  objective(item: ItemState<TParameters>, eta: Float64Array, gradient?: Float64Array, hessian?: Float64Array): number;
  /** Takes one Fisher-scoring step up the item's expected complete-data log-posterior and assigns the result. */
  maximize(item: ItemState<TParameters>): void;
  /** Returns the unbounded coordinates of the item's estimated entries. */
  coordinates(item: ItemState<TParameters>): Float64Array;
}

/** The scratch space of the sums over nodes. */
interface Scratch {
  readonly grid: Grid;
  readonly theta: Float64Array;
  readonly probabilities: Float64Array;
  readonly derivatives: Float64Array;
}

/** Returns the M-step of the items of a calibration over a grid, with its scratch space. */
export function itemMaximization<TParameters>(
  items: readonly ItemState<TParameters>[],
  grid: Grid,
): ItemMaximization<TParameters> {
  const categories = items.reduce((most, item) => Math.max(most, item.categoryCount), 0);
  const entries = items.reduce((most, item) => Math.max(most, item.vector.length), 0);
  const scratch: Scratch = {
    grid,
    theta: new Float64Array(grid.dimensions),
    probabilities: new Float64Array(categories),
    derivatives: new Float64Array(categories * entries),
  };

  const objective = (
    item: ItemState<TParameters>,
    eta: Float64Array,
    gradient?: Float64Array,
    hessian?: Float64Array,
  ): number => {
    assign(item, eta);
    const size = item.free.length;
    gradient?.fill(0, 0, size);
    hessian?.fill(0, 0, size * size);
    const value = likelihoodTerm(item, scratch, gradient, hessian) + priorTerm(item, gradient, hessian);
    if (gradient !== undefined) toUnboundedScale(item, gradient, hessian);
    return value;
  };

  return {
    assign,
    objective,
    coordinates,
    maximize(item) {
      const size = item.free.length;
      if (size === 0) return;
      const start = coordinates(item);
      const gradient = new Float64Array(size);
      const hessian = new Float64Array(size * size);
      const value = objective(item, start, gradient, hessian);
      const direction = scoringDirection(gradient, hessian);
      const trial = new Float64Array(size);
      for (let step = 1, halving = 0; halving <= MAX_HALVINGS; step /= 2, halving++) {
        for (let l = 0; l < size; l++) trial[l] = (start[l] ?? 0) + step * (direction[l] ?? 0);
        if (objective(item, trial) <= value) return;
      }
      assign(item, start);
    },
  };
}

function assign<TParameters>(item: ItemState<TParameters>, eta: Float64Array): void {
  item.free.forEach((j, l) => {
    const { lower, upper } = item.entries[l] ?? UNBOUNDED;
    item.vector[j] = fromUnbounded(eta[l] ?? 0, lower, upper);
  });
  item.parameters = item.model.fromVector(item.vector, item.dimensions);
}

function coordinates<TParameters>(item: ItemState<TParameters>): Float64Array {
  return Float64Array.from(item.free, (j, l) => {
    const { lower, upper } = item.entries[l] ?? UNBOUNDED;
    return toUnbounded(item.vector[j] ?? 0, lower, upper);
  });
}

/**
 * Returns -Σ count log P over an item's nodes and categories and, when `gradient` is given, adds its gradient at the
 * item's estimated entries and, when `hessian` is given, the expected information Σ n P'P'ᵀ / P of the item's counts,
 * n persons at a node.
 */
function likelihoodTerm<TParameters>(
  item: ItemState<TParameters>,
  scratch: Scratch,
  gradient: Float64Array | undefined,
  hessian: Float64Array | undefined,
): number {
  const { count } = scratch.grid;
  let value = 0;
  for (let q = 0; q < count; q++) {
    let persons = 0;
    for (let k = 0; k < item.categoryCount; k++) persons += item.counts[k * count + q] ?? 0;
    if (persons > 0) value += nodeTerm(item, q, persons, scratch, gradient, hessian);
  }
  return value;
}

/** Returns the term of {@link likelihoodTerm} at node q, where `persons` persons are expected. */
function nodeTerm<TParameters>(
  item: ItemState<TParameters>,
  q: number,
  persons: number,
  scratch: Scratch,
  gradient: Float64Array | undefined,
  hessian: Float64Array | undefined,
): number {
  const { dimensions, count, nodes } = scratch.grid;
  const { theta, probabilities, derivatives } = scratch;
  for (let d = 0; d < dimensions; d++) theta[d] = nodes[q * dimensions + d] ?? 0;
  item.model.probabilities(item.parameters, theta, probabilities);
  if (gradient !== undefined) item.model.parameterDerivatives(item.parameters, theta, derivatives);
  let value = 0;
  for (let k = 0; k < item.categoryCount; k++) {
    const observed = item.counts[k * count + q] ?? 0;
    const probability = Math.max(probabilities[k] ?? 0, MIN_PROBABILITY);
    if (observed > 0) value -= observed * Math.log(probability);
    if (gradient !== undefined) {
      addCategory(item, k, observed / probability, persons / probability, scratch, gradient, hessian);
    }
  }
  return value;
}

/** Subtracts `weight` times category k's derivatives from the gradient and adds `information` times their outer product. */
function addCategory<TParameters>(
  item: ItemState<TParameters>,
  k: number,
  weight: number,
  information: number,
  { derivatives }: Scratch,
  gradient: Float64Array,
  hessian: Float64Array | undefined,
): void {
  const { free } = item;
  const size = free.length;
  const length = item.vector.length;
  for (let l = 0; l < size; l++) {
    const first = derivatives[k * length + (free[l] ?? 0)] ?? 0;
    gradient[l] = (gradient[l] ?? 0) - weight * first;
    if (hessian === undefined) continue;
    for (let m = 0; m < size; m++) {
      hessian[l * size + m] =
        (hessian[l * size + m] ?? 0) + information * first * (derivatives[k * length + (free[m] ?? 0)] ?? 0);
    }
  }
}

/** Returns minus the log priors of the item's entries, adding their derivatives on the entries' own scale. */
function priorTerm<TParameters>(
  item: ItemState<TParameters>,
  gradient: Float64Array | undefined,
  hessian: Float64Array | undefined,
): number {
  const size = item.free.length;
  let value = 0;
  item.free.forEach((j, l) => {
    const prior = item.entries[l]?.prior;
    if (prior === undefined) return;
    const x = item.vector[j] ?? 0;
    value -= prior.logDensity(x);
    if (gradient !== undefined) gradient[l] = (gradient[l] ?? 0) - prior.logDensityDerivative(x);
    if (hessian !== undefined)
      hessian[l * size + l] = (hessian[l * size + l] ?? 0) - prior.logDensitySecondDerivative(x);
  });
  return value;
}

/**
 * Moves the gradient and the Hessian to the unbounded scale: J g and J H J, with J the diagonal of dx/dη. Fisher
 * scoring leaves out the term in the second derivative of the transformation, which vanishes for unbounded parameters.
 */
function toUnboundedScale<TParameters>(
  item: ItemState<TParameters>,
  gradient: Float64Array,
  hessian: Float64Array | undefined,
): void {
  const size = item.free.length;
  const scales = Float64Array.from(item.free, (j, l) => {
    const { lower, upper } = item.entries[l] ?? UNBOUNDED;
    return unboundedDerivative(item.vector[j] ?? 0, lower, upper);
  });
  for (let l = 0; l < size; l++) gradient[l] = (gradient[l] ?? 0) * (scales[l] ?? 0);
  if (hessian === undefined) return;
  for (let l = 0; l < size; l++) {
    for (let m = 0; m < size; m++)
      hessian[l * size + m] = (hessian[l * size + m] ?? 0) * (scales[l] ?? 0) * (scales[m] ?? 0);
  }
}

/** Returns the Fisher-scoring direction -H⁻¹ g, or the steepest descent direction -g where H is singular. */
function scoringDirection(gradient: Float64Array, hessian: Float64Array): Float64Array {
  const size = gradient.length;
  const inverse = invert(hessian, size);
  const direction = new Float64Array(size);
  let slope = 0;
  for (let l = 0; l < size; l++) {
    let sum = 0;
    for (let m = 0; m < size; m++) sum -= (inverse?.[l * size + m] ?? 0) * (gradient[m] ?? 0);
    direction[l] = sum;
    slope += sum * (gradient[l] ?? 0);
  }
  if (inverse !== null && slope < 0) return direction;
  return gradient.map((entry) => -entry);
}

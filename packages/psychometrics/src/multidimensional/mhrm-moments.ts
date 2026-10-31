import { unboundedDerivative } from "../irt/bounds.js";
import { MIN_PROBABILITY } from "../irt/probability.js";
import { type ItemState, UNBOUNDED } from "./calibration-setup.js";
import { assignCoordinates, itemCoordinates } from "./m-step.js";
import { symmetrized } from "./matrix.js";
import type { ItemResponses } from "./mhrm-chain.js";

/** The relative step of the central differences of the complete-data score, about the cube root of epsilon. */
const STEP = 1e-5;

/** Scratch space for the sums over responses, sized for the items and dimensions of a calibration. */
export interface Scratch {
  readonly point: Float64Array;
  readonly probabilities: Float64Array;
  readonly derivatives: Float64Array;
}

/** Returns scratch space for items of the most categories and parameters, and for `dimensions` dimensions. */
export function createScratch<TParameters>(items: readonly ItemState<TParameters>[], dimensions: number): Scratch {
  const categories = items.reduce((most, item) => Math.max(most, item.categoryCount), 0);
  const entries = items.reduce((most, item) => Math.max(most, item.vector.length), 0);
  return {
    point: new Float64Array(dimensions),
    probabilities: new Float64Array(categories),
    derivatives: new Float64Array(categories * entries),
  };
}

/** The draws and the responses an item's sums run over. */
export interface Draws {
  readonly theta: Float64Array;
  readonly responses: ItemResponses;
  readonly dimensions: number;
}

/** Returns dx/dη at each estimated entry of the item. */
function scalesOf<TParameters>(item: ItemState<TParameters>): Float64Array {
  return Float64Array.from(item.free, (j, l) => {
    const { lower, upper } = item.entries[l] ?? UNBOUNDED;
    return unboundedDerivative(item.vector[j] ?? 0, lower, upper);
  });
}

/**
 * Writes the complete-data score of an item's estimated entries on the unbounded scale into `gradient`: the sum over its
 * responses of ∂ ln P(x | θ) / ∂η at the draws.
 */
export function itemMoments<TParameters>(
  item: ItemState<TParameters>,
  index: number,
  { theta, responses, dimensions }: Draws,
  scratch: Scratch,
  gradient: Float64Array,
): void {
  const size = item.free.length;
  const length = item.vector.length;
  const scales = scalesOf(item);
  const { point, probabilities, derivatives } = scratch;
  gradient.fill(0, 0, size);
  for (let r = responses.starts[index] ?? 0; r < (responses.starts[index + 1] ?? 0); r++) {
    const person = responses.respondents[r] ?? 0;
    for (let d = 0; d < dimensions; d++) point[d] = theta[person * dimensions + d] ?? 0;
    item.model.probabilities(item.parameters, point, probabilities);
    item.model.parameterDerivatives(item.parameters, point, derivatives);
    const code = responses.codes[r] ?? 0;
    const observed = Math.max(probabilities[code] ?? 0, MIN_PROBABILITY);
    for (let l = 0; l < size; l++) {
      gradient[l] =
        (gradient[l] ?? 0) + ((derivatives[code * length + (item.free[l] ?? 0)] ?? 0) / observed) * (scales[l] ?? 1);
    }
  }
}

/**
 * Writes the complete-data score of each of an item's responses on the unbounded scale, ∂ ln P(x | θ) / ∂η at the draw of
 * the response's person, into `scores`: the entries of response r start at `offsets[r]`, the responses numbered in the
 * item-major order of {@link ItemResponses}.
 */
export function responseScores<TParameters>(
  item: ItemState<TParameters>,
  index: number,
  { theta, responses, dimensions }: Draws,
  scratch: Scratch,
  scores: Float64Array,
  offsets: Int32Array,
): void {
  const size = item.free.length;
  const length = item.vector.length;
  const rates = scalesOf(item);
  const { point, probabilities, derivatives } = scratch;
  for (let r = responses.starts[index] ?? 0; r < (responses.starts[index + 1] ?? 0); r++) {
    const person = responses.respondents[r] ?? 0;
    for (let d = 0; d < dimensions; d++) point[d] = theta[person * dimensions + d] ?? 0;
    item.model.probabilities(item.parameters, point, probabilities);
    item.model.parameterDerivatives(item.parameters, point, derivatives);
    const code = responses.codes[r] ?? 0;
    const observed = Math.max(probabilities[code] ?? 0, MIN_PROBABILITY);
    for (let l = 0; l < size; l++) {
      scores[(offsets[r] ?? 0) + l] =
        ((derivatives[code * length + (item.free[l] ?? 0)] ?? 0) / observed) * (rates[l] ?? 1);
    }
  }
}

/**
 * Adds the log prior's derivatives of an item's entries to the score and the information, on the unbounded scale: the
 * derivative of ln p(x) is p'(x) dx/dη, and the information leaves out the second derivative of the transformation.
 */
export function priorMoments<TParameters>(
  item: ItemState<TParameters>,
  gradient: Float64Array,
  information?: Float64Array,
): void {
  const size = item.free.length;
  const scales = scalesOf(item);
  item.free.forEach((j, l) => {
    const prior = item.entries[l]?.prior;
    if (prior === undefined) return;
    const x = item.vector[j] ?? 0;
    const scale = scales[l] ?? 1;
    gradient[l] = (gradient[l] ?? 0) + prior.logDensityDerivative(x) * scale;
    if (information !== undefined) {
      information[l * size + l] =
        (information[l * size + l] ?? 0) - prior.logDensitySecondDerivative(x) * scale * scale;
    }
  });
}

/**
 * Writes the observed complete-data information of an item's entries on the unbounded scale into `out`, by rows: minus
 * the derivative of the complete-data score with respect to the entries, by differences of its exact value, forward
 * from the score at the item's parameters or, with `central`, centered on them. The item is left at its parameters.
 */
export function observedItemInformation<TParameters>(
  item: ItemState<TParameters>,
  index: number,
  draws: Draws,
  scratch: Scratch,
  out: Float64Array,
  central: boolean,
): void {
  const size = item.free.length;
  const eta = itemCoordinates(item);
  const moved = Float64Array.from(eta);
  const here = new Float64Array(size);
  const up = new Float64Array(size);
  const down = new Float64Array(size);
  /** Writes the score at the coordinates `at` into `into`. */
  const scoreAt = (at: Float64Array, into: Float64Array): void => {
    assignCoordinates(item, at);
    itemMoments(item, index, draws, scratch, into);
  };
  if (!central) scoreAt(eta, here);
  const below = central ? down : here;
  for (let l = 0; l < size; l++) {
    const step = STEP * Math.max(1, Math.abs(eta[l] ?? 0));
    moved[l] = (eta[l] ?? 0) + step;
    scoreAt(moved, up);
    if (central) {
      moved[l] = (eta[l] ?? 0) - step;
      scoreAt(moved, down);
    }
    moved[l] = eta[l] ?? 0;
    const width = central ? 2 * step : step;
    for (let m = 0; m < size; m++) out[m * size + l] = -((up[m] ?? 0) - (below[m] ?? 0)) / width;
  }
  assignCoordinates(item, eta);
  out.set(symmetrized(out, size));
}

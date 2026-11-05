import { type CompressedResponses, MIN_PROBABILITY } from "../irt/internal.js";
import { mirror } from "../math/internal.js";
import type { ItemState } from "./calibration-setup.js";
import type { Latent } from "./correlation.js";
import type { ExpectationStep } from "./expectation.js";
import type { Grid } from "./grid.js";
import { itemScales } from "./m-step.js";
import { type DimensionSet, restriction, subgridCount, subgridNodes } from "./subgrid.js";

/**
 * The scores of a calibration's estimated parameters, the derivatives of the log density of a person's responses and
 * theta, averaged over the person's posterior: their products summed over persons.
 */
export interface ScoreMoments {
  /** Σ_p E_p[s] E_p[s]ᵀ over persons, by rows: the cross-product information. */
  readonly outer: Float64Array;
  /** Σ_p E_p[s sᵀ], by rows: the cross-product with the posterior variance of the scores. `null` when not asked for. */
  readonly second: Float64Array | null;
}

/** What the scores need of a fit: its items, grid, the distribution of each group and E-step. */
export interface ScoredFit<TParameters> {
  readonly items: readonly ItemState<TParameters>[];
  readonly grid: Grid;
  /** The distribution of each group. */
  readonly latents: readonly Latent[];
  /** The index of each person's group, in the order of the persons of the responses. */
  readonly groupOf: Int32Array;
  readonly eStep: ExpectationStep;
  readonly responses: CompressedResponses;
}

/**
 * The scores of one parameter group at the nodes of the subgrid it depends on: an item's entries, or the entries of a
 * group's distribution, by category (0 for the distributions), then node, then entry.
 */
interface ScoreUnit {
  /** The slot in the parameter vector of its first entry. */
  readonly offset: number;
  /** The number of entries. */
  readonly size: number;
  /** The dimensions its scores depend on. */
  readonly set: DimensionSet;
  /** The number of nodes of its subgrid. */
  readonly nodes: number;
  readonly table: Float64Array;
}

/** A marginal distribution of the posterior over the subgrid of a set of dimensions. */
interface Marginal {
  readonly set: DimensionSet;
  /** The index in its subgrid of each node of the full grid. */
  readonly fromFull: Int32Array;
  readonly weights: Float64Array;
}

/** Returns the number of estimated parameters: the items' estimated entries, then those of the distributions. */
export function parameterCount<TParameters>({
  items,
  latents,
}: Pick<ScoredFit<TParameters>, "items" | "latents">): number {
  return items.reduce((sum, item) => sum + item.free.length, 0) + latents.reduce((sum, { size }) => sum + size, 0);
}

/**
 * Returns the moments of the scores of the estimated parameters over the persons' posteriors.
 *
 * @remarks
 * The score of an item depends on theta through the dimensions it loads on, so its expectation over a person's
 * posterior needs only the posterior's marginal over those dimensions, and the product of the scores of two items only
 * the marginal over the union of their dimensions. The marginals have far fewer nodes than the grid for items that load
 * on few dimensions. The scores of a group's distribution depend on every dimension, and only the persons of the group
 * have them. With `scaled`, the scores are those of the items' entries on the unbounded scale of calibration, the
 * derivatives of ln P times dx/dη.
 *
 * @param withSecond - Whether to also sum the second moments E_p[s sᵀ], which Louis' formula needs.
 */
export function posteriorScoreMoments<TParameters>(
  fit: ScoredFit<TParameters>,
  scaled: boolean,
  withSecond: boolean,
): ScoreMoments {
  const { items, grid, groupOf, eStep, responses } = fit;
  const size = parameterCount(fit);
  const axis = Float64Array.from({ length: grid.points }, (_, k) => grid.nodes[k * grid.dimensions] ?? 0);
  const units = scoreUnits(fit, axis, scaled);
  const latentUnits = fit.latents.map((latent, g) => {
    const before = fit.latents.slice(0, g).filter(({ size }) => size > 0).length;
    return latent.size > 0 ? items.length + before : -1;
  });
  const marginals = marginalsOf(units, grid, withSecond);
  const restrictions = restrictionsOf(units, marginals, grid, withSecond);
  const state: Accumulation = {
    units,
    size,
    person: { answered: new Int32Array(items.length + 1), codes: new Int32Array(items.length + 1), count: 0 },
    marginals,
    shared: { marginals, restrictions, width: 1 << grid.dimensions },
    means: new Float64Array((items.length + 1) * units.reduce((most, unit) => Math.max(most, unit.size), 0)),
    widest: units.reduce((most, unit) => Math.max(most, unit.size), 0),
    outer: new Float64Array(size * size),
    second: withSecond ? new Float64Array(size * size) : null,
  };
  eStep.forEachPosterior((index, posterior) => {
    answeredBy(state.person, units, responses, index, latentUnits[groupOf[index] ?? 0] ?? -1);
    accumulate(state, posterior);
  });
  mirror(state.outer, size);
  if (state.second !== null) mirror(state.second, size);
  return { outer: state.outer, second: state.second };
}

/** The scratch space and the sums of {@link posteriorScoreMoments}. */
interface Accumulation {
  readonly units: readonly ScoreUnit[];
  readonly size: number;
  readonly person: Answered;
  readonly marginals: ReadonlyMap<DimensionSet, Marginal>;
  readonly shared: Marginals;
  /** The posterior mean of each answered unit's scores, `widest` entries per unit. */
  readonly means: Float64Array;
  readonly widest: number;
  readonly outer: Float64Array;
  readonly second: Float64Array | null;
}

/** Adds a person's posterior moments of the scores to the sums. */
function accumulate(state: Accumulation, posterior: Float64Array): void {
  const { units, size, person, marginals, shared, means, widest, outer, second } = state;
  marginalize(marginals, posterior);
  for (let a = 0; a < person.count; a++) {
    meanScore(units[person.answered[a] ?? 0], person.codes[a] ?? 0, marginals, posterior, means, a * widest);
  }
  for (let a = 0; a < person.count; a++) {
    for (let b = a; b < person.count; b++) {
      addProduct(outer, size, units, person, a, b, means, widest);
      const pair = second === null ? undefined : pairOf(units, person, a, b, shared, posterior);
      if (second !== null && pair !== undefined) addSecondMoment(second, size, pair);
    }
  }
}

/** A person's answered units in the order of their slots, with the person's response to each. */
interface Answered {
  readonly answered: Int32Array;
  readonly codes: Int32Array;
  count: number;
}

/**
 * Sets the units a person answered, by item, then the unit of the person's group, and the responses to them. A person's
 * responses come in item order, as `compressResponses` gives them, so that the units are in the order of their slots.
 */
function answeredBy(
  person: Answered,
  units: readonly ScoreUnit[],
  responses: CompressedResponses,
  index: number,
  latentIndex: number,
): void {
  person.count = 0;
  for (let r = responses.offsets[index] ?? 0; r < (responses.offsets[index + 1] ?? 0); r++) {
    const item = responses.items[r] ?? 0;
    if ((units[item]?.size ?? 0) > 0) appendAnswered(person, item, responses.codes[r] ?? 0);
  }
  if (latentIndex >= 0) appendAnswered(person, latentIndex, 0);
}

/** Adds a unit and its response to the person's. */
function appendAnswered(person: Answered, unit: number, code: number): void {
  person.answered[person.count] = unit;
  person.codes[person.count] = code;
  person.count++;
}

/** Returns the score units of the items, in item order, and then of each group's distribution that has estimated entries. */
function scoreUnits<TParameters>(
  { items, grid, latents }: ScoredFit<TParameters>,
  axis: Float64Array,
  scaled: boolean,
): ScoreUnit[] {
  let offset = 0;
  const units = items.map((item) => {
    const unit = itemUnit(item, offset, axis, scaled);
    offset += item.free.length;
    return unit;
  });
  for (const latent of latents) {
    const entries = latent.size;
    if (entries === 0) continue;
    const scores = new Float64Array(entries * grid.count);
    latent.scores(scores);
    const table = new Float64Array(grid.count * entries);
    for (let q = 0; q < grid.count; q++)
      for (let p = 0; p < entries; p++) table[q * entries + p] = scores[p * grid.count + q] ?? 0;
    units.push({ offset, size: entries, set: (1 << grid.dimensions) - 1, nodes: grid.count, table });
    offset += entries;
  }
  return units;
}

/** Returns an item's score unit: ∂ ln P(x | θ) / ∂ξ at the nodes of the subgrid over the dimensions it loads on. */
function itemUnit<TParameters>(
  item: ItemState<TParameters>,
  offset: number,
  axis: Float64Array,
  scaled: boolean,
): ScoreUnit {
  const { dimensions, free, vector } = item;
  let set: DimensionSet = 0;
  for (const j of free) if (j < dimensions) set |= 1 << j;
  // A slope held at a value other than 0 moves the probabilities as an estimated one does, so the scores of the item's
  // other entries depend on its dimension. An anchor has no scores, and needs no dimension.
  if (free.length > 0) for (let d = 0; d < dimensions; d++) if ((vector[d] ?? 0) !== 0) set |= 1 << d;
  const nodes = subgridCount(set, axis.length);
  const coordinates = new Float64Array(nodes * dimensions);
  subgridNodes(set, axis, dimensions, coordinates);
  const scales = scaled ? itemScales(item) : new Float64Array(free.length).fill(1);
  return { offset, size: free.length, set, nodes, table: itemScores(item, coordinates, scales) };
}

/**
 * Returns ∂ ln P_k / ∂ξ_l times `scales[l]` of an item at each of the coordinates, by category, then node, then entry.
 */
function itemScores<TParameters>(
  item: ItemState<TParameters>,
  coordinates: Float64Array,
  scales: Float64Array,
): Float64Array {
  const { dimensions, categoryCount, free, vector } = item;
  const size = free.length;
  const nodes = coordinates.length / dimensions;
  const table = new Float64Array(categoryCount * nodes * size);
  const theta = new Float64Array(dimensions);
  const probabilities = new Float64Array(categoryCount);
  const derivatives = new Float64Array(categoryCount * vector.length);
  for (let r = 0; r < nodes; r++) {
    for (let d = 0; d < dimensions; d++) theta[d] = coordinates[r * dimensions + d] ?? 0;
    item.model.probabilities(item.parameters, theta, probabilities);
    item.model.parameterDerivatives(item.parameters, theta, derivatives);
    for (let k = 0; k < categoryCount; k++) {
      const probability = Math.max(probabilities[k] ?? 0, MIN_PROBABILITY);
      for (let l = 0; l < size; l++) {
        table[(k * nodes + r) * size + l] =
          ((derivatives[k * vector.length + (free[l] ?? 0)] ?? 0) / probability) * (scales[l] ?? 1);
      }
    }
  }
  return table;
}

/** Returns a marginal of the posterior for each set of dimensions that a unit or, with `pairs`, two units depend on. */
function marginalsOf(units: readonly ScoreUnit[], grid: Grid, pairs: boolean): Map<DimensionSet, Marginal> {
  const sets = new Set<DimensionSet>();
  const used = units.filter((unit) => unit.size > 0);
  for (const unit of used) {
    sets.add(unit.set);
    if (pairs) for (const other of used) sets.add(unit.set | other.set);
  }
  const marginals = new Map<DimensionSet, Marginal>();
  const full = (1 << grid.dimensions) - 1;
  for (const set of sets) {
    const size = subgridCount(set, grid.points);
    const fromFull = set === full ? new Int32Array(0) : restriction(full, set, grid.points);
    marginals.set(set, { set, fromFull, weights: new Float64Array(size) });
  }
  return marginals;
}

/** Returns the index of each node of a union's subgrid in the subgrid of each of its parts, for every pair of units. */
function restrictionsOf(
  units: readonly ScoreUnit[],
  marginals: ReadonlyMap<DimensionSet, Marginal>,
  grid: Grid,
  pairs: boolean,
): Map<number, Int32Array> {
  const maps = new Map<number, Int32Array>();
  if (!pairs) return maps;
  const width = 1 << grid.dimensions;
  for (const unit of units) {
    for (const other of units) {
      const whole = unit.set | other.set;
      for (const part of [unit.set, other.set]) {
        const key = whole * width + part;
        if (whole !== part && !maps.has(key) && marginals.has(whole))
          maps.set(key, restriction(whole, part, grid.points));
      }
    }
  }
  return maps;
}

/** Writes the marginals of a posterior over the full grid into the buffers of `marginals`. */
function marginalize(marginals: ReadonlyMap<DimensionSet, Marginal>, posterior: Float64Array): void {
  for (const { fromFull, weights } of marginals.values()) {
    if (fromFull.length === 0) {
      weights.set(posterior);
      continue;
    }
    weights.fill(0);
    for (let q = 0; q < posterior.length; q++) {
      const r = fromFull[q] ?? 0;
      weights[r] = (weights[r] ?? 0) + (posterior[q] ?? 0);
    }
  }
}

/** Writes the posterior mean of a unit's scores at a response into `means` from `start`. */
function meanScore(
  unit: ScoreUnit | undefined,
  code: number,
  marginals: ReadonlyMap<DimensionSet, Marginal>,
  posterior: Float64Array,
  means: Float64Array,
  start: number,
): void {
  if (unit === undefined) return;
  const weights = marginals.get(unit.set)?.weights ?? posterior;
  means.fill(0, start, start + unit.size);
  for (let r = 0; r < unit.nodes; r++) {
    const weight = weights[r] ?? 0;
    const row = (code * unit.nodes + r) * unit.size;
    for (let l = 0; l < unit.size; l++)
      means[start + l] = (means[start + l] ?? 0) + weight * (unit.table[row + l] ?? 0);
  }
}

/** Adds the outer product of two units' posterior means to the block of `matrix` of the first unit's slots. */
function addProduct(
  matrix: Float64Array,
  size: number,
  units: readonly ScoreUnit[],
  { answered }: Answered,
  a: number,
  b: number,
  means: Float64Array,
  widest: number,
): void {
  const unitA = units[answered[a] ?? 0];
  const unitB = units[answered[b] ?? 0];
  if (unitA === undefined || unitB === undefined) return;
  for (let l = 0; l < unitA.size; l++) {
    const left = means[a * widest + l] ?? 0;
    for (let m = 0; m < unitB.size; m++) {
      const index = (unitA.offset + l) * size + unitB.offset + m;
      matrix[index] = (matrix[index] ?? 0) + left * (means[b * widest + m] ?? 0);
    }
  }
}

/** The marginals of the posterior and the maps from the subgrid of a union to those of its parts. */
interface Marginals {
  readonly marginals: ReadonlyMap<DimensionSet, Marginal>;
  readonly restrictions: ReadonlyMap<number, Int32Array>;
  /** The number of subsets of the dimensions, which keys the maps. */
  readonly width: number;
}

/** Two units at their responses, with the posterior marginal over the union of their dimensions. */
interface UnitPair {
  readonly first: ScoreUnit;
  readonly second: ScoreUnit;
  /** The first row of each unit's table at its response. */
  readonly firstRows: number;
  readonly secondRows: number;
  /** The index of each node of the union's subgrid in each unit's subgrid, `undefined` for the identity. */
  readonly toFirst: Int32Array | undefined;
  readonly toSecond: Int32Array | undefined;
  readonly weights: Float64Array;
}

/** Returns the pair of the answered units a and b of a person, or `undefined` for a unit without entries. */
function pairOf(
  units: readonly ScoreUnit[],
  { answered, codes }: Answered,
  a: number,
  b: number,
  { marginals, restrictions, width }: Marginals,
  posterior: Float64Array,
): UnitPair | undefined {
  const first = units[answered[a] ?? 0];
  const second = units[answered[b] ?? 0];
  if (first === undefined || second === undefined) return undefined;
  const whole = first.set | second.set;
  return {
    first,
    second,
    firstRows: (codes[a] ?? 0) * first.nodes,
    secondRows: (codes[b] ?? 0) * second.nodes,
    toFirst: restrictions.get(whole * width + first.set),
    toSecond: restrictions.get(whole * width + second.set),
    weights: marginals.get(whole)?.weights ?? posterior,
  };
}

/** Adds E[s_a s_bᵀ] over the posterior, at the responses of the pair, to the block of the first unit's slots. */
function addSecondMoment(matrix: Float64Array, size: number, pair: UnitPair): void {
  const { first, second, weights, toFirst, toSecond } = pair;
  for (let r = 0; r < weights.length; r++) {
    const weight = weights[r] ?? 0;
    if (weight === 0) continue;
    const firstStart = (pair.firstRows + (toFirst?.[r] ?? r)) * first.size;
    const secondStart = (pair.secondRows + (toSecond?.[r] ?? r)) * second.size;
    for (let l = 0; l < first.size; l++) {
      const left = weight * (first.table[firstStart + l] ?? 0);
      const row = (first.offset + l) * size + second.offset;
      for (let m = 0; m < second.size; m++) {
        matrix[row + m] = (matrix[row + m] ?? 0) + left * (second.table[secondStart + m] ?? 0);
      }
    }
  }
}

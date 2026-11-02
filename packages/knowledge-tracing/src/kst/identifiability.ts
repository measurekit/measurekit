import { DataError } from "@measurekit/core";
import { entry } from "../data/internal.js";
import { rank } from "../math/internal.js";
import { constraintsOf, type Restrictions, sharesRates } from "./constraints.js";
import type { Blim } from "./estimation.js";
import { type KstParameters, START_RATE } from "./parameters.js";
import { type CompiledStructure, holds } from "./structure.js";

/**
 * The most cells of the Jacobian that local identifiability builds: 2 ^ items rows by the parameters. Ten million cells
 * are 80 MB of doubles; 20 items and 10,000 states would be 84 GB.
 */
const MAX_JACOBIAN_CELLS = 1e7;

/** The local identifiability of a BLIM at its parameters. */
export interface Identifiability {
  /** The free parameters: the states' probabilities but one, and the guess and slip of each item that is not fixed. */
  readonly parameters: number;
  /** The rank of the Jacobian of the probabilities of the patterns of responses, in these parameters. */
  readonly rank: number;
  /** Whether the rank is the number of parameters: no change of them leaves every pattern's probability as it was. */
  readonly identifiable: boolean;
}

/** What one pattern of responses needs: the model, and the tables it fills. */
interface Setup {
  readonly states: number;
  readonly items: number;
  /** 1 at state × items + item when the state masters the item. */
  readonly mastery: Uint8Array;
  readonly probabilities: readonly number[];
  readonly blim: Blim;
  /** The column of each item's slip and guess in the Jacobian, or -1 where the rate is fixed. */
  readonly slipColumn: Int32Array;
  readonly guessColumn: Int32Array;
  readonly columns: Float64Array[];
  readonly factors: Float64Array;
  readonly prefix: Float64Array;
}

/**
 * Returns the local identifiability of a model over the elements of a structure, each an item, at its parameters.
 *
 * @remarks
 * The Jacobian has a row for each of the 2 ^ items patterns of responses and a column for each free parameter: the
 * derivative of the pattern's probability with respect to the probability of a state, the last state's being what
 * remains, and to the guess and slip of an item. An item without parameters is at 0.1. Rates held by `fixed` are not
 * parameters. The derivatives are sums of products that leave out the item, so a rate of 0 or 1 does not divide.
 *
 * @throws `DataError` when rates are shared, a fixed rate names an element outside the structure, or the Jacobian
 * would have more cells than the limit.
 */
export function localIdentifiability(
  compiled: CompiledStructure,
  parameters: KstParameters,
  restrictions: Restrictions,
): Identifiability {
  if (sharesRates(restrictions)) {
    throw new DataError(
      "The local identifiability of a model with equal rates has no reference: remove the option equal, or fix the rates.",
    );
  }
  const { domain } = compiled;
  const items = domain.length;
  const blim: Blim = {
    probabilities: Float64Array.from(parameters.probabilities),
    guess: Float64Array.from(domain, (id) => entry(parameters.items, id)?.guess ?? START_RATE),
    slip: Float64Array.from(domain, (id) => entry(parameters.items, id)?.slip ?? START_RATE),
  };
  const { guess, slip } = constraintsOf(restrictions, domain, blim, "is not an element of the structure");
  let width = compiled.count - 1;
  const slipColumn = Int32Array.from(domain, (_, i) => (slip.held[i] === 1 ? -1 : width++));
  const guessColumn = Int32Array.from(domain, (_, i) => (guess.held[i] === 1 ? -1 : width++));
  const rows = 2 ** items;
  if (rows * width > MAX_JACOBIAN_CELLS) {
    throw new DataError(
      `Local identifiability needs a Jacobian of 2 ^ ${items} patterns by ${width} parameters, more than the limit of ${MAX_JACOBIAN_CELLS} cells. Check a structure with fewer items or states, or hold more rates with fixed.`,
    );
  }
  const setup: Setup = {
    states: compiled.count,
    items,
    mastery: Uint8Array.from({ length: compiled.count * items }, (_, at) =>
      holds(compiled, Math.floor(at / items), at % items) ? 1 : 0,
    ),
    probabilities: parameters.probabilities,
    blim,
    slipColumn,
    guessColumn,
    columns: Array.from({ length: width }, () => new Float64Array(rows)),
    factors: new Float64Array(items),
    prefix: new Float64Array(items + 1),
  };
  for (let pattern = 0; pattern < rows; pattern++) fillRow(setup, pattern);
  const found = rank(setup.columns);
  return { parameters: width, rank: found, identifiable: found === width };
}

/** Writes the derivatives of one pattern's probability into its row of every column. */
function fillRow(setup: Setup, pattern: number): void {
  const { states, columns } = setup;
  let last = 0;
  for (let k = 0; k < states; k++) {
    const probability = addState(setup, pattern, k);
    if (k < states - 1) {
      const column = columns[k];
      if (column !== undefined) column[pattern] = (column[pattern] ?? 0) + probability;
    } else {
      last = probability;
    }
  }
  for (let k = 0; k < states - 1; k++) {
    const column = columns[k];
    if (column !== undefined) column[pattern] = (column[pattern] ?? 0) - last;
  }
}

/** Returns the probability of a pattern of responses given a state, and adds the state's share of the rate columns. */
function addState(setup: Setup, pattern: number, k: number): number {
  const { items, factors, prefix } = setup;
  prefix[0] = 1;
  for (let q = 0; q < items; q++) {
    factors[q] = factorOf(setup, pattern, k, q);
    prefix[q + 1] = (prefix[q] ?? 1) * (factors[q] ?? 1);
  }
  addRateDerivatives(setup, pattern, k);
  return prefix[items] ?? 1;
}

/**
 * Adds to the rate columns the state's share of the derivatives of the pattern's probability: its probability times the
 * product of the other items' factors, signed by the slope of the item's own factor.
 */
function addRateDerivatives(setup: Setup, pattern: number, k: number): void {
  const { items, factors, prefix, mastery } = setup;
  let suffix = 1;
  for (let q = items - 1; q >= 0; q--) {
    const mastered = mastery[k * items + q] === 1;
    const column = setup.columns[(mastered ? setup.slipColumn : setup.guessColumn)[q] ?? -1];
    if (column !== undefined) {
      // The factor falls with the slip where the answer is right and rises with it where wrong; the guess does the opposite.
      const slope = (((pattern >>> q) & 1) === 1) === mastered ? -1 : 1;
      column[pattern] = (column[pattern] ?? 0) + (setup.probabilities[k] ?? 0) * (prefix[q] ?? 1) * suffix * slope;
    }
    suffix *= factors[q] ?? 1;
  }
}

/** Returns the probability of the answer to item q in a pattern, given that state k masters the item or not. */
function factorOf({ mastery, items, blim }: Setup, pattern: number, k: number, q: number): number {
  const mastered = mastery[k * items + q] === 1;
  const rate = (mastered ? blim.slip[q] : blim.guess[q]) ?? 0;
  return (((pattern >>> q) & 1) === 1) === mastered ? 1 - rate : rate;
}

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
  readonly probabilities: readonly number[];
  /**
   * At 2 (state × items + item) + answer, the probability of the answer, 1 for right and 0 for wrong, to the item in the
   * state: 1 - slip or slip where the state masters the item, guess or 1 - guess where it does not.
   */
  readonly factor: Float64Array;
  /** At the same index, the sign of the derivative of that probability by the rate it holds: -1 for 1 - rate, 1 for rate. */
  readonly slope: Float64Array;
  /** At state × items + item, the column of the rate that the probability holds, slip or guess, or -1 where it is fixed. */
  readonly rateColumn: Int32Array;
  readonly columns: Float64Array[];
  /** The derivatives of one pattern, by column, summed before they are written to the columns. */
  readonly row: Float64Array;
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
  const cells = compiled.count * items;
  const mastered = (cell: number): boolean => holds(compiled, Math.floor(cell / items), cell % items);
  const rateOf = (cell: number): number => (mastered(cell) ? blim.slip[cell % items] : blim.guess[cell % items]) ?? 0;
  // The answer of index answer % 2 to the item, 1 right, has probability 1 - rate where the state masters the item.
  const isComplement = (answer: number): boolean => (answer % 2 === 1) === mastered(Math.floor(answer / 2));
  const setup: Setup = {
    states: compiled.count,
    items,
    probabilities: parameters.probabilities,
    factor: Float64Array.from({ length: 2 * cells }, (_, answer) => {
      const rate = rateOf(Math.floor(answer / 2));
      return isComplement(answer) ? 1 - rate : rate;
    }),
    slope: Float64Array.from({ length: 2 * cells }, (_, answer) => (isComplement(answer) ? -1 : 1)),
    rateColumn: Int32Array.from(
      { length: cells },
      (_, cell) => (mastered(cell) ? slipColumn : guessColumn)[cell % items] ?? -1,
    ),
    columns: Array.from({ length: width }, () => new Float64Array(rows)),
    row: new Float64Array(width),
    factors: new Float64Array(items),
    prefix: new Float64Array(items + 1),
  };
  for (let pattern = 0; pattern < rows; pattern++) fillRow(setup, pattern);
  const found = rank(setup.columns);
  return { parameters: width, rank: found, identifiable: found === width };
}

/**
 * Writes the derivatives of one pattern's probability into its row of every column. They are summed in a row of their
 * own and written once, as the columns are far apart in memory.
 */
function fillRow(setup: Setup, pattern: number): void {
  const { states, columns, row } = setup;
  row.fill(0);
  let last = 0;
  for (let k = 0; k < states; k++) {
    const probability = addState(setup, pattern, k);
    if (k < states - 1) row[k] = (row[k] ?? 0) + probability;
    else last = probability;
  }
  for (let k = 0; k < states - 1; k++) row[k] = (row[k] ?? 0) - last;
  columns.forEach((column, at) => {
    column[pattern] = row[at] ?? 0;
  });
}

/** Returns the probability of a pattern of responses given a state, and adds the state's share of the rate columns. */
function addState(setup: Setup, pattern: number, k: number): number {
  const { items, factor, factors, prefix } = setup;
  prefix[0] = 1;
  for (let q = 0; q < items; q++) {
    factors[q] = factor[2 * (k * items + q) + ((pattern >>> q) & 1)] ?? 0;
    prefix[q + 1] = (prefix[q] ?? 1) * (factors[q] ?? 1);
  }
  addRateDerivatives(setup, pattern, k);
  return prefix[items] ?? 1;
}

/**
 * Adds to the row of the rate columns the state's share of the derivatives of the pattern's probability: its probability
 * times the product of the other items' factors, signed by the slope of the item's own factor.
 */
function addRateDerivatives(setup: Setup, pattern: number, k: number): void {
  const { items, factors, prefix, row, rateColumn, slope, probabilities } = setup;
  let suffix = 1;
  for (let q = items - 1; q >= 0; q--) {
    const cell = k * items + q;
    const at = rateColumn[cell] ?? -1;
    if (at >= 0) {
      const sign = slope[2 * cell + ((pattern >>> q) & 1)] ?? 0;
      row[at] = (row[at] ?? 0) + (probabilities[k] ?? 0) * (prefix[q] ?? 1) * suffix * sign;
    }
    suffix *= factors[q] ?? 1;
  }
}

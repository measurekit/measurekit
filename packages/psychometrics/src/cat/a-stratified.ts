import { checkArray, checkPositiveInteger, DataError } from "@measurekit/core";
import { DEFAULT_THETA_RANGE, modalTheta } from "../irt/ability.js";
import type { IrtItem } from "../irt/model.js";
import { MIN_PROBABILITY } from "../irt/probability.js";
import type { PoolItem } from "./items.js";
import type { ItemSelectionRule } from "./selection.js";

/**
 * The spacing of the grid on which an item's information is searched for maxima. The information of a polytomous item
 * can have two maxima a few hundredths of theta apart with almost equal heights, as a graded item of slope 2.8 and
 * thresholds 0, 0.11, 0.76 and 1.48 has at 0.45 and 0.51; a coarser grid steps over the pair and may keep the lower one.
 */
const PEAK_SCAN_STEP = 0.01;

/** The strata of a pool: each item's stratum and the theta at which its information peaks. */
interface Strata {
  readonly stratum: Int32Array;
  readonly peak: Float64Array;
}

/**
 * Returns a-stratified selection for items of any model.
 *
 * @remarks
 * The pool is split into `lengths.length` strata of equal size by the items' maximum information. The first
 * `lengths[0]` items come from the stratum of the least informative items, the next `lengths[1]` from the next stratum,
 * and so on; after them, from the last stratum. Within the stratum, the item whose information peaks closest to the
 * current estimate is selected, catR's `thOpt`. For the 1PL and 2PL the maximum information is a² / 4 at θ = b, so the
 * strata are Chang and Ying's, by a, with b matched to theta; for the other models this is maximum information
 * stratification. When the stratum has no eligible item, the next stratum with one is used, otherwise the nearest
 * earlier one. Peaks are searched from -6 to 6. The strata of a pool are computed at its first selection and kept for
 * the pool array, so a pool must not change afterward.
 *
 * @param lengths - The number of items from each stratum, in order of increasing information.
 * @throws `DataError` when `lengths` is not a non-empty array of positive integers.
 * @see {@link https://doi.org/10.1177/01466219922031338 | Chang and Ying (1999)}
 * @see {@link https://pubmed.ncbi.nlm.nih.gov/17296025/ | Barrada et al. (2006)}
 */
export function aStratified(lengths: readonly number[]): ItemSelectionRule {
  checkArray(lengths, "The lengths of a-stratified selection must be an array");
  if (lengths.length === 0) throw new DataError("a-stratified selection needs at least one stratum.");
  lengths.forEach((length, s) => {
    checkPositiveInteger(length, `The length of stratum ${s + 1}`);
  });
  const strataCount = lengths.length;
  const strataByPool = new WeakMap<readonly PoolItem[], Strata>();
  return {
    scores({ pool, administered, estimate, eligible }, out) {
      let strata = strataByPool.get(pool);
      if (strata === undefined) {
        strata = stratify(pool, strataCount);
        strataByPool.set(pool, strata);
      }
      const { stratum, peak } = strata;
      const chosen = eligibleStratum(stratum, eligible, scheduledStratum(lengths, administered.length), strataCount);
      pool.forEach((_, index) => {
        if (eligible[index] !== 1) return;
        out[index] =
          stratum[index] === chosen ? -Math.abs((peak[index] ?? 0) - estimate.theta) : Number.NEGATIVE_INFINITY;
      });
    },
  };
}

/** Splits a pool into strata of equal size by the items' maximum information, the lowest in stratum 0. */
function stratify(pool: readonly PoolItem[], strataCount: number): Strata {
  const peak = new Float64Array(pool.length);
  const maximum = new Float64Array(pool.length);
  pool.forEach((item, index) => {
    const theta = informationPeak(item);
    peak[index] = theta;
    maximum[index] = item.model.information(item.parameters, theta);
  });
  const order = Array.from(pool.keys()).sort((i, j) => (maximum[i] ?? 0) - (maximum[j] ?? 0) || i - j);
  const stratum = new Int32Array(pool.length);
  order.forEach((index, rank) => {
    stratum[index] = Math.floor((rank * strataCount) / pool.length);
  });
  return { stratum, peak };
}

/** Returns the stratum of the next item after `administered` items: by `lengths`, then the last. */
function scheduledStratum(lengths: readonly number[], administered: number): number {
  let remaining = administered;
  for (let s = 0; s < lengths.length; s++) {
    remaining -= lengths[s] ?? 0;
    if (remaining < 0) return s;
  }
  return lengths.length - 1;
}

/** Returns `scheduled` if it has an eligible item, else the next stratum with one, else the nearest earlier one. */
function eligibleStratum(stratum: Int32Array, eligible: Uint8Array, scheduled: number, strataCount: number): number {
  const available = new Uint8Array(strataCount);
  stratum.forEach((s, index) => {
    if (eligible[index] === 1) available[s] = 1;
  });
  for (let s = scheduled; s < strataCount; s++) if (available[s] === 1) return s;
  for (let s = scheduled - 1; s >= 0; s--) if (available[s] === 1) return s;
  return scheduled;
}

/**
 * Returns the theta in [-6, 6] at which an item's information is largest: of the roots of its derivative
 * I'(θ) = Σ_k (2 P'_k P''_k / P_k - P'_k³ / P_k²) found on a grid of 0.01, and the ends, the one with the most
 * information.
 */
export function informationPeak({ model, parameters }: IrtItem): number {
  const count = model.categoryCount(parameters);
  const probabilities = new Float64Array(count);
  const first = new Float64Array(count);
  const second = new Float64Array(count);
  const slope = (theta: number): number => {
    model.probabilities(parameters, theta, probabilities);
    model.thetaDerivatives(parameters, theta, first, second);
    let sum = 0;
    for (let k = 0; k < count; k++) {
      const p = Math.max(probabilities[k] ?? 0, MIN_PROBABILITY);
      const d = first[k] ?? 0;
      sum += (d * (2 * (second[k] ?? 0) - (d * d) / p)) / p;
    }
    return sum;
  };
  return modalTheta(slope, DEFAULT_THETA_RANGE, (theta) => model.information(parameters, theta), PEAK_SCAN_STEP);
}

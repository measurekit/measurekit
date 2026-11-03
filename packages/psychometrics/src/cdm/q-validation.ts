import { checkOptions, checkUnitInterval } from "@measurekit/core";
import type { ResponseMatrix } from "../data/internal.js";
import { type CdmCalibration, calibratedModel } from "./calibration.js";
import { forEachPosterior } from "./expectation.js";
import { bitPattern, profileIndices, profilesOf } from "./q-matrix.js";

const DEFAULT_EPS = 0.95;
/**
 * Added to a profile's expected correct and incorrect responses, so that a profile without persons has probability 1/2.
 */
const SMOOTHING = 1e-10;

/** Options of {@link validateQMatrix}. */
// biome-ignore lint/style/useNamingConvention: Q-matrix and q-vector are the names of these in the field.
export interface QMatrixValidationOptions {
  /**
   * The share of the full q-vector's variance a q-vector must exceed, as GDINA's `eps`.
   *
   * @defaultValue `0.95`
   */
  readonly eps?: number;
}

/** An item's q-vector and the one the data suggest. */
// biome-ignore lint/style/useNamingConvention: Q-matrix and q-vector are the names of these in the field.
export interface QVectorSuggestion {
  /** The item's id. */
  readonly id: string;
  /** The item's attributes in the calibration. */
  readonly attributes: readonly string[];
  /**
   * The suggested attributes: the fewest whose PVAF exceeds `eps`, of those the largest PVAF; the item's attributes
   * when its PVAF is undefined.
   */
  readonly suggested: readonly string[];
  /**
   * The PVAF of the suggested attributes; `null` when the item's success probabilities do not vary over the profiles,
   * as for an item without responses.
   */
  readonly pvaf: number | null;
}

/**
 * Validates a Q-matrix by the PVAF method, as GDINA's `Qval(method = "PVAF")`. For each item and each q-vector q over
 * the calibration's attributes, ς²(q) = Σ_l w_l (P̄_l - P̄)² is the variance of the item's success probabilities over
 * the latent groups l of q, weighted by their proportions w, with each profile's success probability its persons'
 * expected share correct under the calibration; PVAF(q) = ς²(q) / ς² of all attributes. The suggestion is the q-vector
 * of the fewest attributes whose PVAF exceeds `eps`, of those the one with the largest PVAF, the first in profile order
 * on ties. A latent group whose profiles have no proportion adds nothing to ς². The groups of every q-vector are summed
 * from those of the q-vectors with one attribute more, in time that grows as 3^K for K attributes.
 *
 * @param matrix - The responses of the persons.
 * @param calibration - The calibrated model.
 * @param options - The proportion of variance accounted for that an attribute vector must reach.
 * @returns For each item, its attributes, the suggested attributes and the PVAF of the suggestion.
 * @throws `DataError` when the matrix is invalid, its items are not the calibration's, a response is neither 0 nor 1,
 * or `eps` is not from 0 to 1.
 * @see {@link https://doi.org/10.1007/s11336-015-9467-8 | de la Torre and Chiu (2016)}
 */
// biome-ignore lint/style/useNamingConvention: Q-matrix and q-vector are the names of these in the field.
export function validateQMatrix(
  matrix: ResponseMatrix,
  calibration: CdmCalibration,
  options: QMatrixValidationOptions = {},
): QVectorSuggestion[] {
  checkOptions(options);
  const { eps = DEFAULT_EPS } = options;
  checkUnitInterval(eps, "The option eps");
  const { data, design, state } = calibratedModel(matrix, calibration);
  const profileCount = design.profiles.length;
  const correct = calibration.items.map(() => new Float64Array(profileCount));
  const persons = calibration.items.map(() => new Float64Array(profileCount));
  forEachPosterior(data, design, state, (person, posterior) => {
    for (let j = 0; j < data.itemCount; j++) {
      const response = data.responses[person * data.itemCount + j] ?? -1;
      const itemPersons = persons[j];
      const itemCorrect = correct[j];
      if (response < 0 || itemPersons === undefined || itemCorrect === undefined) continue;
      posterior.forEach((p, c) => {
        itemPersons[c] = (itemPersons[c] ?? 0) + p;
        if (response === 1) itemCorrect[c] = (itemCorrect[c] ?? 0) + p;
      });
    }
  });
  const attributeCount = design.attributes.length;
  const candidates = profilesOf(attributeCount).slice(1);
  const masks = candidates.map((q) => bitPattern(q, q.keys()));
  const indexOf = profileIndices(attributeCount);
  const weights = Float64Array.from(indexOf, (c) => state.proportions[c] ?? 0);
  return calibration.items.map((item, j) => {
    const sums = Float64Array.from(indexOf, (c) => {
      const success = ((correct[j]?.[c] ?? 0) + SMOOTHING) / ((persons[j]?.[c] ?? 0) + 2 * SMOOTHING);
      return (state.proportions[c] ?? 0) * success;
    });
    const variances = groupVariances(weights, sums, attributeCount);
    const full = variances[(1 << attributeCount) - 1] ?? 0;
    if (!(full > 0)) return { id: item.id, attributes: item.attributes, suggested: item.attributes, pvaf: null };
    const pvaf = masks.map((mask) => (variances[mask] ?? 0) / full);
    let best = candidates.length - 1;
    for (let size = 1; size <= attributeCount; size++) {
      let top = -1;
      candidates.forEach((q, i) => {
        const count = q.reduce((sum, value) => sum + value, 0);
        if (count === size && (top < 0 || (pvaf[i] ?? 0) > (pvaf[top] ?? 0))) top = i;
      });
      if (top >= 0 && (pvaf[top] ?? 0) > eps) {
        best = top;
        break;
      }
    }
    return {
      id: item.id,
      attributes: item.attributes,
      suggested: design.attributes.filter((_, k) => candidates[best]?.[k] === 1),
      pvaf: pvaf[best] ?? 0,
    };
  });
}

/**
 * Returns ς²(q) = Σ_l W_l (S_l / W_l - P̄)² of every q-vector q, by its bit pattern, over the latent groups l of q: the
 * profiles that agree on its attributes, with W_l the sum of their weights, S_l of their weighted success probabilities,
 * and P̄ the weighted mean of all. A group without weight adds nothing.
 *
 * @remarks
 * The groups of q are the marginal table over its attributes of the table of the profiles. Each q is reached from the
 * full table by summing out its missing attributes in increasing order, so that each marginal table is summed once from
 * the table of one attribute more: 3^K sums for K attributes, where grouping the profiles anew for each q takes 4^K.
 *
 * @param weights - Each profile's weight, by its bit pattern: bit k for attribute k.
 * @param sums - Each profile's weight times its success probability, by its bit pattern.
 */
function groupVariances(weights: Float64Array, sums: Float64Array, attributeCount: number): Float64Array {
  let totalWeight = 0;
  let totalSum = 0;
  for (let c = 0; c < weights.length; c++) {
    totalWeight += weights[c] ?? 0;
    totalSum += sums[c] ?? 0;
  }
  const mean = totalSum / totalWeight;
  const variances = new Float64Array(weights.length);
  // The marginal tables at each depth of the search: one attribute fewer for each depth, so a child reuses its depth's.
  const tables = Array.from({ length: attributeCount + 1 }, (_, depth) => ({
    weights: new Float64Array(2 ** (attributeCount - depth)),
    sums: new Float64Array(2 ** (attributeCount - depth)),
  }));
  tables[0]?.weights.set(weights);
  tables[0]?.sums.set(sums);
  const visit = (mask: number, depth: number, from: number): void => {
    const table = tables[depth];
    const child = tables[depth + 1];
    if (table === undefined) return;
    variances[mask] = tableVariance(table, mean);
    if (child === undefined) return;
    // Attribute k sits at the position of the attributes of the mask below it, in the table's compressed index.
    let position = 0;
    for (let k = 0; k < attributeCount; k++) {
      if (((mask >> k) & 1) === 0) continue;
      if (k >= from) {
        sumOut(table, child, position);
        visit(mask & ~(1 << k), depth + 1, k + 1);
      }
      position++;
    }
  };
  visit(weights.length - 1, 0, 0);
  return variances;
}

/** Returns Σ_l W_l (S_l / W_l - mean)² over the groups of a marginal table that have weight. */
function tableVariance(table: { readonly weights: Float64Array; readonly sums: Float64Array }, mean: number): number {
  let variance = 0;
  for (let g = 0; g < table.weights.length; g++) {
    const weight = table.weights[g] ?? 0;
    if (weight > 0) variance += weight * ((table.sums[g] ?? 0) / weight - mean) ** 2;
  }
  return variance;
}

/** Writes into `child` the marginal table of `table` with the attribute at bit `position` of its index summed out. */
function sumOut(
  table: { readonly weights: Float64Array; readonly sums: Float64Array },
  child: { readonly weights: Float64Array; readonly sums: Float64Array },
  position: number,
): void {
  const low = (1 << position) - 1;
  for (let g = 0; g < child.weights.length; g++) {
    const absent = (g & low) | ((g & ~low) << 1);
    const present = absent | (1 << position);
    child.weights[g] = (table.weights[absent] ?? 0) + (table.weights[present] ?? 0);
    child.sums[g] = (table.sums[absent] ?? 0) + (table.sums[present] ?? 0);
  }
}

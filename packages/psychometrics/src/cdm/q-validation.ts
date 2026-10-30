import { checkOptions, checkUnitInterval } from "@measurekit/core";
import type { ResponseMatrix } from "../data/response-matrix.js";
import { type CdmCalibration, calibratedModel, forEachPosterior } from "./calibration.js";
import { profilesOf } from "./q-matrix.js";

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
 * on ties.
 *
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
  const candidates = profilesOf(design.attributes.length).slice(1);
  const weights = state.proportions;
  return calibration.items.map((item, j) => {
    const success = Float64Array.from({ length: profileCount }, (_, c) => {
      return ((correct[j]?.[c] ?? 0) + SMOOTHING) / ((persons[j]?.[c] ?? 0) + 2 * SMOOTHING);
    });
    const variances = candidates.map((q) => groupVariance(q, design.profiles, success, weights));
    const full = variances.at(-1) ?? 0;
    if (!(full > 0)) return { id: item.id, attributes: item.attributes, suggested: item.attributes, pvaf: null };
    const pvaf = variances.map((value) => value / full);
    let best = candidates.length - 1;
    for (let size = 1; size <= design.attributes.length; size++) {
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

/** Returns Σ_l w_l (P̄_l - P̄)² over the latent groups of q-vector q, the profiles that agree on its attributes. */
function groupVariance(
  q: readonly number[],
  profiles: readonly (readonly number[])[],
  success: Float64Array,
  weights: Float64Array,
): number {
  const groups = new Map<string, { weight: number; sum: number }>();
  let mean = 0;
  profiles.forEach((profile, c) => {
    const key = profile.map((value, k) => (q[k] === 1 ? value : 0)).join("");
    const w = weights[c] ?? 0;
    const group = groups.get(key) ?? { weight: 0, sum: 0 };
    group.weight += w;
    group.sum += w * (success[c] ?? 0);
    groups.set(key, group);
    mean += w * (success[c] ?? 0);
  });
  let variance = 0;
  for (const { weight, sum } of groups.values()) variance += weight * (sum / weight - mean) ** 2;
  return variance;
}

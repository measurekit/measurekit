import { DataError } from "@measurekit/core";
import { pseudoInverse } from "../math/pseudo-inverse.js";
import { type CdmState, type DichotomousRows, forEachPosterior } from "./expectation.js";
import type { CdmDesign } from "./q-matrix.js";

/**
 * The most success probabilities, over all items, that the incomplete information holds: its p² cells stay within 1e7,
 * 80 MB, as an item of k attributes has 2^k of them.
 */
const MAX_INCOMPLETE_PROBABILITIES = 3162;

/**
 * Checks that the incomplete information of a design's items fits the limit of its size.
 *
 * @throws `DataError` when the items have more than {@link MAX_INCOMPLETE_PROBABILITIES} success probabilities.
 */
export function checkIncompleteSize(design: CdmDesign): void {
  const probabilities = design.itemAttributes.reduce((sum, indices) => sum + 2 ** indices.length, 0);
  if (probabilities > MAX_INCOMPLETE_PROBABILITIES) {
    throw new DataError(
      `The option standardErrors "incomplete" takes at most ${MAX_INCOMPLETE_PROBABILITIES} success probabilities over all items, 2^k for an item of k attributes, found ${probabilities}.`,
    );
  }
}

/**
 * Returns the covariance matrix of each item's success probabilities, each stored by rows, from the outer product of
 * the gradient: the inverse of the sum over persons of the outer products of their scores for the success probabilities
 * of all items, as GDINA's `SE.type = 2`, its incomplete information.
 *
 * @remarks
 * A person's score for the success probability P of a latent group of an item is the posterior probability that the
 * person belongs to the group, times (x - P) / (P (1 - P)) for the response x; a missing response scores 0. The
 * information is singular when a latent group has no expected persons, or when there are fewer persons than
 * parameters, and its Moore-Penrose inverse is taken, as GDINA's.
 */
export function incompleteCovariances(data: DichotomousRows, design: CdmDesign, state: CdmState): Float64Array[] {
  const sizes = state.probabilities.map((probabilities) => probabilities.length);
  const starts = sizes.map((_, j) => sizes.slice(0, j).reduce((sum, size) => sum + size, 0));
  const total = sizes.reduce((sum, size) => sum + size, 0);
  const information = new Float64Array(total * total);
  const score = new Float64Array(total);
  const mass = new Float64Array(Math.max(...sizes));
  forEachPosterior(data, design, state, (person, posterior) => {
    score.fill(0);
    state.probabilities.forEach((probabilities, j) => {
      const response = data.responses[person * data.itemCount + j] ?? -1;
      if (response < 0) return;
      groupMasses(posterior, design.groupOf[j], mass);
      probabilities.forEach((p, group) => {
        score[(starts[j] ?? 0) + group] = ((mass[group] ?? 0) * (response - p)) / (p * (1 - p));
      });
    });
    addOuterProduct(information, score, total);
  });
  mirrorUpperTriangle(information, total);
  const covariance = pseudoInverse(information, total);
  return sizes.map((size, j) => {
    const block = new Float64Array(size * size);
    for (let a = 0; a < size; a++) {
      for (let b = 0; b < size; b++)
        block[a * size + b] = covariance[((starts[j] ?? 0) + a) * total + (starts[j] ?? 0) + b] ?? 0;
    }
    return block;
  });
}

/** Writes into `mass` the sum of the posterior over the profiles of each latent group, `groupOf` giving a profile's. */
function groupMasses(posterior: Float64Array, groupOf: Int32Array | undefined, mass: Float64Array): void {
  mass.fill(0);
  posterior.forEach((probability, profile) => {
    const group = groupOf?.[profile] ?? 0;
    mass[group] = (mass[group] ?? 0) + probability;
  });
}

/** Adds the outer product of `score` with itself to the upper triangle of the `size` by `size` matrix `information`. */
function addOuterProduct(information: Float64Array, score: Float64Array, size: number): void {
  for (let a = 0; a < size; a++) {
    const scoreA = score[a] ?? 0;
    if (scoreA === 0) continue;
    const row = a * size;
    for (let b = a; b < size; b++) information[row + b] = (information[row + b] ?? 0) + scoreA * (score[b] ?? 0);
  }
}

/** Copies the upper triangle of the `size` by `size` matrix `information` into its lower triangle. */
function mirrorUpperTriangle(information: Float64Array, size: number): void {
  for (let a = 0; a < size; a++) {
    for (let b = a + 1; b < size; b++) information[b * size + a] = information[a * size + b] ?? 0;
  }
}

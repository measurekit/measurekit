import type { Random } from "@measurekit/core";
import { logProbability } from "../irt/probability.js";
import type { CompressedResponses } from "../irt/responses.js";
import type { ItemState } from "./calibration-setup.js";
import type { Correlations } from "./mhrm-latent.js";

/**
 * The persons who answered an item and their responses, by item: the form MH-RM sums over, as the draws of a person
 * enter every item the person answered.
 */
export interface ItemResponses {
  /** The start of each item's responses in `respondents` and `codes`, and their end as the last entry. */
  readonly starts: Int32Array;
  /** The respondent of each response: the index among the persons with at least one response. */
  readonly respondents: Int32Array;
  /** Each response. */
  readonly codes: Int32Array;
  /** The number of persons with at least one response. */
  readonly count: number;
}

/** Returns the responses by item of the persons with at least one response. */
export function responsesByItem(data: CompressedResponses): ItemResponses {
  const personCount = data.offsets.length - 1;
  const { respondentOf, count } = respondentIndex(data);
  const starts = cumulativeCounts(data.items, data.itemIds.length);
  const fill = Int32Array.from(starts);
  const respondents = new Int32Array(data.items.length);
  const codes = new Int32Array(data.items.length);
  for (let person = 0; person < personCount; person++) {
    for (let r = data.offsets[person] ?? 0; r < (data.offsets[person + 1] ?? 0); r++) {
      const item = data.items[r] ?? 0;
      const slot = fill[item] ?? 0;
      fill[item] = slot + 1;
      respondents[slot] = respondentOf[person] ?? 0;
      codes[slot] = data.codes[r] ?? 0;
    }
  }
  return { starts, respondents, codes, count };
}

/** Returns the index among the respondents of each person, -1 for a person without a response, and their number. */
function respondentIndex(data: CompressedResponses): { respondentOf: Int32Array; count: number } {
  const respondentOf = new Int32Array(data.offsets.length - 1).fill(-1);
  let count = 0;
  for (let person = 0; person < respondentOf.length; person++) {
    if ((data.offsets[person + 1] ?? 0) > (data.offsets[person] ?? 0)) respondentOf[person] = count++;
  }
  return { respondentOf, count };
}

/** Returns the start of each of `size` groups when the entries of `groups` are listed by group, and the end as the last entry. */
function cumulativeCounts(groups: ArrayLike<number>, size: number): Int32Array {
  const starts = new Int32Array(size + 1);
  for (let k = 0; k < groups.length; k++) starts[(groups[k] ?? 0) + 1] = (starts[(groups[k] ?? 0) + 1] ?? 0) + 1;
  for (let g = 0; g < size; g++) starts[g + 1] = (starts[g + 1] ?? 0) + (starts[g] ?? 0);
  return starts;
}

/** The responses by respondent: for each, the indices of its responses in the item-major order of {@link ItemResponses}. */
export interface PersonResponses {
  /** The start of each respondent's responses in `responses`, and their end as the last entry. */
  readonly starts: Int32Array;
  /** The item-major index of each response, by respondent, in item order. */
  readonly responses: Int32Array;
  /** The item of each response, by its item-major index. */
  readonly itemOf: Int32Array;
}

/** Returns the responses of each respondent, with the item of each response. */
export function responsesByPerson(data: ItemResponses): PersonResponses {
  const itemOf = new Int32Array(data.respondents.length);
  for (let i = 0; i < data.starts.length - 1; i++) itemOf.fill(i, data.starts[i] ?? 0, data.starts[i + 1] ?? 0);
  const starts = cumulativeCounts(data.respondents, data.count);
  const fill = Int32Array.from(starts);
  const responses = new Int32Array(data.respondents.length);
  // Item-major order lists a person's responses in item order.
  data.respondents.forEach((person, r) => {
    responses[fill[person] ?? 0] = r;
    fill[person] = (fill[person] ?? 0) + 1;
  });
  return { starts, responses, itemOf };
}

/**
 * The Metropolis-Hastings chain of the persons' thetas: one draw per person from the posterior given the items'
 * current parameters, advanced by random-walk steps.
 */
export interface Chain {
  /** The draws, by rows of `dimensions` entries, one row per respondent. */
  readonly theta: Float64Array;
  /** The number of respondents. */
  readonly respondents: number;
  /** Recomputes the log densities of the draws, after the items' parameters or the correlations changed. */
  refresh(): void;
  /** Makes one Metropolis-Hastings step of every draw with normal proposals of variance `variance`, and returns the share accepted. */
  sweep(random: Random, variance: number): number;
}

/**
 * Returns the chain of the persons who answered the items, with every theta at 0.
 *
 * @remarks
 * The target of a person's draw is proportional to Π_j P_j(x_j | θ) N(θ; 0, Φ), over the items the person answered. The
 * proposals move all dimensions at once by independent normal steps of one variance.
 */
export function createChain<TParameters>(
  items: readonly ItemState<TParameters>[],
  data: ItemResponses,
  correlations: Correlations,
): Chain {
  const { dimensions, precision } = correlations;
  const respondents = data.count;
  const theta = new Float64Array(respondents * dimensions);
  const proposal = new Float64Array(respondents * dimensions);
  const logLikelihood = new Float64Array(respondents);
  const proposedLogLikelihood = new Float64Array(respondents);
  const logPrior = new Float64Array(respondents);
  const point = new Float64Array(dimensions);
  const probabilities = new Float64Array(items.reduce((most, item) => Math.max(most, item.categoryCount), 0));

  /** Adds ln P(x | θ) of every response, at the thetas in `at`, to `out`. */
  const addLogLikelihood = (at: Float64Array, out: Float64Array): void => {
    items.forEach((item, i) => {
      for (let r = data.starts[i] ?? 0; r < (data.starts[i + 1] ?? 0); r++) {
        const person = data.respondents[r] ?? 0;
        for (let d = 0; d < dimensions; d++) point[d] = at[person * dimensions + d] ?? 0;
        item.model.probabilities(item.parameters, point, probabilities);
        out[person] = (out[person] ?? 0) + logProbability(probabilities[data.codes[r] ?? 0] ?? 0);
      }
    });
  };
  /** Returns -θᵀ Ω θ / 2 of person p's row of `at`. */
  const kernel = (at: Float64Array, person: number): number => {
    let sum = 0;
    for (let i = 0; i < dimensions; i++) {
      for (let j = 0; j < dimensions; j++) {
        sum +=
          (at[person * dimensions + i] ?? 0) *
          (precision[i * dimensions + j] ?? 0) *
          (at[person * dimensions + j] ?? 0);
      }
    }
    return -sum / 2;
  };

  /** Moves person p's draw to the proposal, whose log likelihood and log prior are known. */
  const accept = (p: number, proposedPrior: number): void => {
    for (let d = 0; d < dimensions; d++) theta[p * dimensions + d] = proposal[p * dimensions + d] ?? 0;
    logLikelihood[p] = proposedLogLikelihood[p] ?? 0;
    logPrior[p] = proposedPrior;
  };

  return {
    theta,
    respondents,
    refresh() {
      logLikelihood.fill(0);
      addLogLikelihood(theta, logLikelihood);
      for (let p = 0; p < respondents; p++) logPrior[p] = kernel(theta, p);
    },
    sweep(random, variance) {
      const spread = Math.sqrt(variance);
      for (let k = 0; k < proposal.length; k++) proposal[k] = (theta[k] ?? 0) + spread * random.normal();
      proposedLogLikelihood.fill(0);
      addLogLikelihood(proposal, proposedLogLikelihood);
      let accepted = 0;
      for (let p = 0; p < respondents; p++) {
        const proposedPrior = kernel(proposal, p);
        const change = (proposedLogLikelihood[p] ?? 0) + proposedPrior - (logLikelihood[p] ?? 0) - (logPrior[p] ?? 0);
        if (Math.log(random.uniform()) >= change) continue;
        accepted++;
        accept(p, proposedPrior);
      }
      return accepted / respondents;
    },
  };
}

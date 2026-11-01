import { normalizeLogs } from "@measurekit/core";
import { logProbability } from "../irt/internal.js";
import type { CdmDesign } from "./q-matrix.js";

/** Responses by person, then item: 0, 1, or -1 when missing. */
export interface DichotomousRows {
  readonly itemCount: number;
  readonly responses: Int8Array;
  /** The number of persons with at least one response. */
  readonly respondents: number;
  /** Whether every person with a response answered every item. */
  readonly complete: boolean;
}

/** The parameters during EM: each item's success probability per latent group, and the profile proportions. */
export interface CdmState {
  readonly probabilities: readonly Float64Array[];
  readonly proportions: Float64Array;
}

/** The expected counts of an E-step: correct responses and persons per item group, persons per profile. */
export interface ExpectedCounts {
  readonly correct: readonly Float64Array[];
  readonly persons: readonly Float64Array[];
  readonly profiles: Float64Array;
  readonly logLikelihood: number;
}

/**
 * Runs the E-step: each person's posterior over the profiles, from the proportions and the success probabilities of
 * the answered items, gives the expected counts and the marginal log-likelihood.
 *
 * @remarks
 * A profile's expected correct responses to an item are the sum of the posteriors of the persons who answered it
 * correctly, and its expected persons the sum over those who answered it, which for complete data is the profile's
 * expected persons: the counts of a latent group are those of its profiles, added once after the persons.
 */
export function expectedCounts(data: DichotomousRows, design: CdmDesign, state: CdmState): ExpectedCounts {
  const { itemCount, responses, complete } = data;
  const profileCount = design.profiles.length;
  const profiles = new Float64Array(profileCount);
  const correct = new Float64Array(itemCount * profileCount);
  const answered = complete ? null : new Float64Array(itemCount * profileCount);
  let logLikelihood = 0;
  forEachPosterior(data, design, state, (person, posterior, logMarginal) => {
    logLikelihood += logMarginal;
    addTo(profiles, 0, posterior);
    for (let j = 0; j < itemCount; j++) {
      const response = responses[person * itemCount + j] ?? -1;
      if (response === 1) addTo(correct, j * profileCount, posterior);
      if (response >= 0 && answered !== null) addTo(answered, j * profileCount, posterior);
    }
  });
  return {
    persons: state.probabilities.map((p, j) =>
      sumByGroup(answered ?? profiles, answered === null ? 0 : j * profileCount, design.groupOf[j], p.length),
    ),
    correct: state.probabilities.map((p, j) => sumByGroup(correct, j * profileCount, design.groupOf[j], p.length)),
    profiles,
    logLikelihood,
  };
}

/** Adds `values` to the entries of `counts` from `start`. */
function addTo(counts: Float64Array, start: number, values: Float64Array): void {
  for (let c = 0; c < values.length; c++) counts[start + c] = (counts[start + c] ?? 0) + (values[c] ?? 0);
}

/** Returns, for each of `groups` latent groups, the sum of the counts of its profiles, which `counts` holds from `start`. */
function sumByGroup(
  counts: Float64Array,
  start: number,
  groupOf: Int32Array | undefined,
  groups: number,
): Float64Array {
  const sums = new Float64Array(groups);
  for (let c = 0; c < (groupOf?.length ?? 0); c++) {
    const group = groupOf?.[c] ?? 0;
    sums[group] = (sums[group] ?? 0) + (counts[start + c] ?? 0);
  }
  return sums;
}

/** What the log-likelihood of a person's responses over the profiles is built from. */
interface LogLikelihoodTables {
  /** Each profile's log proportion plus the log probability of an incorrect response to every item. */
  readonly base: Float64Array;
  /** Each item's rise in the log-likelihood of each profile when the response is correct, by item, then profile. */
  readonly rises: Float64Array;
  /** Each item's log probability of an incorrect response in each latent group. */
  readonly incorrect: readonly Float64Array[];
}

/**
 * Returns the tables of the log-likelihood of a response pattern: the pattern of incorrect responses to every item
 * (`base`), then one `rises` row added for each correct response, and the item's `incorrect` row taken away for each
 * missing one. A response's probability enters as at least `MIN_PROBABILITY`, so that responses no profile can give
 * still have a posterior.
 */
function logLikelihoodTables(design: CdmDesign, state: CdmState): LogLikelihoodTables {
  const profileCount = design.profiles.length;
  const base = state.proportions.map(Math.log);
  const rises = new Float64Array(state.probabilities.length * profileCount);
  const incorrect = state.probabilities.map((p, j) => {
    const log = p.map((value) => logProbability(1 - value));
    const logCorrect = p.map(logProbability);
    const groupOf = design.groupOf[j] ?? new Int32Array(profileCount);
    for (let c = 0; c < profileCount; c++) {
      const group = groupOf[c] ?? 0;
      base[c] = (base[c] ?? 0) + (log[group] ?? 0);
      rises[j * profileCount + c] = (logCorrect[group] ?? 0) - (log[group] ?? 0);
    }
    return log;
  });
  return { base, rises, incorrect };
}

/**
 * Calls `visit` with each person who answered an item, the posterior over the profiles and the log of the person's
 * marginal likelihood. `posterior` is reused between calls.
 */
export function forEachPosterior(
  data: DichotomousRows,
  design: CdmDesign,
  state: CdmState,
  visit: (person: number, posterior: Float64Array, logMarginal: number) => void,
): void {
  const { itemCount, responses } = data;
  const tables = logLikelihoodTables(design, state);
  const posterior = new Float64Array(design.profiles.length);
  for (let person = 0; person < responses.length / itemCount; person++) {
    if (setLogPosterior(posterior, tables, design, responses, person * itemCount)) {
      visit(person, posterior, normalizeLogs(posterior));
    }
  }
}

/**
 * Writes the log of a person's posterior over the profiles, before normalization, into `posterior`, from the person's
 * responses to the items, which `responses` holds from `start`. Returns whether the person answered an item.
 */
function setLogPosterior(
  posterior: Float64Array,
  { base, rises, incorrect }: LogLikelihoodTables,
  design: CdmDesign,
  responses: Int8Array,
  start: number,
): boolean {
  const profileCount = posterior.length;
  posterior.set(base);
  let answered = false;
  for (let j = 0; j < incorrect.length; j++) {
    const response = responses[start + j] ?? -1;
    if (response >= 0) answered = true;
    if (response === 1) {
      for (let c = 0; c < profileCount; c++) posterior[c] = (posterior[c] ?? 0) + (rises[j * profileCount + c] ?? 0);
    }
    const groupOf = design.groupOf[j];
    const log = incorrect[j];
    if (response < 0 && groupOf !== undefined && log !== undefined) subtractGroupValues(posterior, log, groupOf);
  }
  return answered;
}

/** Takes `values[group(c)]` off `out[c]` for each profile c. */
function subtractGroupValues(out: Float64Array, values: Float64Array, groupOf: Int32Array): void {
  for (let c = 0; c < out.length; c++) out[c] = (out[c] ?? 0) - (values[groupOf[c] ?? 0] ?? 0);
}

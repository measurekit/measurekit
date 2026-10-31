import { DataError, normalizeLogs, type Stopping } from "@measurekit/core";
import { expectationMaximization } from "../math/expectation-maximization.js";
import { minimum } from "../math/extremes.js";

/** A learner's answers to one item: the item, the states that master it, the count and the sum of outcomes. */
export interface Row {
  readonly item: number;
  readonly mask: Uint8Array;
  readonly count: number;
  readonly sum: number;
}

/** Learners with the same answers, weighted by their number. */
export interface Group {
  readonly weight: number;
  readonly rows: readonly Row[];
}

/** BLIM parameters in arrays: the states' probabilities and each item's guess and slip. */
export interface Blim {
  readonly probabilities: Float64Array;
  readonly guess: Float64Array;
  readonly slip: Float64Array;
}

/** The outcome of a BLIM fit. */
export interface BlimFit {
  readonly parameters: Blim;
  readonly logLikelihood: number;
  readonly iterations: number;
  readonly converged: boolean;
}

/**
 * Writes into `out` the posterior of a group's states and returns the log of its likelihood: ln Σ_K π(K) L(K), with
 * L(K) the product over rows of (1 - slip)^y slip^(n - y) where K masters the item and guess^y (1 - guess)^(n - y)
 * where it does not. Each state's log-likelihood is summed on its own, so a rate of 0 or 1 excludes states exactly.
 */
function posterior(group: Group, parameters: Blim, logPrior: Float64Array, out: Float64Array): number {
  out.fill(0);
  for (const row of group.rows) addRow(row, parameters, out);
  for (let k = 0; k < out.length; k++) out[k] = (logPrior[k] ?? 0) + (out[k] ?? 0);
  return normalizeLogs(out);
}

/** Adds a row's log-likelihood to each state's. */
function addRow({ item, mask, count, sum }: Row, parameters: Blim, out: Float64Array): void {
  const guess = parameters.guess[item] ?? 0;
  const slip = parameters.slip[item] ?? 0;
  const known = xlogy(sum, 1 - slip) + xlogy(count - sum, slip);
  const unknown = xlogy(sum, guess) + xlogy(count - sum, 1 - guess);
  for (let k = 0; k < out.length; k++) out[k] = (out[k] ?? 0) + (mask[k] === 1 ? known : unknown);
}

/**
 * The bound that keeps estimated guess and slip from 0 and 1, as pks's `blim`, so that every outcome keeps a positive
 * probability.
 */
const MIN_RATE = 1e-6;

/** How far above the least discrepancy a state may be and still tie with it, for the rounding of the sums. */
const TIE_TOLERANCE = 1e-9;

/** The expected counts of an E-step, summed over the groups as each one's posterior is found. */
interface Counts {
  /** The states' summed posteriors, weighted by the groups' learners. */
  readonly probabilities: Float64Array;
  /** Four per item, as {@link addGroup} adds them. */
  readonly sums: Float64Array;
  /** The number of learners. */
  total: number;
}

function emptyCounts(parameters: Blim): Counts {
  return {
    probabilities: new Float64Array(parameters.probabilities.length),
    sums: new Float64Array(4 * parameters.guess.length),
    total: 0,
  };
}

/**
 * Adds a group's posterior to the counts: to each state's sum, and per item the expected wrong answers and answers
 * where the state masters the item, and right answers and answers where it does not. Both masses are summed, as
 * 1 - mastered can round below 0.
 */
function addGroup(counts: Counts, group: Group, posterior: Float64Array): void {
  const { probabilities } = counts;
  counts.total += group.weight;
  for (let k = 0; k < posterior.length; k++) {
    probabilities[k] = (probabilities[k] ?? 0) + group.weight * (posterior[k] ?? 0);
  }
  for (const row of group.rows) addRowCounts(counts.sums, row, group.weight, posterior);
}

function addRowCounts(
  sums: Float64Array,
  { item, mask, count, sum }: Row,
  weight: number,
  posterior: Float64Array,
): void {
  let mastered = 0;
  let unmastered = 0;
  for (let k = 0; k < posterior.length; k++) {
    if (mask[k] === 1) mastered += posterior[k] ?? 0;
    else unmastered += posterior[k] ?? 0;
  }
  const at = 4 * item;
  sums[at] = (sums[at] ?? 0) + weight * mastered * (count - sum);
  sums[at + 1] = (sums[at + 1] ?? 0) + weight * mastered * count;
  sums[at + 2] = (sums[at + 2] ?? 0) + weight * unmastered * sum;
  sums[at + 3] = (sums[at + 3] ?? 0) + weight * unmastered * count;
}

/**
 * Returns the parameters that maximize the expected complete-data log-likelihood given the counts: the states' mean
 * posterior; an item's slip, the expected wrong answers where its state masters it over the answers there, and its
 * guess, the expected right answers where it does not over the answers there, each kept 1e-6 from 0 and 1. A rate
 * without expected answers keeps its value.
 */
function maximize(counts: Counts, previous: Blim): Blim {
  const { sums, total } = counts;
  const probabilities = counts.probabilities.map((sum) => sum / total);
  const ratio = (at: number, kept: number): number => {
    const answers = sums[at + 1] ?? 0;
    return answers > 0 ? Math.min(Math.max((sums[at] ?? 0) / answers, MIN_RATE), 1 - MIN_RATE) : kept;
  };
  return {
    probabilities,
    slip: previous.slip.map((value, i) => ratio(4 * i, value)),
    guess: previous.guess.map((value, i) => ratio(4 * i + 2, value)),
  };
}

/**
 * Fits a BLIM by maximum likelihood: EM from `start` until the log-likelihood changes by at most the tolerance. Each
 * E-step adds a group's posterior to the counts as it is found, so memory does not grow with the groups.
 *
 * @throws `DataError` when a group's answers have probability 0 under a step's parameters.
 */
export function maximumLikelihood(groups: readonly Group[], start: Blim, options: Stopping): BlimFit {
  const scratch = new Float64Array(start.probabilities.length);
  return expectationMaximization(
    start,
    (parameters) => {
      const counts = emptyCounts(parameters);
      const logLikelihood = sumLogLikelihoods(groups, parameters, scratch, (group) => addGroup(counts, group, scratch));
      return { logLikelihood, parameters: maximize(counts, parameters) };
    },
    options,
  );
}

/**
 * Fits a BLIM by minimum discrepancy: each group is spread evenly over the states that need the fewest careless errors
 * and lucky guesses to give its answers, and the parameters are those states' frequencies and the errors they need. One
 * pass; the log-likelihood is that of the estimates.
 *
 * @throws `DataError` when a group's answers have probability 0 under the estimates.
 * @see {@link https://doi.org/10.1016/j.endm.2013.05.145 | Heller and Wickelmaier (2013)}
 */
export function minimumDiscrepancy(groups: readonly Group[], start: Blim): BlimFit {
  const scratch = new Float64Array(start.probabilities.length);
  const counts = emptyCounts(start);
  for (const group of groups) {
    nearestStates(group, scratch);
    addGroup(counts, group, scratch);
  }
  const parameters = maximize(counts, start);
  const logLikelihood = sumLogLikelihoods(groups, parameters, scratch, () => undefined);
  return { parameters, logLikelihood, iterations: 1, converged: true };
}

/**
 * Writes into `out` an even spread over the states of least discrepancy from a group's answers: the careless errors
 * and lucky guesses a state needs to give them.
 */
function nearestStates(group: Group, out: Float64Array): void {
  out.fill(0);
  for (const { mask, count, sum } of group.rows) {
    for (let k = 0; k < out.length; k++) out[k] = (out[k] ?? 0) + (mask[k] === 1 ? count - sum : sum);
  }
  const least = minimum(out);
  let ties = 0;
  for (let k = 0; k < out.length; k++) {
    out[k] = (out[k] ?? 0) - least <= TIE_TOLERANCE ? 1 : 0;
    ties += out[k] ?? 0;
  }
  for (let k = 0; k < out.length; k++) out[k] = (out[k] ?? 0) / ties;
}

/**
 * Returns the log-likelihood of the groups, finding each one's posterior in `scratch` and passing the group to `use`
 * while it is there.
 *
 * @throws `DataError` when a group's answers have probability 0.
 */
function sumLogLikelihoods(
  groups: readonly Group[],
  parameters: Blim,
  scratch: Float64Array,
  use: (group: Group) => void,
): number {
  const logPrior = parameters.probabilities.map(Math.log);
  let logLikelihood = 0;
  for (const group of groups) {
    const value = posterior(group, parameters, logPrior, scratch);
    if (value === Number.NEGATIVE_INFINITY) {
      throw new DataError(
        "A learner's outcomes have probability 0 under the parameters, as with a guess or slip of 0.",
      );
    }
    logLikelihood += group.weight * value;
    use(group);
  }
  return logLikelihood;
}

/** Returns x ln y, 0 when x is 0. */
function xlogy(x: number, y: number): number {
  return x === 0 ? 0 : x * Math.log(y);
}
